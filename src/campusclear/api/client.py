"""老师专用客户端接口。

定位：这是「学生处老师」用的客户端，不是运维面板。
它只做两件事——**实时看**（轻量增强）与**回放查**（重量增强）。

关键约定（应你的要求）：**所有流都由服务器提供，客户端只消费**。
- 实时：服务器把增强帧编码成 MJPEG（``multipart/x-mixed-replace``），
  客户端用普通 ``<img>`` 就能看，浏览器原生支持、零依赖、离线可用；
- 回放：老师选一段录像 → 服务器起一个增强任务（重量档，慢慢跑）→
  产出 mp4，客户端用 ``<video>`` 播放。

认证：这是「老师自己电脑上的单机客户端」——服务只绑定 127.0.0.1，
访问者只能是本机，因此**本组接口不做登录**（单用户本地工具无需账号体系）。
运维面板 ``/api/v1`` 的 JWT/RBAC 鉴权保持原样，与客户端互不影响。

安全：录像名一律做 basename 化并校验落在录像目录内，杜绝路径穿越。
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..core.camera import CameraConfig
from ..core.demo_source import DemoScene
from ..core.modes import EnhanceMode, build_for_mode, policy_for
from ..core.planner import build_plan
from ..core.quality import QualityAnalyzer
from ..core.snapshot import enhance_still, list_presets
from ..core.sources import (
    build_input_args,
    describe_methods,
    detect_scheme,
    vendor_live_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".ts"}

# 演示用合成源（无摄像头时让客户端也有画面）。仅用于演示，不能当画质依据。
_demo_scene = DemoScene()
_demo_enhancer = None
_demo_policy = policy_for(EnhanceMode.REALTIME)


# --------------------------------------------------------------------------- #
# 依赖与工具
# --------------------------------------------------------------------------- #
def _settings(request: Request):
    return request.app.state.settings


def _supervisor(request: Request):
    return request.app.state.supervisor


def local_user(request: Request) -> dict:
    """本机客户端：**不做登录**。

    这是「老师自己电脑上的单机程序」，服务只绑定 127.0.0.1，访问者只能是本机。
    对单用户本地工具来说，登录界面只是负担，因此客户端这组接口不做鉴权。
    运维面板 /api/v1 的鉴权（JWT/RBAC）保持原样，不受影响。
    """
    return {
        "username": "local",
        "role": "admin",
        "client_ip": request.client.host if request.client else "",
    }


def _recordings_dir(settings) -> Path:
    path = Path(settings.data_dir) / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _out_dir(settings) -> Path:
    path = Path(settings.data_dir) / "client_out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_recording(settings, name: str) -> Path:
    """把录像名解析为目录内的真实路径，防止路径穿越。"""
    base = _recordings_dir(settings).resolve()
    candidate = (base / Path(name).name).resolve()
    if candidate.parent != base or not candidate.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"录像不存在: {name}")
    return candidate


# --------------------------------------------------------------------------- #
# 实时
# --------------------------------------------------------------------------- #
@router.get("/client/streams", tags=["client"])
def list_streams(request: Request, user: dict = Depends(local_user)) -> dict:
    """返回客户端可看的实时源：真实通道 + 一个演示源。

    每路附带 ``view_port``：MJPEG 长连接受浏览器「同源 ~6 并发」限制，
    多端口轮转分配可以把观看流量摊开（详见 config.view_port_count 注释）。
    前端用 ``主机:view_port`` 拼图片地址；``view_port`` 为 null 时退回主端口。
    """
    settings = _settings(request)
    supervisor = _supervisor(request)
    view_ports = list(getattr(settings, "active_view_ports", []) or [])
    streams = []

    def _view_port() -> int | None:
        if not view_ports:
            return None
        return view_ports[len(streams) % len(view_ports)]

    for cam in request.app.state.store.list_cameras():
        pipeline = supervisor.get_pipeline(cam.id)
        streams.append({
            "id": cam.id,
            "name": cam.name,
            "location": cam.location,
            "kind": "camera",
            "tags": list(cam.tags),
            "sub": _is_sub_stream(cam.source_rtsp),
            "online": bool(pipeline and pipeline.state.connected and pipeline.state.running),
            "url": f"{settings.api_prefix}/client/live/{cam.id}.mjpeg",
            "view_port": _view_port(),
        })
    # 演示源永远在线，保证没有摄像头时客户端不是一片空白
    streams.append({
        "id": "demo",
        "name": "演示画面（合成）",
        "location": "本机合成",
        "kind": "demo",
        "online": True,
        "url": f"{settings.api_prefix}/client/live/demo.mjpeg",
        "view_port": _view_port(),
    })
    return {"streams": streams, "fps": 15}


def _demo_jpeg(enhance: bool, elapsed: float, analyzer: QualityAnalyzer) -> bytes | None:
    """生成一帧演示画面的 JPEG（CPU 密集部分，放在线程池里跑）。"""
    global _demo_enhancer
    if enhance and _demo_enhancer is None:
        _demo_enhancer, _, _ = build_for_mode(EnhanceMode.REALTIME)
    frame = _demo_scene.frame(elapsed)
    out = frame
    if enhance:
        plan = build_plan(analyzer.analyze(frame), _demo_policy)
        out = _demo_enhancer.enhance(frame, plan)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


def _mjpeg_demo(enhance: bool = True):
    """同步版演示流，保留给测试与离线抓帧。"""
    analyzer = QualityAnalyzer()
    t0 = time.time()
    interval = 1.0 / 15
    while True:
        tick = time.time()
        jpeg = _demo_jpeg(enhance, time.time() - t0, analyzer)
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(0.0, interval - (time.time() - tick)))


async def _mjpeg_demo_async(request: Request, enhance: bool = True):
    """异步版演示流：等待用 asyncio.sleep，不占用 AnyIO 工作线程。"""
    analyzer = QualityAnalyzer()
    t0 = time.time()
    interval = 1.0 / 15
    while not await request.is_disconnected():
        tick = time.time()
        jpeg = await run_in_threadpool(_demo_jpeg, enhance, time.time() - t0, analyzer)
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(max(0.0, interval - (time.time() - tick)))


async def _mjpeg_camera(request: Request, pipeline, raw: bool = False):
    """真实通道 → 该通道的最新帧（增强或原图）→ MJPEG。

    两个关键点：
    1. **即使暂时没有新帧，也把上一帧重发**。否则一旦某段时间没有新帧，
       这条 multipart 流就没有字节输出，浏览器会显示“卡住”；
    2. 用 async 生成器 + asyncio.sleep。同步生成器的每次 next() 都会占住
       AnyIO 线程池（默认 40 个），40 路 MJPEG 就能把所有同步接口饿死。
    """
    last: bytes | None = None
    interval = 1.0 / 15
    while not await request.is_disconnected():
        if not pipeline.state.running:
            # 通道已停止/删除：主动结束长连接，不能无限重发最后一帧
            break
        jpeg = await run_in_threadpool(pipeline.snapshot_jpeg_cached, raw, 85)
        if jpeg:
            last = jpeg
        if last:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
        await asyncio.sleep(interval)


def _live_response(request: Request, stream_id: str, raw: bool) -> StreamingResponse:
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    if stream_id == "demo":
        gen = _mjpeg_demo_async(request, enhance=not raw)
    else:
        pipeline = _supervisor(request).get_pipeline(stream_id)
        if pipeline is None or not pipeline.state.running:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "该通道未在运行")
        gen = _mjpeg_camera(request, pipeline, raw=raw)
    return StreamingResponse(
        gen, media_type="multipart/x-mixed-replace; boundary=frame", headers=headers
    )


@router.get("/client/live/{stream_id}.mjpeg", tags=["client"])
def live_mjpeg(stream_id: str, request: Request) -> StreamingResponse:
    """增强后的实时画面。"""
    return _live_response(request, stream_id, raw=False)


@router.get("/client/live/{stream_id}/raw.mjpeg", tags=["client"])
def live_raw_mjpeg(stream_id: str, request: Request) -> StreamingResponse:
    """未增强的原始画面（供「原图/增强」对比）。"""
    return _live_response(request, stream_id, raw=True)


# --------------------------------------------------------------------------- #
# 数据源管理（连到学校监控服务器）
# --------------------------------------------------------------------------- #
@router.get("/client/methods", tags=["client"])
def list_methods(user: dict = Depends(local_user)) -> dict:
    """返回支持的取流连接方式，供界面下拉选择。"""
    return {"methods": describe_methods()}


@router.post("/client/sources", status_code=status.HTTP_201_CREATED, tags=["client"])
def add_source(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """新增一个数据源：直接给 URL，或按厂商构造实时取流地址。"""
    store = request.app.state.store
    supervisor = _supervisor(request)

    if payload.get("brand"):
        try:
            url = vendor_live_url(
                payload["brand"],
                payload["host"],
                channel=int(payload.get("channel", 1)),
                sub=bool(payload.get("sub")),
                username=payload.get("username", ""),
                password=payload.get("password", ""),
                port=int(payload.get("port", 554)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"构造取流地址失败: {exc}") from exc
    else:
        url = (payload.get("url") or "").strip()

    if not url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "请提供取流地址，或选择厂商并填写主机")

    cid = str(payload.get("id") or f"src_{uuid.uuid4().hex[:6]}")
    # 与运维接口的 CameraCreate 保持同一约束；否则客户端可以用任意 id
    # 覆盖已有通道（包括改掉它的取流地址），或把 output_name 拼进奇怪路径。
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", cid):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "通道 id 只能包含字母、数字、_ 和 -（1-32 位）")
    if store.get_camera(cid) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"数据源 {cid} 已存在")

    camera = CameraConfig(
        id=cid,
        name=(payload.get("name") or "").strip() or cid,
        location=(payload.get("location") or "").strip(),
        source_rtsp=url,
        enabled=True,
        tags=list(payload.get("tags") or (["测试"] if payload.get("kind") == "camera" else [])),
    )
    store.upsert_camera(camera)
    store.add_audit(
        "client_add_source", username=user["username"], target=cid,
        detail=f"scheme={detect_scheme(url) or 'file'}", client_ip=user.get("client_ip", ""),
    )

    started, error = True, ""
    try:
        supervisor.start_camera(camera)
    except Exception as exc:  # noqa: BLE001 - 已入库但启动失败：界面可提示，改地址后重试
        logger.exception("通道 %s 启动失败", cid)
        started, error = False, str(exc)

    return {"id": cid, "name": camera.name, "started": started, "error": error}


def _mask_url_creds(url: str) -> str:
    """批量导入回显时抹掉 URL 里的账号密码，避免接口把凭证吐回去。"""
    if "://" not in url or "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"


@router.post("/client/sources/batch", status_code=status.HTTP_201_CREATED, tags=["client"])
def add_sources_batch(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """批量导入数据源（每行一个取流地址）。

    背景：接入学校流媒体平台时，源地址往往分布在**多个端口**上（每路一个端口
    是常见形态）。用单条添加逐个录入时，厂商构造路径只有**一个端口框**，
    一不留神几百路就全被填成同一个端口——该端口的并发会话立刻被打满，
    表现为「所有路都打不开」。本接口按行解析、**逐行保留各自的端口**，
    并按 ``CAMPUSCLEAR_START_STAGGER_SECONDS`` 错峰启动，避免连接风暴。

    payload::

        {"urls": "rtsp://...:8001/... # 1号机\\nrtsp://...:8002/... # 2号机",
         "start": true}   # start 可选，默认 true；false 只入库不启动

    行格式：``地址`` 或 ``地址 # 名称``（名称可选，# 后内容作为通道名）。
    """
    store = request.app.state.store
    supervisor = _supervisor(request)
    settings = _settings(request)

    raw = payload.get("urls")
    lines = raw.splitlines() if isinstance(raw, str) else list(raw or [])
    max_batch = 1000
    if len(lines) > max_batch:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"单次最多导入 {max_batch} 行（当前 {len(lines)} 行），请分批"
        )

    items: list[dict] = []
    cameras: list[CameraConfig] = []
    for index, line in enumerate(lines, 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        name = ""
        if " #" in text:
            text, _, tail = text.partition(" #")
            name = tail.strip()
            text = text.strip()
        if not text:
            continue
        cid = f"src_{uuid.uuid4().hex[:6]}"
        camera = CameraConfig(
            id=cid,
            name=name or f"导入-{index}",
            source_rtsp=text,
            enabled=True,
            tags=["批量导入"],
        )
        store.upsert_camera(camera)
        cameras.append(camera)
        items.append({"id": cid, "name": camera.name, "url": _mask_url_creds(text)})

    if not cameras:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有解析到有效的取流地址")

    started_count = 0
    if payload.get("start", True):
        stagger = float(getattr(settings, "start_stagger_seconds", 0.5))
        for index, camera in enumerate(cameras):
            error = ""
            try:
                supervisor.start_camera(camera)
                started_count += 1
            except Exception as exc:  # noqa: BLE001 - 单路失败不拖垮整批
                logger.exception("批量导入通道 %s 启动失败", camera.id)
                error = str(exc)
            items[index]["started"] = error == ""
            items[index]["error"] = error
            if stagger > 0 and index < len(cameras) - 1:
                time.sleep(stagger)
    else:
        for item in items:
            item["started"] = None  # 未尝试启动

    store.add_audit(
        "client_add_sources_batch", username=user["username"],
        target=f"{len(cameras)} sources", detail=f"started={started_count}",
        client_ip=user.get("client_ip", ""),
    )
    failed = sum(1 for i in items if i.get("started") is False)
    return {
        "added": len(cameras),
        "started": started_count,
        "failed": failed,
        "items": items,
    }


@router.delete("/client/sources/{source_id}", tags=["client"])
def remove_source(source_id: str, request: Request, user: dict = Depends(local_user)) -> dict:
    store = request.app.state.store
    _supervisor(request).stop_camera(source_id)
    if not store.delete_camera(source_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "数据源不存在")
    store.add_audit(
        "client_remove_source", username=user["username"], target=source_id,
        client_ip=user.get("client_ip", ""),
    )
    return {"removed": source_id}


def _is_sub_stream(url: str) -> bool:
    """判断地址当前指向主码流还是子码流（覆盖海康/大华/宇视常见形态）。"""
    if re.search(r"Streaming/Channels/\d*[23](?=[?/]|$)", url):
        return True
    lower = url.lower()
    if "realmonitor" in lower:
        return bool(re.search(r"[?&]subtype=1(?:&|$)", lower))
    return bool(re.search(r"/media/video2(?=[?/]|$)", lower))


def _stream_url_variant(url: str, sub: bool) -> str:
    """在主/子码流之间切换。

    不能要求通道号正好处在 URL 结尾：现场海康地址经常带
    ``?transportmode=unicast&profile=...``，旧实现会静默返回原地址，
    接口却回报「切换成功」。大华/宇视也一并支持。
    """
    match = re.search(r"(Streaming/Channels/)(\d+)(?=[?/]|$)", url)
    if match:
        number = match.group(2)
        if len(number) >= 2:
            number = number[:-1] + ("2" if sub else "1")
        return url[: match.start(2)] + number + url[match.end(2):]

    lower = url.lower()
    if "realmonitor" in lower:
        # 大华：subtype=0 主码流，1 子码流
        return re.sub(
            r"([?&]subtype=)[01](?=&|$)",
            lambda found: found.group(1) + ("1" if sub else "0"),
            url,
        )

    match = re.search(r"(/media/video)([12])(?=[?/]|$)", url)
    if match:
        # 宇视：video1 主码流，video2 子码流
        return url[: match.start(2)] + ("2" if sub else "1") + url[match.end(2):]
    return url


@router.post("/client/sources/{source_id}/stream", tags=["client"])
def set_source_stream(
    source_id: str, payload: dict, request: Request, user: dict = Depends(local_user)
) -> dict:
    """切换某路的**主码流 / 子码流**（实时观看用：子码流流畅，主码流更清晰）。"""
    store = request.app.state.store
    supervisor = _supervisor(request)
    camera = store.get_camera(source_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "数据源不存在")

    sub = bool(payload.get("sub"))
    camera.source_rtsp = _stream_url_variant(camera.source_rtsp, sub)
    store.upsert_camera(camera)

    started, error = True, ""
    try:
        supervisor.start_camera(camera)
    except RuntimeError as exc:
        started, error = False, str(exc)
    store.add_audit(
        "client_set_stream", username=user["username"], target=source_id,
        detail="sub" if sub else "main", client_ip=user.get("client_ip", ""),
    )
    return {"id": source_id, "sub": sub, "started": started, "error": error}


# --------------------------------------------------------------------------- #
# 截图增强（全屏时「截图并增强」）
# --------------------------------------------------------------------------- #
def _snapshots_dir(settings) -> Path:
    path = Path(settings.data_dir) / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _main_stream_url(url: str) -> str:
    """把「子码流」地址换成「主码流」（用于截图取高清帧）。"""
    return _stream_url_variant(url, sub=False)


def _grab_jpeg_from_url(url: str, ffmpeg_bin: str, timeout: float = 25.0) -> "np.ndarray | None":
    """用 ffmpeg 从指定 RTSP 地址抓**一帧**（高分辨率、用于截图）。"""
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        # 不能固定加 -rtsp_transport：源也可能是 HTTP-FLV/HLS/本地文件，
        # RTSP 私有参数会让它们直接报 "Option not found" 再白等超时。
        *build_input_args(url, rtsp_transport="tcp"),
        "-i", url,
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)


def _grab_one_frame(request: Request, stream_id: str, source: str = "pipeline") -> np.ndarray | None:
    """抓取某路当前的一帧。

    取帧优先级是**现场体验的关键**，不是随手写的：

    * 默认（``source="pipeline"``）直接从通道管线里取「最近一帧原图」。管线一直在
      解码，这一帧本来就在内存里，耗时是毫秒级。
    * 早先的实现默认去摄像头**新开一路 RTSP 拉主码流**。真实海康摄像头的并发会话
      有上限，新开连接不仅要等握手和关键帧（几秒），还会把正在预览的那一路挤出
      去（表现为「点一下截图，画面卡很久然后断流」），最坏情况要等满 25 秒超时。
    * ``source="camera"`` 保留给「高清抓帧」：确实需要原始主码流分辨率时才用，
      超时也收紧到 8 秒，失败就退回管线帧，绝不让界面干等。
    """
    if stream_id == "demo":
        return _demo_scene.frame(time.time())

    settings = _settings(request)
    camera = request.app.state.store.get_camera(stream_id)

    if source == "camera" and camera is not None:
        frame = _grab_jpeg_from_url(
            _main_stream_url(camera.source_rtsp), settings.ffmpeg_bin, timeout=8.0
        )
        if frame is not None:
            return frame

    pipeline = _supervisor(request).get_pipeline(stream_id)
    if pipeline is not None:
        jpeg = pipeline.snapshot_raw_jpeg()
        if jpeg:
            return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    # 管线没起来（通道离线/刚加进来）时才退回到直连抓帧
    if camera is not None:
        return _grab_jpeg_from_url(
            _main_stream_url(camera.source_rtsp), settings.ffmpeg_bin, timeout=8.0
        )
    return None


@router.get("/client/presets", tags=["client"])
def snapshot_presets(user: dict = Depends(local_user)) -> dict:
    """截图增强可选的预设标准。"""
    return {"presets": list_presets()}


class _StillJobs:
    """截图增强队列：**单线程串行**执行。

    为什么必须串行：一次超分（任何倍率，模型内部固定按 4× 推理）在纯 CPU 上约
    10 秒，取证档 + 1080P 源会到 40 秒以上，且推理期间会把所有核心吃满。
    如果连点几下就让请求并发，结果是「每张都更慢 + 实时预览一起卡死」，
    用户侧的观感就是整页卡住。

    串行之后：点击是即时的（只是往列表塞一张占位卡），任务在后台一张张跑，
    完成一张替换一张；排队中的卡会显示队列位置，用户知道系统在干活。
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cc-still")

    def submit(self, task) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "state": "queued",
            "created_at": time.time(),
            "elapsed_ms": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._pool.submit(self._run, job_id, task)
        return job

    def _run(self, job_id: str, task) -> None:
        job = self._jobs[job_id]
        started = time.time()
        job["state"] = "running"
        try:
            job["result"] = task()
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - 失败要留在任务状态里给前端显示
            logger.exception("截图增强失败")
            job["error"] = str(exc)
            job["state"] = "failed"
        finally:
            job["elapsed_ms"] = int((time.time() - started) * 1000)
            with self._lock:
                if job_id in self._order:
                    self._order.remove(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = dict(job)
            if job["state"] == "queued" and job_id in self._order:
                snapshot["position"] = self._order.index(job_id) + 1
            return snapshot


still_jobs = _StillJobs()


def _render_still(settings, frame, preset_key: str, prefix: str, label_format: str,
                  store, user: dict, target: str, audit_action: str,
                  extra: dict | None = None) -> dict:
    """把一帧按预设增强、存盘并返回结果字典（同步与异步两条路径共用）。"""
    output, preset = enhance_still(frame, preset_key, models_dir=settings.models_dir)
    ok, buf = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("编码失败")
    name = f"{prefix}-{preset.key}-{int(time.time())}-{uuid.uuid4().hex[:6]}.jpg"
    (_snapshots_dir(settings) / name).write_bytes(buf.tobytes())
    store.add_audit(
        audit_action, username=user["username"], target=target,
        detail=preset.key, client_ip=user.get("client_ip", ""),
    )
    result = {
        "id": name,
        "url": f"{settings.api_prefix}/client/snapshots/{name}",
        "preset": preset.key,
        "label": label_format.format(label=preset.label),
        "width": output.shape[1],
        "height": output.shape[0],
        "source_size": f"{frame.shape[1]}x{frame.shape[0]}",
    }
    if extra:
        result.update(extra)
    return result


@router.post("/client/snapshot", status_code=status.HTTP_201_CREATED, tags=["client"])
def take_snapshot(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """抓当前帧 → 按预设增强 → 存盘，返回可访问地址。

    ``wait: false`` 时改为入队并立刻返回 ``{id, state}``，前端拿 id 轮询
    ``/client/snapshot-jobs/{id}``。默认仍是同步返回，兼容旧调用与测试。
    """
    settings = _settings(request)
    stream_id = payload.get("stream_id") or "demo"
    preset_key = payload.get("preset") or "standard"
    # pipeline=复用管线内存帧（毫秒级）；camera=直连抓主码流（慢，但分辨率最高）
    source = "camera" if payload.get("source") == "camera" else "pipeline"
    frame = _grab_one_frame(request, stream_id, source)
    if frame is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "抓不到该点位的画面")

    store = request.app.state.store
    task = lambda: _render_still(  # noqa: E731 - 轻量闭包，无需具名函数
        settings, frame, preset_key, "live", "{label}", store, user, stream_id, "client_snapshot"
    )
    if payload.get("wait") is False:
        job = still_jobs.submit(task)
        return {"id": job["id"], "state": job["state"], "queued": True}
    return task()


@router.get("/client/snapshot-jobs/{job_id}", tags=["client"])
def snapshot_job(job_id: str) -> dict:
    """查询异步截图任务：queued（带队列位置）/ running / done / failed。"""
    job = still_jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return job


@router.get("/client/snapshots", tags=["client"])
def list_snapshots(request: Request, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    base = _snapshots_dir(settings)
    items = [
        {
            "name": p.name,
            "url": f"{settings.api_prefix}/client/snapshots/{p.name}",
            "mtime": int(p.stat().st_mtime),
        }
        for p in sorted(base.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    return {"snapshots": items}


@router.get("/client/snapshots/{name}", tags=["client"])
def get_snapshot(name: str, request: Request) -> FileResponse:
    base = _snapshots_dir(_settings(request)).resolve()
    path = (base / Path(name).name).resolve()
    if path.parent != base or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "截图不存在")
    return FileResponse(str(path), media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# 回放
# --------------------------------------------------------------------------- #
@router.get("/client/recordings", tags=["client"])
def list_recordings(request: Request, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    base = _recordings_dir(settings)
    items = []
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXT:
            items.append({
                "name": path.name,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "mtime": int(path.stat().st_mtime),
            })
    return {"recordings": items, "dir": str(base)}


@router.get("/client/recordings/{name}/video", tags=["client"])
def serve_recording(name: str, request: Request, user: dict = Depends(local_user)):
    """直接返回录像文件，供前端 ``<video>`` 播放（含 range/seek）。

    Starlette ``FileResponse`` 原生支持 HTTP Range 请求，HTML5 ``<video>``
    拖进度条时浏览器会发 ``Range: bytes=...``，无需自己实现分块。
    """
    path = _safe_recording(_settings(request), name)
    media = "video/mp4" if path.suffix.lower() in (".mp4", ".m4v") else "video/x-msvideo"
    return FileResponse(str(path), media_type=media, filename=name)


@router.post("/client/snapshot-recording", status_code=status.HTTP_201_CREATED, tags=["client"])
def snapshot_from_recording(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    """从**录像文件**的指定时间点抓一帧，按预设增强后存盘。

    与 ``take_snapshot``（从实时流抓帧）的区别：这里源是已落盘的录像 MP4，
    用 ``cv2.VideoCapture`` 按 ``CAP_PROP_POS_MSEC`` 精确 seek 到目标时间，
    读一帧 → ``enhance_still`` → 存盘。单帧可用最重预设（取证 4×）。

    ``wait: false`` 时入队并立刻返回，前端轮询 ``/client/snapshot-jobs/{id}``；
    默认同步返回（兼容旧调用）。seek 与推理都放在任务里，请求本身不阻塞。
    """
    settings = _settings(request)
    name = payload.get("recording", "")
    path = _safe_recording(settings, name)

    t = float(payload.get("time") or 0.0)
    preset_key = payload.get("preset") or "forensic"
    store = request.app.state.store

    def render() -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开录像: {name}")
        try:
            if t > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"在 {t:.1f}s 处抓不到帧（录像可能短于该时长）")
        return _render_still(
            settings, frame, preset_key, "rec", "{label} @ " + f"{t:.1f}s",
            store, user, name, "client_snapshot_recording",
            extra={"source_fps": round(fps, 1)},
        )

    if payload.get("wait") is False:
        job = still_jobs.submit(render)
        return {"id": job["id"], "state": job["state"], "queued": True}
    try:
        return render()
    except RuntimeError as exc:
        message = str(exc)
        code = status.HTTP_400_BAD_REQUEST if message.startswith("无法打开录像") else status.HTTP_404_NOT_FOUND
        raise HTTPException(code, message) from exc


class _Jobs:
    """极简回放任务管理：单线程排队执行，避免离开时把机器压垮。"""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cc-playback")

    def submit(self, settings, payload: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "state": "queued",
            "created_at": time.time(),
            "params": payload,
            "progress_frames": 0,
            "result": None,
            "error": None,
            "output": str(_out_dir(settings) / f"{job_id}.mp4"),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, settings, job)
        return job

    def _run(self, settings, job: dict) -> None:
        from ..core.playback import enhance_clip

        job["state"] = "running"
        params = job["params"]
        try:
            source = _safe_recording(settings, params["recording"])
            result = enhance_clip(
                source,
                job["output"],
                mode=params.get("mode", "forensic"),
                scale=params.get("scale"),
                region=params.get("region"),
                start=params.get("start"),
                end=params.get("end"),
                progress_cb=lambda n: job.__setitem__("progress_frames", n),
            )
            job["result"] = result.as_dict()
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - 任务失败要留在状态里供前端显示
            logger.exception("回放增强任务失败")
            job["error"] = str(exc)
            job["state"] = "failed"

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)


jobs = _Jobs()


@router.post("/client/jobs", status_code=status.HTTP_201_CREATED, tags=["client"])
def create_job(request: Request, payload: dict, user: dict = Depends(local_user)) -> dict:
    settings = _settings(request)
    _safe_recording(settings, payload.get("recording", ""))  # 校验存在且不含穿越
    mode = payload.get("mode", "forensic")
    if mode not in {m.value for m in EnhanceMode}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"未知档位: {mode}")
    job = jobs.submit(settings, payload)
    request.app.state.store.add_audit(
        "client_playback", username=user["username"], target=payload.get("recording", ""),
        detail=f"mode={mode}", client_ip=user.get("client_ip", ""),
    )
    return {"id": job["id"], "state": job["state"]}


@router.get("/client/jobs", tags=["client"])
def list_jobs(request: Request, user: dict = Depends(local_user)) -> dict:
    return {"jobs": jobs.list()}


@router.get("/client/jobs/{job_id}", tags=["client"])
def job_status(job_id: str, request: Request, user: dict = Depends(local_user)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在")
    return {k: v for k, v in job.items() if k != "output"}


@router.get("/client/jobs/{job_id}/result", tags=["client"])
def job_result(job_id: str, request: Request) -> FileResponse:
    job = jobs.get(job_id)
    if job is None or job["state"] != "done":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "结果不存在或尚未完成")
    return FileResponse(job["output"], media_type="video/mp4", filename=f"{job_id}.mp4")
