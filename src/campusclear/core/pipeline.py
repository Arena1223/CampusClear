"""单路视频处理管线。

线程模型（每路两条线程，刻意保持简单）：

    [FFmpeg 解码进程] → reader 线程 → 有界帧队列 → worker 线程 → [FFmpeg 编码进程]

背压与丢帧策略：
- 帧队列有界（默认 8 帧）。队列满时**丢最旧的一帧**而不是等新帧。
  实时监控里，「延迟 3 秒的完整画面」价值远低于「延迟 200ms 的当前画面」；
- 丢弃在 reader 端完成，worker 端永远不会因为读队而阻塞，保证编码端节奏稳定。

推理帧率与输出帧率解耦：
- 源流 25fps，但推理只需要 8-12fps 就能覆盖安防取证需求；
- 未到推理时机的帧直接复用上一轮增强结果，输出仍是 25fps 的连续流，
  既保持安防平台侧「有流不断」的预期，又把 GPU 负载压到 1/2 ~ 1/3。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraConfig, ResolvedCamera
from .gateway import rtsp_endpoint_reachable
from .ingest import RtspReader
from .metrics import StreamMetrics
from .output import RtspPusher
from .planner import EnhancePlan, LoadGovernor, PlannerPolicy, build_plan
from .quality import QualityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    running: bool = False
    connected: bool = False
    output_ready: bool = False


class StreamPipeline:
    """一路摄像头的完整处理管线。"""

    def __init__(
        self,
        resolved: ResolvedCamera,
        settings,
        enhancer,
        governor: LoadGovernor,
        *,
        policy: PlannerPolicy | None = None,
    ) -> None:
        self.resolved = resolved
        self.camera: CameraConfig = resolved.camera
        # 输出尺寸的**唯一口径**：一旦启动就不允许逐帧变化。
        # rawvideo 管道按固定帧长读取，ECONOMY/SAFE 档即使把 SR 倍率降到 1，
        # 也必须把最终画布补回这个倍率，否则推流端会把相邻两帧拼成一帧。
        self.output_scale = resolved.scale if resolved.enable_sr else 1
        self.settings = settings
        self.enhancer = enhancer
        self.governor = governor
        self.policy = policy or PlannerPolicy(
            enable_lowlight=resolved.enable_lowlight,
            enable_denoise=resolved.enable_denoise,
            enable_deblur=resolved.enable_deblur,
            enable_sr=resolved.enable_sr,
            base_scale=resolved.scale,
            enable_static_reuse=settings.enable_static_reuse,
        )

        self.analyzer = QualityAnalyzer(
            low_light_threshold=settings.low_light_threshold,
            noise_threshold=settings.noise_threshold,
            blur_threshold=settings.blur_threshold,
            motion_threshold=settings.motion_threshold,
            reuse_max_noise=settings.static_reuse_max_noise,
        )
        self.metrics = StreamMetrics(camera_id=self.camera.id)
        self.state = PipelineState()

        self.reader = RtspReader(
            self.camera.source_rtsp,
            ffmpeg_bin=settings.ffmpeg_bin,
            timeout_us=settings.rtsp_timeout_us,
            width=self.camera.width,
            height=self.camera.height,
            fps=self.camera.fps,
            transport=getattr(settings, "rtsp_transport", "tcp"),
            reconnect_backoff=tuple(settings.reconnect_backoff_seconds),
            force_size=getattr(settings, "force_decode_size", True),
            max_width=getattr(settings, "max_decode_width", 0),
            max_height=getattr(settings, "max_decode_height", 0),
        )
        self.pusher: RtspPusher | None = None
        # 回推重建退避：目标不可达时绝不能每帧重建 ffmpeg 进程
        self._push_backoff = 1.0
        self._push_retry_at = 0.0
        # 没有 RTSP 网关时只警告一次，并按 5 分钟慢重试（现场可能后补网关）
        self._gateway_missing_logged = False

        self._queue: queue.Queue = queue.Queue(maxsize=settings.queue_size)
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        self._last_output: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        # MJPEG 快照缓存：同一帧被多个观看者请求时只编码一次。
        # 顺序号分「原图/增强」两路，静止复用时不递增增强序号，
        # 这样复用旧帧期间缓存始终命中，不会每 66ms 重编码一次。
        self._raw_seq = 0
        self._output_seq = 0
        self._jpeg_lock = threading.Lock()
        self._jpeg_cache: dict[tuple[bool, int], tuple[int, bytes]] = {}
        self._last_infer_at = 0.0
        self._infer_interval = 1.0 / max(1, settings.infer_fps)
        self._frozen_counter = 0
        self._last_signature: bytes | None = None
        self._first_frame_logged = False
        # 画面冻结重连请求：worker 线程检测到冻结后**不能直接杀 ffmpeg 进程**
        # ——那会让 reader 线程正在 read 的管道突然 EOF，两条线程各自走重连逻辑，
        # 互相覆盖 open() 出来的新进程，形成竞态。改为设标记，由 reader 线程
        # 统一在循环顶部消费。
        self._frozen_reconnect_requested = False

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self.state.running:
            return
        # 每次启动换一个新的 Event，线程持有自己那一轮的引用。
        # 这样即使上一轮 worker 卡在长推理里、join 超时后才醒来，它看到的
        # 仍是「已停止」的旧事件，不会和新一轮线程一起往队列/推流器写帧。
        self._stop_event = threading.Event()
        self._push_backoff = 1.0
        self._push_retry_at = 0.0
        self._frozen_reconnect_requested = False
        # 健康检查按「本次启动」计时：重启后从零开始观察，避免拿着上一轮
        # 的 last_frame_at 立刻又触发一次重启。
        self.metrics.started_at = time.time()
        self.metrics.last_frame_at = 0.0
        self._first_frame_logged = False
        self.reader.resume()

        probe = self.reader.probe()
        if probe is not None:
            self.reader.configure(probe)
            logger.info(
                "通道 %s 探测到 %dx%d@%.2f (%s)",
                self.camera.id, probe.width, probe.height, probe.fps, probe.codec,
            )
        self._configure_output()

        self.reader.open()
        self.state.connected = True

        stop_event = self._stop_event
        self._threads = [
            threading.Thread(
                target=self._reader_loop,
                args=(stop_event,),
                name=f"cc-read-{self.camera.id}",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_loop,
                args=(stop_event,),
                name=f"cc-work-{self.camera.id}",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        self.state.running = True
        logger.info("通道 %s 已启动", self.camera.id)

    def stop(self, timeout: float = 5.0) -> None:
        """停止通道。可重复调用，也覆盖「start 中途失败」的清理场景。"""
        # 先停 reader：它会置停止标志并 terminate ffmpeg，从而立刻打断
        # 阻塞读和退避等待；否则 join 只能干等 5 秒，然后旧线程又自己重连。
        self._stop_event.set()
        self.reader.stop()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        if self.pusher is not None:
            self.pusher.close()
        self.state.running = False
        self.state.connected = False
        self.state.output_ready = False
        with self._frame_lock:
            self._last_output = None
            self._last_raw = None
            self._raw_seq += 1
            self._output_seq += 1
        with self._jpeg_lock:
            self._jpeg_cache.clear()
        logger.info("通道 %s 已停止", self.camera.id)

    def restart(self) -> None:
        logger.info("通道 %s 重启", self.camera.id)
        self.stop()
        self.analyzer.reset()
        self._last_signature = None
        self._frozen_counter = 0
        self._frozen_reconnect_requested = False
        self.start()

    # ------------------------------------------------------------------ #
    # 输出配置
    # ------------------------------------------------------------------ #
    def _configure_output(self) -> bool:
        """建立回推器。返回 True 表示「因目标没有 RTSP 网关而主动跳过」。

        跳过与失败要区分开：没有网关是**部署形态问题**（单机客户端本来就没有
        网关），不该按退避节奏反复重连；网关在但推流被拒才是需要重试的故障。
        """
        # 单机客户端形态：内网没有 RTSP 网关，回推必然失败。
        # 直接不建推流器，画面仍通过 _last_output 走 MJPEG 提供给浏览器。
        if not getattr(self.settings, "output_enabled", True):
            self.pusher = None
            self.state.output_ready = False
            return False
        out_width = self.reader.width * self.output_scale
        out_height = self.reader.height * self.output_scale
        target = f"{self.settings.rtsp_output_base}/{self.camera.output_name}"
        if not rtsp_endpoint_reachable(target):
            self.pusher = None
            self.state.output_ready = False
            if not self._gateway_missing_logged:
                self._gateway_missing_logged = True
                logger.warning(
                    "通道 %s 回推跳过：%s 上没有 RTSP 网关（浏览器观看与截图不受影响；"
                    "需要回推时把 mediamtx 放进 bin/ 或设 CAMPUSCLEAR_RTSP_OUTPUT_BASE 指向已有网关）",
                    self.camera.id, self.settings.rtsp_output_base,
                )
            return True
        # 用探测到的真实源帧率，而不是配置默认的 25：15fps 的流按 25fps
        # 打时间戳会让回放速度偏快。
        source_fps = max(1, int(round(self.reader.fps or self.resolved.fps)))
        self.pusher = RtspPusher(
            target,
            width=out_width,
            height=out_height,
            fps=source_fps,
            bitrate_kbps=self._suggest_bitrate(out_width, out_height),
            ffmpeg_bin=self.settings.ffmpeg_bin,
            username=self.settings.rtsp_output_user,
            password=self.settings.rtsp_output_password,
        )
        try:
            self.pusher.start()
            self.state.output_ready = True
        except OSError as exc:
            self.state.output_ready = False
            self.metrics.last_error = f"回推启动失败: {exc}"
            logger.error("通道 %s 回推启动失败: %s", self.camera.id, exc)
        return False

    def _ensure_output(self) -> None:
        """按需建立/重建回推，失败走指数退避。

        这里是「实时清晰化」能不能长期稳定运行的关键：FFmpeg 推流失败是**常事**
        （网关重启、网络抖动、目标地址写错），而原实现在写帧失败后会立刻重建，
        25fps 下等于每秒创建 25 个 ffmpeg 进程 —— 机器会被瞬间打死，
        连 MJPEG 出图都会一起卡住。改为退避后，最坏情况是「回推暂时不可用」，
        实时预览始终不受影响。
        """
        if not getattr(self.settings, "output_enabled", True):
            return
        now = time.monotonic()
        if self.pusher is not None and self.pusher.alive:
            return
        if self._push_retry_at and now < self._push_retry_at:
            return

        if self._configure_output():
            # 没有网关：不是故障，是部署形态。5 分钟后再看一眼，
            # 现场补上网关后能自动恢复，同时避免每 60 秒空拉一次 ffmpeg。
            self._push_retry_at = now + 300.0
            return
        if self.state.output_ready and self.pusher is not None and self.pusher.alive:
            self._push_backoff = 1.0
            self._push_retry_at = 0.0
        else:
            self._push_backoff = min(60.0, self._push_backoff * 2)
            self._push_retry_at = now + self._push_backoff

    def _note_push_failure(self, reason: str) -> None:
        """记录一次回推失败并进入退避窗口。"""
        self.metrics.last_error = reason
        if self.pusher is not None:
            self.pusher.close()
        self.state.output_ready = False
        self._push_backoff = min(60.0, self._push_backoff * 2)
        self._push_retry_at = time.monotonic() + self._push_backoff
        logger.warning("通道 %s 回推失败（%s），%.0fs 后重试", self.camera.id, reason, self._push_backoff)

    @staticmethod
    def _suggest_bitrate(width: int, height: int) -> int:
        """按分辨率线性给码率：1080p 约 4Mbps，2K 约 8Mbps。

        内网带宽通常充裕，但摄像机侧解码与存储 NVR 的写盘能力有限，
        盲目提高码率反而会让 NVR 掉帧，因此不做激进设置。
        """
        pixels = width * height
        base = 4_000_000 / (1920 * 1080)
        return int(max(2000, min(20000, pixels * base / 1000)))

    # ------------------------------------------------------------------ #
    # reader 线程
    # ------------------------------------------------------------------ #
    def _reader_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            # worker 线程检测到画面冻结时只设标记，重连动作由本线程统一执行：
            # 避免两条线程同时 close/open 同一个 ffmpeg 进程的竞态。
            if self._frozen_reconnect_requested:
                self._frozen_reconnect_requested = False
                self.metrics.last_error = "检测到画面冻结，触发重连"
                self._frozen_counter = 0
                self.analyzer.reset()
                if not self._reconnect():
                    continue

            if not self.reader.alive:
                self.state.connected = False
                if not self._reconnect():
                    continue

            try:
                frame = self.reader.read(timeout=3.0)
            except Exception as exc:  # noqa: BLE001 - 单路异常不能拖垮整机
                self.metrics.last_error = f"读取异常: {exc}"
                frame = None

            if frame is None:
                if stop_event.is_set():
                    break
                self.metrics.last_error = self.reader.last_error or "读取失败"
                self.state.connected = False
                self._reconnect()
                continue

            if not self._first_frame_logged:
                logger.info(
                    "通道 %s 首帧到达: %dx%d dtype=%s 均值=%.1f",
                    self.camera.id, frame.shape[1], frame.shape[0],
                    frame.dtype, float(frame.mean()),
                )
                self._first_frame_logged = True
            self.metrics.frames_read += 1
            self.metrics.last_frame_at = time.time()
            self._offer(frame)

    def _reconnect(self) -> bool:
        """统一走 RtspReader 的指数退避（可被 stop() 立即打断）。"""
        self.metrics.reconnects += 1
        if not self.reader.reconnect():
            self.metrics.last_error = self.reader.last_error or "重连失败"
            return False
        self.state.connected = True
        self.analyzer.reset()
        return True

    def _offer(self, frame: np.ndarray) -> None:
        """把帧放进队列；满则丢弃最旧的一帧。

        这就是「保实时不保完整」的具体实现。丢帧计数会暴露在指标页面上，
        让值班人员能看出「画面在动但服务器已经跟不上了」。
        """
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.metrics.frames_dropped += 1
        except queue.Empty:  # pragma: no cover - 竞态下的兜底
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover
            self.metrics.frames_dropped += 1

    # ------------------------------------------------------------------ #
    # worker 线程
    # ------------------------------------------------------------------ #
    def _worker_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                output = self._process_frame(frame)
            except Exception as exc:  # noqa: BLE001
                self.metrics.last_error = f"处理异常: {exc}"
                logger.exception("通道 %s 处理异常", self.camera.id)
                output = frame

            # 推理可能几十秒；醒来后本通道可能已经停止/重启过。用本轮
            # 线程持有的 stop_event 判断，避免旧 worker 往新管线里写帧。
            if stop_event.is_set() or output is None:
                continue

            # 先更新对外快照：**无论回推是否可用，客户端 MJPEG 都必须拿到最新画面**。
            # 这正是「单机客户端不依赖任何 RTSP 网关」的结构性保证。
            # 静止复用返回的是同一个 ndarray，此时不递增序号，MJPEG 缓存可命中。
            with self._frame_lock:
                if self._last_output is not output:
                    self._last_output = output
                    self._output_seq += 1

            self._ensure_output()
            if self.pusher is None or not self.pusher.alive:
                continue
            if not self.pusher.write(output):
                self._note_push_failure("写帧失败（推流目标不可达或编码进程退出）")
                continue
            self.metrics.frames_written += 1

    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        # 留一份未增强的原始帧，供客户端「原图 / 增强」对比使用
        with self._frame_lock:
            self._last_raw = frame
            self._raw_seq += 1
        quality = self.analyzer.analyze(frame)

        # ---- 画面冻结检测 ----
        # 老旧摄像头偶发「输出静止帧」故障（NVR 侧看起来在线，实际画面卡死）。
        # 用 8x8 缩略图的字节序列做指纹，连续多次相同即判定冻结。
        # 注意：worker 线程**不能直接重连** —— 那会杀掉 reader 正在 read 的
        # ffmpeg 进程，导致竞态。只设标记，由 reader 线程统一处理。
        signature = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA).tobytes()
        if signature == self._last_signature:
            self._frozen_counter += 1
            if self._frozen_counter >= self.settings.frozen_frame_restart_times:
                logger.warning(
                    "通道 %s 检测到画面冻结（连续 %d 帧相同），已请求重连",
                    self.camera.id, self._frozen_counter,
                )
                self._frozen_reconnect_requested = True
                self._frozen_counter = 0
                self.analyzer.reset()
        else:
            self._frozen_counter = 0
        self._last_signature = signature

        level = self.governor.level
        plan = build_plan(quality, self.policy, level)
        self.metrics.record_plan(plan.reason, plan.stages())

        # ---- 复用判定 ----
        reuse = self._should_reuse(plan)
        if reuse and self._last_output is not None:
            self.metrics.frames_reused += 1
            return self._last_output

        roi = self._extract_roi(frame, quality) if self.camera.roi_enabled else None

        start = time.perf_counter()
        output = self.enhancer.enhance(frame, plan, roi)
        self.metrics.record_latency(time.perf_counter() - start)

        self.metrics.frames_enhanced += 1
        return self._match_output_size(frame, output)

    def _match_output_size(self, frame: np.ndarray, output: np.ndarray) -> np.ndarray:
        """把增强结果归一到启动时确定的输出画布。

        这里必须覆盖三种情况，缺一都会让 rawvideo 帧长错位：
        1. 模型原生倍率（如 4×）大于请求倍率（如 2×）→ 缩小；
        2. 负载降档把 SR 关掉，enhancer 原样返回输入尺寸 → 补到输出倍率；
        3. 兜底实现只返回同尺寸 → 补到输出倍率。
        """
        target_width = frame.shape[1] * self.output_scale
        target_height = frame.shape[0] * self.output_scale
        if (output.shape[1], output.shape[0]) == (target_width, target_height):
            return output
        interpolation = (
            cv2.INTER_AREA if output.shape[1] > target_width else cv2.INTER_CUBIC
        )
        return cv2.resize(
            output, (target_width, target_height), interpolation=interpolation
        )

    def _should_reuse(self, plan: EnhancePlan) -> bool:
        """是否跳过本轮推理。"""
        if plan.reuse_previous:
            return True
        # 推理降频：不到间隔就复用上一结果
        now = time.monotonic()
        if now - self._last_infer_at < self._infer_interval:
            return True
        self._last_infer_at = now
        return False

    # ------------------------------------------------------------------ #
    def _extract_roi(self, frame: np.ndarray, quality) -> list[tuple] | None:
        """从运动掩膜中提取 ROI 外接矩形。

        ROI 的价值在于把有限算力压在「有东西在动」的区域。
        校园监控里绝大多数时间画面是空走廊、空操场，全帧超分纯属浪费。
        """
        try:
            mask = self.analyzer.motion_mask(frame)
        except Exception:  # noqa: BLE001
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        height, width = frame.shape[:2]
        frame_area = float(height * width)
        padding = self.settings.roi_padding

        boxes: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            # 太小的连通域多半是噪点，过滤掉
            if area < frame_area * 0.0005:
                continue
            boxes.append(
                (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(width, x + w + padding),
                    min(height, y + h + padding),
                    area,
                )
            )

        if not boxes:
            return None

        boxes.sort(key=lambda item: item[4], reverse=True)
        boxes = boxes[: self.settings.roi_max_boxes]

        # 合并相近的框，减少条数也减少拼接开销
        boxes = _merge_boxes([(b[0], b[1], b[2], b[3]) for b in boxes])

        covered = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / frame_area
        if covered > self.policy.roi_max_area_ratio or covered < self.policy.roi_min_area_ratio:
            # 覆盖太少说明运动是零星噪点，覆盖太多说明不如整帧处理
            return None
        return boxes

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #
    def snapshot_jpeg(self, quality: int = 85) -> bytes | None:
        return self.snapshot_jpeg_cached(raw=False, quality=quality)

    def snapshot_raw_jpeg(self, quality: int = 85) -> bytes | None:
        """未增强的原始帧（对比用）。"""
        return self.snapshot_jpeg_cached(raw=True, quality=quality)

    def snapshot_jpeg_cached(self, raw: bool = False, quality: int = 85) -> bytes | None:
        """按帧序号缓存 JPEG 编码结果。

        MJPEG 每个观看者都会高频取图；如果每人都独立 imencode，8 路 × 5 人
        就是每秒数百次 1080p 编码。同一帧只编码一次，其余观看者直接拿缓存。
        """
        with self._frame_lock:
            frame = self._last_raw if raw else self._last_output
            seq = self._raw_seq if raw else self._output_seq
        if frame is None:
            return None

        key = (bool(raw), int(quality))
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(key)
            if cached is not None and cached[0] == seq:
                return cached[1]

        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            return None
        encoded = buffer.tobytes()

        with self._jpeg_lock:
            # 编码期间可能已有别的线程为同一帧完成编码，直接复用
            cached = self._jpeg_cache.get(key)
            if cached is not None and cached[0] == seq:
                return cached[1]
            self._jpeg_cache[key] = (seq, encoded)
        return encoded

    def snapshot(self) -> dict:
        data = self.metrics.snapshot()
        data.update(
            {
                "name": self.camera.name,
                "location": self.camera.location,
                "enabled": self.camera.enabled,
                "running": self.state.running,
                "connected": self.state.connected,
                "output_ready": self.state.output_ready,
                "output_url": f"{self.settings.rtsp_output_base}/{self.camera.output_name}",
                "source_host": _host_of(self.camera.source_rtsp),
                "plan": self.metrics.snapshot().get("last_plan", ""),
                "queue_depth": self._queue.qsize(),
            }
        )
        if self.pusher is not None:
            data["encoder"] = self.pusher.stats()
        return data


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = 32) -> list[tuple[int, int, int, int]]:
    """把距离小于 ``gap`` 的矩形合并，直到不再变化。

    朴素 N² 实现即可：ROI 数量上限只有 6，性能完全不是瓶颈。
    """
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        for box in merged:
            absorbed = False
            for index, existing in enumerate(result):
                if _near(existing, box, gap):
                    result[index] = (
                        min(existing[0], box[0]),
                        min(existing[1], box[1]),
                        max(existing[2], box[2]),
                        max(existing[3], box[3]),
                    )
                    absorbed = True
                    changed = True
                    break
            if not absorbed:
                result.append(box)
        merged = result
    return merged


def _near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _host_of(url: str) -> str:
    """只保留主机部分，接口返回时不要把源流完整地址和凭证暴露出去。"""
    if "://" not in url:
        return url
    _, _, rest = url.partition("://")
    if "@" in rest:
        _, _, rest = rest.partition("@")
    return rest.split("/")[0]
