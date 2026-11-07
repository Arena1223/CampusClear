#!/usr/bin/env python3
"""服务器实时/回放推流方式诊断。

现场上机时，最怕的不是「连不上」，而是不知道对方服务器到底支持哪种方式：
RTSP 主/子码流、RTMP、HTTP-FLV、HLS，还是海康/大华/宇视的私有回放 URL。
本工具把能试的方式全部试一遍，并把每一步的原始结果写进日志：

  1. DNS / TCP 连通性、端口、耗时；
  2. RTSP OPTIONS（Server、Public、WWW-Authenticate）；
  3. 每个候选 URL 的 RTSP DESCRIBE / HTTP 头；
  4. ffprobe 的完整 stdout/stderr（分辨率、编码、帧率）；
  5. ffmpeg 实际解码 3 帧；
  6. 用项目自己的 RtspReader 再读 3 帧，确认「本程序确实能拿到画面」。

用法：
  # 海康：实时 + 回放都试
  python3 tools/diagnose_server.py --host 192.168.1.64 --user admin --password xxx \
      --brand hikvision --channel 1 \
      --start "2026-10-29 16:00:00" --end "2026-10-29 16:05:00"

  # 不知道品牌：三家都试
  python3 tools/diagnose_server.py --host 10.0.0.9 --user admin --password xxx --brand auto

  # 平台只给了一个地址
  python3 tools/diagnose_server.py --url "http://10.0.0.9:8080/live/cam01.flv"

  # 只看会试哪些地址，不真正连接
  python3 tools/diagnose_server.py --host 10.0.0.9 --brand auto --mode playback --dry-run

日志默认写到 logs/diagnose-YYYYmmdd-HHMMSS.log，终端只打印摘要。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campusclear.compat import ffprobe_of, run_kwargs  # noqa: E402
from campusclear.core.ingest import RtspReader  # noqa: E402
from campusclear.core.sources import build_input_args, detect_scheme  # noqa: E402

LOG = logging.getLogger("diagnose")
SECRETS: list[str] = []


# --------------------------------------------------------------------------- #
# 日志与脱敏
# --------------------------------------------------------------------------- #
def setup_logging(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    LOG.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(out_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    LOG.addHandler(file_handler)
    LOG.addHandler(console)

    # 把项目内部（ingest/pipeline）的日志也接进来，否则 RtspReader 实读
    # 那一节的「启动拉流/首帧/重连原因」不会出现在诊断日志里。
    library = logging.getLogger("campusclear")
    library.setLevel(logging.DEBUG)
    library.handlers.clear()
    library.propagate = False
    library.addHandler(file_handler)
    library.addHandler(console)


def sanitize(text: object) -> str:
    value = str(text)
    for secret in SECRETS:
        if secret:
            value = value.replace(secret, "***")
    value = re.sub(
        r"(?i)\b(rtsp|rtsps|rtmp|rtmps|http|https)://([^/\s:@]+):([^@\s/]+)@",
        r"\1://\2:***@",
        value,
    )
    value = re.sub(r"(?i)\b(password|passwd|pwd|token)=([^&\s]+)", r"\1=***", value)
    return value


def mask_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" in rest:
        _, _, host = rest.partition("@")
        return f"{scheme}://***@{host}"
    return url


def log_block(title: str, text: str, limit: int = 4000) -> None:
    text = sanitize(text).strip()
    if len(text) > limit:
        text = text[:limit] + f"\n...（已截断，原长 {len(text)}）"
    LOG.info("%s:\n%s", title, text or "（空）")


# --------------------------------------------------------------------------- #
# URL / 时间
# --------------------------------------------------------------------------- #
def parse_dt(value: str) -> datetime:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 14:
            raise ValueError(f"时间格式不对: {value!r}（示例 2026-10-29 16:00:00）")
        parsed = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    if parsed.tzinfo is None:
        # 输入按本机本地时间理解；诊断时同时生成 UTC 和本地两种写法，
        # 让设备自己告诉我们它认哪一种。
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def time_variants(start: datetime, end: datetime) -> dict[str, tuple[str, str]]:
    utc_start = start.astimezone(timezone.utc)
    utc_end = end.astimezone(timezone.utc)
    variants = {
        "utc_z": (
            utc_start.strftime("%Y%m%dT%H%M%SZ"),
            utc_end.strftime("%Y%m%dT%H%M%SZ"),
        ),
        "local_compact": (
            start.strftime("%Y%m%dT%H%M%S"),
            end.strftime("%Y%m%dT%H%M%S"),
        ),
        "local_space": (
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    }
    return variants


def q(value: str) -> str:
    return quote(value, safe=":-T")


def host_parts(host: str, port: int) -> tuple[str, int, str]:
    if "://" in host:
        parsed = urlsplit(host)
        return parsed.hostname or host, parsed.port or port, parsed.scheme
    return host, port, "rtsp"


def build_rtsp(host: str, port: int, path: str, user: str, password: str) -> str:
    auth = ""
    if user:
        auth = f"{quote(user, safe='')}:{quote(password, safe='')}@"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"rtsp://{auth}{host}:{port}{path}"


@dataclass
class Candidate:
    mode: str
    brand: str
    label: str
    url: str


def build_candidates(args, start: datetime, end: datetime) -> list[Candidate]:
    if args.url:
        return [Candidate("direct", "custom", "用户指定地址", args.url)]

    host, port, _ = host_parts(args.host, args.port)
    brands = ["hikvision", "dahua", "uniview"] if args.brand == "auto" else [args.brand]
    out: list[Candidate] = []

    if args.mode in ("live", "all"):
        for brand in brands:
            if brand == "hikvision":
                for sub in (False, True):
                    suffix = "02" if sub else "01"
                    path = f"/Streaming/Channels/{args.channel}{suffix}"
                    out.append(Candidate(
                        "live", brand, f"海康 {'子' if sub else '主'}码流",
                        build_rtsp(host, port, path, args.user, args.password),
                    ))
            elif brand == "dahua":
                for sub in (False, True):
                    path = f"/cam/realmonitor?channel={args.channel}&subtype={1 if sub else 0}"
                    out.append(Candidate(
                        "live", brand, f"大华 {'子' if sub else '主'}码流",
                        build_rtsp(host, port, path, args.user, args.password),
                    ))
            elif brand == "uniview":
                for sub in (False, True):
                    path = f"/media/video{2 if sub else 1}"
                    out.append(Candidate(
                        "live", brand, f"宇视 {'子' if sub else '主'}码流",
                        build_rtsp(host, port, path, args.user, args.password),
                    ))

    if args.mode in ("playback", "all"):
        variants = time_variants(start, end)
        for brand in brands:
            for variant_name, (s_val, e_val) in variants.items():
                if brand == "hikvision":
                    path = (
                        f"/Streaming/tracks/{args.channel}01"
                        f"?starttime={q(s_val)}&endtime={q(e_val)}"
                    )
                elif brand == "dahua":
                    path = (
                        f"/cam/playback?channel={args.channel}&subtype=0"
                        f"&starttime={q(s_val)}&endtime={q(e_val)}"
                    )
                elif brand == "uniview":
                    path = (
                        f"/unicast/c{args.channel}/s0/playback"
                        f"?starttime={q(s_val)}&endtime={q(e_val)}"
                    )
                else:
                    continue
                out.append(Candidate(
                    "playback", brand, f"{brand} 回放/{variant_name}",
                    build_rtsp(host, port, path, args.user, args.password),
                ))

    # 去重（同 URL 可能被不同标签生成）
    seen: set[str] = set()
    unique: list[Candidate] = []
    for item in out:
        if item.url in seen:
            continue
        seen.add(item.url)
        unique.append(item)
    if args.max_candidates and len(unique) > args.max_candidates:
        unique = unique[: args.max_candidates]
    return unique


# --------------------------------------------------------------------------- #
# 网络 / 协议探测
# --------------------------------------------------------------------------- #
def tcp_check(host: str, port: int, timeout: float) -> bool:
    LOG.info("== TCP 连接 %s:%d ==", host, port)
    started = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        LOG.info("DNS 解析: %s", ", ".join(sorted({item[4][0] for item in infos})))
    except OSError as exc:
        LOG.error("DNS 解析失败: %s", exc)
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        LOG.error("TCP 连接失败（%.2fs）: %s", time.perf_counter() - started, exc)
        return False
    LOG.info("TCP 连接成功（%.2fs）", time.perf_counter() - started)
    return True


def scan_common_ports(host: str, timeout: float) -> list[int]:
    """扫一遍常见推流端口，帮助判断对方到底是 RTSP、RTMP 还是 HTTP 服务。"""
    ports = [554, 8554, 1935, 80, 8080, 8000, 8443, 443, 9000]
    open_ports: list[int] = []
    LOG.info("== 常见推流端口扫描 %s ==", host)
    for port in ports:
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=min(timeout, 1.0)):
                pass
        except OSError:
            LOG.info("  %5d 关闭/超时", port)
            continue
        open_ports.append(port)
        LOG.info("  %5d 开放（%.2fs）", port, time.perf_counter() - started)
    if open_ports:
        LOG.info("开放端口: %s", ", ".join(str(port) for port in open_ports))
    return open_ports


def _rtsp_request(url: str, method: str, timeout: float, user: str, password: str) -> str:
    """发一条最小 RTSP 请求，返回原始响应文本（用于看 Server/401/SDP）。"""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or 554
    target = f"rtsp://{host}:{port}{parts.path or '/'}"
    if parts.query:
        target += "?" + parts.query

    lines = [
        f"{method} {target} RTSP/1.0",
        "CSeq: 1",
        "User-Agent: CampusClear-Diagnose",
    ]
    if user:
        raw = f"{user}:{password}".encode("utf-8")
        lines.append("Authorization: Basic " + base64.b64encode(raw).decode("ascii"))
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8", errors="replace")

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(request)
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        length = 0
        for line in head.decode("utf-8", "replace").splitlines():
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    length = 0
        while len(body) < min(length, 65536):
            chunk = sock.recv(4096)
            if not chunk:
                break
            body += chunk
    return (head + b"\r\n\r\n" + body).decode("utf-8", "replace")


def rtsp_probe(url: str, timeout: float, user: str, password: str, method: str) -> None:
    if detect_scheme(url) not in ("rtsp", "rtsps"):
        return
    try:
        text = _rtsp_request(url, method, timeout, user, password)
        log_block(f"RTSP {method} 响应", text, limit=3000)
    except (OSError, ValueError) as exc:
        LOG.warning("RTSP %s 失败: %s", method, exc)


def http_probe(url: str, timeout: float) -> None:
    LOG.info("-- HTTP GET（Range 0-255）--")
    request = urllib.request.Request(
        url, method="GET",
        headers={"Range": "bytes=0-255", "User-Agent": "CampusClear-Diagnose"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(256)
            LOG.info("HTTP 状态: %s，耗时 %.2fs", response.status, time.perf_counter() - started)
            for key in ("Content-Type", "Content-Length", "Server", "Location", "Accept-Ranges"):
                if response.headers.get(key):
                    LOG.info("HTTP %s: %s", key, response.headers.get(key))
            LOG.info("前 256 字节(hex): %s", body[:64].hex())
            text = body.decode("utf-8", "replace")
            if text.startswith("#EXTM3U"):
                LOG.info("识别为 HLS 播放列表")
                log_block("HLS 前几行", "\n".join(text.splitlines()[:8]))
            elif body.startswith(b"FLV"):
                LOG.info("识别为 HTTP-FLV（FLV 头）")
            elif body.startswith(b"\xff\xd8"):
                LOG.info("识别为 JPEG/MJPEG")
            elif body[:1] == b"\x47":
                LOG.info("疑似 MPEG-TS（0x47 同步字节）")
    except urllib.error.HTTPError as exc:
        LOG.warning("HTTP 状态: %s %s", exc.code, exc.reason)
        for key, value in exc.headers.items():
            if key.lower() in ("server", "www-authenticate", "content-type", "location"):
                LOG.warning("HTTP %s: %s", key, value)
    except (urllib.error.URLError, OSError) as exc:
        LOG.error("HTTP 请求失败（%.2fs）: %s", time.perf_counter() - started, exc)


# --------------------------------------------------------------------------- #
# 媒体探测
# --------------------------------------------------------------------------- #
def ffprobe_command(ffprobe_bin: str, url: str, timeout: float, transport: str) -> list[str]:
    return [
        ffprobe_bin,
        "-hide_banner",
        "-v", "error",
        *build_input_args(
            url, rtsp_transport=transport, timeout_us=int(timeout * 1_000_000)
        ),
        "-select_streams", "v:0",
        "-show_entries", "stream=index,codec_name,width,height,avg_frame_rate,bit_rate",
        "-show_entries", "format=format_name,duration",
        "-of", "json",
        url,
    ]


def probe_ffprobe(
    ffprobe_bin: str, url: str, timeout: float, transport: str
) -> tuple[bool, dict | None]:
    command = ffprobe_command(ffprobe_bin, url, timeout, transport)
    LOG.info("-- ffprobe --")
    LOG.debug("命令: %s", sanitize(" ".join(command)))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout + 5,
            check=False, encoding="utf-8", errors="replace", **run_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.error("ffprobe 失败（%.2fs）: %s", time.perf_counter() - started, exc)
        return False, None
    elapsed = time.perf_counter() - started
    LOG.info("ffprobe 退出码=%s，耗时 %.2fs", result.returncode, elapsed)
    log_block("ffprobe stderr", result.stderr or "", limit=3000)
    LOG.debug("ffprobe stdout:\n%s", sanitize(result.stdout or ""))
    if result.returncode != 0:
        return False, None
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        LOG.error("ffprobe 输出不是合法 JSON")
        return False, None
    streams = data.get("streams") or []
    if not streams:
        LOG.error("ffprobe 成功但没有视频流")
        return False, data
    stream = streams[0]
    LOG.info(
        "视频流: %sx%s codec=%s fps=%s bitrate=%s",
        stream.get("width"), stream.get("height"), stream.get("codec_name"),
        stream.get("avg_frame_rate"), stream.get("bit_rate"),
    )
    return True, data


def decode_check(ffmpeg_bin: str, url: str, timeout: float, transport: str) -> bool:
    command = [
        ffmpeg_bin, "-hide_banner", "-v", "error",
        *build_input_args(
            url, rtsp_transport=transport, timeout_us=int(timeout * 1_000_000)
        ),
        "-i", url,
        "-t", "2",
        "-frames:v", "3",
        "-f", "null", "-",
    ]
    LOG.info("-- ffmpeg 实际解码（最多 3 帧 / 2 秒）--")
    LOG.debug("命令: %s", sanitize(" ".join(command)))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout + 8,
            check=False, encoding="utf-8", errors="replace", **run_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.error("ffmpeg 解码失败（%.2fs）: %s", time.perf_counter() - started, exc)
        return False
    LOG.info("ffmpeg 退出码=%s，耗时 %.2fs", result.returncode, time.perf_counter() - started)
    log_block("ffmpeg stderr", result.stderr or "", limit=3000)
    return result.returncode == 0


def runtime_check(
    url: str, timeout: float, transport: str, ffmpeg_bin: str, frames: int
) -> bool:
    """用项目自己的 RtspReader 读几帧，确认本程序能拿到画面。"""
    LOG.info("-- 项目 RtspReader 实读 %d 帧 --", frames)
    reader = RtspReader(
        url,
        ffmpeg_bin=ffmpeg_bin,
        timeout_us=int(timeout * 1_000_000),
        transport=transport,
        force_size=True,
        max_width=640,
    )
    try:
        probe = reader.probe(timeout=min(timeout, 10.0))
        if probe is not None:
            reader.configure(probe)
            LOG.info(
                "RtspReader 探测: %dx%d@%.2f %s",
                probe.width, probe.height, probe.fps, probe.codec,
            )
        reader.open()
        got = 0
        for index in range(frames):
            frame = reader.read(timeout=min(timeout, 10.0))
            if frame is None:
                LOG.error("第 %d 帧读取失败: %s", index + 1, reader.last_error or "未知")
                break
            got += 1
            LOG.info(
                "第 %d 帧: %dx%d dtype=%s 均值=%.1f",
                index + 1, frame.shape[1], frame.shape[0], frame.dtype, float(frame.mean()),
            )
        LOG.info("RtspReader 实际读到 %d/%d 帧", got, frames)
        return got > 0
    except Exception as exc:  # noqa: BLE001 - 诊断工具不能自己崩
        LOG.exception("RtspReader 异常: %s", exc)
        return False
    finally:
        reader.close()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
@dataclass
class CandidateResult:
    candidate: Candidate
    ffprobe_ok: bool = False
    decode_ok: bool = False
    runtime_ok: bool = False
    error: str = ""
    extra: dict = field(default_factory=dict)


def run(args) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else ROOT / "logs" / f"diagnose-{timestamp}.log"
    setup_logging(out_path)
    SECRETS.extend([args.password or "", args.user or ""])
    SECRETS[:] = [item for item in SECRETS if item]

    LOG.info("=" * 78)
    LOG.info("CampusClear 服务器推流诊断")
    LOG.info("时间: %s", datetime.now().astimezone().isoformat(timespec="seconds"))
    LOG.info("主机: %s", socket.gethostname())
    LOG.info("Python: %s", sys.version.replace("\n", " "))
    LOG.info("平台: %s", sys.platform)
    LOG.info("日志: %s", out_path)
    ffmpeg_bin = args.ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = args.ffprobe or ffprobe_of(ffmpeg_bin) or shutil.which("ffprobe") or "ffprobe"
    LOG.info("ffmpeg: %s", ffmpeg_bin)
    LOG.info("ffprobe: %s", ffprobe_bin)
    for tool in (ffmpeg_bin, ffprobe_bin):
        try:
            version = subprocess.run(
                [tool, "-version"], capture_output=True, text=True, timeout=5,
                check=False, encoding="utf-8", errors="replace", **run_kwargs(),
            ).stdout.splitlines()[0]
            LOG.info("%s 版本: %s", Path(tool).name, version)
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            LOG.warning("无法获取 %s 版本: %s", tool, exc)

    LOG.info("模式: %s  品牌: %s  通道: %s  传输: %s",
             args.mode, args.brand, args.channel, args.transport)
    if args.host:
        LOG.info("目标: %s:%s", args.host, args.port)

    # 时间区间
    now = datetime.now().astimezone()
    if args.end:
        end = parse_dt(args.end)
    else:
        end = now
    if args.start:
        start = parse_dt(args.start)
    else:
        start = end - timedelta(minutes=5)
    if start >= end:
        LOG.error("起始时间必须早于结束时间")
        return 2
    LOG.info("回放时间段(本地): %s ~ %s", start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))
    LOG.info("回放时间段(UTC) : %s ~ %s",
             start.astimezone(timezone.utc).isoformat(timespec="seconds"),
             end.astimezone(timezone.utc).isoformat(timespec="seconds"))

    candidates = build_candidates(args, start, end)
    LOG.info("候选地址: %d 个", len(candidates))
    for index, candidate in enumerate(candidates, 1):
        LOG.info("[%02d] %-8s %-24s %s", index, candidate.mode, candidate.label, mask_url(candidate.url))
    if args.dry_run:
        LOG.info("--dry-run：只生成候选，不发起连接 --")
        return 0

    target_host, target_port, _ = host_parts(
        args.host or _host_of(candidates[0].url), args.port
    )
    if target_host:
        if not tcp_check(target_host, target_port, args.timeout):
            LOG.warning("TCP 不可达，仍会继续尝试候选地址（可能是 DNS/代理/端口问题）")
        if args.scan_ports:
            scan_common_ports(target_host, args.timeout)

    if args.host:
        rtsp_probe(
            f"rtsp://{target_host}:{target_port}/",
            args.timeout, args.user, args.password, "OPTIONS",
        )

    results: list[CandidateResult] = []
    runtime_done: set[str] = set()

    for index, candidate in enumerate(candidates, 1):
        LOG.info("")
        LOG.info("=" * 78)
        LOG.info("[%02d/%02d] %s | %s | %s", index, len(candidates),
                 candidate.mode, candidate.label, mask_url(candidate.url))
        result = CandidateResult(candidate=candidate)
        scheme = detect_scheme(candidate.url)

        if scheme in ("http", "https"):
            http_probe(candidate.url, args.timeout)
        elif scheme in ("rtsp", "rtsps") and not args.no_rtsp_describe:
            rtsp_probe(candidate.url, min(args.timeout, 4.0), args.user, args.password, "DESCRIBE")

        ok, data = probe_ffprobe(ffprobe_bin, candidate.url, args.timeout, args.transport)
        result.ffprobe_ok = ok
        if data:
            result.extra = data
        if ok:
            result.decode_ok = decode_check(ffmpeg_bin, candidate.url, args.timeout, args.transport)
        else:
            result.error = "ffprobe 失败"

        if ok and result.decode_ok and args.runtime_check:
            if candidate.mode not in runtime_done:
                result.runtime_ok = runtime_check(
                    candidate.url, args.timeout, args.transport, ffmpeg_bin, args.runtime_frames
                )
                if result.runtime_ok:
                    runtime_done.add(candidate.mode)
        results.append(result)

    LOG.info("")
    LOG.info("=" * 78)
    LOG.info("诊断摘要")
    success = [r for r in results if r.ffprobe_ok and r.decode_ok]
    for result in results:
        mark = "✓" if (result.ffprobe_ok and result.decode_ok) else "✗"
        extra = ""
        if result.ffprobe_ok:
            streams = (result.extra or {}).get("streams") or []
            if streams:
                item = streams[0]
                extra = (
                    f"{item.get('width')}x{item.get('height')} "
                    f"{item.get('codec_name')} {item.get('avg_frame_rate')}"
                )
        if result.runtime_ok:
            extra += " | RtspReader 已实读"
        LOG.info("%s [%s] %-24s %s %s", mark, result.candidate.mode,
                 result.candidate.label, extra, mask_url(result.candidate.url))
    if success:
        LOG.info("可用方式: %d 个（见上方 ✓）", len(success))
        for result in success:
            LOG.info("  → %s: %s", result.candidate.mode, mask_url(result.candidate.url))
    else:
        LOG.warning("没有找到可用方式。请重点看每个候选的 ffprobe stderr：")
        LOG.warning("  401/403 → 账号密码或认证方式不对")
        LOG.warning("  404     → 路径/通道号不对")
        LOG.warning("  454     → 该时间段没有录像或会话不存在")
        LOG.warning("  timeout → 网络/端口/防火墙，或设备只允许特定源 IP")
    LOG.info("完整日志: %s", out_path)
    return 0 if success else 1


def _host_of(url: str) -> str:
    parsed = urlsplit(url)
    return parsed.hostname or ""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="服务器实时/回放推流方式诊断")
    ap.add_argument("--host", default="", help="服务器/摄像头/NVR 地址")
    ap.add_argument("--port", type=int, default=554, help="RTSP 端口，默认 554")
    ap.add_argument("--user", default="", help="用户名")
    ap.add_argument("--password", default="", help="密码")
    ap.add_argument("--brand", default="auto", choices=["auto", "hikvision", "dahua", "uniview"])
    ap.add_argument("--channel", type=int, default=1, help="通道号，默认 1")
    ap.add_argument("--start", default="", help="录像开始时间，如 '2026-10-29 16:00:00'")
    ap.add_argument("--end", default="", help="录像结束时间")
    ap.add_argument("--url", default="", help="直接诊断一个地址（优先于 host/brand）")
    ap.add_argument("--mode", default="all", choices=["live", "playback", "all"])
    ap.add_argument("--transport", default="tcp", choices=["tcp", "udp"])
    ap.add_argument("--timeout", type=float, default=8.0, help="单次探测超时秒数")
    ap.add_argument("--ffmpeg", default="", help="ffmpeg 路径（默认 PATH）")
    ap.add_argument("--ffprobe", default="", help="ffprobe 路径（默认由 ffmpeg 推导）")
    ap.add_argument("--out", default="", help="日志文件路径")
    ap.add_argument("--runtime-check", dest="runtime_check", action="store_true", default=True,
                    help="用项目 RtspReader 实读几帧（默认开启）")
    ap.add_argument("--no-runtime", dest="runtime_check", action="store_false",
                    help="跳过 RtspReader 实读")
    ap.add_argument("--runtime-frames", type=int, default=3, help="实读帧数，默认 3")
    ap.add_argument("--no-rtsp-describe", action="store_true", help="跳过逐候选 RTSP DESCRIBE")
    ap.add_argument("--scan-ports", action="store_true",
                    help="扫描 554/8554/1935/80/8080 等常见推流端口")
    ap.add_argument("--max-candidates", type=int, default=40, help="最多尝试多少个候选")
    ap.add_argument("--dry-run", action="store_true", help="只生成候选地址，不连接")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.url and not args.host:
        build_parser().error("必须提供 --host 或 --url")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
