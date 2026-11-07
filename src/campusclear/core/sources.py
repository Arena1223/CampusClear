"""取流来源：支持监控服务器/摄像头常见的多种推流方式。

监控现场的取流方式五花八门，本项目把它们统一收敛到「一个 URL + 一组
按协议自适应的 FFmpeg 参数」上——因为 FFmpeg 本身就能播这些协议，
没有必要为每种协议写一套读取器。

支持的连接方式（``describe_methods()`` 也会输出）：

====================  ====================================================
方式                     URL 形态
====================  ====================================================
RTSP（主流）            ``rtsp://user:pass@host:554/Streaming/Channels/101``
RTSP over TLS          ``rtsps://host:322/...``
RTMP                   ``rtmp://host:1935/live/xxx``
HTTP-FLV               ``http://host:8080/live/xxx.flv``
HLS                    ``http://host:8080/live/xxx.m3u8``
HTTP(MJPEG/其他)        ``http://host/video.mjpg``
RTP / UDP（裸流、组播）  ``rtp://239.0.0.1:5000``  ``udp://...``
本地录像文件             ``/path/to/clip.mp4``  ``file://...``
====================  ====================================================

另外提供**厂商回放地址构造**（按时间段取录像），覆盖海康 / 大华 / 宇视，
这些路径各厂商、各固件差异很大，一律以 ffprobe 实测为准（见 docs/09）。
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    schemes: tuple[str, ...]
    note: str


METHODS: tuple[Method, ...] = (
    Method("rtsp", "RTSP", ("rtsp", "rtsps"), "最常用；内网低延迟，默认走 TCP"),
    Method("rtmp", "RTMP", ("rtmp", "rtmps"), "部分平台的二次分发"),
    Method("http_flv", "HTTP-FLV", ("http",), "URL 以 .flv 结尾，低延迟"),
    Method("hls", "HLS", ("http", "https"), "URL 以 .m3u8 结尾，通用但延迟高"),
    Method("http", "HTTP(MJPEG/其他)", ("http", "https"), "由 FFmpeg 自动识别"),
    Method("rtp", "RTP/UDP", ("rtp", "udp", "srt"), "裸流、组播/单播"),
    Method("file", "本地录像文件", ("file", ""), "回放增强用；也可直接给路径"),
)


def describe_methods() -> list[dict]:
    return [
        {"key": m.key, "label": m.label, "schemes": list(m.schemes), "note": m.note}
        for m in METHODS
    ]


def detect_scheme(url: str) -> str:
    """返回小写 scheme；本地路径返回空串。"""
    if "://" not in url:
        return ""
    return urlparse(url).scheme.lower()


def is_live_scheme(url: str) -> bool:
    return detect_scheme(url) in {"rtsp", "rtsps", "rtmp", "rtmps", "rtp", "udp", "srt"}


def build_input_args(
    url: str,
    *,
    rtsp_transport: str = "tcp",
    timeout_us: int = 5_000_000,
) -> list[str]:
    """按 URL 协议返回 FFmpeg 的**输入端**参数（不含 -i 与 URL）。

    统一的低延迟诉求：``nobuffer`` + ``discardcorrupt`` + ``low_delay``。
    不同协议只在「连接/传输层」参数上有区别。
    """
    scheme = detect_scheme(url)

    if scheme in ("rtsp", "rtsps"):
        return [
            "-rtsp_transport", rtsp_transport,
            "-rtsp_flags", "prefer_tcp",
            "-timeout", str(timeout_us),
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
        ]
    if scheme in ("rtmp", "rtmps"):
        return [
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
        ]
    if scheme in ("http", "https"):
        lower = url.lower()
        if lower.endswith(".m3u8") or ".m3u8" in lower:
            # HLS 有天然延迟，尽量贴近直播边缘
            return ["-fflags", "nobuffer", "-live_start_index", "-1", "-flags", "low_delay"]
        # HTTP-FLV / MJPEG / 其它由 FFmpeg 自行探测
        return ["-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "1000000"]
    if scheme in ("rtp", "udp", "srt"):
        return ["-fflags", "nobuffer", "-flags", "low_delay"]
    # file:// 与本地路径：普通读取即可
    return []


# --------------------------------------------------------------------------- #
# 厂商回放地址构造（按时间段取录像）
# --------------------------------------------------------------------------- #
def _fmt(ts: str) -> str:
    """把 ``YYYY-MM-DD HH:MM:SS`` 或 ``YYYYMMDDhhmmss`` 归一为 YYYYMMDDThhmmssZ。"""
    digits = "".join(ch for ch in ts if ch.isdigit())
    if len(digits) < 14:
        raise ValueError(f"时间格式不对: {ts!r}（示例 2026-10-10 21:30:00）")
    d = digits[:14]
    return f"{d[:8]}T{d[8:14]}Z"


def build_playback_url(
    brand: str,
    host: str,
    *,
    channel: int = 1,
    start: str,
    end: str,
    username: str = "",
    password: str = "",
    port: int = 554,
) -> str:
    """构造厂商 RTSP 回放（按时间取录像）地址。

    ⚠️ 各厂商/固件的回放路径差异很大，这里给的是常见形态，
    **下发前一律用 ``tools/probe_source.py`` 实测**（docs/09 有同样的告诫）。
    """
    auth = f"{quote(username)}:{quote(password)}@" if username else ""
    brand = brand.lower()
    ch = int(channel)

    if brand in ("hikvision", "hik", "海康", "海康威视"):
        track = f"{ch}01"
        return (
            f"rtsp://{auth}{host}:{port}/Streaming/tracks/{track}"
            f"?starttime={_fmt(start)}&endtime={_fmt(end)}"
        )
    if brand in ("dahua", "大华"):
        return (
            f"rtsp://{auth}{host}:{port}/cam/playback"
            f"?channel={ch}&subtype=0&starttime={_fmt(start)}&endtime={_fmt(end)}"
        )
    if brand in ("uniview", "unv", "宇视"):
        return (
            f"rtsp://{auth}{host}:{port}/unicast/c{ch}/s0/playback"
            f"?starttime={_fmt(start)}&endtime={_fmt(end)}"
        )
    raise ValueError(f"暂不支持的厂商: {brand!r}（可选：海康 hikvision / 大华 dahua / 宇视 uniview）")


def vendor_live_url(
    brand: str,
    host: str,
    *,
    channel: int = 1,
    sub: bool = False,
    username: str = "",
    password: str = "",
    port: int = 554,
) -> str:
    """构造厂商实时取流地址（主/子码流）。"""
    auth = f"{quote(username)}:{quote(password)}@" if username else ""
    brand = brand.lower()
    ch = int(channel)

    if brand in ("hikvision", "hik", "海康", "海康威视"):
        suffix = "02" if sub else "01"
        return f"rtsp://{auth}{host}:{port}/Streaming/Channels/{ch}{suffix}"
    if brand in ("dahua", "大华"):
        return f"rtsp://{auth}{host}:{port}/cam/realmonitor?channel={ch}&subtype={1 if sub else 0}"
    if brand in ("uniview", "unv", "宇视"):
        return f"rtsp://{auth}{host}:{port}/media/video{2 if sub else 1}"
    raise ValueError(f"暂不支持的厂商: {brand!r}")
