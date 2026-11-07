#!/usr/bin/env python3
"""取流探测工具：现场排障用。

三件事：
1. 列出本项目支持的**全部推流连接方式**（``--methods``）；
2. 对**任意** URL（rtsp/rtmp/http-flv/hls/rtp/本地文件）做 ffprobe 探测；
3. 按**厂商**自动构造实时/回放地址再探测（海康/大华/宇视）。

用法:
    python tools/probe_source.py --methods
    python tools/probe_source.py --url "rtsp://user:pass@10.0.0.5:554/Streaming/Channels/101"
    python tools/probe_source.py --url "http://10.0.0.9:8080/live/cam01.flv"
    python tools/probe_source.py --brand hikvision --host 10.0.0.5 --user admin --password xxx \
        --channel 1 --sub
    python tools/probe_source.py --brand dahua --host 10.0.0.6 --user admin --password xxx \
        --playback --start "2026-10-10 21:00:00" --end "2026-10-10 21:05:00"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.core.ingest import _mask_url  # noqa: E402
from campusclear.core.sources import (  # noqa: E402
    build_input_args,
    build_playback_url,
    describe_methods,
    detect_scheme,
    vendor_live_url,
)


def ffprobe(url: str, ffmpeg_bin: str = "ffmpeg") -> None:
    ffprobe_bin = ffmpeg_bin.replace("ffmpeg", "ffprobe")
    args = build_input_args(url, rtsp_transport="tcp")
    command = [
        ffprobe_bin, "-v", "error", *args,
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,codec_name,bit_rate",
        "-of", "default=noprint_wrappers=1",
        url,
    ]
    print(f"== 探测 {_mask_url(url)}  (协议: {detect_scheme(url) or '本地文件'}) ==")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  ffprobe 不可用/超时: {exc}")
        return
    if result.returncode != 0:
        print(f"  ✗ 不可用：{result.stderr.strip()[:300]}")
        return
    for line in result.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            print(f"  {k:16} {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description="取流探测")
    ap.add_argument("--methods", action="store_true", help="列出支持的连接方式")
    ap.add_argument("--url", default=None, help="直接探测一个 URL")
    ap.add_argument("--brand", default=None, help="hikvision / dahua / uniview")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=554)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--sub", action="store_true", help="子码流")
    ap.add_argument("--playback", action="store_true", help="构造回放地址")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--no-probe", action="store_true", help="只打印地址，不探测")
    args = ap.parse_args()

    if args.methods:
        print("支持的推流连接方式：")
        for m in describe_methods():
            print(f"  {m['label']:16} {m['note']}")
        return 0

    urls: list[str] = []
    if args.url:
        urls.append(args.url)
    if args.brand and args.host:
        if args.playback:
            urls.append(build_playback_url(
                args.brand, args.host, channel=args.channel,
                start=args.start, end=args.end,
                username=args.user, password=args.password, port=args.port,
            ))
        else:
            urls.append(vendor_live_url(
                args.brand, args.host, channel=args.channel, sub=args.sub,
                username=args.user, password=args.password, port=args.port,
            ))

    if not urls:
        ap.print_help()
        return 1

    for url in urls:
        print(f"构造地址: {_mask_url(url)}")
        if not args.no_probe:
            ffprobe(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
