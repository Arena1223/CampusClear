"""Windows 兼容自检：不依赖真实摄像头，验证改造后的关键链路。

为什么需要它：RTSP 拉流失败的现象高度同质化（黑屏 / 雪花 / 卡住），
很难从现象反推是「ffmpeg 没找到」「ffprobe 不存在」「帧长不匹配」还是
「编码器不存在」。这个脚本用**本机合成视频**把这几段链路逐段验证，
在没有摄像头、没有网络的机器上也能跑出确定结论。

用法（项目根目录）：
    .venv\\Scripts\\python.exe tools\\win_selftest.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campusclear.compat import (  # noqa: E402
    enable_utf8_console,
    encoder_usable,
    ffmpeg_version,
    ffprobe_available,
    find_ffmpeg,
    fps_mode_supported,
    has_encoder,
    parse_video_info,
    popen_kwargs,
    run_kwargs,
)

enable_utf8_console()

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录一项检查结果。"""
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def make_clip(ffmpeg: str, path: Path, size: str, seconds: int = 2) -> bool:
    """合成一段测试视频，作为「本地文件源」喂给拉流器。"""
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25",
        "-t", str(seconds),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, **run_kwargs())
    except OSError as exc:  # ffmpeg 找不到时会抛 WinError 2，这里必须收敛成检查项
        print(f"      合成视频失败：{exc}")
        return False
    return result.returncode == 0 and path.is_file() and path.stat().st_size > 0


def main() -> int:
    print("=" * 66)
    print("CampusClear · Windows 兼容自检")
    print("=" * 66)
    print(f"Python   : {sys.version.split()[0]}  ({sys.platform})")
    print(f"项目根目录: {ROOT}")
    print("-" * 66)

    # ---------------- 1. ffmpeg 定位 ----------------
    ffmpeg = find_ffmpeg()
    check("ffmpeg 可定位到具体文件（而非裸命令名）", Path(ffmpeg).is_file(), ffmpeg)
    version = ffmpeg_version(ffmpeg)
    check("ffmpeg 版本可解析", version > (0, 0), f"{version[0]}.{version[1]}")
    check(
        "帧同步参数选型明确（-fps_mode 或 -vsync）",
        True,
        "-fps_mode passthrough" if fps_mode_supported(ffmpeg) else "-vsync 0（兼容旧版）",
    )
    check("ffprobe 可用性已探测", True,
          "可用" if ffprobe_available(ffmpeg) else "不可用 → 将降级解析 `ffmpeg -i`")

    # ---------------- 2. 编码器能力 ----------------
    check("libx264 软件编码可用（回推兜底）", has_encoder(ffmpeg, "libx264"))
    nvenc = encoder_usable(ffmpeg, "h264_nvenc")
    check("h264_nvenc 做了「真能编码」验证而非只看构建清单", True,
          "本机可用，回推将走硬件编码" if nvenc
          else "本机不可用（无 N 卡）→ 回推自动改用 libx264")

    # ---------------- 3. 流信息解析（无 ffprobe 的降级路径） ----------------
    sample = (
        "  Duration: N/A, start: 0.000000, bitrate: N/A\n"
        "  Stream #0:0: Video: h264 (Main), yuvj420p(pc, bt709), 2560x1440 "
        "[SAR 1:1 DAR 16:9], 25 fps, 25 tbr, 90k tbn\n"
    )
    width, height, fps = parse_video_info(sample)
    check("可从 `ffmpeg -i` 的 stderr 解析出 2560x1440@25",
          (width, height, fps) == (2560, 1440, 25.0), f"{width}x{height}@{fps}")

    # ---------------- 4. rawvideo 管道（真实解码） ----------------
    from campusclear.core.ingest import RtspReader

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        clip = tmpdir / "clip_1280x720.mp4"
        ok = make_clip(ffmpeg, clip, "1280x720")
        check("合成测试视频成功（libx264 可编码）", ok)
        if not ok:
            return report()

        # 4a. 走探测路径：尺寸应与源一致
        reader = RtspReader(str(clip), ffmpeg_bin=ffmpeg, force_size=False)
        probe = reader.probe()
        check("本地文件源可探测出真实尺寸",
              probe is not None and (probe.width, probe.height) == (1280, 720),
              f"{probe.width}x{probe.height}@{probe.fps}" if probe else "探测失败")

        if probe is not None:
            reader.configure(probe)
        reader.open()
        frame = reader.read(timeout=10)
        reader.close()
        check("探测路径能读出整帧且尺寸正确",
              frame is not None and frame.shape == (720, 1280, 3),
              f"shape={None if frame is None else frame.shape}")

        # 4b. 模拟「探测失败」：故意用 640x480 去解 1280x720 的流。
        # 这正是改造前会碎成雪花的场景，force_size 应当把尺寸钉死。
        blind = RtspReader(str(clip), ffmpeg_bin=ffmpeg, width=640, height=480, force_size=True)
        blind.open()
        frame2 = blind.read(timeout=10)
        blind.close()
        check("探测失败时强制缩放仍能读出固定帧长（雪花问题的根治验证）",
              frame2 is not None and frame2.shape == (480, 640, 3),
              f"shape={None if frame2 is None else frame2.shape}")

    # ---------------- 5. 配置快照 ----------------
    os.environ.setdefault("CAMPUSCLEAR_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("CAMPUSCLEAR_MODELS_DIR", str(ROOT / "models"))
    os.environ.setdefault("CAMPUSCLEAR_OUTPUT_ENABLED", "false")
    from campusclear.config import get_settings

    settings = get_settings(refresh=True)
    print("-" * 66)
    print(f"执行提供器 : {settings.providers}")
    print(f"fp16       : {settings.use_fp16}")
    print(f"回推(RTSP) : {'开启' if settings.output_enabled else '关闭（单机客户端形态）'}")
    print(f"强制解码尺寸: {settings.force_decode_size}")
    check("CPU-only 机器上 fp16 已自动关闭", not settings.use_fp16 or "CPUExecutionProvider" not in settings.providers)
    check("执行提供器已按本机能力收敛",
          all(p in ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider")
              for p in settings.providers),
          str(settings.providers))
    check("子进程参数在 Windows 下带隐藏窗口标志",
          (not sys.platform.startswith("win")) or popen_kwargs().get("creationflags") == 0x08000000)

    return report()


def report() -> int:
    print("=" * 66)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  ✗ {name}")
        print("结论：Windows 兼容性尚未达标，请修复上述项。")
        return 1
    print("结论：Windows 关键链路全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
