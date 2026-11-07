"""跨平台兼容层。

Linux 上「装个 ffmpeg」是一条 apt 命令；到了 Windows 就没这么理所当然。
本项目要在 Windows 上做到「拷走文件夹双击即用」，以下四件事必须集中处理：

1. **ffmpeg 从哪来**：按「环境变量 → 程序自带 bin/ → PyInstaller 解包目录 →
   imageio-ffmpeg 内置静态二进制 → PATH」依次查找。这样源码运行与打包运行
   走的是同一套解析逻辑，不会出现「源码能跑、打包后找不到 ffmpeg」。
2. **子进程不能弹黑框**：每路拉流/回推都是独立 ffmpeg 进程，不加
   ``CREATE_NO_WINDOW`` 时 Windows 会不断闪出控制台窗口。
3. **控制台是 GBK**：中文日志和「✓/✗」这类符号在 cp936 下会触发
   UnicodeEncodeError（日志系统会吞掉，但满屏 ``--- Logging error ---``）。
4. **ffprobe 可能根本不存在**：imageio-ffmpeg 只带 ffmpeg。好在
   ``ffmpeg -i <url>`` 本身就会把流信息打到 stderr，解析它即可等价探测。

依赖方向：本模块只依赖标准库，可被 config / core / packaging 任意层引用。
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform.startswith("win")

# Windows 上隐藏子进程控制台窗口；其它平台为 0（不影响行为）
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


# --------------------------------------------------------------------------- #
# 子进程参数
# --------------------------------------------------------------------------- #
def popen_kwargs() -> dict:
    """``subprocess.Popen`` 的平台附加参数（Windows 下隐藏窗口）。"""
    if IS_WINDOWS:
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def run_kwargs() -> dict:
    """``subprocess.run`` 的平台附加参数。"""
    return popen_kwargs()


# --------------------------------------------------------------------------- #
# ffmpeg 定位
# --------------------------------------------------------------------------- #
def _candidate_dirs() -> list[Path]:
    """按优先级给出可能存放自带 ffmpeg 的目录（只用于「固定文件名」查找）。"""
    dirs: list[Path] = []

    # 1) PyInstaller 解包目录（onedir 时与 exe 同级，onefile 时是临时目录）
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        dirs += [Path(bundle) / "bin", Path(bundle)]

    frozen = getattr(sys, "frozen", False)
    if frozen:
        exe_dir = Path(sys.executable).parent
        dirs += [exe_dir / "bin", exe_dir]

    # 2) 源码运行：项目根目录下的 bin/（src/campusclear/compat.py → 上溯三级）
    here = Path(__file__).resolve()
    for up in (here.parents[2], here.parents[1]):
        dirs += [up / "bin", up]

    # 去重且保持顺序
    seen: set[str] = set()
    unique: list[Path] = []
    for item in dirs:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _imageio_ffmpeg_binaries() -> list[Path]:
    """列出 imageio-ffmpeg 自带的静态二进制。

    注意**不要**用 ``imageio_ffmpeg.get_ffmpeg_exe()``：二进制缺失时它会去
    GitHub 下载——而本项目的全部前提就是「无公网依赖」。这里直接扫包内
    ``binaries/`` 目录，找不到就是找不到，绝不联网。

    它的文件名形如 ``ffmpeg-win-x86_64-v7.1.exe``，带平台与版本后缀，
    所以必须按前缀扫描，不能按固定名 ``ffmpeg.exe`` 去找。
    """
    try:
        import imageio_ffmpeg
    except Exception:  # noqa: BLE001 - 可选依赖
        return []
    try:
        binaries = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
        if not binaries.is_dir():
            return []
        found = [
            item for item in sorted(binaries.iterdir())
            if item.is_file() and item.name.startswith("ffmpeg")
        ]
    except OSError:  # pragma: no cover
        return []
    return found


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """解析可用的 ffmpeg 可执行文件。

    查找顺序：环境变量（由 config 层处理）→ 程序自带 ``bin/`` →
    PyInstaller 解包目录 → imageio-ffmpeg 内置静态二进制 → 裸命令 ``ffmpeg``。
    返回**完整路径**优先，实在找不到才退回裸命令名交给 PATH 解析。
    """
    names = ("ffmpeg.exe", "ffmpeg") if IS_WINDOWS else ("ffmpeg",)
    for directory in _candidate_dirs():
        for name in names:
            candidate = directory / name
            try:
                if candidate.is_file():
                    logger.info("使用自带 ffmpeg: %s", candidate)
                    return str(candidate)
            except OSError:  # pragma: no cover - 权限/长路径异常
                continue

    for candidate in _imageio_ffmpeg_binaries():
        logger.info("使用 imageio-ffmpeg 内置二进制: %s", candidate)
        return str(candidate)

    logger.warning(
        "未找到自带的 ffmpeg，将按 PATH 查找裸命令「ffmpeg」；"
        "若拉流失败请设置环境变量 CAMPUSCLEAR_FFMPEG 指向 ffmpeg 可执行文件"
    )
    return "ffmpeg"


def ffprobe_of(ffmpeg_bin: str) -> str:
    """由 ffmpeg 路径推导 ffprobe 路径（找不到时返回同名裸命令）。"""
    candidate = re.sub(r"ffmpeg(\.exe)?$", lambda m: "ffprobe" + (m.group(1) or ""), ffmpeg_bin)
    if candidate != ffmpeg_bin and Path(candidate).is_file():
        return candidate
    return "ffprobe"


def ffprobe_available(ffmpeg_bin: str) -> bool:
    """ffprobe 是否真的可用（imageio-ffmpeg 就不带它）。"""
    binary = ffprobe_of(ffmpeg_bin)
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            timeout=8,
            check=False,
            **run_kwargs(),
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------- #
# ffmpeg 能力探测
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def ffmpeg_version(ffmpeg_bin: str) -> tuple[int, int]:
    """解析 ``ffmpeg -version`` 的主次版本号；失败返回 (0, 0)。"""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            encoding="utf-8",
            errors="replace",
            **run_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    match = re.search(r"ffmpeg version\s+n?(\d+)\.(\d+)", result.stdout or "")
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def fps_mode_supported(ffmpeg_bin: str) -> bool:
    """``-fps_mode`` 自 5.1 起提供，取代了已废弃的 ``-vsync``。

    FFmpeg 8 起 ``-vsync`` 被移除，因此不能写死老参数；
    版本识别不出来时保守使用老参数（老参数在新版上只是告警）。
    """
    major, minor = ffmpeg_version(ffmpeg_bin)
    if major == 0:
        return False
    return (major, minor) >= (5, 1)


@lru_cache(maxsize=32)
def has_encoder(ffmpeg_bin: str, encoder: str) -> bool:
    """该 ffmpeg **构建**里是否编译了指定编码器（如 ``h264_nvenc``）。

    注意：构建里有 ≠ 机器上能用。静态 ffmpeg 普遍把 nvenc 编了进去，
    但没有 NVIDIA 硬件的机器一调用就失败。要判断「确实能用」请用
    ``encoder_usable()``。
    """
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            encoding="utf-8",
            errors="replace",
            **run_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"\b{re.escape(encoder)}\b", result.stdout or "") is not None


@lru_cache(maxsize=32)
def encoder_usable(ffmpeg_bin: str, encoder: str) -> bool:
    """该编码器在本机**真的能编码**（做一次 1 帧最小编码验证）。

    为什么不能只看 ``-encoders``：构建里带 h264_nvenc 的 ffmpeg，
    在没有 N 卡的机器上会以「Cannot load nvcuda.dll」失败。
    这个探测只跑一次（lru_cache），换取的是「回推不会因为编码器选错而起不来」。
    """
    if not has_encoder(ffmpeg_bin, encoder):
        return False
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "nullsrc=s=128x128:d=0.2",
        "-frames:v", "1",
        "-c:v", encoder,
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=30, check=False, **run_kwargs()
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        logger.info("编码器 %s 在本机不可用，将回退软件编码", encoder)
    return result.returncode == 0


# --------------------------------------------------------------------------- #
# 控制台编码
# --------------------------------------------------------------------------- #
def enable_utf8_console() -> None:
    """把标准流切到 UTF-8，避免 Windows 控制台（cp936）打断中文日志。"""
    if not IS_WINDOWS:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 非文件流
            pass


# --------------------------------------------------------------------------- #
# 由 ffmpeg stderr 探测视频参数（ffprobe 的降级替代）
# --------------------------------------------------------------------------- #
_VIDEO_LINE = re.compile(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_RAW_LINE = re.compile(r"(\d{2,5})x(\d{2,5})")


def parse_video_info(stderr_text: str) -> tuple[int, int, float]:
    """从 ``ffmpeg -i`` 的 stderr 里抽出 (宽, 高, 帧率)。

    找不到的项返回 0，由调用方决定回退策略。
    """
    width = height = 0
    fps = 0.0
    for line in stderr_text.splitlines():
        if "Video:" not in line:
            continue
        match = _VIDEO_LINE.search(line) or _RAW_LINE.search(line)
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            fps_match = _FPS.search(line)
            if fps_match:
                try:
                    fps = round(float(fps_match.group(1)), 2)
                except ValueError:
                    fps = 0.0
            break
    return width, height, fps


def probe_with_ffmpeg(
    ffmpeg_bin: str,
    url: str,
    *,
    input_args: list[str] | None = None,
    timeout: float = 10.0,
) -> tuple[int, int, float]:
    """用 ``ffmpeg -i`` 探测流参数（无需 ffprobe）。

    故意让它「打不出输出就退出」：只给 ``-i`` 不给输出文件时，ffmpeg 会打印
    输入流信息然后以非零码退出，这正是我们要的——不建立完整解码会话。

    ``input_args`` 必须是**按 URL 协议自适应**的（见 ``core/sources.build_input_args``）：
    ``-rtsp_transport`` / ``-timeout`` 是 RTSP 解复用器的私有参数，
    套在本地 mp4 上会让 ffmpeg 直接报「Option not found」而探测失败。
    """
    command = [
        ffmpeg_bin,
        "-hide_banner",
        *(input_args or []),
        "-i", url,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
            encoding="utf-8",
            errors="replace",
            **run_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffmpeg 探测失败: %s", exc)
        return (0, 0, 0.0)
    return parse_video_info(result.stderr or "")


# --------------------------------------------------------------------------- #
# ONNX Runtime 执行提供器
# --------------------------------------------------------------------------- #
def resolve_providers(requested: list[str]) -> list[str]:
    """把请求的执行提供器过滤成**本机真正可用**的那部分。

    典型场景：配置里默认写了 TensorRT/CUDA，但内网机器只有 CPU 版
    onnxruntime。直接交给框架会在每次建会话时刷一堆告警，甚至因
    ``use_fp16`` 在 CPU 上不受支持而报错；这里提前收敛为
    ``CPUExecutionProvider``。
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - 没装 onnxruntime 时不做假设
        return requested

    resolved = [name for name in requested if name in available]
    if not resolved:
        resolved = ["CPUExecutionProvider"]
    if resolved != requested:
        logger.info("执行提供器按本机能力收敛为: %s", resolved)
    return resolved


def cpu_only(providers: list[str]) -> bool:
    """仅剩 CPU 时，fp16 无意义且部分算子不支持，需要显式关掉。"""
    return all(name == "CPUExecutionProvider" for name in providers)
