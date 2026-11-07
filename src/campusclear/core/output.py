"""增强结果回推。

输出仍是一条标准 RTSP 流，推给内网自建的 MediaMTX（轻量 RTSP/RTMP/HLS 网关），
安防平台按原有方式拉取即可 —— **既有平台零改造**是本项目的重要工程约束：
不要求学校更换 NVR、不要求厂家配合改协议、不要求前端页面重新开发。

编码优先级：NVENC 硬件编码 → libx264 软件编码。
1080p 编码在 CPU 上约占 2-3 核，若服务器核心紧张则硬件编码收益明显。
"""

from __future__ import annotations

import logging
import subprocess

import numpy as np

from ..compat import encoder_usable, popen_kwargs

logger = logging.getLogger(__name__)


class RtspPusher:
    """把 BGR 帧序列编码后推送到 RTSP 服务器。"""

    def __init__(
        self,
        target_url: str,
        *,
        width: int,
        height: int,
        fps: int = 25,
        bitrate_kbps: int = 4000,
        gop: int = 50,
        ffmpeg_bin: str = "ffmpeg",
        prefer_hardware: bool = True,
        username: str = "",
        password: str = "",
    ) -> None:
        self.target_url = _inject_credentials(target_url, username, password)
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.gop = gop
        self.ffmpeg_bin = ffmpeg_bin
        self.prefer_hardware = prefer_hardware

        self._process: subprocess.Popen | None = None
        self.encoder = "none"
        self.frames_written = 0
        self.write_failures = 0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        encoder_args = self._encoder_args()
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            *encoder_args,
            "-b:v", f"{self.bitrate_kbps}k",
            "-maxrate", f"{int(self.bitrate_kbps * 1.3)}k",
            "-bufsize", f"{self.bitrate_kbps * 2}k",
            "-g", str(self.gop),
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.target_url,
        ]
        logger.info("启动回推: %s (%s)", _mask_url(self.target_url), self.encoder)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3,
            **popen_kwargs(),           # Windows 下隐藏 ffmpeg 控制台窗口
        )

    def _encoder_args(self) -> list[str]:
        """选择编码器。

        这里必须先探测再选择，不能靠「让 FFmpeg 自己回退」：
        FFmpeg 遇到构建里不存在的编码器是**直接报错退出**，不是降级到软件编码。
        而「构建里有 h264_nvenc」也不等于「本机能用」——没有 N 卡的机器上
        一调用就报「Cannot load nvcuda.dll」。因此用 encoder_usable 做真实验证。
        """
        if self.prefer_hardware and encoder_usable(self.ffmpeg_bin, "h264_nvenc"):
            self.encoder = "h264_nvenc"
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4",       # p1 最快 / p7 最慢，p4 是实时场景的均衡点
                "-tune", "ll",         # 低延迟调优
                "-rc", "cbr",
                "-zerolatency", "1",
            ]
        if self.prefer_hardware:
            logger.info("未检测到 h264_nvenc（无 NVIDIA 编码器），回推改用 libx264 软件编码")
        self.encoder = "libx264"
        return [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-profile:v", "main",
        ]

    # ------------------------------------------------------------------ #
    def write(self, frame: np.ndarray) -> bool:
        """写入一帧。返回 False 表示管道不可用，需要重建。"""
        # 最后一道尺寸防线：rawvideo 没有帧头，尺寸错了 ffmpeg 不会报错，
        # 只会把字节流按错误的边界切开，表现为画面撕裂/跳段。宁可拒绝写帧，
        # 也不要让错误尺寸污染整条输出流。
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width:
            self.write_failures += 1
            logger.error(
                "回推帧尺寸不匹配: 期望 %dx%d，实际 %s（%s）",
                self.width,
                self.height,
                "x".join(str(v) for v in frame.shape[:2]) if frame.ndim >= 2 else "?",
                self.target_url,
            )
            return False

        process = self._process
        if process is None or process.stdin is None:
            return False
        if process.poll() is not None:
            logger.warning("回推进程已退出（返回码 %s），需要重建", process.returncode)
            return False
        try:
            process.stdin.write(frame.tobytes())
            self.frames_written += 1
            return True
        except (BrokenPipeError, OSError) as exc:
            self.write_failures += 1
            logger.warning("写帧失败: %s", exc)
            return False

    def restart(self) -> bool:
        self.close()
        try:
            self.start()
            return True
        except OSError as exc:
            logger.error("回推重建失败: %s", exc)
            return False

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                try:
                    process.stdin.write(b"")  # 冲一下缓冲
                except (BrokenPipeError, OSError):
                    pass
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        except OSError:  # pragma: no cover
            pass

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stats(self) -> dict:
        return {
            "encoder": self.encoder,
            "frames_written": self.frames_written,
            "write_failures": self.write_failures,
            "alive": self.alive,
        }


def _inject_credentials(url: str, username: str, password: str) -> str:
    """把 RTSP 凭证写进 URL。输出流必须带鉴权，不能让内网任意主机直接拉走画面。"""
    if not username:
        return url
    if "@" in url.split("://", 1)[-1].split("/", 1)[0]:
        return url
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    from urllib.parse import quote

    return f"{scheme}://{quote(username)}:{quote(password)}@{rest}"


def _mask_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"
