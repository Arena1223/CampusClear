"""RTSP 拉流。

为什么不用 ``cv2.VideoCapture``？

1. 无法精细控制 RTSP 传输方式与超时，内网交换机抖动时会静默卡死而不报错；
2. 内置缓冲不可控，最长可积压数秒，对「实时清晰化」是致命的；
3. 出错后的重连语义不明确，长时间运行容易出现「进程还活着但画面冻住」。

改用 FFmpeg 子进程 + ``rawvideo`` 管道：
- 参数完全可控（TCP 传输、低延迟、超时自恢复、丢弃损坏帧）；
- 一帧固定 ``H*W*3`` 字节，读满即一帧，边界清晰，不存在「半个包」问题；
- 进程死了就是死了，可被监督器明确感知并重建。

代价是每路多一个 FFmpeg 进程（约 20-40MB 内存），在校园内网服务器上完全可接受。
"""

from __future__ import annotations

import collections
import io
import logging
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..compat import (
    ffprobe_available,
    ffprobe_of,
    fps_mode_supported,
    popen_kwargs,
    probe_with_ffmpeg,
    run_kwargs,
)
from .sources import build_input_args, detect_scheme

logger = logging.getLogger(__name__)


@dataclass
class StreamProbe:
    """探测到的流参数。"""

    width: int
    height: int
    fps: float
    codec: str = "unknown"


class RtspSourceError(RuntimeError):
    """拉流不可恢复错误。"""


class RtspReader:
    """单路 RTSP 读取器（自带重连）。"""

    def __init__(
        self,
        url: str,
        *,
        ffmpeg_bin: str = "ffmpeg",
        timeout_us: int = 5_000_000,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        transport: str = "tcp",
        reconnect_backoff: tuple[int, int] = (1, 30),
        force_size: bool = True,
        max_width: int = 0,
        max_height: int = 0,
    ) -> None:
        self.url = url
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_us = timeout_us
        self.transport = transport
        self.reconnect_backoff = reconnect_backoff
        # 强制解码输出尺寸 == self.width/height：rawvideo 管道是按固定帧长读的，
        # 源分辨率与预期不一致时会出现「帧边界错位 → 画面雪花」。
        self.force_size = force_size
        # 解码分辨率上限（0 = 不限制）。这是纯 CPU 机器上唯一能换来帧率的手段：
        # 增强耗时大致随像素数线性增长，4MP（2560x1440）在 CPU 上跑不到 1 fps，
        # 先缩到 1280 宽再增强，观感「动得起来」远比「分辨率高但卡成幻灯片」有用。
        # 注意它是**上限**，等比缩放且只缩不放，宽高都取偶数（H.264/yuv420 要求）。
        self.max_width = max(0, int(max_width or 0))
        self.max_height = max(0, int(max_height or 0))

        # 调用方是否显式钉死了尺寸。钉死时 probe 只用来校准 fps，不改尺寸
        # ——原实现用「等于 1920x1080 就当占位符」来判断，遇到真的 1080p 流
        # 会误判，这里换成显式标记。
        self._size_pinned = bool(width) or bool(height)

        # 目标输出尺寸。源流尺寸未知时先用 1080p 占位，probe 后修正。
        # 一律过一遍 _fit：显式尺寸也可能超上限、也可能是奇数。
        self.width, self.height = self._fit(width or 1920, height or 1080)
        self.fps = fps or 25

        self._process: subprocess.Popen | None = None
        self._frame_bytes = self.width * self.height * 3
        self._lock = threading.Lock()
        # ``reconnects`` 是生命周期累计值，``_consecutive_failures`` 只用于
        # 计算退避。成功读到一帧就清零后者，避免「偶尔断一次」之后退避
        # 永久停在 30 秒，恢复要等半分钟。
        self.reconnects = 0
        self._consecutive_failures = 0
        self.last_error: str = ""
        # 显式停止标志：``close()`` 是重连内部的「关旧进程」，不能置位；
        # 只有 ``stop()``（通道销毁）才置位。这样退避等待可以被立刻打断，
        # 也不会出现「停完又自己 open 回来」的孤儿 ffmpeg。
        self._stop_event = threading.Event()

        # stderr 持续排空：ffmpeg 的 stderr 是 PIPE，不读满就会让进程写阻塞。
        # 用一条守护线程把 stderr 行循环塞进有界双端队列，进程退出时
        # 调用方可以 ``drain_stderr()`` 取出来看真正的退出原因——
        # 否则日志只能写「原因：未知」，排查无从下手。
        self._stderr_buf: collections.deque[str] = collections.deque(maxlen=50)
        self._stderr_stream: io.BufferedReader | None = None
        self._stderr_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def _fit(self, width: int, height: int) -> tuple[int, int]:
        """按 ``max_width`` / ``max_height`` 等比缩小（只缩不放），宽高取偶数。

        三点都是踩过的：
        1. **只缩不放**：上限是上界。源本身比上限还小就原样放行——
           把 720p 源放大到 1920 再增强，纯粹是拿算力换更大的马赛克。
        2. **整数运算**：``3840 * (720/2160)`` 在 IEEE754 下是
           1279.9999999999998，``int()`` 一截就变成 1278。差两个像素肉眼看不出，
           但「按上限算出来却不是上限」这种事会在现场反复误导排查的人。
        3. **宽高取偶**：yuv420p 的色度平面是 2x2 下采样，奇数宽高会让
           ffmpeg 的 ``scale`` 直接报 "width not divisible by 2" 拒跑。
        """
        if width <= 0 or height <= 0:
            return width, height
        origin = (width, height)

        if self.max_width > 0 and width > self.max_width:
            height = max(1, height * self.max_width // width)
            width = self.max_width
        if self.max_height > 0 and height > self.max_height:
            width = max(1, width * self.max_height // height)
            height = self.max_height

        width = max(2, (width // 2) * 2)
        height = max(2, (height // 2) * 2)

        if (width, height) != origin:
            logger.info(
                "解码尺寸由 %dx%d 调整为 %dx%d（上限/偶数对齐；增强耗时随像素数"
                "增长，纯 CPU 机器上这是换帧率最有效的一档）",
                origin[0], origin[1], width, height,
            )
        return width, height

    def configure(self, probe: StreamProbe) -> None:
        """按探测结果设定实际读取尺寸（调用方未显式指定尺寸时才采纳探测值）。"""
        with self._lock:
            if not self._size_pinned and probe.width and probe.height:
                self.width, self.height = probe.width, probe.height
            # 无论尺寸来自探测还是调用方，都要过一遍 _fit：它同时负责
            # 「应用分辨率上限」与「宽高对齐到偶数」两件事。
            self.width, self.height = self._fit(self.width, self.height)
            self.fps = probe.fps or self.fps
            self._frame_bytes = self.width * self.height * 3

    def probe(self, timeout: float = 10.0) -> StreamProbe | None:
        """探测流参数：优先 ffprobe，**没有 ffprobe 时退化为解析 ``ffmpeg -i``**。

        为什么要有降级路径：Windows / 离线部署常见「只有一个静态 ffmpeg」的场景
        （例如 PyPI 的 imageio-ffmpeg 就只带 ffmpeg 不带 ffprobe）。此时若直接放弃
        探测，reader 会按占位尺寸 1920x1080 去解析 rawvideo 管道，而 4MP 摄像头实际
        输出 2560x1440 —— 帧边界整段错位，画面会碎成雪花。

        两者都失败才返回 None，由调用方按默认值处理。
        """
        if ffprobe_available(self.ffmpeg_bin):
            probed = self._probe_ffprobe(timeout)
            if probed is not None:
                return probed
        else:
            logger.info("未发现 ffprobe，改用 ffmpeg -i 解析流参数")

        width, height, fps = probe_with_ffmpeg(
            self.ffmpeg_bin,
            self.url,
            input_args=build_input_args(
                self.url, rtsp_transport=self.transport, timeout_us=self.timeout_us
            ),
            timeout=timeout,
        )
        if not width or not height:
            logger.warning(
                "流参数探测失败，沿用占位尺寸 %dx%d（解码端会按此尺寸强制缩放）",
                self.width, self.height,
            )
            return None
        return StreamProbe(width=width, height=height, fps=fps or self.fps, codec="unknown")

    def _probe_ffprobe(self, timeout: float = 10.0) -> StreamProbe | None:
        """用 ffprobe 探测流参数；不可用或异常时返回 None 交给上层降级。"""
        ffprobe = ffprobe_of(self.ffmpeg_bin)
        command = [
            ffprobe,
            "-v", "error",
            "-rtsp_transport", self.transport,
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
            "-of", "default=noprint_wrappers=1",
            self.url,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                **run_kwargs(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffprobe 不可用或超时: %s", exc)
            return None
        logger.debug(
            "ffprobe 原始输出 %s: stdout=%r stderr=%r",
            _mask_url(self.url),
            (result.stdout or "")[:1000],
            (result.stderr or "")[:1000],
        )
        if result.returncode != 0:
            logger.warning("ffprobe 探测失败: %s", result.stderr.strip()[:200])
            return None

        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()

        try:
            width = int(values.get("width", 0)) or self.width
            height = int(values.get("height", 0)) or self.height
        except ValueError:
            return None

        fps = self.fps
        raw_rate = values.get("avg_frame_rate", "")
        if "/" in raw_rate:
            num, _, den = raw_rate.partition("/")
            try:
                if float(den) > 0:
                    fps = round(float(num) / float(den), 2)
            except ValueError:
                pass
        return StreamProbe(width=width, height=height, fps=fps, codec=values.get("codec_name", "unknown"))

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """启动 FFmpeg 解码进程。

        输入参数按 URL 协议自适应（rtsp / rtmp / http-flv / hls / rtp / 本地文件），
        见 ``core/sources.py``——不再只认 RTSP。
        """
        # 帧同步参数：-vsync 自 FFmpeg 5.1 起被 -fps_mode 取代、8.0 起移除，
        # 因此按实测版本号选择写法，而不是写死某一个。
        sync_args = (
            ["-fps_mode", "passthrough"]
            if fps_mode_supported(self.ffmpeg_bin)
            else ["-vsync", "0"]
        )
        # 强制把解码结果缩放到 self.width×self.height：
        # rawvideo 管道是按固定帧长（H*W*3）读取的，只要输出尺寸与预期不一致，
        # 帧边界就会整体错位，表现为「画面碎成雪花」而非黑屏——
        # 这类故障极难从现象反推原因，所以在解码端直接钉死尺寸。
        scale_args = (
            ["-vf", f"scale={self.width}:{self.height}:flags=bilinear"]
            if self.force_size
            else []
        )
        # 本地录像/文件也是合法的「源」。若不加以限制，ffmpeg 会用最快速度把整段
        # 文件解完（30 秒的片段可能 2 秒读完），随即 EOF → 触发重连 → 再解一遍，
        # 于是通道在监控面板上反复上下线。`-re` 让它按原始帧率吐出，
        # 行为与直播源对齐，也让「拿一段录像当测试源」这种用法真正可用。
        replay_args = ["-re"] if detect_scheme(self.url) in ("", "file") else []
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            *replay_args,
            *build_input_args(
                self.url, rtsp_transport=self.transport, timeout_us=self.timeout_us
            ),
            "-i", self.url,
            *scale_args,
            "-an",                      # 丢弃音频：安防回传只需画面
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            *sync_args,                 # 不做帧率重采样，保持源时序
            "-",
        ]
        logger.info(
            "启动拉流: %s (%dx%d, transport=%s)",
            _mask_url(self.url), self.width, self.height, self.transport,
        )
        logger.debug(
            "拉流命令: %s",
            " ".join(_mask_url(str(part)) if "://" in str(part) else str(part) for part in command),
        )
        with self._lock:
            # 停止后残留的线程若再调 open，必须在这里被挡住，否则会重新拉起
            # 一个没人读、没人关的 ffmpeg 进程。
            if self._stop_event.is_set():
                raise RtspSourceError("读取器已停止，拒绝重新打开")
            # 先关旧的（防泄漏：上一次 open 异常或外部调用方忘记 close）
            self._close_locked()
            self._stderr_buf.clear()
            # bufsize=0 有两个目的：
            # 1. 避免每路 1080p 默认分配 ~12MB 读缓冲（100 路就是 1.2GB）；
            # 2. stdout 变成裸 FileIO，配合 select/os.read 才能实现真正的读超时。
            # stderr 外面单独包一层 BufferedReader，保留 readline 的按行语义。
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **popen_kwargs(),           # Windows 下隐藏 ffmpeg 控制台窗口
            )
            process = self._process
            self._stderr_stream = (
                io.BufferedReader(process.stderr, buffer_size=64 * 1024)
                if process.stderr is not None
                else None
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr_loop,
                args=(self._stderr_stream,),
                name=f"cc-stderr-{id(self)}",
                daemon=True,
            )
            self._stderr_thread.start()

    def _drain_stderr_loop(self, stream: io.BufferedReader | None) -> None:
        """持续读取 ffmpeg stderr，存入有界队列。

        为什么必须开线程而不是「进程退出时再 read」：Windows 管道的
        ``read()`` 是阻塞的，而 stderr PIPE 有固定缓冲（约 4KB）。ffmpeg
        在 ``-loglevel error`` 下输出不多，但一旦缓冲写满，进程的 stdout
        写也会跟着卡住——表现为「画面突然冻住」。
        """
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_buf.append(line)
        except (OSError, ValueError):
            # 进程已关闭 / fd 被外部 close：静默退出
            pass

    def drain_stderr(self) -> str:
        """取走累积的 stderr 内容（进程退出后诊断用）。"""
        return "\n".join(self._stderr_buf) if self._stderr_buf else ""

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """读取一帧。返回 ``None`` 表示本路当前不可用，调用方应触发重连。

        ``timeout`` 是**真超时**：底层用 select（POSIX）/ PeekNamedPipe
        （Windows）轮询管道，避免 ffmpeg 活着但不吐数据时把线程永久卡在
        ``read()`` 里——UDP/RTP 丢包、网络半开都会出现这种「静默僵死」。
        """
        with self._lock:
            process = self._process
        if process is None or process.stdout is None:
            return None
        try:
            fd = process.stdout.fileno()
        except (OSError, ValueError):
            return None

        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while len(buffer) < self._frame_bytes:
            if self._stop_event.is_set():
                self.last_error = "读取器已停止"
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.last_error = "读取超时"
                return None
            chunk = _read_pipe_chunk(
                fd, self._frame_bytes - len(buffer), min(remaining, 0.5), process
            )
            if chunk is None:
                # 本次轮询没有数据，回去检查停止标志/剩余时间
                continue
            if not chunk:
                if self._stop_event.is_set():
                    return None
                # EOF = ffmpeg 已退出。读 stderr 看 ffmpeg 报了什么错，
                # 而不是只写一句「流中断」就重连——那样日志永远只有「原因：未知」。
                rc = process.poll()
                stderr_tail = self.drain_stderr()[-500:]
                if stderr_tail:
                    self.last_error = f"ffmpeg 退出(rc={rc}): {stderr_tail}"
                    logger.error(
                        "拉流进程退出 %s rc=%s stderr:\n%s",
                        _mask_url(self.url), rc, stderr_tail,
                    )
                else:
                    self.last_error = f"解码进程已退出(rc={rc}，无 stderr 输出)"
                return None
            buffer.extend(chunk)
        # 真正读满一帧才重置退避，避免「Popen 成功但流不可用」时退避被反复清零
        self._consecutive_failures = 0
        frame = np.frombuffer(bytes(buffer), dtype=np.uint8)
        return frame.reshape((self.height, self.width, 3))

    def reconnect(self) -> bool:
        """按指数退避重建拉流连接。返回是否成功打开。

        退避等待用 ``Event.wait`` 而不是 ``sleep``：通道销毁时 ``stop()``
        会立即把它叫醒并返回 False，不会出现「join 超时后旧线程又自己
        open 回来」的双 reader/孤儿进程。
        """
        if self._stop_event.is_set():
            return False
        self.close()
        self.reconnects += 1
        self._consecutive_failures += 1
        base, maximum = self.reconnect_backoff
        delay = min(maximum, base * (2 ** min(self._consecutive_failures - 1, 5)))
        logger.warning(
            "第 %d 次重连 %s，等待 %ds（原因：%s）",
            self._consecutive_failures, _mask_url(self.url), delay,
            self.last_error or "未知",
        )
        if self._stop_event.wait(delay):
            return False
        if self._stop_event.is_set():
            return False
        try:
            self.open()
            return True
        except (OSError, RtspSourceError) as exc:
            self.last_error = f"重连失败: {exc}"
            return False

    def stop(self) -> None:
        """通道销毁时调用：置停止标志并关闭进程。

        与 ``close()`` 的区别：``close()`` 只关闭当前进程，重连流程还会
        重新打开；``stop()`` 之后任何 ``open()`` 都会被拒绝。
        """
        self._stop_event.set()
        self.close()

    def resume(self) -> None:
        """通道重启时调用：允许重新打开（仅 ``Pipeline.start`` 使用）。"""
        self._stop_event.clear()
        self._consecutive_failures = 0

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """在已持有 ``self._lock`` 的前提下关闭进程。

        **先 terminate、再关管道**：先关 stdout 会让另一个线程里阻塞的
        ``os.read``/close 互相等待，停流或关服务时可能直接挂死。terminate
        之后子进程关闭写端，阻塞读会自然收到 EOF。
        """
        process = self._process
        self._process = None
        stream = self._stderr_stream
        self._stderr_stream = None
        thread = self._stderr_thread
        self._stderr_thread = None

        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        pass
            except OSError:  # pragma: no cover
                pass

            for pipe in (process.stdout, process.stderr if stream is None else None):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        # stderr 线程只读本地管道，join 很短；避免它在下一次 open 后
        # 把旧进程的错误写进新缓冲区。
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1)

    @property
    def alive(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None


def _read_pipe_chunk(
    fd: int, size: int, timeout: float, process: subprocess.Popen
) -> bytes | None:
    """从管道 fd 读一块数据。

    返回 ``b""`` 表示 EOF，``None`` 表示本次等待超时但仍可继续。
    POSIX 用 select；Windows 的 select 只支持 socket，改用 PeekNamedPipe
    轮询（间隔 50ms，CPU 开销可忽略）。
    """
    if sys.platform.startswith("win"):
        return _read_pipe_chunk_windows(fd, size, timeout, process)
    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (OSError, ValueError):
        return b""
    if not ready:
        return None
    try:
        return os.read(fd, size)
    except (OSError, ValueError):
        return b""


def _read_pipe_chunk_windows(
    fd: int, size: int, timeout: float, process: subprocess.Popen
) -> bytes | None:
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(fd)
    except OSError:
        return b""
    deadline = time.monotonic() + timeout
    while True:
        available = _peek_named_pipe(handle)
        if available:
            try:
                return os.read(fd, min(size, available))
            except (OSError, ValueError):
                return b""
        # 进程已退出：即使 Peek 显示 0，也再 read 一次拿 EOF/剩余数据
        if process.poll() is not None:
            try:
                return os.read(fd, size)
            except (OSError, ValueError):
                return b""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.05, remaining))


_kernel32 = None


def _peek_named_pipe(handle: int) -> int:
    """Windows：返回管道中可读字节数；失败返回 0。"""
    global _kernel32
    import ctypes
    from ctypes import wintypes

    if _kernel32 is None:
        _kernel32 = ctypes.windll.kernel32
        _kernel32.PeekNamedPipe.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        _kernel32.PeekNamedPipe.restype = wintypes.BOOL
    available = wintypes.DWORD()
    ok = _kernel32.PeekNamedPipe(
        wintypes.HANDLE(handle), None, 0, None, ctypes.byref(available), None
    )
    return int(available.value) if ok else 0


def _mask_url(url: str) -> str:
    """日志脱敏：抹掉 RTSP URL 中的账号密码。"""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"
