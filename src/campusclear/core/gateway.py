"""本地 RTSP 网关（MediaMTX）的探测与托管。

为什么需要这一层：

「增强结果回推」输出的是标准 RTSP 流，但 RTSP 的 ANNOUNCE/RECORD **必须有
一个服务端接收**。项目默认目标地址是 ``rtsp://127.0.0.1:8554/<通道名>``，
如果本机没有网关在监听，FFmpeg 只能连接失败——而失败表现是「播放器打不开
这个地址」，用户看到的却是「回推功能坏了」，极难自查。

因此这里做两件事：

1. **探测**：批量建流前先看目标端口通不通；不通就明确跳过回推并只记一条
   日志，而不是每 60 秒反复拉起注定失败的 FFmpeg 进程。
2. **托管**：若 ``bin/`` 目录下自带 ``mediamtx``，且目标是本机地址，则在启动
   时顺带拉起网关，退出时关掉；没有网关二进制时安静地跳过。

网关只监听回环/内网地址，不改变「无公网依赖」这一硬约束。
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from ..compat import popen_kwargs

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def rtsp_endpoint_reachable(url: str, timeout: float = 0.6) -> bool:
    """目标 RTSP 端口是否可建立 TCP 连接。

    只做 TCP 握手，不发 RTSP 报文：目的是区分「没有网关」与「网关拒绝这次
    推流」，前者应当跳过，后者仍要重试。
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.hostname:
        return False
    port = parts.port or 554
    try:
        with socket.create_connection((parts.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_local_endpoint(url: str) -> bool:
    """目标是否指向本机（只有本机才由我们托管网关）。"""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


def find_gateway_binary(ffmpeg_bin: str | None) -> Path | None:
    """在 FFmpeg 同目录（即打包后的 ``bin/``）里找网关可执行文件。

    接受 ``mediamtx.exe`` / ``mediamtx`` 两种命名，找不到返回 None。
    刻意不在 PATH 里搜索：现场机器上不该因为装了别的东西就悄悄启用网关。
    """
    if not ffmpeg_bin:
        return None
    base = Path(ffmpeg_bin)
    candidates = []
    for directory in (base.parent, base.parent / "bin"):
        candidates.extend([directory / "mediamtx.exe", directory / "mediamtx"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class LocalGateway:
    """被托管的网关进程（或「没有网关」这一事实）。"""

    def __init__(self, process: subprocess.Popen | None = None, reason: str = "",
                 job: object | None = None) -> None:
        self.process = process
        self.reason = reason
        # Windows Job 句柄必须一直持有：句柄一关，KILL_ON_JOB_CLOSE 立刻生效
        self._job = job

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            logger.info("本地 RTSP 网关已停止")
        except OSError:  # pragma: no cover - 退出阶段尽力而为
            pass


def _assign_kill_on_close(process: subprocess.Popen) -> object | None:
    """Windows：把网关进程放进 Job Object，父进程消失时它必须一起结束。

    为什么需要：正常退出走 lifespan 会 ``stop()``，但现场更常见的是「任务管理器
    结束进程」「程序崩溃」「直接关电源」。没有这层保护，网关会变成孤儿进程一直
    占着 8554，下一次启动只能去「接管」而不是「托管」，配置改动也不生效。

    失败不是致命问题（拿不到 Job 就退回普通子进程），所以异常一律吞掉。
    """
    if not sys.platform.startswith("win"):  # pragma: no cover - 非 Windows 无此问题
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(job)
            return None
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle))):  # noqa: SLF001
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception as exc:  # noqa: BLE001 - 拿不到就退回普通子进程
        logger.debug("网关 Job Object 绑定失败（不影响回推）: %s", exc)
        return None


def ensure_local_gateway(settings, *, wait_seconds: float = 5.0) -> LocalGateway:
    """按需拉起本机网关。

    触发条件全部满足才动手：回推已启用 → 目标是本机 → ``bin/`` 里有二进制
    → 端口当前没人监听。任一不满足都返回未托管的 LocalGateway，并给出原因，
    由调用方决定是否提示。
    """
    base = getattr(settings, "rtsp_output_base", "") or ""
    if not getattr(settings, "output_enabled", True):
        return LocalGateway(reason="回推已关闭（CAMPUSCLEAR_OUTPUT_ENABLED=0）")
    if not is_local_endpoint(base):
        return LocalGateway(reason=f"回推目标是外部网关 {base}，不由本机托管")
    if rtsp_endpoint_reachable(base):
        return LocalGateway(reason=f"检测到已有 RTSP 网关 {base}")

    binary = find_gateway_binary(getattr(settings, "ffmpeg_bin", None))
    if binary is None:
        return LocalGateway(
            reason=(
                f"{base} 上没有 RTSP 网关，且 bin/ 目录未提供 mediamtx；"
                "回推将跳过（浏览器观看不受影响）"
            )
        )

    logger.info("启动本地 RTSP 网关: %s", binary)
    try:
        process = subprocess.Popen(
            [str(binary)],
            cwd=str(binary.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs(),
        )
    except OSError as exc:
        logger.error("本地 RTSP 网关启动失败: %s", exc)
        return LocalGateway(reason=f"网关启动失败: {exc}")

    job = _assign_kill_on_close(process)
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        if rtsp_endpoint_reachable(base, timeout=0.3):
            logger.info("本地 RTSP 网关就绪（%s）", base)
            return LocalGateway(process=process, reason=f"已托管 {base}", job=job)
        if process.poll() is not None:
            break
        time.sleep(0.2)

    logger.warning("本地 RTSP 网关未在 %.1fs 内就绪，回推可能仍不可用", wait_seconds)
    return LocalGateway(process=process if process.poll() is None else None,
                        reason="网关启动后未就绪", job=job)
