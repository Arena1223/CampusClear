"""拉流读取器的生命周期与超时。

现场最容易出问题的不是「连不上」，而是「连上了但不动」：
- 进程活着、管道没数据，旧的 read() 会永久阻塞；
- 通道重启时旧线程还在退避睡眠，醒来后又把 ffmpeg 拉起来；
- 停止时先关管道再 terminate，可能互相等锁。

这些用纯内存管道即可复现，不需要真实摄像头或 ffmpeg。
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from campusclear.core.ingest import RtspReader, RtspSourceError


class _FakeProcess:
    """只暴露 read() 需要的接口：stdout.fileno() 与 poll()。"""

    def __init__(self, read_fd: int) -> None:
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.stderr = None
        self._returncode = None

    def poll(self):
        return self._returncode


def _reader_with_pipe(**kwargs) -> tuple[RtspReader, _FakeProcess, int]:
    read_fd, write_fd = os.pipe()
    reader = RtspReader("rtsp://192.0.2.10:554/Streaming/Channels/101", **kwargs)
    reader.width, reader.height = 2, 2
    reader._frame_bytes = reader.width * reader.height * 3  # noqa: SLF001
    process = _FakeProcess(read_fd)
    reader._process = process  # noqa: SLF001
    return reader, process, write_fd


def test_read_times_out_instead_of_blocking_forever():
    reader, process, write_fd = _reader_with_pipe()
    try:
        start = time.monotonic()
        assert reader.read(timeout=0.3) is None
        elapsed = time.monotonic() - start
        assert elapsed < 1.5, "read() 必须在 timeout 附近返回，不能卡在管道上"
        assert reader.last_error == "读取超时"
    finally:
        process.stdout.close()
        os.close(write_fd)


def test_stop_interrupts_reconnect_backoff_immediately():
    reader = RtspReader(
        "rtsp://192.0.2.10:554/Streaming/Channels/101",
        reconnect_backoff=(30, 30),
    )
    reader.resume()
    results: list[bool] = []

    thread = threading.Thread(target=lambda: results.append(reader.reconnect()))
    thread.start()
    time.sleep(0.1)
    start = time.monotonic()
    reader.stop()
    thread.join(timeout=2)
    elapsed = time.monotonic() - start

    assert not thread.is_alive(), "stop() 必须能叫醒退避等待"
    assert elapsed < 1.0
    assert results == [False], "停止后 reconnect 必须返回 False"


def test_open_is_refused_after_stop():
    reader = RtspReader("rtsp://192.0.2.10:554/Streaming/Channels/101")
    reader.stop()
    with pytest.raises(RtspSourceError):
        reader.open()


def test_resume_allows_open_again():
    reader = RtspReader("rtsp://192.0.2.10:554/Streaming/Channels/101")
    reader.stop()
    reader.resume()
    assert reader._stop_event.is_set() is False  # noqa: SLF001
