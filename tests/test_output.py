"""回推编码器的输入契约。

rawvideo 没有帧头：写错尺寸时 ffmpeg 不会报错，只会按错误边界切分字节流，
最终表现为画面撕裂/跳段。因此编码器必须在写帧前做尺寸校验。
"""

from __future__ import annotations

import numpy as np

from campusclear.core.output import RtspPusher


class _FakeStdin:
    def __init__(self) -> None:
        self.written = 0

    def write(self, data: bytes) -> int:
        self.written += len(data)
        return len(data)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()

    def poll(self):
        return None


def test_pusher_rejects_frame_with_wrong_size():
    pusher = RtspPusher("rtsp://127.0.0.1:8554/cc_test", width=64, height=48)
    pusher._process = _FakeProcess()  # noqa: SLF001

    assert pusher.write(np.zeros((48, 64, 3), dtype=np.uint8)) is True
    assert pusher.write(np.zeros((96, 128, 3), dtype=np.uint8)) is False
    assert pusher.write_failures == 1
    assert pusher._process.stdin.written == 48 * 64 * 3  # noqa: SLF001
