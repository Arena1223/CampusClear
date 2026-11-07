"""MJPEG 观看路径：异步生成器 + 同一帧只编码一次。"""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np
import pytest

from campusclear.api.client import _mjpeg_camera
from campusclear.core.camera import CameraConfig
from campusclear.core.enhancer import ClassicalEnhancer
from campusclear.core.pipeline import StreamPipeline
from campusclear.core.planner import LoadGovernor


@pytest.fixture()
def pipeline(settings):
    camera = CameraConfig(
        id="mjpeg_01",
        name="MJPEG",
        source_rtsp="rtsp://192.0.2.10:554/Streaming/Channels/101",
    )
    return StreamPipeline(
        camera.resolve(settings),
        settings,
        ClassicalEnhancer(),
        LoadGovernor(),
    )


def test_snapshot_jpeg_cache_encodes_each_frame_once(pipeline, monkeypatch):
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)
    pipeline._last_output = frame  # noqa: SLF001
    pipeline._output_seq = 1  # noqa: SLF001

    calls = {"n": 0}
    real_imencode = cv2.imencode

    def counting_imencode(*args, **kwargs):
        calls["n"] += 1
        return real_imencode(*args, **kwargs)

    monkeypatch.setattr("campusclear.core.pipeline.cv2.imencode", counting_imencode)

    first = pipeline.snapshot_jpeg_cached(raw=False, quality=85)
    second = pipeline.snapshot_jpeg_cached(raw=False, quality=85)
    third = pipeline.snapshot_jpeg_cached(raw=False, quality=85)

    assert first and first == second == third
    assert calls["n"] == 1, "同一帧被多个观看者请求时只能编码一次"

    # 新帧才需要重新编码
    pipeline._last_output = frame.copy()  # noqa: SLF001
    pipeline._output_seq = 2  # noqa: SLF001
    pipeline.snapshot_jpeg_cached(raw=False, quality=85)
    assert calls["n"] == 2


class _OneShotRequest:
    def __init__(self) -> None:
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > 1


def test_async_mjpeg_generator_yields_and_exits(pipeline):
    frame = np.full((48, 64, 3), 120, dtype=np.uint8)
    pipeline._last_output = frame  # noqa: SLF001
    pipeline._output_seq = 1  # noqa: SLF001
    pipeline.state.running = True

    async def first_chunk():
        gen = _mjpeg_camera(_OneShotRequest(), pipeline, raw=False)
        try:
            return await gen.__anext__()
        finally:
            await gen.aclose()

    chunk = asyncio.run(first_chunk())
    assert chunk.startswith(b"--frame")
    assert b"image/jpeg" in chunk and b"JFIF" in chunk
