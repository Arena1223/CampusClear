"""多路调度器：增量 reload、锁粒度与健康重启边界。

这里全部使用假管线，不启动 ffmpeg。重点钉住三个现场事故点：
1. ``/system/reload`` 不能把没变化的通道全部重启；
2. 启动一路要 probe 十几秒，期间状态接口必须还能响应；
3. 已删除的通道不能被监控线程重新拉起来。
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from types import SimpleNamespace

import pytest

import campusclear.core.scheduler as scheduler_module
from campusclear.core.camera import CameraConfig
from campusclear.core.scheduler import StreamSupervisor


class FakePipeline:
    instances: list["FakePipeline"] = []
    start_gate: threading.Event | None = None

    def __init__(self, resolved, settings, enhancer, governor) -> None:
        self.camera = resolved.camera
        self.resolved = resolved
        self.state = SimpleNamespace(running=False, connected=False, output_ready=False)
        self.metrics = SimpleNamespace(
            last_frame_at=0.0,
            started_at=time.time(),
            reconnects=0,
            snapshot=lambda: {"camera_id": self.camera.id},
        )
        self._queue = queue.Queue()
        self.started = 0
        self.stopped = 0
        self.restarted = 0
        FakePipeline.instances.append(self)

    def start(self) -> None:
        if FakePipeline.start_gate is not None:
            FakePipeline.start_gate.wait(timeout=3)
        self.started += 1
        self.state.running = True

    def stop(self) -> None:
        self.stopped += 1
        self.state.running = False

    def restart(self) -> None:
        self.restarted += 1
        self.stop()
        self.start()

    def snapshot(self) -> dict:
        return {"camera_id": self.camera.id, "running": self.state.running}


class FakeStore:
    def __init__(self, cameras: list[CameraConfig]) -> None:
        self.cameras = cameras

    def list_cameras(self) -> list[CameraConfig]:
        # 与 CameraStore 一致：每次返回全新对象，避免调用方原地修改后
        # 影响已经在跑的管线所持有的配置快照。
        return [copy.deepcopy(camera) for camera in self.cameras]


def _camera(camera_id: str = "cam_01", name: str = "东门") -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name=name,
        source_rtsp=f"rtsp://192.0.2.10:554/Streaming/Channels/101",
    )


@pytest.fixture(autouse=True)
def _isolate_fakes(monkeypatch, settings):
    FakePipeline.instances = []
    FakePipeline.start_gate = None
    monkeypatch.setattr(scheduler_module, "StreamPipeline", FakePipeline)
    monkeypatch.setattr(settings, "max_streams", 8)
    monkeypatch.setattr(settings, "start_stagger_seconds", 0.0)
    yield
    FakePipeline.start_gate = None


def test_reload_does_not_restart_unchanged_camera(settings):
    camera = _camera()
    supervisor = StreamSupervisor(settings, FakeStore([camera]))

    supervisor.reload()
    assert len(FakePipeline.instances) == 1
    first = FakePipeline.instances[0]
    assert first.started == 1

    supervisor.reload()
    assert FakePipeline.instances == [first], "配置没变时不应重建管线"
    assert first.started == 1 and first.stopped == 0

    camera.name = "东门（改造后）"
    supervisor.reload()
    assert len(FakePipeline.instances) == 2
    assert first.stopped == 1
    assert FakePipeline.instances[1].started == 1


def test_reload_removes_disabled_camera(settings):
    camera = _camera()
    supervisor = StreamSupervisor(settings, FakeStore([camera]))
    supervisor.reload()
    pipeline = FakePipeline.instances[0]

    camera.enabled = False
    supervisor.reload()
    assert pipeline.stopped == 1
    assert supervisor.snapshot()["stream_count"] == 0


def test_status_snapshot_is_not_blocked_by_slow_start(settings):
    camera = _camera()
    supervisor = StreamSupervisor(settings, FakeStore([camera]))
    gate = threading.Event()
    FakePipeline.start_gate = gate

    thread = threading.Thread(target=supervisor.reload)
    thread.start()
    try:
        # 等 reload 进入 FakePipeline.start() 并卡在 gate 上
        deadline = time.monotonic() + 2
        while not FakePipeline.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        assert FakePipeline.instances, "reload 应该已经创建了管线"

        start = time.monotonic()
        status = supervisor.snapshot()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, "慢启动期间 /system/status 不能被 _lock 卡住"
        assert status["stream_count"] == 0  # 还没 start 完成，暂未登记
    finally:
        gate.set()
        thread.join(timeout=3)
    assert supervisor.snapshot()["stream_count"] == 1


def test_restart_of_removed_camera_is_rejected(settings):
    supervisor = StreamSupervisor(settings, FakeStore([]))
    assert supervisor.restart_camera("gone") is False
    assert supervisor.stop_camera("gone") is False


def test_max_streams_is_enforced(settings, monkeypatch):
    monkeypatch.setattr(settings, "max_streams", 1)
    cameras = [_camera("cam_01"), _camera("cam_02")]
    supervisor = StreamSupervisor(settings, FakeStore(cameras))

    supervisor.reload()
    assert len(FakePipeline.instances) == 1
    assert supervisor.snapshot()["stream_count"] == 1
    assert supervisor.snapshot()["max_streams"] == 1
