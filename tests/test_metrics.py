"""运行指标：并发快照与 GPU 采集降级。"""

from __future__ import annotations

import subprocess
import threading
import time

import campusclear.core.metrics as metrics_module
from campusclear.core.metrics import StreamMetrics, read_gpu_util


def test_snapshot_is_safe_while_worker_updates():
    metrics = StreamMetrics("cam_01")
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            metrics.record_latency(0.012)
            metrics.record_plan("测试", ["denoise", "sr"])

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(300):
            data = metrics.snapshot()
            assert data["camera_id"] == "cam_01"
            assert data["avg_latency_ms"] > 0
            assert data["stage_usage"]["sr"] > 0
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1)


def test_gpu_transient_failure_is_retried_not_disabled(monkeypatch):
    monkeypatch.setattr(metrics_module.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(metrics_module, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(metrics_module, "_GPU_RETRY_AT", 0.0)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)

    monkeypatch.setattr(metrics_module.subprocess, "run", timeout)
    assert read_gpu_util() is None
    assert metrics_module._GPU_AVAILABLE is True, "瞬时超时不能永久关闭 GPU 采集"
    assert metrics_module._GPU_RETRY_AT > 0

    # 冷却期内不再调用子进程
    monkeypatch.setattr(metrics_module, "_GPU_RETRY_AT", time.monotonic() + 60)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("冷却期内不应调用 nvidia-smi")

    monkeypatch.setattr(metrics_module.subprocess, "run", should_not_run)
    assert read_gpu_util() is None

    # 冷却结束后恢复采样
    monkeypatch.setattr(metrics_module, "_GPU_RETRY_AT", 0.0)

    def ok(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="42, 1024, 8192, 60\n", stderr=""
        )

    monkeypatch.setattr(metrics_module.subprocess, "run", ok)
    data = read_gpu_util()
    assert data is not None
    assert data["utilization"] == 42.0
    assert metrics_module._GPU_RETRY_AT == 0.0
