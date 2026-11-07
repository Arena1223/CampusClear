"""运行指标采集。

三条底线要求：
1. 只读本地数据，不发起任何网络请求；
2. 无第三方监控依赖（不引入 Prometheus 客户端也可运行，可选导出）；
3. 采集本身的开销可以忽略（纯计数 + 滑动窗口，无锁竞争热点）。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ..compat import run_kwargs


@dataclass
class StreamMetrics:
    """单路通道运行时指标。"""

    camera_id: str
    started_at: float = field(default_factory=time.time)

    frames_read: int = 0
    frames_dropped: int = 0          # 队列背压丢弃
    frames_enhanced: int = 0         # 真正走过推理
    frames_reused: int = 0           # 静止帧/降帧复用
    frames_written: int = 0
    inference_seconds: float = 0.0

    last_frame_at: float = 0.0
    last_error: str = ""
    reconnects: int = 0

    stage_usage: dict[str, int] = field(default_factory=dict)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=120))
    _plan_reasons: deque = field(default_factory=lambda: deque(maxlen=20))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ #
    def record_latency(self, seconds: float) -> None:
        with self._lock:
            self._latencies.append(seconds)
            self.inference_seconds += seconds

    def record_plan(self, reason: str, stages: list[str]) -> None:
        with self._lock:
            self._plan_reasons.append(reason)
            for stage in stages:
                self.stage_usage[stage] = self.stage_usage.get(stage, 0) + 1

    # ------------------------------------------------------------------ #
    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            values = list(self._latencies)
        return _average_ms(values)

    @property
    def p95_latency_ms(self) -> float:
        with self._lock:
            values = list(self._latencies)
        return _percentile_ms(values, 0.95)

    @property
    def effective_fps(self) -> float:
        uptime = max(1e-6, time.time() - self.started_at)
        return self.frames_written / uptime

    @property
    def drop_ratio(self) -> float:
        total = self.frames_read
        return self.frames_dropped / total if total else 0.0

    @property
    def reuse_ratio(self) -> float:
        total = self.frames_enhanced + self.frames_reused
        return self.frames_reused / total if total else 0.0

    def snapshot(self) -> dict:
        # worker 线程会持续 append 延迟/写 stage_usage；这里必须一次性拷贝
        # 再计算，否则 sorted(deque) 可能抛 "deque mutated during iteration"，
        # 让 /system/metrics 偶发 500。
        with self._lock:
            reasons = list(self._plan_reasons)[-1:] or [""]
            latencies = list(self._latencies)
            stage_usage = dict(self.stage_usage)
        return {
            "camera_id": self.camera_id,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "frames_enhanced": self.frames_enhanced,
            "frames_reused": self.frames_reused,
            "frames_written": self.frames_written,
            "effective_fps": round(self.effective_fps, 2),
            "avg_latency_ms": round(_average_ms(latencies), 2),
            "p95_latency_ms": round(_percentile_ms(latencies, 0.95), 2),
            "drop_ratio": round(self.drop_ratio, 4),
            "reuse_ratio": round(self.reuse_ratio, 4),
            "reconnects": self.reconnects,
            "stage_usage": stage_usage,
            "last_plan": reasons[0],
            "last_error": self.last_error,
        }


def _average_ms(values: list[float]) -> float:
    return (sum(values) / len(values) * 1000.0) if values else 0.0


def _percentile_ms(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index] * 1000.0


# --------------------------------------------------------------------------- #
# GPU 采集
# --------------------------------------------------------------------------- #
_GPU_LOCK = threading.Lock()
_GPU_AVAILABLE = True
# nvidia-smi 偶发超时/驱动忙时不能永久放弃采集，否则负载治理从此只看队列。
_GPU_RETRY_AT = 0.0
_GPU_RETRY_SECONDS = 60.0


def read_gpu_util(device_id: int = 0) -> dict | None:
    """读取 GPU 利用率与显存。

    优先使用 ``nvidia-smi``（CUDA 镜像里必然存在，无需额外 Python 依赖）；
    不可用时返回 None，由上层按「无观测」处理，绝不因此报错或阻塞出流。

    「没有 nvidia-smi」才永久关闭采集；超时、驱动忙、临时非零返回只做
    60 秒冷却后重试。原实现一次超时就把 GPU 观测永久关掉，负载治理会
    在全系统最需要它的时候失效。
    """
    global _GPU_AVAILABLE, _GPU_RETRY_AT
    if not _GPU_AVAILABLE:
        return None
    now = time.monotonic()
    with _GPU_LOCK:
        if now < _GPU_RETRY_AT:
            return None
        if shutil.which("nvidia-smi") is None:
            _GPU_AVAILABLE = False
            return None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_id}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                **run_kwargs(),
            )
        except (OSError, subprocess.TimeoutExpired):
            _GPU_RETRY_AT = time.monotonic() + _GPU_RETRY_SECONDS
            return None
        if result.returncode != 0:
            _GPU_RETRY_AT = time.monotonic() + _GPU_RETRY_SECONDS
            return None
        _GPU_RETRY_AT = 0.0
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 4:
            return None
        try:
            used, total = float(parts[1]), float(parts[2])
            return {
                "utilization": float(parts[0]),
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_ratio": round(used / total, 4) if total else 0.0,
                "temperature_c": float(parts[3]),
            }
        except (ValueError, IndexError):
            return None
