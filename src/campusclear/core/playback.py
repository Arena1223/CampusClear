"""回放增强：对一段已录像片段做离线增强（取证 / 放大档）。

与实时链路的区别
----------------
- 实时（``realtime`` 档）：要跟得上帧率，用快算法；
- 回放（``forensic`` 档）：不要求实时，用重算法（NL-means / 重 AI），慢慢跑。

本模块把「对一段视频做增强」抽成一个纯函数 ``enhance_clip``，
供 API 的任务队列与命令行工具共用，避免两处逻辑漂移。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .modes import EnhanceMode, Engine, build_for_mode
from .planner import build_plan
from .quality import QualityAnalyzer


@dataclass
class ClipResult:
    frames: int = 0
    enhanced_frames: int = 0
    reuse_frames: int = 0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    stage_hits: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        single = round(1000 / self.median_ms, 2) if self.median_ms else None
        return {
            "frames": self.frames,
            "enhanced_frames": self.enhanced_frames,
            "reuse_frames": self.reuse_frames,
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "single_route_fps": single,
            "stage_hits": self.stage_hits,
            "elapsed_s": round(self.elapsed_s, 1),
        }


def enhance_clip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mode: EnhanceMode | str = EnhanceMode.FORENSIC,
    engine: Engine | str = Engine.AUTO,
    models_dir: str | Path | None = None,
    scale: int | None = None,
    region: str | None = None,
    start: float | None = None,
    end: float | None = None,
    max_frames: int | None = None,
    progress_cb=None,
) -> ClipResult:
    """对视频片段做离线增强，写出 ``output_path``（mp4）。"""
    mode = EnhanceMode(mode)
    enhancer, policy, _info = build_for_mode(mode, models_dir=models_dir, engine=engine)
    if scale is not None:
        policy.base_scale = scale

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    analyzer = QualityAnalyzer()
    box = [int(v) for v in region.split(",")] if region else None

    writer = None
    result = ClipResult()
    times: list[float] = []
    prev = None
    t_begin = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end and t > end:
            break
        if max_frames and result.frames >= max_frames:
            break
        if box:
            x, y, w, h = box
            frame = frame[y : y + h, x : x + w]

        quality = analyzer.analyze(frame)
        plan = build_plan(quality, policy)
        if plan.reuse_previous and prev is not None:
            output = prev
            result.reuse_frames += 1
        else:
            t0 = time.perf_counter()
            output = enhancer.enhance(frame, plan)
            times.append((time.perf_counter() - t0) * 1000)
            result.enhanced_frames += 1
        for stage in plan.stages():
            result.stage_hits[stage] = result.stage_hits.get(stage, 0) + 1
        prev = output

        if writer is None:
            h, w = output.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(output)
        result.frames += 1
        if progress_cb:
            progress_cb(result.frames)

    cap.release()
    if writer is not None:
        writer.release()

    result.elapsed_s = time.perf_counter() - t_begin
    if times:
        result.median_ms = statistics.median(times)
        result.p95_ms = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    if result.frames == 0:
        raise RuntimeError("没有处理任何帧（区间为空或视频损坏）")
    return result
