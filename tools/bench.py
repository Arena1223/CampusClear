#!/usr/bin/env python3
"""性能基准测试（离线运行，不产生任何外网流量）。

用途：
1. **验收**：在目标服务器上给出「单阶段延迟 / 单帧端到端延迟 / 等效可承载路数」
   的实测数据，直接作为申报材料与验收报告的数据来源；
2. **选型**：对比不同执行提供器（TensorRT / CUDA / CPU）与不同分辨率下的耗时，
   为「该给这台服务器配多少路」提供依据；
3. **回归**：升级模型或代码后确认性能没有劣化。

用法：
    # 不依赖任何权重，只测经典算法与图像 I/O 开销
    python tools/bench.py --synthetic --width 1920 --height 1080 --runs 30

    # 有权重时测真实推理
    python tools/bench.py --models-dir ./models --width 1920 --height 1080 --runs 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.core.enhancer import ClassicalEnhancer, OnnxEnhancer  # noqa: E402
from campusclear.core.planner import EnhancePlan, PlannerPolicy, build_plan  # noqa: E402
from campusclear.core.quality import QualityAnalyzer  # noqa: E402
from campusclear.models.registry import ModelRegistry  # noqa: E402


def synth_frame(width: int, height: int, mode: str) -> np.ndarray:
    """构造指定退化类型的合成帧，用于在没有真实监控画面时也能压测。"""
    rng = np.random.default_rng(20261002)
    base = np.full((height, width, 3), 70, dtype=np.float32)
    # 加上一些低频结构，避免全是纯色导致算法走捷径；末尾补一个通道维度做广播
    ys = np.linspace(0, 6 * np.pi, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 4 * np.pi, width, dtype=np.float32)[None, :]
    pattern = 40 * np.sin(ys) * np.cos(xs)
    base += pattern[:, :, None]
    base = np.clip(base, 0, 255)

    if mode == "lowlight":
        base *= 0.18
    elif mode == "noisy":
        base += rng.normal(0, 18, base.shape)
    elif mode == "blurry":
        import cv2

        base = cv2.GaussianBlur(base, (0, 0), 4.0)
    return np.clip(base, 0, 255).astype(np.uint8)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def bench_analyzer(width: int, height: int, runs: int) -> dict:
    analyzer = QualityAnalyzer()
    frame = synth_frame(width, height, "noisy")
    times = []
    for index in range(runs + 5):
        start = time.perf_counter()
        quality = analyzer.analyze(frame)
        elapsed = time.perf_counter() - start
        if index >= 5:  # 跳过背景模型预热
            times.append(elapsed)
    median = statistics.median(times)
    return {
        "module": "quality_analyzer",
        "median_ms": round(median * 1000, 2),
        "p95_ms": round(percentile(times, 0.95) * 1000, 2),
        "throughput_fps": round(1.0 / median, 2),
        "single_card_max_streams_12fps": int(1.0 / median / 12),
        "sample": quality.as_dict(),
    }


def bench_enhancer(mode: str, width: int, height: int, runs: int, models_dir: Path | None) -> dict:
    enhancer = None
    if models_dir is not None:
        registry = ModelRegistry(models_dir, verify_hash=False)
        candidate = OnnxEnhancer(registry, use_fp16=True)
        if candidate.available():
            enhancer = candidate
    if enhancer is None:
        enhancer = ClassicalEnhancer()

    policy = PlannerPolicy()
    analyzer = QualityAnalyzer()
    frame = synth_frame(width, height, mode)
    quality = analyzer.analyze(frame)
    plan = build_plan(quality, policy)
    plan.reuse_previous = False  # 基准测试必须真的跑推理

    times = []
    for index in range(runs + 3):
        start = time.perf_counter()
        enhancer.enhance(frame, plan)
        elapsed = time.perf_counter() - start
        if index >= 3:
            times.append(elapsed)

    median = statistics.median(times)
    return {
        "module": f"enhancer[{getattr(enhancer, 'name', '?')}]",
        "mode": mode,
        "resolution": f"{width}x{height}",
        "stages": plan.stages(),
        "scale": plan.scale,
        "roi_only": plan.roi_only,
        "median_ms": round(median * 1000, 2),
        "p95_ms": round(percentile(times, 0.95) * 1000, 2),
        "throughput_fps": round(1.0 / median, 2),
        "single_card_max_streams_12fps": int(1.0 / median / 12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CampusClear 性能基准")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--models-dir", default=None, help="不指定则只测经典算法兜底")
    parser.add_argument("--synthetic", action="store_true", help="只跑合成数据（无需权重）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models_dir = None if (args.synthetic or not args.models_dir) else Path(args.models_dir).resolve()

    results = [bench_analyzer(args.width, args.height, args.runs)]
    for mode in ("lowlight", "noisy", "blurry"):
        results.append(bench_enhancer(mode, args.width, args.height, args.runs, models_dir))

    payload = {
        "python": sys.version.split()[0],
        "mode": "classical" if models_dir is None else "onnx",
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"CPU/GPU 基准（Python {payload['python']}，推理链路 {payload['mode']}）")
    print("-" * 96)
    print(f"{'模块':<28}{'场景':<10}{'分辨率':<12}{'中位(ms)':>10}{'P95(ms)':>10}{'吞吐(fps)':>11}{'单卡可载路数':>14}")
    print("-" * 96)
    for row in results:
        print(
            f"{row['module']:<28}{row.get('mode', '-'):<10}{row.get('resolution', f'{args.width}x{args.height}'):<12}"
            f"{row['median_ms']:>10}{row['p95_ms']:>10}"
            f"{row.get('throughput_fps', 0):>11}{row.get('single_card_max_streams_12fps', '-'):>14}"
        )
    print("-" * 96)
    print("说明：「单卡可载路数」按每路 12fps 推理帧率估算，实际部署需再扣除编解码与网络开销，建议按 70% 折算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
