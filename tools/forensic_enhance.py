#!/usr/bin/env python3
"""取证 / 放大增强：对已录像片段按需做「重量级」离线增强。

与 ``offline_trial.py`` 的分工
-----------------------------
``offline_trial.py`` 是**试机**：合成场景、看个大概、验证链路能跑。
本工具是**取证**：拿真实录像片段（或指定区域），用你选定的档位认真跑一遍，
产出可用于申报 / 复核的对比证据，并如实打印每一帧走了哪条链路。

三档（详见 ``campusclear.core.modes``）
--------------------------------------
    realtime  轻量实时   双边降噪 + 可选 1x        快，适合确认「实时链路长什么样」
    balanced  标准       双边降噪 + 2x 超分        画质/速度折中
    forensic  取证/放大  NL-means + 2~4x 放大      慢，但画质最好（离线按需）

三种引擎
--------
    auto       有 ONNX 权重走 ONNX，否则经典算法（默认）
    classical  强制经典 OpenCV 算法
    onnx       强制 ONNX（无权重会报错）
    ffmpeg     纯 FFmpeg 滤镜（处理整段视频最快，不逐帧进 Python）

用法
----
    # 整段片段，取证档，2 倍放大
    python tools/forensic_enhance.py --input ./clip.mp4 --mode forensic --scale 2

    # 只放大画面里的一个区域（车牌 / 人脸），从第 12 秒起取 5 秒
    python tools/forensic_enhance.py --input ./clip.mp4 --mode forensic \
        --region 900,300,320,240 --start 12 --end 17 --scale 3

    # 用 FFmpeg 滤镜快速跑一版（对比「无模型增强」的效果）
    python tools/forensic_enhance.py --input ./clip.mp4 --engine ffmpeg --mode balanced

    # 处理单张图片
    python tools/forensic_enhance.py --input ./frame.jpg --mode forensic --scale 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.core.modes import (  # noqa: E402
    PRESETS,
    EnhanceMode,
    Engine,
    build_for_mode,
    policy_for,
)
from campusclear.core.planner import build_plan  # noqa: E402
from campusclear.core.quality import QualityAnalyzer  # noqa: E402


# --------------------------------------------------------------------------- #
# FFmpeg 滤镜链路（无模型，纯 CPU，直接处理整段视频）
# --------------------------------------------------------------------------- #
FFMPEG_PRESETS = {
    "realtime": "hqdn3d=1.5:1.5:4:4,eq=gamma=1.05,cas=0.3",
    "balanced": "hqdn3d=3:3:6:6,eq=gamma=1.10:contrast=1.04,cas=0.4",
    "forensic": "hqdn3d=4:4:8:8,eq=gamma=1.15:contrast=1.05,unsharp=5:5:0.8:5:5:0.0",
}


def run_ffmpeg(args) -> int:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("找不到 ffmpeg，无法使用 ffmpeg 引擎", file=sys.stderr)
        return 2
    if not Path(args.input).exists():
        print(f"输入不存在: {args.input}", file=sys.stderr)
        return 2

    out = Path(args.out) / f"{Path(args.input).stem}_{args.mode}_ffmpeg.mp4"
    filters = [FFMPEG_PRESETS[args.mode]]
    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        filters.insert(0, f"crop={w}:{h}:{x}:{y}")
    if args.scale and args.scale > 1:
        filters.append(f"scale=iw*{args.scale}:ih*{args.scale}:flags=lanczos")
    vf = ",".join(filters)

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", args.input]
    if args.end:
        cmd += ["-t", str(max(0.0, args.end - (args.start or 0)))]
    cmd += ["-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(out)]
    print(f"[ffmpeg] {' '.join(cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    print(f"[ffmpeg] 退出码 {rc}，用时 {time.perf_counter() - t0:.1f}s -> {out}")
    print(f"[ffmpeg] 滤镜链: {vf}")
    return rc


# --------------------------------------------------------------------------- #
# Python 逐帧增强
# --------------------------------------------------------------------------- #
def annotate(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 9 + 20, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(source: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    if enhanced.shape[:2] != source.shape[:2]:
        source = cv2.resize(source, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    divider = np.full((source.shape[0], 2, 3), 235, dtype=np.uint8)
    return np.hstack([source, divider, enhanced])


def process_image(args, enhancer, policy, analyzer) -> int:
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"读不到图片: {args.input}", file=sys.stderr)
        return 2
    if args.region:
        x, y, w, h = (int(v) for v in args.region.split(","))
        frame = frame[y : y + h, x : x + w]
    quality = analyzer.analyze(frame)
    plan = build_plan(quality, policy)
    plan.reuse_previous = False  # 单帧无上一帧可复用
    t0 = time.perf_counter()
    enhanced = enhancer.enhance(frame, plan)
    dt = (time.perf_counter() - t0) * 1000
    base = Path(args.out) / Path(args.input).stem
    cv2.imwrite(str(base) + f"_{args.mode}_enhanced.jpg", enhanced)
    cv2.imwrite(str(base) + f"_{args.mode}_side_by_side.jpg", side_by_side(frame, enhanced))
    print(f"图片增强完成：{args.input}  {plan.stages()}  耗时 {dt:.1f} ms")
    print(f"  计划: {plan.reason}")
    print(f"  输出: {base}_{args.mode}_enhanced.jpg / _side_by_side.jpg")
    return 0


def process_video(args, enhancer, policy, analyzer) -> int:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"打不开视频: {args.input}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    out_base = Path(args.out) / f"{Path(args.input).stem}_{args.mode}"
    enh_path = str(out_base) + "_enhanced.mp4"
    sbs_path = str(out_base) + "_side_by_side.mp4"
    writer = sbs_writer = None

    times, stage_hits, reuse_count = [], {}, 0
    n = 0
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if args.end and t > args.end:
            break
        if args.max_frames and n >= args.max_frames:
            break
        if args.region:
            x, y, w, h = (int(v) for v in args.region.split(","))
            frame = frame[y : y + h, x : x + w]

        quality = analyzer.analyze(frame)
        plan = build_plan(quality, policy)
        if plan.reuse_previous and prev is not None:
            enhanced = prev
            reuse_count += 1
        else:
            # 只对「真实增强」的帧计时。静止复用的帧近乎零耗时，
            # 若一并计入会把中位耗时压到 0、算出荒谬的容量——
            # 生产工程与原型测试都专门守过这条口径（test_latency_only_counts_...）。
            t0 = time.perf_counter()
            enhanced = enhancer.enhance(frame, plan)
            times.append((time.perf_counter() - t0) * 1000)
        for s in plan.stages():
            stage_hits[s] = stage_hits.get(s, 0) + 1
        prev = enhanced

        if writer is None:
            h, w = enhanced.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(enh_path, fourcc, fps, (w, h))
            sbs_writer = cv2.VideoWriter(sbs_path, fourcc, fps, (w * 2 + 2, h))
        writer.write(enhanced)
        sbs_writer.write(side_by_side(annotate(frame, "SOURCE"), annotate(enhanced, f"ENHANCED-{args.mode}")))
        n += 1

    cap.release()
    if writer:
        writer.release()
    if sbs_writer:
        sbs_writer.release()

    if n == 0:
        print("没有处理任何帧", file=sys.stderr)
        return 2

    if times:
        med = statistics.median(times)
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
    else:
        med = p95 = 0.0
    summary = {
        "input": str(args.input),
        "mode": args.mode,
        "frames": n,
        "enhanced_frames": len(times),
        "reuse_frames": reuse_count,
        "reuse_ratio": round(reuse_count / n, 3),
        "fps_source": fps,
        "median_ms": round(med, 2),
        "p95_ms": round(p95, 2),
        "single_route_fps": round(1000 / med, 2) if med else None,
        "stage_hits": stage_hits,
        "realtime_ok": (1000 / med) >= 12 if med else None,
    }
    with open(str(out_base) + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"处理 {n} 帧（真实增强 {len(times)} / 复用 {reuse_count}）")
    print(f"  真实增强帧：中位 {med:.1f} ms  P95 {p95:.1f} ms  单路≈{summary['single_route_fps']} fps")
    print(f"  阶段命中 {stage_hits}   复用率 {summary['reuse_ratio'] * 100:.0f}%")
    rt = summary["realtime_ok"]
    print(f"  实时性(≥12fps): {'满足' if rt else '不满足——属正常，此档为离线按需' if rt is not None else '无增强帧'}")
    print(f"  输出: {enh_path}")
    print(f"        {sbs_path}")
    print(f"        {out_base}_summary.json")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="取证/放大离线增强")
    ap.add_argument("--input", required=True, help="输入视频或图片")
    ap.add_argument("--mode", choices=[m.value for m in EnhanceMode], default="forensic")
    ap.add_argument("--engine", choices=[e.value for e in Engine], default="auto")
    ap.add_argument("--scale", type=int, default=None, help="放大倍率（默认取档位预设）")
    ap.add_argument("--region", default=None, help="只处理指定区域 x,y,w,h（取证放大用）")
    ap.add_argument("--start", type=float, default=None, help="起始秒")
    ap.add_argument("--end", type=float, default=None, help="结束秒")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理帧数")
    ap.add_argument("--models-dir", default=None, help="ONNX 权重目录（有则走 AI）")
    ap.add_argument("--denoise", choices=["bilateral", "nlmeans"], default=None, help="覆盖经典链路降噪器")
    ap.add_argument("--out", default="./forensic_out")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    mode = EnhanceMode(args.mode)

    if args.engine == Engine.FFMPEG.value:
        return run_ffmpeg(args)

    enhancer, policy, info = build_for_mode(
        mode,
        models_dir=args.models_dir,
        engine=args.engine,
    )
    if args.denoise and info.get("engine") == "classical":
        from campusclear.core.enhancer import ClassicalEnhancer

        enhancer = ClassicalEnhancer(denoise_mode=args.denoise, force_nlmeans=(args.denoise == "nlmeans"))
        info["denoise"] = args.denoise
    if args.scale is not None:
        policy.base_scale = args.scale

    preset = PRESETS[mode]
    print(f"[档位] {preset.label}（{mode.value}）  引擎={info['engine']}  "
          f"降噪={info.get('denoise', '-')}  倍率={policy.base_scale}")
    print(f"[实时性] 本档{'需满足' if preset.realtime else '不要求'}实时契约")

    analyzer = QualityAnalyzer()
    ext = Path(args.input).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return process_image(args, enhancer, policy, analyzer)
    return process_video(args, enhancer, policy, analyzer)


if __name__ == "__main__":
    raise SystemExit(main())
