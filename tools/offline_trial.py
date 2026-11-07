#!/usr/bin/env python3
"""离线试机：不接摄像头、不需要 GPU、不需要 ffmpeg，直接看增强效果。

为什么需要这个工具
------------------
接一台真实摄像头试机的成本被高估了。真正的麻烦在于：
  · 家里的摄像头（米家/小米系）多数不提供标准 RTSP，折腾型号与固件不划算；
  · 没装 ffmpeg / MediaMTX 就跑不通「拉流 → 回推」链路；
  · 没有 NVIDIA 显卡时实时性差，看不出效果还容易误判系统不行。

而「画质到底有没有变好」这件事，其实只依赖增强管线本身，
与流传输、GPU、RTSP 全部无关。本工具把增强管线单独摘出来跑在
**合成场景或本地视频文件**上，产出可肉眼对比的并排视频与单帧对比图。

顺带解决三个正经用途：
  1. 开发期调试：改完算法立刻看效果，不用接线；
  2. 申报素材：直接产出「原始 / 增强后」的成对对比图（docs/06 §4.2 要求）；
  3. 性能标定：在真实硬件上测每帧处理耗时与各阶段触发比例。

用法
----
    # 生成四类模拟场景并逐一出对比视频（推荐先跑这个）
    python tools/offline_trial.py --all --out ./trial_output

    # 只跑夜间低光场景，并开启 2 倍放大
    python tools/offline_trial.py --scene night --scale 2 --out ./trial_output

    # 用自己拍的视频试（手机随手拍一段昏暗走廊/夜路效果最好）
    python tools/offline_trial.py --input ./my_corridor.mp4 --out ./trial_output

输出
----
    <name>_source.mp4        原始（退化）画面，循环播放便于对比
    <name>_enhanced.mp4      增强后画面
    <name>_side_by_side.mp4  左原图 / 右增强，动图直观看效果
    <name>_compare.jpg       单帧左右对比图，可直接放进申报材料
    summary.json             每帧处理计划、耗时、各阶段触发统计

注意
----
· 没有离线权重时超分退化为插值，超分效果不真实；本工具默认 --scale 1，
  只展示低光/降噪/去模糊这三个在 CPU 环境下真实生效的阶段。
· 画面标注使用英文：OpenCV 的 putText 不支持中文字形。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campusclear.core.enhancer import ClassicalEnhancer, OnnxEnhancer  # noqa: E402
from campusclear.core.planner import PlannerPolicy, build_plan  # noqa: E402
from campusclear.core.quality import QualityAnalyzer  # noqa: E402
from campusclear.models.registry import ModelRegistry  # noqa: E402

SCENES = ("day", "night", "noise", "blur", "static")


# --------------------------------------------------------------------------- #
# 合成场景：模拟校园走廊监控画面
# --------------------------------------------------------------------------- #
def _corridor_base(width: int, height: int) -> np.ndarray:
    """画一条有透视感的走廊：后墙、地面、天花板、门、柱子、顶灯。

    刻意做成有纹理且有明确边缘的画面——纯色块会让所有画质指标都失去区分度，
    测出来的效果也不代表真实场景。
    """
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    horizon = int(height * 0.42)

    # 天花板（上）与地面（下），用垂直渐变区分明暗
    ceil_grad = np.linspace(120, 95, horizon + 1, dtype=np.float32)[:, None]
    canvas[: horizon + 1, :, :] = ceil_grad[:, :, None]
    floor_grad = np.linspace(88, 52, height - horizon - 1, dtype=np.float32)[:, None]
    canvas[horizon + 1 :, :, :] = floor_grad[:, :, None]

    # 后墙：比天花板略暗的一块矩形
    wall_x0, wall_x1 = int(width * 0.28), int(width * 0.72)
    canvas[: horizon, wall_x0:wall_x1, :] = 108

    # 地面纹理：横向地砖缝，越远越密（营造透视感）
    y = horizon + 4
    step = 3.0
    while y < height:
        canvas[int(y) : int(y) + 1, :, :] = 44
        y += step
        step *= 1.16

    # 门：深色矩形 + 亮色门框（门框是重要的高频边缘，影响模糊度判据）
    door_x0, door_x1 = int(width * 0.44), int(width * 0.56)
    door_y0, door_y1 = int(horizon * 0.35), horizon
    canvas[door_y0:door_y1, door_x0:door_x1, :] = 46
    cv2.rectangle(
        canvas, (door_x0 - 3, door_y0 - 3), (door_x1 + 3, door_y1), (150, 150, 150), 2
    )

    # 柱子：左右各一根，形成纵深
    for ratio in (0.15, 0.85):
        px = int(width * ratio)
        cv2.rectangle(canvas, (px, int(horizon * 0.15)), (px + 18, height), (96, 96, 96), -1)
        cv2.line(canvas, (px, int(horizon * 0.15)), (px, height), (128, 128, 128), 1)

    # 顶灯：两个圆形光斑
    for cx_ratio in (0.34, 0.66):
        cx = int(width * cx_ratio)
        cv2.circle(canvas, (cx, int(horizon * 0.12)), 9, (230, 232, 235), -1)

    # 墙上的告示牌（小尺寸高频细节，用来检验超分/清晰化能力）
    cv2.rectangle(
        canvas, (int(width * 0.31), int(horizon * 0.55)),
        (int(width * 0.36), int(horizon * 0.68)), (168, 168, 168), -1,
    )
    for k in range(3):
        y_line = int(horizon * (0.58 + k * 0.03))
        cv2.line(
            canvas, (int(width * 0.315), y_line), (int(width * 0.355), y_line), (70, 70, 70), 1
        )
    return canvas


def _draw_person(canvas: np.ndarray, cx: int, cy: int, scale: float) -> None:
    """画一个行走的人形剪影。尺寸随位置变化以模拟远近。

    尺寸刻意画得足够大（占画幅约 1%）：运动检测的判据是运动像素占比，
    太小的人形会低于阈值而被判为静止场景——这不是算法的缺陷，
    而是「目标必须占画幅一定比例才能被背景建模捕捉」这一客观限制。

    颜色用偏亮的中灰，模拟被补光灯照亮的浅色衣物。
    这一项同样不能省：若人形与暗背景灰度接近（早期版本差不到 2 级），
    压暗后目标与背景将无法区分，运动检测必然失效——
    这也是真实夜间的客观困难，不应靠调算法阈值来掩盖。
    """
    height = int(88 * scale)
    width_body = max(6, int(22 * scale))
    torso_top = cy - height // 2
    torso_bottom = cy + height // 4
    colour = (168, 168, 172)

    cv2.circle(canvas, (cx, torso_top + int(height * 0.10)), max(5, int(10 * scale)), colour, -1)
    cv2.rectangle(
        canvas, (cx - width_body // 2, torso_top + int(height * 0.20)),
        (cx + width_body // 2, torso_bottom), colour, -1,
    )
    leg_len = int(height * 0.30)
    cv2.line(canvas, (cx - width_body // 3, torso_bottom), (cx - width_body // 3, torso_bottom + leg_len), colour, max(3, int(6 * scale)))
    cv2.line(canvas, (cx + width_body // 3, torso_bottom), (cx + width_body // 3, torso_bottom + leg_len), colour, max(3, int(6 * scale)))


def synth_frame(scene: str, index: int, total: int, width: int, height: int) -> np.ndarray:
    """生成第 ``index`` 帧。退化程度由 ``scene`` 决定。"""
    base = _corridor_base(width, height)

    # 人形位置：在静止场景里固定在画面中央偏左，其他场景来回走动
    progress = index / max(1, total - 1)
    if scene == "static":
        person_x = int(width * 0.40)
    else:
        person_x = int(width * (0.26 + 0.48 * progress))
    depth = 1.15 - 0.25 * progress
    _draw_person(base, person_x, int(height * 0.60), depth)

    # 轻微整体抖动：真实手持/云台摄像头的微小位移
    if scene != "static":
        shift = int(round(2 * np.sin(index * 0.35)))
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
        base = cv2.warpAffine(base, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    frame = np.clip(base, 0, 255)

    rng = np.random.default_rng(1000 + index)

    if scene == "night":
        # 顺序必须遵循真实成像链路：先光学/失焦模糊，再由传感器增益引入噪声。
        # 反过来写（先加噪再模糊）会把噪声一起模糊掉，测出来的噪声极低，
        # 与真实设备完全对不上——这是本工具开发过程中实际踩过的坑。
        frame = cv2.GaussianBlur(frame * 0.20 + 10, (0, 0), 1.3)
        frame = frame + rng.normal(0, 11, frame.shape)   # 高增益噪声
    elif scene == "noise":
        frame = cv2.GaussianBlur(frame * 0.62, (0, 0), 0.8)
        frame = frame + rng.normal(0, 17, frame.shape)   # 强噪声但亮度尚可
    elif scene == "blur":
        frame = frame * 0.85
        frame = cv2.GaussianBlur(frame, (0, 0), 3.4)     # 慢快门运动拖影
        frame = frame * 0.9 + 12                         # 模糊同时往往伴随对比度下降
    elif scene == "day":
        frame = frame * 1.06 - 6                         # 轻微过曝
        frame = frame + rng.normal(0, 3, frame.shape)

    return np.clip(frame, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 处理
# --------------------------------------------------------------------------- #
def pick_enhancer(models_dir: Path | None, denoise_mode: str = "bilateral"):
    """按有无离线权重选择推理实现，与生产链路的逻辑保持一致。"""
    if models_dir is not None:
        registry = ModelRegistry(models_dir, verify_hash=False)
        candidate = OnnxEnhancer(registry, use_fp16=True)
        if candidate.available():
            print(f"[推理链路] ONNX（可用阶段：{', '.join(registry.available_stages())}）")
            return candidate, registry.available_stages()
        print("[推理链路] 未发现可用权重，退化为经典 CPU 算法")
    else:
        print("[推理链路] 未指定 --models-dir，使用经典 CPU 算法")
    if denoise_mode == "nlmeans":
        print(
            "          警告：已选用 NL-means 降噪，画质明显更好，"
            "但实测 640x360 约 500ms/帧、1080p 预计 4 秒以上。\n"
            "          只适合离线出对比图，不能用于实时链路。"
        )
    print("          说明：此链路不含生成式超分，--scale >1 时用插值放大")
    return ClassicalEnhancer(denoise_mode=denoise_mode), []


def annotate(image: np.ndarray, text: str) -> np.ndarray:
    """左上角打标签。用英文是因为 OpenCV 的 putText 无法渲染中文。"""
    output = image.copy()
    cv2.rectangle(output, (0, 0), (len(text) * 8 + 16, 26), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def stack_side_by_side(source: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """左右并排。尺寸不一致时把源帧插值放大到同尺寸，属「插值 vs 增强」的对照。"""
    if enhanced.shape[:2] != source.shape[:2]:
        source = cv2.resize(
            source, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
    divider = np.full((source.shape[0], 2, 3), 240, dtype=np.uint8)
    return np.hstack([annotate(source, "SOURCE"), divider, annotate(enhanced, "ENHANCED")])


def run_scene(
    scene: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    frames: int,
    fps: int,
    scale: int,
    models_dir: Path | None,
    max_seconds: float,
    denoise_mode: str = "bilateral",
) -> dict:
    enhancer, available_stages = pick_enhancer(models_dir, denoise_mode)

    analyzer = QualityAnalyzer()
    # 本工具**不执行**静止帧复用，每帧都真实处理。原因有三：
    #   1. 复用会直接返回上一帧，静止场景的对比视频会变成静止不动，看不出效果；
    #   2. 复用的帧耗时近乎为 0，混进延迟统计会让中位数严重失真；
    #   3. 复用条件（is_static + 噪声闸门）只在 planner 里维护一份，
    #      在工具里再手写一遍必然漏条件——这是实际踩过的坑。
    # 复用的收益改为「可复用帧比例」单独统计并报告，代表生产环境下能省下的算力。
    policy = PlannerPolicy(
        enable_lowlight=True,
        enable_denoise=True,
        enable_deblur=True,
        enable_sr=scale > 1,
        base_scale=scale,
        enable_static_reuse=False,
    )

    name = f"{scene}_{width}x{height}"
    source_path = out_dir / f"{name}_source.mp4"
    enhanced_path = out_dir / f"{name}_enhanced.mp4"
    side_path = out_dir / f"{name}_side_by_side.mp4"
    compare_path = out_dir / f"{name}_compare.jpg"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    source_writer = cv2.VideoWriter(str(source_path), fourcc, fps, (width, height))
    out_w, out_h = width * scale, height * scale
    enhanced_writer = cv2.VideoWriter(str(enhanced_path), fourcc, fps, (out_w, out_h))
    side_writer = cv2.VideoWriter(str(side_path), fourcc, fps, (out_w * 2 + 2, out_h))

    if not source_writer.isOpened() or not enhanced_writer.isOpened():
        raise RuntimeError(
            "无法写入 mp4，请确认 opencv 带 mp4v 编码器（opencv-python-headless 通常可用）"
        )

    stage_counter: dict[str, int] = {}
    durations: list[float] = []
    static_count = 0
    reusable_count = 0
    sample_quality = None
    sample_plan = None
    # 取样帧取中段：前 20 帧背景模型尚未收敛，运动占比恒为 1，
    # 拿它当样本会给出误导性的判据。
    compare_index = max(1, frames // 2)

    started = time.perf_counter()
    for index in range(frames):
        frame = synth_frame(scene, index, frames, width, height)
        quality = analyzer.analyze(frame)
        if quality.is_static:
            static_count += 1
        # 复用条件以 FrameQuality 的判定为准，不在本工具里另立一套标准
        if quality.is_static and quality.reuse_allowed:
            reusable_count += 1

        plan = build_plan(quality, policy)

        begin = time.perf_counter()
        enhanced = enhancer.enhance(frame, plan, None)
        durations.append(time.perf_counter() - begin)

        if enhanced.shape[:2] != (out_h, out_w):
            enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        for stage in plan.stages():
            stage_counter[stage] = stage_counter.get(stage, 0) + 1

        source_writer.write(frame)
        enhanced_writer.write(enhanced)
        side_writer.write(stack_side_by_side(frame, enhanced))

        if index == compare_index:
            sample_quality = quality
            sample_plan = plan
            cv2.imwrite(str(compare_path), stack_side_by_side(frame, enhanced))

        if (time.perf_counter() - started) > max_seconds:
            print(f"  [警告] 达到 {max_seconds:.0f}s 时间上限，提前结束（已处理 {index + 1} 帧）")
            break

    source_writer.release()
    enhanced_writer.release()
    side_writer.release()

    median_ms = statistics.median(durations) * 1000 if durations else 0.0
    p95_ms = (
        sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))] * 1000
        if durations
        else 0.0
    )
    processed = len(durations)

    return {
        "scene": scene,
        "resolution": f"{width}x{height}",
        "scale": scale,
        "frames": processed,
        "enhancer": getattr(enhancer, "name", "unknown"),
        "denoise_mode": denoise_mode,
        "onnx_stages_available": available_stages,
        "median_ms": round(median_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "fps_single_stream": round(1000 / median_ms, 2) if median_ms else 0.0,
        "static_frames": static_count,
        "reusable_frames": reusable_count,
        "reusable_ratio": round(reusable_count / processed, 3) if processed else 0.0,
        "stage_trigger": stage_counter,
        "sample_frame_quality": sample_quality.as_dict() if sample_quality else None,
        "sample_frame_plan": sample_plan.as_dict() if sample_plan else None,
        "files": {
            "source": str(source_path),
            "enhanced": str(enhanced_path),
            "side_by_side": str(side_path),
            "compare": str(compare_path),
        },
    }


def run_video(
    video_path: Path,
    out_dir: Path,
    *,
    scale: int,
    models_dir: Path | None,
    max_frames: int,
    denoise_mode: str = "bilateral",
) -> dict:
    """用本地视频文件试机。想看出效果，视频最好是昏暗或抖动/模糊的场景。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    total = min(total, max_frames)

    enhancer, available_stages = pick_enhancer(models_dir, denoise_mode)
    analyzer = QualityAnalyzer()
    policy = PlannerPolicy(
        base_scale=scale, enable_sr=scale > 1, enable_static_reuse=False
    )

    name = f"video_{video_path.stem}"
    out_w, out_h = width * scale, height * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    side_path = out_dir / f"{name}_side_by_side.mp4"
    side_writer = cv2.VideoWriter(str(side_path), fourcc, fps, (out_w * 2 + 2, out_h))
    enhanced_path = out_dir / f"{name}_enhanced.mp4"
    enhanced_writer = cv2.VideoWriter(str(enhanced_path), fourcc, fps, (out_w, out_h))
    compare_path = out_dir / f"{name}_compare.jpg"

    durations: list[float] = []
    stage_counter: dict[str, int] = {}
    index = 0
    while index < total:
        ok, frame = capture.read()
        if not ok:
            break
        quality = analyzer.analyze(frame)
        plan = build_plan(quality, policy)
        for stage in plan.stages():
            stage_counter[stage] = stage_counter.get(stage, 0) + 1

        begin = time.perf_counter()
        enhanced = enhancer.enhance(frame, plan, None)
        durations.append(time.perf_counter() - begin)
        if enhanced.shape[:2] != (out_h, out_w):
            enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        enhanced_writer.write(enhanced)
        side_writer.write(stack_side_by_side(frame, enhanced))
        if index == total // 2:
            cv2.imwrite(str(compare_path), stack_side_by_side(frame, enhanced))
        index += 1

    capture.release()
    enhanced_writer.release()
    side_writer.release()

    median_ms = statistics.median(durations) * 1000 if durations else 0.0
    return {
        "scene": f"video:{video_path.name}",
        "resolution": f"{width}x{height}",
        "scale": scale,
        "frames": index,
        "enhancer": getattr(enhancer, "name", "unknown"),
        "denoise_mode": denoise_mode,
        "onnx_stages_available": available_stages,
        "median_ms": round(median_ms, 2),
        "fps_single_stream": round(1000 / median_ms, 2) if median_ms else 0.0,
        "stage_trigger": stage_counter,
        "files": {
            "enhanced": str(enhanced_path),
            "side_by_side": str(side_path),
            "compare": str(compare_path),
        },
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CampusClear 离线试机工具（无需摄像头 / GPU / ffmpeg）"
    )
    parser.add_argument("--scene", choices=SCENES, help="要生成的模拟场景")
    parser.add_argument("--all", action="store_true", help="逐个跑完全部模拟场景")
    parser.add_argument("--input", help="改用本地视频文件作为输入")
    parser.add_argument("--out", default="./trial_output", help="输出目录")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=90, help="每个场景的帧数")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--scale", type=int, default=1, choices=[1, 2, 4],
                        help="超分倍率。无权重时只是插值，默认 1")
    parser.add_argument("--models-dir", default=None, help="指定离线权重目录（可选）")
    parser.add_argument("--denoise", default="bilateral", choices=["bilateral", "nlmeans"],
                        help="降噪器。bilateral 实时可用；nlmeans 画质更好但极慢，"
                             "只适合离线生成高质量对比图")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="单场景处理时间上限，防止 CPU 环境跑太久")
    parser.add_argument("--json", action="store_true", help="额外打印 JSON 摘要")
    args = parser.parse_args()

    if not args.scene and not args.all and not args.input:
        parser.print_help()
        print("\n提示：先试 `--all`，一次生成四种典型场景的对比视频。")
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None

    print(f"输出目录: {out_dir}")
    print("-" * 78)

    results: list[dict] = []
    if args.input:
        results.append(
            run_video(
                Path(args.input).resolve(), out_dir, scale=args.scale,
                models_dir=models_dir, max_frames=args.frames,
                denoise_mode=args.denoise,
            )
        )
    else:
        scenes = SCENES if args.all else [args.scene]
        for scene in scenes:
            print(f"[场景 {scene}] 生成 {args.frames} 帧 {args.width}x{args.height}")
            result = run_scene(
                scene, out_dir, width=args.width, height=args.height, frames=args.frames,
                fps=args.fps, scale=args.scale, models_dir=models_dir,
                max_seconds=args.max_seconds, denoise_mode=args.denoise,
            )
            if result["sample_frame_quality"]:
                quality = result["sample_frame_quality"]
                static_note = "判为静止" if quality["is_static"] else "判为有运动"
                print(
                    f"  质量判据: 亮度 {quality['brightness']:.1f} / 噪声 {quality['noise_sigma']:.1f}"
                    f" / 模糊度 {quality['blur_score']:.0f} / 运动 {quality['motion_ratio']:.4f}"
                    f" → {static_note}"
                )
            if result["sample_frame_plan"]:
                plan = result["sample_frame_plan"]
                print(f"  增强计划: {'+'.join(plan['stages']) or '（跳过）'}  ← {plan['reason']}")
            print(
                f"  处理耗时: 中位 {result['median_ms']} ms / P95 {result['p95_ms']} ms"
                f"  →  单路约 {result['fps_single_stream']} fps"
            )
            print(
                f"  静止帧 {result['static_frames']} / 可复用 {result['reusable_frames']} 帧"
                f"（{result['reusable_ratio'] * 100:.0f}%）"
            )
            print(f"  阶段触发: {result['stage_trigger']}")
            results.append(result)

    summary = {"results": results}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 78)
    for result in results:
        print(f"{result['scene']:<18}{result['median_ms']:>9} ms   {result['files']['side_by_side']}")
    print("-" * 78)
    print("*_side_by_side.mp4 是左原图右增强的对照视频，用播放器打开就能看出差别。")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
