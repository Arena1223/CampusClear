"""单帧「截图增强」：对当前画面按预设标准做一次增强。

用途：全屏查看时点「截图并增强」，把这一帧抓下来，按用户选的**预设标准**
离线处理一遍，结果进入右侧列表。与实时链路共用同一套增强器，因此
「看到什么」和「截下来增强成什么」口径一致。

预设是可选的「标准」，例如轻量 / 标准 / 超分 2×/4× / 取证放大。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .modes import build_for_mode
from .planner import EnhancePlan
from .quality import QualityAnalyzer


@dataclass(frozen=True)
class StillPreset:
    key: str
    label: str
    mode: str        # 用哪个增强档位（realtime/balanced/forensic）选增强器
    lowlight: bool
    denoise: bool
    deblur: bool
    sr: bool
    scale: int


STILL_PRESETS: tuple[StillPreset, ...] = (
    StillPreset("light", "轻量：低光 + 降噪", "realtime", True, True, False, False, 1),
    StillPreset("standard", "标准：低光 + 降噪 + 去模糊", "balanced", True, True, True, False, 1),
    StillPreset("sr2", "超分 2×", "balanced", False, False, False, True, 2),
    StillPreset("sr4", "超分 4×", "balanced", False, False, False, True, 4),
    StillPreset("forensic", "取证放大：全阶段 4×", "balanced", True, True, True, True, 4),
)
PRESET_MAP = {p.key: p for p in STILL_PRESETS}


def list_presets() -> list[dict]:
    return [{"key": p.key, "label": p.label} for p in STILL_PRESETS]


def enhance_still(
    frame: np.ndarray,
    preset_key: str,
    *,
    models_dir: str | Path | None = None,
) -> tuple[np.ndarray, StillPreset]:
    """对单帧按预设增强，返回 (增强后帧, 预设)。

    若超分模型原生倍率（如 ONNX 的 4×）与预设要求不一致，按预设目标尺寸缩放对齐，
    保证「超分 2×」就真的输出 2×。
    """
    preset = PRESET_MAP.get(preset_key) or PRESET_MAP["standard"]
    enhancer, _policy, _info = build_for_mode(preset.mode, models_dir=models_dir)
    quality = QualityAnalyzer().analyze(frame)

    plan = EnhancePlan(
        run_lowlight=preset.lowlight,
        run_denoise=preset.denoise,
        run_deblur=preset.deblur,
        run_sr=preset.sr,
        scale=preset.scale,
        noise_sigma=quality.noise_sigma,
        reason=f"截图增强：{preset.label}",
    )
    output = enhancer.enhance(frame, plan)

    if preset.sr and preset.scale:
        height, width = frame.shape[:2]
        target = (width * preset.scale, height * preset.scale)
        if (output.shape[1], output.shape[0]) != target:
            interp = cv2.INTER_AREA if output.shape[1] > target[0] else cv2.INTER_LANCZOS4
            output = cv2.resize(output, target, interpolation=interp)
    return output, preset
