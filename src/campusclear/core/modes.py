"""增强档位：三套链路，按用途与算力切换。

本模块把「用哪条增强链路」抽象成一个档位（``EnhanceMode``），
让实时播放与按需取证用同一套代码、同一套接口（``enhance(frame, plan, roi)``），
只是参数与所选实现不同。

三档定位（与 docs 里的工程取舍一致）：

================  ============  ==========================  ==================
档位               用途           降噪                         备注
================  ============  ==========================  ==================
``realtime``       实时多路上墙    双边滤波（快）                必须跟得上推理帧率
``balanced``       常规回放/导出   双边滤波 + 2 倍超分            画质与速度折中
``forensic``       取证 / 放大       NL-means（慢，画质最好）       离线按需，慢无所谓
================  ============  ==========================  ==================

选择实现时遵循「有 GPU 走重模型、没有就退经典」：
- 有可用 ONNX 权重 → ``OnnxEnhancer``（重 AI）；
- 没有权重 / 纯 CPU → ``ClassicalEnhancer``（经典算法）。
两条路径的 ``enhance`` 签名完全一致，因此对上层透明。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .enhancer import ClassicalEnhancer, OnnxEnhancer
from .planner import PlannerPolicy


class EnhanceMode(str, Enum):
    REALTIME = "realtime"    # 轻量实时：优先保证帧率
    BALANCED = "balanced"    # 标准：画质与速度折中
    FORENSIC = "forensic"    # 取证/放大：优先画质，允许离线慢跑


#: 引擎选择。``auto`` 表示有 ONNX 权重走 ONNX，否则经典。
class Engine(str, Enum):
    AUTO = "auto"
    CLASSICAL = "classical"
    ONNX = "onnx"
    FFMPEG = "ffmpeg"        # 纯 FFmpeg 滤镜，不经 Python 逐帧（处理视频文件最快）


@dataclass(frozen=True)
class ModePreset:
    label: str
    denoise_mode: str        # 经典链路降噪器：bilateral / nlmeans
    base_scale: int
    enable_static_reuse: bool
    enable_sr: bool          # 倍率=1 的档位不应触发超分（否则命中却无效果，误导统计）
    realtime: bool           # 是否必须满足实时契约


PRESETS: dict[EnhanceMode, ModePreset] = {
    EnhanceMode.REALTIME: ModePreset("轻量实时", "bilateral", 1, True, False, True),
    EnhanceMode.BALANCED: ModePreset("标准", "bilateral", 2, True, True, False),
    EnhanceMode.FORENSIC: ModePreset("取证/放大", "nlmeans", 2, False, True, False),
}


def policy_for(mode: EnhanceMode) -> PlannerPolicy:
    """按档位生成规划策略。"""
    preset = PRESETS[EnhanceMode(mode)]
    return PlannerPolicy(
        enable_lowlight=True,
        enable_denoise=True,
        enable_deblur=True,
        enable_sr=preset.enable_sr,
        base_scale=preset.base_scale,
        enable_static_reuse=preset.enable_static_reuse,
    )


def build_for_mode(
    mode: EnhanceMode,
    *,
    models_dir: Path | str | None = None,
    engine: Engine | str = Engine.AUTO,
):
    """构建某档位的增强器。

    返回 ``(enhancer, policy, info)``；``info`` 记录实际选中的引擎与原因，
    便于在报告 / 日志里如实标注（申报材料里「用了什么链路」必须可交代）。
    """
    mode = EnhanceMode(mode)
    preset = PRESETS[mode]
    engine = Engine(engine)

    if engine is Engine.FFMPEG:
        raise ValueError("FFmpeg 引擎不经 Python 增强器，请使用 forensic_enhance.py 的 ffmpeg 分支")

    want_onnx = engine in (Engine.AUTO, Engine.ONNX)
    if want_onnx and models_dir is not None:
        from ..models.registry import ModelRegistry

        registry = ModelRegistry(Path(models_dir), verify_hash=False)
        candidate = OnnxEnhancer(registry, use_fp16=True)
        if candidate.available():
            info = {
                "engine": "onnx",
                "stages": list(registry.available_stages()),
                "mode": mode.value,
                "label": preset.label,
            }
            return candidate, policy_for(mode), info
        if engine is Engine.ONNX:
            raise RuntimeError(f"指定了 ONNX 引擎，但 {models_dir} 下没有可用权重")

    enhancer = ClassicalEnhancer(
        denoise_mode=preset.denoise_mode,
        force_nlmeans=(preset.denoise_mode == "nlmeans"),
    )
    info = {
        "engine": "classical",
        "denoise": preset.denoise_mode,
        "mode": mode.value,
        "label": preset.label,
    }
    return enhancer, policy_for(mode), info
