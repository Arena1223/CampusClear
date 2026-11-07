"""增强策略规划与负载治理。

核心思想：**算力按需分配**。

老旧校园监控的共性问题不是「所有帧都差」，而是「大部分时间画面静止且尚可，
关键事件发生时的那几秒到几十秒画质不够用」。如果对每一路每一帧都跑
低光+降噪+去模糊+超分四连，单张 T4 连一路 1080p@25fps 都跑不满，
更谈不上多路接入。

因此本模块做两件事：
1. 依据帧质量评估结果，只为「确实需要的阶段」分配算力；
2. 维护全局负载档位（NORMAL / ECONOMY / SAFE），GPU 吃紧时逐级降载，
   优先牺牲画质而不是丢流——安防场景里「有画面」永远比「画面漂亮」重要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .quality import FrameQuality


class LoadLevel(str, Enum):
    """全局负载档位。由 GPU 利用率与队列积压共同驱动。"""

    NORMAL = "NORMAL"    # 全功能
    ECONOMY = "ECONOMY"  # 降倍率、提运动阈值，保留全部阶段
    SAFE = "SAFE"        # 只做低光+降噪，超分与去模糊暂停，优先保连通


@dataclass
class EnhancePlan:
    """单帧的增强执行计划。"""

    run_lowlight: bool = False
    run_denoise: bool = False
    run_deblur: bool = False
    run_sr: bool = False
    scale: int = 2
    roi_only: bool = False
    reuse_previous: bool = False
    reason: str = ""
    # 噪声水平随计划下发，执行器据此选择降噪强度与提亮幅度。
    # 放在计划里而不是让执行器自己估计，是因为 analyzer 的读数
    # 带降采样口径，执行器若自己再估一遍很可能对不上。
    noise_sigma: float = 0.0

    def stages(self) -> list[str]:
        result = []
        if self.run_lowlight:
            result.append("lowlight")
        if self.run_denoise:
            result.append("denoise")
        if self.run_deblur:
            result.append("deblur")
        if self.run_sr:
            result.append("sr")
        return result

    def as_dict(self) -> dict:
        data = {
            "stages": self.stages(),
            "scale": self.scale,
            "roi_only": self.roi_only,
            "reuse_previous": self.reuse_previous,
            "reason": self.reason,
        }
        return data


@dataclass
class PlannerPolicy:
    """可热更新的策略参数。不同画质诉求的校区可挂不同策略。"""

    enable_lowlight: bool = True
    enable_denoise: bool = True
    enable_deblur: bool = True
    enable_sr: bool = True
    base_scale: int = 2
    enable_static_reuse: bool = True
    # ROI 增强的最小面积占比：目标太小就没必要单独裁剪，不如整帧处理
    roi_min_area_ratio: float = 0.01
    # 单帧 ROI 面积占比超过该值直接整帧处理，避免裁剪-拼接开销大于收益
    roi_max_area_ratio: float = 0.6


def build_plan(
    quality: FrameQuality,
    policy: PlannerPolicy,
    level: LoadLevel = LoadLevel.NORMAL,
) -> EnhancePlan:
    """根据帧质量与负载档位生成增强计划。

    决策规则（全部可解释，便于现场排障）：

    - 静止帧 且 复用被允许：整帧跳过推理，直接复用上一增强结果 → 省掉 100% 算力
      （同时要求 ``is_static`` 与 ``reuse_allowed``：高噪声环境下运动检测会漏检，
       此时必须禁用复用，否则画面在动却复用旧帧）
    - 有运动：按需开启各阶段
        · 亮度 < 阈值            → 低光增强
        · 噪声 > 阈值 或 低光     → 降噪
        · 模糊度 < 阈值           → 去模糊
        · 画面清晰度够但分辨率低  → 超分
    - 负载档位降级：ECONOMY 降倍率，SAFE 只保留低光+降噪
    """
    plan = EnhancePlan(scale=policy.base_scale, noise_sigma=quality.noise_sigma)

    # ---- 优先级 0：静止帧复用 ----
    # 两个条件都要查，缺一不可：
    # · is_static       —— 画面确实没有运动；
    # · reuse_allowed   —— 运动检测本身可信（高噪声环境下会漏检，见 quality.analyze）。
    # 之所以不把 is_static 折进 reuse_allowed 里省一次判断，是因为 FrameQuality
    # 也可能由其他代码路径手工构造，此时两个字段未必一致；
    # 复用错帧的代价（画面在动却卡住）远高于多算一帧，因此这里做双重校验。
    if policy.enable_static_reuse and quality.is_static and quality.reuse_allowed:
        plan.reuse_previous = True
        plan.reason = f"画面静止(motion={quality.motion_ratio:.4f})，复用上一增强帧"
        return plan

    # ---- SAFE 档位：只保底 ----
    if level is LoadLevel.SAFE:
        plan.run_lowlight = policy.enable_lowlight and quality.is_low_light
        plan.run_denoise = policy.enable_denoise and (quality.is_noisy or quality.is_low_light)
        plan.scale = 1
        plan.reason = "负载达 SAFE 档，仅保留低光与降噪保底"
        return plan

    # ---- 正常按需开启 ----
    reasons: list[str] = []
    if policy.enable_lowlight and quality.is_low_light:
        plan.run_lowlight = True
        reasons.append(f"低光(亮度{quality.brightness:.1f})")

    # 低光必然伴随高增益噪声，降噪与低光绑定开启
    if policy.enable_denoise and (quality.is_noisy or quality.is_low_light):
        plan.run_denoise = True
        reasons.append(f"噪声估计{quality.noise_sigma:.1f}")

    if policy.enable_deblur and quality.is_blurry:
        plan.run_deblur = True
        reasons.append(f"模糊(拉普拉斯方差{quality.blur_score:.0f})")

    if policy.enable_sr:
        # 已经模糊的画面先修清晰度再放大，否则会把模糊一起放大，得不偿失
        if quality.is_blurry and plan.run_deblur:
            reasons.append("画面偏模糊，本轮跳过超分待清晰度恢复")
        else:
            plan.run_sr = True
            reasons.append("细节缺失")

    # 至少有一个阶段被选中，否则仍需降噪做兜底，避免输出直通劣化画面
    if not plan.stages():
        if policy.enable_denoise:
            plan.run_denoise = True
        reasons.append("质量达标，仅做轻量处理")

    # ---- ECONOMY 档位：降倍率 ----
    if level is LoadLevel.ECONOMY:
        plan.scale = 1 if policy.base_scale <= 2 else 2
        if plan.scale == 1 and plan.run_sr:
            # 超分模型的原生倍率是固定的（如 realesr ×4），请求 1× 它也会
            # 照跑全量推理再把结果缩回去——纯浪费算力。倍率降到 1 时直接关掉。
            plan.run_sr = False
        reasons.append("负载达 ECONOMY 档，超分倍率下调")
    else:
        plan.scale = policy.base_scale

    plan.reason = "、".join(reasons) if reasons else "默认策略"
    return plan


@dataclass
class LoadGovernor:
    """全局负载档位机。

    双阈值 + 迟滞：利用率超过高水位进入降载，低于低水位才回升，
    避免在阈值附近反复抖动导致画面质量忽好忽坏。
    """

    high_watermark: float = 85.0
    low_watermark: float = 60.0
    level: LoadLevel = LoadLevel.NORMAL
    last_gpu_util: float = 0.0
    last_reason: str = "初始状态"
    samples: list[float] = field(default_factory=list)
    window: int = 15

    def observe(self, gpu_util: float | None, queue_pressure: float = 0.0) -> LoadLevel:
        """喂入最新观测。``gpu_util`` 为 None 表示无采集手段（如纯 CPU 部署）。"""
        if gpu_util is not None:
            self.last_gpu_util = gpu_util
            self.samples.append(gpu_util)
            if len(self.samples) > self.window:
                self.samples.pop(0)
        # 用滑动均值而非瞬时值：瞬时尖峰大多来自引擎首次编译或显存抖动
        smoothed = sum(self.samples) / len(self.samples) if self.samples else 0.0

        pressure = max(smoothed, queue_pressure * 100.0)

        if self.level is LoadLevel.NORMAL and pressure >= self.high_watermark:
            self.level = LoadLevel.ECONOMY
            self.last_reason = f"负载 {pressure:.0f}% 超过高水位 {self.high_watermark:.0f}%，降载至 ECONOMY"
        elif self.level is LoadLevel.ECONOMY:
            if pressure >= self.high_watermark + 8:
                self.level = LoadLevel.SAFE
                self.last_reason = f"降载后仍达 {pressure:.0f}%，进一步降载至 SAFE"
            elif pressure <= self.low_watermark:
                self.level = LoadLevel.NORMAL
                self.last_reason = f"负载回落至 {pressure:.0f}%，恢复 NORMAL"
        elif self.level is LoadLevel.SAFE and pressure <= self.low_watermark - 10:
            self.level = LoadLevel.ECONOMY
            self.last_reason = f"负载回落至 {pressure:.0f}%，从 SAFE 回升至 ECONOMY"

        return self.level

    def snapshot(self) -> dict:
        smoothed = sum(self.samples) / len(self.samples) if self.samples else 0.0
        return {
            "level": self.level.value,
            "gpu_util_smoothed": round(smoothed, 1),
            "gpu_util_latest": round(self.last_gpu_util, 1),
            "reason": self.last_reason,
        }
