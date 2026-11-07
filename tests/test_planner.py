"""增强策略规划与负载治理测试。

策略模块决定了「一帧究竟花多少算力」，是性能与画质的平衡点，
必须保证：静止场景一定跳过推理、负载过高一定降级、且降级不会导致丢流。
"""

from __future__ import annotations

from campusclear.core.planner import (
    EnhancePlan,
    LoadGovernor,
    LoadLevel,
    PlannerPolicy,
    build_plan,
)
from campusclear.core.quality import FrameQuality


def make_quality(**overrides) -> FrameQuality:
    base = dict(
        brightness=120.0,
        contrast=45.0,
        noise_sigma=2.0,
        blur_score=300.0,
        motion_ratio=0.05,
        is_low_light=False,
        is_noisy=False,
        is_blurry=False,
        is_static=False,
    )
    base.update(overrides)
    return FrameQuality(**base)


def test_static_frame_reuses_previous_result():
    plan = build_plan(make_quality(is_static=True, motion_ratio=0.001), PlannerPolicy())
    assert plan.reuse_previous is True
    assert plan.stages() == []


def test_noisy_static_frame_must_not_reuse():
    """高噪声下运动检测会漏检，此时判为静止也不能复用。

    复用错帧的后果是人物走过时画面卡顿拖影，属安防场景的严重缺陷，
    因此宁可多花算力也必须真实处理。
    """
    quality = make_quality(is_static=True, reuse_allowed=False, noise_sigma=9.0, is_noisy=True)
    plan = build_plan(quality, PlannerPolicy())
    assert plan.reuse_previous is False
    assert plan.stages(), "复用被禁用时必须真的执行增强，否则输出会停在旧帧"


def test_quality_snapshot_exposes_reuse_decision():
    """reuse_allowed 必须出现在对外指标里，否则现场无法判断复用为何没生效。"""
    quality = make_quality(is_static=True, reuse_allowed=False)
    assert "reuse_allowed" in quality.as_dict()


def test_static_reuse_disabled_falls_back_to_stages():
    policy = PlannerPolicy(enable_static_reuse=False)
    plan = build_plan(make_quality(is_static=True), policy)
    assert plan.reuse_previous is False
    assert plan.stages(), "关闭复用后必须至少执行一个阶段，否则输出不经过任何处理"


def test_dark_and_noisy_frame_enables_lowlight_and_denoise():
    quality = make_quality(brightness=30.0, is_low_light=True, noise_sigma=9.0, is_noisy=True)
    plan = build_plan(quality, PlannerPolicy())
    assert plan.run_lowlight is True
    assert plan.run_denoise is True


def test_blurry_frame_extremely_skips_sr_this_round():
    """模糊帧先修清晰度、本轮跳过超分：避免把模糊一起放大。"""
    quality = make_quality(blur_score=20.0, is_blurry=True)
    plan = build_plan(quality, PlannerPolicy())
    assert plan.run_deblur is True
    assert plan.run_sr is False


def test_clean_frame_only_runs_sr_when_other_stages_disabled():
    """画面质量达标且其他阶段关闭时，只剩超分——这才是最省钱的工作状态。"""
    policy = PlannerPolicy(enable_lowlight=False, enable_denoise=False, enable_deblur=False, enable_sr=True)
    plan = build_plan(make_quality(), policy)
    assert plan.stages() == ["sr"]


def test_denoise_is_the_floor_when_no_stage_is_triggered():
    """所有阶段都不满足触发条件时，仍要过一遍降噪，避免输出完全未处理的画面。"""
    policy = PlannerPolicy(
        enable_lowlight=True, enable_denoise=True, enable_deblur=True, enable_sr=False
    )
    plan = build_plan(make_quality(), policy)
    assert plan.stages() == ["denoise"]
    assert plan.reason


def test_sr_runs_when_frame_clean_and_enabled():
    plan = build_plan(make_quality(), PlannerPolicy(enable_sr=True, enable_denoise=False))
    assert plan.run_sr is True


def test_safe_level_keeps_only_floor_stages():
    quality = make_quality(brightness=30.0, is_low_light=True, is_blurry=True, is_noisy=True)
    plan = build_plan(quality, PlannerPolicy(), LoadLevel.SAFE)
    assert plan.run_lowlight is True
    assert plan.run_denoise is True
    assert plan.run_deblur is False
    assert plan.run_sr is False
    assert plan.scale == 1


def test_economy_level_reduces_scale():
    plan = build_plan(make_quality(), PlannerPolicy(base_scale=4), LoadLevel.ECONOMY)
    assert plan.scale == 2, "ECONOMY 档应把 4 倍超分降到 2 倍"


def test_all_stages_disabled_still_returns_valid_plan():
    policy = PlannerPolicy(
        enable_lowlight=False, enable_denoise=False, enable_deblur=False, enable_sr=False
    )
    plan = build_plan(make_quality(), policy)
    assert isinstance(plan, EnhancePlan)
    assert plan.as_dict()["stages"] == []


def test_plan_serialisation_contains_decision_trace():
    plan = build_plan(make_quality(brightness=30.0, is_low_light=True), PlannerPolicy())
    data = plan.as_dict()
    assert "lowlight" in data["stages"]
    assert data["reason"]


# --------------------------------------------------------------------------- #
# 负载治理
# --------------------------------------------------------------------------- #
def test_governor_degrades_on_high_utilization():
    governor = LoadGovernor(high_watermark=85.0, low_watermark=60.0, window=3)
    for _ in range(6):
        governor.observe(95.0)
    assert governor.level in (LoadLevel.ECONOMY, LoadLevel.SAFE)
    assert governor.last_reason


def test_governor_reaches_safe_under_sustained_overload():
    governor = LoadGovernor(high_watermark=85.0, low_watermark=60.0, window=2)
    for _ in range(15):
        governor.observe(99.0)
    assert governor.level is LoadLevel.SAFE


def test_governor_recovers_when_load_drops():
    governor = LoadGovernor(high_watermark=85.0, low_watermark=60.0, window=2)
    for _ in range(8):
        governor.observe(95.0)
    assert governor.level is not LoadLevel.NORMAL
    for _ in range(20):
        governor.observe(10.0)
    assert governor.level is LoadLevel.NORMAL


def test_governor_handles_missing_gpu_signal():
    """纯 CPU 部署时读不到 GPU，系统必须照常工作而不是崩溃。"""
    governor = LoadGovernor()
    level = governor.observe(None)
    assert level is LoadLevel.NORMAL
    assert governor.snapshot()["gpu_util_smoothed"] == 0.0


def test_queue_pressure_can_trigger_degrade_without_gpu_signal():
    governor = LoadGovernor(high_watermark=85.0, window=2)
    for _ in range(4):
        governor.observe(None, queue_pressure=1.0)
    assert governor.level is not LoadLevel.NORMAL


def test_governor_snapshot_shape():
    governor = LoadGovernor()
    governor.observe(50.0)
    snapshot = governor.snapshot()
    assert set(snapshot) == {"level", "gpu_util_smoothed", "gpu_util_latest", "reason"}
