"""帧质量评估测试。

这些算法是整条链路的决策依据，一旦判错就会出现
「该增强的没增强」或「不该超分的白烧算力」，必须被测试锁死。
"""

from __future__ import annotations

import numpy as np

from campusclear.core.quality import QualityAnalyzer


def test_bright_frame_not_low_light(frame_factory):
    analyzer = QualityAnalyzer()
    quality = analyzer.analyze(frame_factory(mode="normal"))
    assert quality.brightness > 60
    assert quality.is_low_light is False


def test_dark_frame_detected_as_low_light(frame_factory):
    analyzer = QualityAnalyzer()
    quality = analyzer.analyze(frame_factory(mode="dark"))
    assert quality.brightness < 60
    assert quality.is_low_light is True


def test_noisy_frame_has_higher_noise_estimate(frame_factory):
    clean = QualityAnalyzer().analyze(frame_factory(mode="normal")).noise_sigma
    noisy = QualityAnalyzer().analyze(frame_factory(mode="noisy")).noise_sigma
    assert noisy > clean, f"含噪帧噪声估计 {noisy:.2f} 应高于干净帧 {clean:.2f}"


def test_blurry_frame_has_lower_laplacian_variance(frame_factory):
    sharp = QualityAnalyzer().analyze(frame_factory(mode="normal")).blur_score
    blurred = QualityAnalyzer().analyze(frame_factory(mode="blurry")).blur_score
    assert blurred < sharp, "模糊帧的拉普拉斯方差应显著更低"


def test_static_scene_becomes_reusable_after_warmup(frame_factory):
    """静止画面在背景模型收敛后应被判为可复用，这是最大的算力节省来源。"""
    analyzer = QualityAnalyzer(motion_threshold=0.004)
    frame = frame_factory(mode="normal")
    for _ in range(30):
        quality = analyzer.analyze(frame)
    assert quality.is_static is True
    assert quality.motion_ratio < 0.004
    assert quality.reuse_allowed is True


def test_reuse_gate_closes_when_noise_exceeds_limit(frame_factory):
    """噪声闸门：高噪声下必须放弃复用，否则会出现「画面在动却卡住」。

    测试手法：把阈值压到极低，等价于「当前噪声水平已超限」，
    从而在确定的静止画面上验证闸门逻辑，不依赖构造特定噪声强度。
    """
    analyzer = QualityAnalyzer(reuse_max_noise=0.01)
    frame = frame_factory(mode="normal")
    for _ in range(30):
        quality = analyzer.analyze(frame)
    assert quality.is_static is True, "画面本身是静止的，运动判据应当成立"
    assert quality.noise_sigma > 0.01
    assert quality.reuse_allowed is False, "噪声超限时必须关闭复用"


def test_reuse_gate_opens_when_noise_within_limit(frame_factory):
    analyzer = QualityAnalyzer(reuse_max_noise=1000.0)
    frame = frame_factory(mode="normal")
    for _ in range(30):
        quality = analyzer.analyze(frame)
    assert quality.reuse_allowed is True


def test_motion_breaks_static_judgement(frame_factory):
    analyzer = QualityAnalyzer()
    frame = frame_factory(mode="normal")
    for _ in range(30):
        analyzer.analyze(frame)

    moved = frame.copy()
    # 在画面中央放一块明显不同的区域，模拟有人走过
    moved[80:200, 100:240] = 255
    quality = analyzer.analyze(moved)
    assert quality.motion_ratio > 0.004
    assert quality.is_static is False


def test_motion_mask_matches_frame_resolution(frame_factory):
    analyzer = QualityAnalyzer()
    frame = frame_factory(width=400, height=300)
    analyzer.analyze(frame)
    mask = analyzer.motion_mask(frame)
    assert mask.shape == (300, 400)


def test_reset_clears_warmup_state(frame_factory):
    analyzer = QualityAnalyzer()
    frame = frame_factory()
    for _ in range(30):
        analyzer.analyze(frame)
    analyzer.reset()
    assert analyzer._warmup == 0  # noqa: SLF001 - 直接验证内部状态是这里最直接的断言
    assert analyzer.analyze(frame).is_static is False


def test_empty_frame_raises():
    analyzer = QualityAnalyzer()
    try:
        analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("空帧应抛出 ValueError")


def test_quality_snapshot_is_serialisable(frame_factory):
    quality = QualityAnalyzer().analyze(frame_factory())
    data = quality.as_dict()
    assert set(data) >= {
        "brightness", "contrast", "noise_sigma", "blur_score",
        "motion_ratio", "is_low_light", "is_noisy", "is_blurry", "is_static",
    }
