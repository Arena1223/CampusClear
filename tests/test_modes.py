"""增强档位（三套链路）测试。

守住几条容易退化的约定：
- 实时档不得触发无意义的超分（倍率=1 时 sr 命中却无效果，会误导统计）；
- 取证档必须真正走重降噪（NL-means 强制开启），否则「取证」名不副实；
- 经典/ONNX 两条路径的 enhance 签名一致，可被上层透明替换。
"""
from __future__ import annotations

import pytest

from campusclear.core.enhancer import ClassicalEnhancer
from campusclear.core.modes import (
    PRESETS,
    EnhanceMode,
    Engine,
    build_for_mode,
    policy_for,
)


def test_presets_match_real_time_contract():
    assert PRESETS[EnhanceMode.REALTIME].realtime is True
    assert PRESETS[EnhanceMode.BALANCED].realtime is False
    assert PRESETS[EnhanceMode.FORENSIC].realtime is False


def test_realtime_scale_one_disables_sr():
    """倍率=1 的实时档不应启用超分——否则计划里 sr 命中却没有实际效果。"""
    policy = policy_for(EnhanceMode.REALTIME)
    assert policy.base_scale == 1
    assert policy.enable_sr is False


def test_balanced_and_forensic_enable_sr():
    assert policy_for(EnhanceMode.BALANCED).enable_sr is True
    assert policy_for(EnhanceMode.FORENSIC).enable_sr is True


def test_forensic_forces_nlmeans():
    enhancer, policy, info = build_for_mode(EnhanceMode.FORENSIC, engine=Engine.CLASSICAL)
    assert isinstance(enhancer, ClassicalEnhancer)
    assert enhancer.denoise_mode == "nlmeans"
    assert enhancer.force_nlmeans is True  # 必须强制，否则噪声不达标就不走重降噪
    assert info["engine"] == "classical"


def test_realtime_uses_bilateral_without_force():
    enhancer, _, info = build_for_mode(EnhanceMode.REALTIME, engine=Engine.CLASSICAL)
    assert isinstance(enhancer, ClassicalEnhancer)
    assert enhancer.denoise_mode == "bilateral"
    assert enhancer.force_nlmeans is False


def test_balanced_is_scale_two():
    _, policy, _ = build_for_mode(EnhanceMode.BALANCED, engine=Engine.CLASSICAL)
    assert policy.base_scale == 2


def test_forensic_disables_static_reuse():
    """取证帧不能被静止复用顶替——取证要的是逐帧增强证据。"""
    assert policy_for(EnhanceMode.FORENSIC).enable_static_reuse is False


def test_ffmpeg_engine_is_rejected_by_builder():
    with pytest.raises(ValueError):
        build_for_mode(EnhanceMode.BALANCED, engine=Engine.FFMPEG)


def test_onnx_engine_requires_weights(tmp_path):
    with pytest.raises(RuntimeError):
        build_for_mode(EnhanceMode.BALANCED, models_dir=tmp_path, engine=Engine.ONNX)


def test_auto_falls_back_to_classical_without_weights(tmp_path):
    enhancer, _, info = build_for_mode(EnhanceMode.BALANCED, models_dir=tmp_path, engine=Engine.AUTO)
    assert isinstance(enhancer, ClassicalEnhancer)
    assert info["engine"] == "classical"
