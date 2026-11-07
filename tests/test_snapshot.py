"""截图增强（单帧按预设增强）测试。"""
from __future__ import annotations

import numpy as np

from campusclear.core.snapshot import PRESET_MAP, enhance_still, list_presets


def _frame(w=60, h=40):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype("uint8")


def test_presets_present():
    keys = {p["key"] for p in list_presets()}
    assert {"light", "standard", "sr2", "sr4", "forensic"} <= keys


def test_light_preset_keeps_size():
    frame = _frame()
    out, preset = enhance_still(frame, "light")
    assert preset.key == "light"
    assert out.shape == frame.shape


def test_sr2_preset_outputs_two_x():
    """超分 2× 必须真的输出 2×（即便底层 ONNX 模型是 4×）。"""
    frame = _frame(60, 40)
    out, _ = enhance_still(frame, "sr2")
    assert (out.shape[1], out.shape[0]) == (120, 80)


def test_sr4_preset_outputs_four_x():
    frame = _frame(60, 40)
    out, _ = enhance_still(frame, "sr4")
    assert (out.shape[1], out.shape[0]) == (240, 160)


def test_unknown_preset_falls_back_to_standard():
    out, preset = enhance_still(_frame(), "no-such-preset")
    assert preset.key == "standard"
    assert out.dtype == np.uint8


def test_all_presets_produce_valid_frames():
    frame = _frame(48, 32)
    for key in PRESET_MAP:
        out, _ = enhance_still(frame, key)
        assert out.dtype == np.uint8 and out.size > 0
