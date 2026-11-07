"""增强器与模型注册表测试。

重点验证三件事：
1. 无权重、无 GPU 时系统仍能产出结果（兜底链路可用）；
2. 图像数据在各格式转换中不丢信息、不串色（BGR/RGB 与 NCHW 往返）；
3. 权重被篡改时必须显式失败，不能被静默忽略。
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from campusclear.core.enhancer import (
    ClassicalEnhancer,
    OnnxEnhancer,
    crop_back,
    estimate_gain,
    from_nchw,
    pad_to_multiple,
    to_nchw,
)
from campusclear.core.pipeline import StreamPipeline, _merge_boxes, _near
from campusclear.core.planner import EnhancePlan, PlannerPolicy, build_plan
from campusclear.core.quality import QualityAnalyzer
from campusclear.models.registry import ModelIntegrityError, ModelRegistry, sha256_of


# --------------------------------------------------------------------------- #
# 几何与格式
# --------------------------------------------------------------------------- #
def test_pad_to_multiple_pads_odd_sizes():
    image = np.zeros((35, 47, 3), dtype=np.uint8)
    padded, info = pad_to_multiple(image, 32)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0
    assert info == (0, 0, 35, 47)


def test_pad_to_multiple_is_identity_when_already_aligned():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    padded, info = pad_to_multiple(image, 32)
    assert padded is image
    assert info == (0, 0, 64, 96)


def test_crop_back_restores_original_region():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    padded, info = pad_to_multiple(image, 32)
    restored = crop_back(padded, info)
    assert restored.shape == image.shape
    assert np.array_equal(restored, image)


def test_nchw_roundtrip_preserves_colors(frame_factory):
    """BGR→RGB→NCHW→RGB→BGR 往返必须与原图完全一致，否则会出现红蓝通道互换这种低级事故。"""
    frame = frame_factory(width=64, height=48)
    tensor = to_nchw(frame)
    assert tensor.shape == (1, 3, 48, 64)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    restored = from_nchw(tensor)
    assert restored.shape == frame.shape
    assert np.abs(restored.astype(int) - frame.astype(int)).max() <= 1


def test_estimate_gain_higher_for_dark_frames(frame_factory):
    bright = estimate_gain(frame_factory(mode="normal"))
    dark = estimate_gain(frame_factory(mode="dark"))
    assert dark > bright
    assert 0.0 <= dark <= 1.0


def test_merge_boxes_joins_neighbours():
    boxes = [(0, 0, 40, 40), (50, 0, 90, 40)]  # 间隔 10px，应合并
    merged = _merge_boxes(boxes, gap=32)
    assert len(merged) == 1
    assert merged[0] == (0, 0, 90, 40)


def test_merge_boxes_keeps_distant_boxes_separate():
    boxes = [(0, 0, 40, 40), (600, 600, 700, 700)]
    merged = _merge_boxes(boxes, gap=32)
    assert len(merged) == 2


def test_near_helper():
    assert _near((0, 0, 10, 10), (20, 0, 30, 30), gap=32) is True
    assert _near((0, 0, 10, 10), (500, 500, 510, 510), gap=32) is False


# --------------------------------------------------------------------------- #
# 兜底增强器
# --------------------------------------------------------------------------- #
def test_classical_enhancer_keeps_shape_without_sr(frame_factory):
    enhancer = ClassicalEnhancer()
    frame = frame_factory(width=128, height=96)
    plan = build_plan(
        QualityAnalyzer().analyze(frame),
        PlannerPolicy(enable_sr=False, enable_static_reuse=False),
    )
    output = enhancer.enhance(frame, plan)
    assert output.shape == frame.shape
    assert output.dtype == np.uint8


def test_classical_enhancer_applies_scale(frame_factory):
    enhancer = ClassicalEnhancer()
    frame = frame_factory(width=64, height=48)
    plan = EnhancePlan(run_sr=True, scale=2)
    output = enhancer.enhance(frame, plan)
    assert output.shape == (96, 128, 3)


def test_classical_enhancer_brightens_dark_frame(frame_factory):
    enhancer = ClassicalEnhancer()
    frame = frame_factory(mode="dark")
    plan = EnhancePlan(run_lowlight=True, run_denoise=True)
    output = enhancer.enhance(frame, plan)
    assert output.mean() > frame.mean()


# --------------------------------------------------------------------------- #
# ONNX 增强器（无权重场景）
# --------------------------------------------------------------------------- #
def test_onnx_enhancer_reports_unavailable_without_weights(settings):
    registry = ModelRegistry(settings.models_dir)
    enhancer = OnnxEnhancer(registry)
    assert enhancer.available() is False
    assert registry.available_stages() == []


def test_onnx_enhancer_falls_back_to_classical_without_weights(settings, frame_factory):
    """没有权重时不报错，且**逐阶段回退经典算法**（而不是整条跳过）。

    这是回归测试：曾经缺失阶段会被静默跳过，导致「只放一个超分权重 →
    低光/降噪全失效」。现在每个阶段独立回退。
    """
    registry = ModelRegistry(settings.models_dir)
    enhancer = OnnxEnhancer(registry)
    frame = frame_factory(width=96, height=64)
    plan = EnhancePlan(run_lowlight=True, run_denoise=True, run_deblur=True, run_sr=True, scale=2)
    output = enhancer.enhance(frame, plan)
    # 超分阶段回退为经典 Lanczos 插值，因此尺寸按 scale 放大
    assert output.shape == (128, 192, 3)
    assert output.dtype == frame.dtype


class _FixedScaleSession:
    """模拟一个原生 4× 的超分模型（与随包发布的 x4v3 权重一致）。"""

    def __init__(self, scale: int = 4) -> None:
        self.scale = scale

    def run(self, _output_names, feeds):
        tensor = next(iter(feeds.values()))
        return [np.repeat(np.repeat(tensor, self.scale, axis=2), self.scale, axis=3)]


def _fake_sr_enhancer(scale: int = 4) -> OnnxEnhancer:
    spec = SimpleNamespace(scale=scale, fp16=False)
    loaded = SimpleNamespace(
        session=_FixedScaleSession(scale),
        input_name="input",
        output_name="output",
        spec=spec,
        provider="CPUExecutionProvider",
    )

    class _Registry:
        def available_stages(self):
            return ["sr"]

        def get(self, _stage):
            return loaded

        def close(self):
            return None

    return OnnxEnhancer(_Registry())


def test_onnx_sr_is_normalized_to_requested_scale(frame_factory):
    """模型原生 4×、计划请求 2× 时，输出必须是 2×。

    这是服务器回推的关键约束：rawvideo 管道按固定帧长读，输出 4× 会被
    ffmpeg 当成 4 帧切分，表现为画面撕裂而不是报错。
    """
    enhancer = _fake_sr_enhancer(scale=4)
    frame = frame_factory(width=64, height=48)
    output = enhancer.enhance(frame, EnhancePlan(run_sr=True, scale=2))
    assert output.shape == (96, 128, 3)


def test_onnx_sr_roi_is_normalized_without_shape_error(frame_factory):
    """ROI 路径同样要归一：4× patch 不能直接贴进 2× 的 ROI 区域。"""
    enhancer = _fake_sr_enhancer(scale=4)
    frame = frame_factory(width=64, height=48)
    roi = [(8, 8, 40, 32)]
    output = enhancer.enhance(frame, EnhancePlan(run_sr=True, scale=2), roi=roi)
    assert output.shape == (96, 128, 3)


def test_onnx_sr_skips_inference_at_scale_one(frame_factory):
    """倍率 1 不应跑 4× 模型再缩回去（ECONOMY 档的省算力路径）。"""
    enhancer = _fake_sr_enhancer(scale=4)
    frame = frame_factory(width=64, height=48)
    output = enhancer.enhance(frame, EnhancePlan(run_sr=True, scale=1))
    assert output.shape == frame.shape


def test_pipeline_normalizes_output_canvas(frame_factory):
    """负载降档导致 enhancer 返回原尺寸时，管线仍要补回固定输出画布。"""
    pipeline = StreamPipeline.__new__(StreamPipeline)
    pipeline.output_scale = 2
    frame = frame_factory(width=64, height=48)
    normalized = pipeline._match_output_size(frame, frame)  # noqa: SLF001
    assert normalized.shape == (96, 128, 3)

    oversized = np.zeros((192, 256, 3), dtype=np.uint8)
    assert pipeline._match_output_size(frame, oversized).shape == (96, 128, 3)  # noqa: SLF001


def test_resolve_region_falls_back_to_full_frame_for_tiny_roi(settings, frame_factory):
    registry = ModelRegistry(settings.models_dir)
    enhancer = OnnxEnhancer(registry)
    frame = frame_factory(width=200, height=150)
    region = enhancer._resolve_region(frame, [(10, 10, 12, 12)])  # noqa: SLF001
    assert region == (0, 0, 150, 200)


def test_resolve_region_unions_multiple_boxes(settings, frame_factory):
    """多目标时应取所有框的最小外接矩形（返回顺序为 top, left, bottom, right）。"""
    registry = ModelRegistry(settings.models_dir)
    enhancer = OnnxEnhancer(registry)
    frame = frame_factory(width=200, height=150)
    region = enhancer._resolve_region(frame, [(20, 20, 60, 60), (100, 80, 160, 140)])  # noqa: SLF001
    assert region == (20, 20, 140, 160)


# --------------------------------------------------------------------------- #
# 模型完整性（安全关键路径）
# --------------------------------------------------------------------------- #
def test_registry_without_manifest_is_empty(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.specs == {}
    assert registry.verify() == {}
    assert registry.describe()


def test_registry_detects_hash_mismatch(tmp_path):
    """权重哈希不匹配必须报错：内网分发靠介质拷贝，哈希是唯一可信凭据。"""
    weights = tmp_path / "tampered.onnx"
    weights.write_bytes(b"not-a-real-model")
    manifest = {
        "models": {
            "sr": {"file": "tampered.onnx", "sha256": "0" * 64, "scale": 2}
        }
    }
    (tmp_path / "models_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    registry = ModelRegistry(tmp_path, verify_hash=True)
    report = registry.verify()
    assert report["sr"].startswith("哈希不匹配")
    with pytest.raises(ModelIntegrityError):
        registry.verify_or_raise()


def test_registry_accepts_correct_hash(tmp_path):
    weights = tmp_path / "ok.onnx"
    weights.write_bytes(b"fake-but-stable")
    digest = sha256_of(weights)
    manifest = {"models": {"denoise": {"file": "ok.onnx", "sha256": digest}}}
    (tmp_path / "models_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    registry = ModelRegistry(tmp_path, verify_hash=True)
    assert registry.verify()["denoise"] == "ok"
    assert registry.available_stages() == ["denoise"]


def test_registry_reports_missing_file(tmp_path):
    manifest = {"models": {"deblur": {"file": "absent.onnx", "sha256": ""}}}
    (tmp_path / "models_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = ModelRegistry(tmp_path)
    assert registry.verify()["deblur"].startswith("缺失")
    assert registry.available_stages() == [], "只有清单、没有权重时不能声称可用"


def test_registry_loads_session_once_under_concurrency(tmp_path, monkeypatch):
    """多路首帧并发时，同一阶段只能创建一个 ORT 会话。"""
    weights = tmp_path / "sr.onnx"
    weights.write_bytes(b"fake-onnx")
    manifest = {
        "models": {"sr": {"file": "sr.onnx", "sha256": "", "scale": 4}}
    }
    (tmp_path / "models_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = ModelRegistry(tmp_path, verify_hash=False)

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def slow_load(spec, path):
        calls["n"] += 1
        started.set()
        release.wait(timeout=2)
        return object()

    monkeypatch.setattr(registry, "_load_session", slow_load)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(registry.get("sr")))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=1)
    time.sleep(0.05)  # 给其余线程一个「同时冲进来」的窗口
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert calls["n"] == 1, "并发 get() 不能重复创建会话"
    assert len(results) == 4
    assert len({id(item) for item in results}) == 1, "所有调用应拿到同一个会话"


def test_registry_ignores_unknown_stage(tmp_path):
    manifest = {"models": {"sr": {"file": "a.onnx"}, "superpixel": {"file": "b.onnx"}}}
    (tmp_path / "models_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = ModelRegistry(tmp_path)
    assert "superpixel" not in registry.specs
    assert "sr" in registry.specs
