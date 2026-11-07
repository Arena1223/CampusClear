"""离线推理引擎。

包含两个实现：

- ``ClassicalEnhancer``：纯 OpenCV/NumPy 的兜底实现。**没有 GPU、没有权重也能跑**，
  保证系统在任意内网机器上都不会因为缺少模型而彻底失效；
- ``OnnxEnhancer``：ONNX Runtime 推理实现，按需加载四个阶段的权重，
  支持 TensorRT / CUDA / CPU 三级执行提供器回退。

一个刻意为之的设计（也是本项目的主要工程取舍之一）：
**超分只在亮度通道（Y）上执行**，色度通道用三次插值上采样。
理由见 docs/03-视频流处理.md：3 通道同时超分要多吃 3 倍算力，
而人眼对亮度细节的敏感度远高于色度；同时这种分离策略能避开
AI 超分常见的色偏问题，对新旧摄像头混合的校园网络尤其重要。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from ..models.registry import STAGES, ModelRegistry
from .planner import EnhancePlan

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def pad_to_multiple(image: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """把 H、W 补齐到 ``multiple`` 的整数倍，返回补齐图与原始尺寸信息。

    卷积网络在 stride 作用下对输入尺寸有整除要求，直接 resize 会让输出与
    输入的像素位置无法对齐（回贴后出现错位）。因此选择反射填充而不是缩放。
    """
    height, width = image.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, height, width)
    padded = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
    )
    return padded, (0, 0, height, width)


def crop_back(padded: np.ndarray, info: tuple[int, int, int, int], scale: int = 1) -> np.ndarray:
    top, left, height, width = info
    return padded[top : top + height * scale, left : left + width * scale]


def to_nchw(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → float32 NCHW，归一化到 [0,1]。"""
    tensor = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...])


def from_nchw(tensor: np.ndarray) -> np.ndarray:
    """float32 NCHW → BGR uint8 HWC。"""
    data = np.clip(tensor[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0
    return np.ascontiguousarray(data[:, :, ::-1].astype(np.uint8))  # RGB→BGR


def estimate_gain(frame: np.ndarray) -> float:
    """粗略估计画面的「欠曝程度」，用于低光增强的强度自适应。

    返回 0-1：越接近 1 表示越需要提亮。取亮度分位数的稳健形式，
    避免少量高光（如路灯）把整体均值拉高。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p70 = float(np.percentile(gray, 70))
    return float(np.clip((128.0 - p70) / 128.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 兜底实现
# --------------------------------------------------------------------------- #
class ClassicalEnhancer:
    """纯 CPU 经典算法兜底链路。

    不做超分（会引入明显涂抹感），只做：降噪 → CLAHE 局部对比度 →
    非锐化掩模。

    **执行顺序是刻意的，不能调换**：必须先降噪、后提亮。
    CLAHE 是局部对比度拉伸，会把噪声和有用细节一起放大；
    若先提亮再降噪，等于先把噪声放大几倍再试图压回去，
    实测结果比不处理还糟（夜间场景增强后噪点反而更明显）。
    这也是图像复原领域的通行做法：降噪永远在最前面。

    **降噪器分两档，按场景选用（``denoise_mode``）：**

    · ``bilateral``（默认，实时）：640x360 约 5ms、1080p 约 25-40ms。
      但对夜间「低对比度目标 + 强噪声」的处理能力有限——
      目标与背景的灰度差可能只有 20 余级，而噪声 σ 就有 10 以上，
      双边滤波要么压不住噪声、要么连目标边缘一起抹平，二者不可兼得。
      这是经典算法的天花板，也正是需要 SCUNet 这类学习型降噪器的原因。

    · ``nlmeans``（离线批处理）：画质显著更好，但实测 640x360 就要 500ms/帧、
      1080p 预计 4 秒以上，**完全不能用于实时链路**。
      只适合离线处理已录像的片段，例如生成申报材料用的高质量对比图。

    默认必须是 bilateral：曾经把 nlmeans 设为默认，实测直接把链路拖到 2fps。
    """

    name = "classical-cpu"

    def __init__(self, denoise_mode: str = "bilateral", force_nlmeans: bool = False) -> None:
        if denoise_mode not in ("bilateral", "nlmeans"):
            raise ValueError(f"未知降噪模式: {denoise_mode}")
        self.denoise_mode = denoise_mode
        # 取证档用：即使噪声没到门限也强制走 NL-means，取最高画质。
        # 默认 False，实时/标准档的行为完全不变。
        self.force_nlmeans = force_nlmeans

    def enhance(self, frame: np.ndarray, plan: EnhancePlan, roi: list[tuple] | None = None) -> np.ndarray:
        output = frame
        # 噪声水平由规划器随计划下发，执行器据此选择降噪强度与提亮幅度，
        # 不必自己再估一遍（也避免与 analyzer 的降采样口径不一致）。
        sigma = plan.noise_sigma
        strong_noise = sigma >= 3.0

        # 1) 降噪必须最先执行：此时噪声还未被任何对比度拉伸放大
        if plan.run_denoise:
            if self.denoise_mode == "nlmeans" and (strong_noise or self.force_nlmeans):
                output = cv2.fastNlMeansDenoisingColored(
                    output, None, h=10, hColor=10,
                    templateWindowSize=7, searchWindowSize=21,
                )
            elif strong_noise:
                # 强噪声下加大双边滤波的范围与混合度。注意 sigmaColor 越大
                # 混合越强（越平滑），同时低对比度边缘也会被抹掉一些——
                # 这正是本链路在夜间场景的固有取舍。
                output = cv2.bilateralFilter(output, d=9, sigmaColor=45, sigmaSpace=9)
            else:
                output = cv2.bilateralFilter(output, d=5, sigmaColor=28, sigmaSpace=5)

        # 2) 提亮 / 局部对比度：在相对干净的图上做，不会放大已抑制的噪声。
        #    噪声偏高时收紧 clip 限制，避免把残余噪声拉得太显眼。
        if plan.run_lowlight or plan.run_denoise:
            lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            if plan.run_lowlight and strong_noise:
                clip_limit = 1.8
            elif plan.run_lowlight:
                clip_limit = 2.5
            else:
                clip_limit = 1.6
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            output = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        # 3) 去模糊：非锐化掩模同样会放大噪声，因此也必须在降噪之后
        if plan.run_deblur:
            blurred = cv2.GaussianBlur(output, (0, 0), 1.2)
            output = cv2.addWeighted(output, 1.6, blurred, -0.6, 0)

        # 兜底链路也支持尺寸提升，但只用插值，不做生成式超分
        if plan.run_sr and plan.scale > 1:
            height, width = output.shape[:2]
            output = cv2.resize(
                output,
                (width * plan.scale, height * plan.scale),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output

    # ------------------------------------------------------------------ #
    # 单阶段执行：供 OnnxEnhancer 在「某阶段没有权重」时逐阶段回退使用。
    # 逻辑与本类 enhance() 保持一致——否则会出现「经典链路和回退路径结果不一样」。
    # ------------------------------------------------------------------ #
    def denoise_only(self, frame: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        strong = sigma >= 3.0
        if self.denoise_mode == "nlmeans" and (strong or self.force_nlmeans):
            return cv2.fastNlMeansDenoisingColored(
                frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
        if strong:
            return cv2.bilateralFilter(frame, d=9, sigmaColor=45, sigmaSpace=9)
        return cv2.bilateralFilter(frame, d=5, sigmaColor=28, sigmaSpace=5)

    def lowlight_only(self, frame: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        strong = sigma >= 3.0
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clip_limit = 1.8 if strong else 2.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    def deblur_only(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
        return cv2.addWeighted(frame, 1.6, blurred, -0.6, 0)

    def sr_only(self, frame: np.ndarray, scale: int) -> np.ndarray:
        if scale and scale > 1:
            height, width = frame.shape[:2]
            return cv2.resize(
                frame, (width * scale, height * scale), interpolation=cv2.INTER_LANCZOS4
            )
        return frame


# --------------------------------------------------------------------------- #
# ONNX 实现
# --------------------------------------------------------------------------- #
@dataclass
class InferenceStat:
    calls: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class OnnxEnhancer:
    """ONNX Runtime 离线推理实现。

    加载策略：惰性。只有某阶段真正被计划命中时才创建会话，
    避免服务启动时把四个模型全部编译进显存（会导致启动慢且常驻占用高）。
    """

    name = "onnx"

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        use_fp16: bool = True,
        tile_max_pixels: int = 1920 * 1080 // 2,
        tile_overlap: int = 16,
        device_id: int = 0,
    ) -> None:
        self.registry = registry
        self.use_fp16 = use_fp16
        self.tile_max_pixels = tile_max_pixels
        self.tile_overlap = tile_overlap
        self.device_id = device_id
        self.stats: dict[str, InferenceStat] = {}
        self._fallback = ClassicalEnhancer()
        self._logged_fallback: set[str] = set()
        # 每个阶段一把锁：多路并发时把 GPU 上的前向计算串行化。
        # 这样做的代价是吞吐上限受单卡约束，收益是显存占用稳定、
        # 不会因为多路同时申请显存触发 OOM 而整机崩掉。安防值守场景下稳定性优先。
        self._stage_locks: dict[str, threading.Lock] = {stage: threading.Lock() for stage in STAGES}
        # 清单里实际提供权重的阶段。缺失的阶段逐个回退到经典算法，
        # 而不是整条链路跳过——否则只放一个超分权重会让低光/降噪静默失效。
        self._available: set[str] = set(registry.available_stages())

    def _lock_for(self, stage: str) -> threading.Lock:
        return self._stage_locks.setdefault(stage, threading.Lock())

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """至少有一个阶段可用才认为推理链路可用。"""
        return bool(self.registry.available_stages())

    def enhance(self, frame: np.ndarray, plan: EnhancePlan, roi: list[tuple] | None = None) -> np.ndarray:
        # 执行顺序：降噪 → 低光 → 去模糊 → 超分。
        # 与 models_manifest.json 里 STAGES 的清单顺序不同，这是刻意的：
        # 降噪必须最先做，否则后续的低光曲线调整与非锐化都会把噪声一起放大。
        # 清单顺序只表示「阶段身份」，不表示执行顺序。
        #
        # 每个阶段独立判断「有没有权重」：有就走 ONNX，没有就用经典算法兜底。
        # 这样权重可以一个一个补，不会因为缺某个阶段而让整条链路退化。
        output = frame
        fallback = self._fallback

        if plan.run_denoise:
            if "denoise" in self._available:
                output = self._apply("denoise", output, roi=roi, scale=1)
            else:
                output = fallback.denoise_only(output, plan.noise_sigma)

        # 低光增强必须整帧执行：只增强局部会造成明显的亮度接缝。
        if plan.run_lowlight:
            if "lowlight" in self._available:
                output = self._apply("lowlight", output, roi=None, scale=1)
            else:
                output = fallback.lowlight_only(output, plan.noise_sigma)

        if plan.run_deblur:
            if "deblur" in self._available:
                output = self._apply("deblur", output, roi=roi, scale=1)
            else:
                output = fallback.deblur_only(output)

        if plan.run_sr:
            if "sr" in self._available:
                output = self._apply_y_sr(output, roi=roi, scale=plan.scale)
            else:
                output = fallback.sr_only(output, plan.scale)

        return output

    # ------------------------------------------------------------------ #
    # 内部：单阶段应用
    # ------------------------------------------------------------------ #
    def _apply(self, stage: str, frame: np.ndarray, roi: list[tuple] | None, scale: int) -> np.ndarray:
        region = self._resolve_region(frame, roi)
        top, left, bottom, right = region
        patch = frame[top:bottom, left:right] if region != (0, 0, frame.shape[0], frame.shape[1]) else frame

        enhanced = self._run_stage(stage, patch, scale)

        if enhanced is None:
            return frame
        if patch is frame:
            return enhanced

        # 回贴：ROI 区域的增强结果覆盖到原帧的副本上
        result = frame.copy()
        result[top:bottom, left:right] = enhanced
        return result

    def _resolve_region(self, frame: np.ndarray, roi: list[tuple] | None) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        if not roi:
            return 0, 0, height, width
        xs = [box[0] for box in roi] + [box[2] for box in roi]
        ys = [box[1] for box in roi] + [box[3] for box in roi]
        left = max(0, int(min(xs)))
        top = max(0, int(min(ys)))
        right = min(width, int(max(xs)))
        bottom = min(height, int(max(ys)))
        if right - left < 16 or bottom - top < 16:
            return 0, 0, height, width
        return top, left, bottom, right

    def _run_stage(self, stage: str, patch: np.ndarray, scale: int) -> np.ndarray | None:
        try:
            loaded = self.registry.get(stage)
        except Exception as exc:  # 完整性异常向上抛，业务上必须人工介入
            raise exc
        if loaded is None:
            if stage not in self._logged_fallback:
                logger.warning("阶段 %s 无可用权重，使用经典算法兜底", stage)
                self._logged_fallback.add(stage)
            return None

        import time

        with self._lock_for(stage):
            # 大图分块：显存与延迟都比一次吃掉整幅图更可控
            if patch.shape[0] * patch.shape[1] > self.tile_max_pixels:
                return self._tiled_inference(stage, loaded, patch, scale)

            start = time.perf_counter()
            result = self._infer(loaded, patch)
            elapsed = (time.perf_counter() - start) * 1000.0
        stat = self.stats.setdefault(stage, InferenceStat())
        stat.calls += 1
        stat.total_ms += elapsed
        return result

    def _tiled_inference(self, stage: str, loaded, patch: np.ndarray, scale: int) -> np.ndarray:
        height, width = patch.shape[:2]
        overlap = self.tile_overlap
        # 让每块面积不超过阈值来反推块边长
        tile_side = int((self.tile_max_pixels) ** 0.5)
        tile_side = max(64, min(tile_side, max(height, width)))
        step = tile_side - overlap
        # 用「模型实际输出倍率」而不是「请求倍率」：ONNX 超分模型可能原生 4×，
        # 而调用方请求的是 2×；若按请求倍率回贴，会与模型输出尺寸对不上而崩溃。
        mscale = loaded.spec.scale if getattr(loaded.spec, "scale", 1) > 1 else scale
        output = None

        for y0 in range(0, height, step):
            for x0 in range(0, width, step):
                y1 = min(height, y0 + tile_side)
                x1 = min(width, x0 + tile_side)
                tile = patch[y0:y1, x0:x1]
                tile_out = self._infer(loaded, tile)

                if output is None:
                    output = np.zeros(
                        (height * mscale, width * mscale, 3), dtype=np.uint8
                    )

                # 裁剪重叠区，只保留本块的「核心」区域，避免缝线
                pad_top = overlap if y0 > 0 else 0
                pad_left = overlap if x0 > 0 else 0
                core = tile_out[pad_top * mscale :, pad_left * mscale :]
                ty, tx = (y0 + pad_top) * mscale, (x0 + pad_left) * mscale
                output[ty : ty + core.shape[0], tx : tx + core.shape[1]] = core

        if output is None:  # pragma: no cover - 理论上不会发生
            return patch
        return output

    def _infer(self, loaded, patch: np.ndarray) -> np.ndarray:
        """执行一次网络前向。优先走 GPU 显存直通（IOBinding），失败退回常规路径。"""
        tensor = to_nchw(patch)
        feed_type = np.float16 if (self.use_fp16 and loaded.spec.fp16) else np.float32
        try:
            tensor = tensor.astype(feed_type)
        except TypeError:  # pragma: no cover
            pass

        result = self._infer_iobinding(loaded, tensor)
        if result is None:
            outputs = loaded.session.run([loaded.output_name], {loaded.input_name: tensor})
            result = outputs[0]
        return from_nchw(result.astype(np.float32))

    def _infer_iobinding(self, loaded, tensor: np.ndarray) -> np.ndarray | None:
        """显存直通推理，省掉 host↔device 两次拷贝。

        在 CPU 执行提供器或某些驱动组合下 IOBinding 不可用，
        此处刻意返回 None 让上层回退，而不是让整条链路报错。
        """
        if not loaded.provider.startswith(("CUDA", "Tensorrt")):
            return None
        try:
            import onnxruntime as ort

            binding = loaded.session.io_binding()
            device = f"cuda:{self.device_id}"
            ort_value = ort.OrtValue.ortvalue_from_numpy(tensor, "cuda", self.device_id)
            binding.bind_input(
                name=loaded.input_name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=tensor.dtype,
                shape=tensor.shape,
                buffer_ptr=ort_value.data_ptr(),
            )
            # 输出形状：空间维度按 scale 放大，其余不变（四阶段网络均为同尺寸或定倍率输出）
            scale = loaded.spec.scale if loaded.spec.scale > 1 else 1
            out_shape = (tensor.shape[0], tensor.shape[1], tensor.shape[2] * scale, tensor.shape[3] * scale)
            out_value = ort.OrtValue.ortvalue_from_shape_and_type(out_shape, np.float32, device)
            binding.bind_output(
                name=loaded.output_name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=out_shape,
                buffer_ptr=out_value.data_ptr(),
            )
            loaded.session.run_with_iobinding(binding)
            return binding.get_outputs()[0].numpy()
        except Exception as exc:  # noqa: BLE001 - 任何异常都只降级，不影响出流
            logger.debug("显存直通不可用，回退常规推理: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Y 通道超分
    # ------------------------------------------------------------------ #
    def _apply_y_sr(self, frame: np.ndarray, roi: list[tuple] | None, scale: int) -> np.ndarray:
        """仅在亮度通道做超分，色度通道用三次插值上采样。

        这样做的收益：
        - 计算量约为 RGB 三通道超分的 1/3（网络按通道数线性增长）；
        - 色度信息本身带宽低、人眼不敏感，插值几乎无损；
        - 避免生成式超分在色度上产生的偏色，保证与原有安防平台色调一致。
        """
        # 1× 不需要超分：模型原生倍率可能是 4×，跑完再缩回去纯属浪费算力
        # （ECONOMY 档会把 scale 降到 1，正是这种情况）。
        if scale <= 1:
            return frame

        region = self._resolve_region(frame, roi)
        top, left, bottom, right = region
        is_full = region == (0, 0, frame.shape[0], frame.shape[1])
        patch = frame if is_full else frame[top:bottom, left:right]

        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

        # 网络期望 3 通道输入，这里把 Y 复制三份送入（单通道网络则直接送 Y）
        gray_bgr = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        enhanced_bgr = self._run_stage("sr", gray_bgr, scale)
        if enhanced_bgr is None:
            return frame
        enhanced_y = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # 模型原生倍率不一定等于调用方请求的倍率（例如清单里是 4×、计划里
        # 请求 2×）。必须在这里归一到「请求倍率」，否则下游 rawvideo 管道
        # 会按错误的帧长切分字节流，输出表现为撕裂/错位而非黑屏，极难排查。
        target_width = patch.shape[1] * scale
        target_height = patch.shape[0] * scale
        if (enhanced_y.shape[1], enhanced_y.shape[0]) != (target_width, target_height):
            interpolation = (
                cv2.INTER_AREA if enhanced_y.shape[1] > target_width else cv2.INTER_CUBIC
            )
            enhanced_y = cv2.resize(
                enhanced_y, (target_width, target_height), interpolation=interpolation
            )

        target_size = (target_width, target_height)
        cr_up = cv2.resize(cr_channel, target_size, interpolation=cv2.INTER_CUBIC)
        cb_up = cv2.resize(cb_channel, target_size, interpolation=cv2.INTER_CUBIC)

        merged = cv2.cvtColor(
            cv2.merge((enhanced_y, cr_up, cb_up)), cv2.COLOR_YCrCb2BGR
        )

        if is_full:
            return merged
        result = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        result[top * scale : bottom * scale, left * scale : right * scale] = merged
        return result

    # ------------------------------------------------------------------ #
    def stats_snapshot(self) -> dict:
        return {
            stage: {"calls": stat.calls, "avg_ms": round(stat.avg_ms, 2)}
            for stage, stat in self.stats.items()
        }

    def close(self) -> None:
        self.registry.close()
