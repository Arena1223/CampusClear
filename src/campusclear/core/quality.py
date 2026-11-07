"""帧质量评估。

用确定性、无依赖的经典 CV 算法估计画面的四个关键退化维度：
亮度不足 / 噪声强度 / 模糊程度 / 运动占比。

为什么不用 AI 做这个判断？
- 判据需要可解释、可调、可复现，评审和运维要能看懂「为什么这一帧没走超分」；
- 经典算法在 1/8 降采样上只要不到 2ms，而 AI 判别器本身也会吃 GPU；
- 离线场景下算法越简单，出问题越容易定位。

运动占比同时被用作「静止帧复用」的开关，是整条链路最大的性能杠杆之一。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Immerkaer 快速噪声估计使用的 3x3 掩膜
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_NOISE_SCALE = float(np.sqrt(np.pi / 2.0) / 6.0)


@dataclass
class FrameQuality:
    """单帧质量评估结果。"""

    brightness: float          # 平均亮度 0-255
    contrast: float            # 灰度标准差
    noise_sigma: float         # 估计噪声标准差
    blur_score: float          # 拉普拉斯方差，越小越模糊
    motion_ratio: float        # 运动像素占比 0-1
    is_low_light: bool
    is_noisy: bool
    is_blurry: bool
    is_static: bool
    # 是否允许「静止帧复用」。默认 True 只是为了让本字段成为可选；
    # 真正决策请一律使用 build_plan()，不要直接看 is_static。
    reuse_allowed: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class QualityAnalyzer:
    """带背景建模的帧质量分析器。

    注意：内部持有 MOG2 背景模型，**每个视频通道必须独占一个实例**，
    不要把同一个分析器跨通道共用，否则背景模型会互相污染。

    **所有判据都在降采样图（``sample_width``，默认 320 px 宽）上计算**，
    因此阈值带分辨率口径，调参时必须一起看：

    · ``noise_threshold`` 是**降采样后**的噪声标准差。1080p 降到 320 宽是 3.375 倍
      降采样，INTER_AREA 平均会让噪声 σ 近似等比缩小。所以阈值 6.0 大致对应
      原始 1080p 上 σ≈20 的噪声，而不是 6。
    · ``blur_threshold`` 同理：降采样会削弱高频，全分辨率下的拉普拉斯方差
      远大于本处的读数（合成 640×360 场景读数为 300 左右，
      而同一画面在 1920×1080 上通常上千）。
    · ``motion_threshold`` 是占比，与分辨率无关，可以直接按比例理解。

    结论：**换 ``sample_width`` 或换输入分辨率，这三个阈值都要重新标定**。
    标定方法见 docs/04 与 docs/07。
    """

    def __init__(
        self,
        *,
        low_light_threshold: float = 60.0,
        noise_threshold: float = 6.0,
        blur_threshold: float = 90.0,
        motion_threshold: float = 0.004,
        reuse_max_noise: float = 3.0,
        sample_width: int = 320,
    ) -> None:
        self.low_light_threshold = low_light_threshold
        self.noise_threshold = noise_threshold
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        # 噪声超过此值即认为背景建模不可信，禁止静止帧复用。理由见 analyze()。
        self.reuse_max_noise = reuse_max_noise
        self.sample_width = sample_width
        # history 调小、varThreshold 调大：监控场景光照变化频繁，需要更快适应
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=24.0, detectShadows=False
        )
        self._warmup = 0

    # ------------------------------------------------------------------ #
    def analyze(self, frame: np.ndarray) -> FrameQuality:
        if frame is None or frame.size == 0:
            raise ValueError("空帧无法评估")

        small = self._downsample(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        noise_sigma = self._estimate_noise(gray)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion_ratio = self._motion_ratio(gray)

        is_low_light = brightness < self.low_light_threshold
        # 低光画面本身噪声估计会被亮度压低，这里在暗场时放宽阈值，避免漏判
        effective_noise_threshold = self.noise_threshold * (0.6 if is_low_light else 1.0)
        is_noisy = noise_sigma > effective_noise_threshold
        is_blurry = blur_score < self.blur_threshold
        # 前若干帧背景模型尚未收敛，强制视为「非静止」，避免开场就复用错误缓存
        is_static = self._warmup >= 20 and motion_ratio < self.motion_threshold

        # ---- 静止复用的安全闸门 ----
        # 高噪声环境下 MOG2 的结论不可信：背景模型给每个像素维护方差估计，
        # 当噪声波动与目标对比度同量级时（例如夜间低对比度场景里
        # 人形与地面只差 20 余灰度，而 varThreshold=24），
        # 运动目标会被统计上判为「正常波动」而漏检，于是画面明明在动却被判静止。
        #
        # 这个漏检的后果不是「省了算力」，而是「画面在动却复用旧帧」——
        # 人物走过时出现卡顿拖影，安防场景下属于严重缺陷（可能漏掉关键瞬间）。
        #
        # 因此这里宁可多算，不可复用错帧：噪声一旦超过 reuse_max_noise，
        # 直接关闭复用，让所有帧都真实处理。
        #
        # 默认 3.0 是偏保守的工程折中，依据：单像素噪声达 3.0 时，
        # 要让差异可靠超出 varThreshold=24（马氏距离约 4.9），信噪比仅 1.6σ，
        # 噪声自身触发阈值的概率已到 10% 量级，判别力不足。
        # 若某点位确实需要在高噪声下复用（例如追求通道数），
        # 应调大 varThreshold 并同步调大本值，且必须做画面滞后验证。
        reuse_allowed = is_static and noise_sigma <= self.reuse_max_noise

        return FrameQuality(
            brightness=brightness,
            contrast=contrast,
            noise_sigma=noise_sigma,
            blur_score=blur_score,
            motion_ratio=motion_ratio,
            is_low_light=is_low_light,
            is_noisy=is_noisy,
            is_blurry=is_blurry,
            is_static=is_static,
            reuse_allowed=reuse_allowed,
        )

    # ------------------------------------------------------------------ #
    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.sample_width:
            return frame
        ratio = self.sample_width / float(width)
        return cv2.resize(
            frame,
            (self.sample_width, max(1, int(round(height * ratio)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Immerkaer 快速噪声估计。

        原理：用高通掩膜卷积，静止区域内响应几乎只由噪声贡献，
        因此响应绝对均值乘上固定系数即可近似噪声标准差。计算量远低于小波方法。

        已实测的局限（重要，勿轻信本字段做告警）：
        **在极暗画面上该估计会显著偏低**。原因是暗场像素值高度集中在 0 附近，
        负向噪声被截断到 0，高通响应被"削平"，估计值随之下降。
        实测中合成夜景（真值 σ≈11）曾被估成 0.5。

        因此工程上不依赖本字段单独判断暗场是否需要降噪，
        而是用「低光」这一条独立条件去兜（见 planner 中
        ``quality.is_noisy or quality.is_low_light``）。
        若后续要本字段在暗场可靠，需要先做黑电平补偿再估计。
        """
        high_pass = cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
        return float(np.abs(high_pass).mean() * _NOISE_SCALE)

    def _motion_ratio(self, gray: np.ndarray) -> float:
        mask = self._bg.apply(gray, learningRate=-1.0)
        self._warmup += 1
        # 二值化 + 开运算：去掉传感器噪点引起的零星误检
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return float(np.count_nonzero(binary)) / float(binary.size)

    def motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """按原始分辨率返回运动掩膜，供 ROI 提取使用。"""
        small = self._downsample(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = self._bg.apply(gray, learningRate=0.0)  # 只查询，不更新背景模型
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        height, width = frame.shape[:2]
        return cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)

    def reset(self) -> None:
        """通道重启后调用，避免旧背景模型残留导致长时间误判。"""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=24.0, detectShadows=False
        )
        self._warmup = 0
