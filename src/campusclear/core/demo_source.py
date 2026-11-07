"""演示用合成视频源。

用途：**没有摄像头时，也让客户端能看到「实时增强」到底长什么样**，
并且走的是生产同一套链路（QualityAnalyzer → build_plan → 实时档增强器 → 编码）。

合成的是一个「昏暗、有噪声、有一个人走过」的走廊场景：
- 暗：整体亮度压低到约 35%，模拟夜间照度不足；
- 噪：叠加高斯噪声，模拟高增益下的彩色噪点；
- 动：一个矩形当作「人」缓慢横穿，用来演示运动检测 / ROI / 静止复用。

它不是真实摄像头，因此**只能用于演示与联调**，不能当作画质评测依据——
这一点在客户端页面上也会标注。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class DemoScene:
    """可随时刻取某一时刻画面的合成走廊。"""

    def __init__(self, width: int = 960, height: int = 540, *, bright: bool = False) -> None:
        self.width = width
        self.height = height
        self.bright = bright
        self._base = self._build_base()
        self._rng = np.random.default_rng(20261010)

    # ------------------------------------------------------------------ #
    def _build_base(self) -> np.ndarray:
        w, h = self.width, self.height
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        horizon = int(h * 0.42)
        # 天花板 / 地面（垂直渐变，制造纵深）
        ceil_grad = np.linspace(120, 95, horizon + 1, dtype=np.float32)[:, None]
        canvas[: horizon + 1, :, :] = ceil_grad[:, :, None]
        floor_grad = np.linspace(90, 55, h - horizon - 1, dtype=np.float32)[:, None]
        canvas[horizon + 1 :, :, :] = floor_grad[:, :, None]
        # 后墙 + 一扇门 + 两根柱子
        x0, x1 = int(w * 0.30), int(w * 0.70)
        canvas[:horizon, x0:x1, :] = 108
        canvas[int(h * 0.15) : int(h * 0.42), int(w * 0.44) : int(w * 0.56), :] = 78   # 门
        for cx in (int(w * 0.18), int(w * 0.82)):
            canvas[int(h * 0.08) : h, cx - 10 : cx + 10, :] = 128                      # 柱子
        # 地面砖缝（越远越密）
        for i in range(1, 9):
            y = horizon + int((h - horizon) * (i / 9) ** 1.6)
            canvas[y : y + 2, :, :] = 70
        return canvas

    # ------------------------------------------------------------------ #
    def frame(self, t: float) -> np.ndarray:
        """刻取第 t 秒的退化帧（BGR uint8）。"""
        img = self._base.copy()
        # 一个横穿画面的人（矩形 + 头）
        px = int(self.width * (0.15 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.6))))
        py = int(self.height * 0.66)
        cv2.rectangle(img, (px - 18, py - 60), (px + 18, py), (150, 150, 155), -1)
        cv2.circle(img, (px, py - 74), 15, (160, 160, 165), -1)

        if not self.bright:
            img = img * 0.35                      # 夜里照度不足
            img = img + self._rng.normal(0, 12, img.shape).astype(np.float32)  # 高增益噪声
        else:
            img = img + self._rng.normal(0, 3, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)
