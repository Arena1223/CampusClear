"""单路视频通道的配置模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CameraConfig:
    """一路摄像头的配置。

    ``scale`` 与各类阶段开关留空（None）时继承全局配置，
    这样现场可以「全局定基线、单路做微调」：
    例如校门口那路需要 4 倍超分，而楼道那路只需降噪即可。
    """

    id: str
    name: str
    source_rtsp: str
    location: str = ""
    enabled: bool = True
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    scale: int | None = None
    enable_lowlight: bool | None = None
    enable_denoise: bool | None = None
    enable_deblur: bool | None = None
    enable_sr: bool | None = None
    # 是否启用 ROI 局部增强（把算力集中在运动目标上）
    roi_enabled: bool = True
    # 输出流名，最终地址为 {rtsp_output_base}/{output_name}
    output_name: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_name:
            self.output_name = f"cc_{self.id}"
        if not self.id:
            raise ValueError("通道 id 不能为空")
        if not self.source_rtsp:
            raise ValueError(f"通道 {self.id} 缺少 source_rtsp")

    def to_row(self) -> dict:
        data = asdict(self)
        data["tags"] = ",".join(self.tags)
        return data

    def resolve(self, settings) -> "ResolvedCamera":
        """与全局配置合并，得到该路的实际生效参数。"""
        return ResolvedCamera(
            camera=self,
            scale=self.scale or settings.sr_scale,
            enable_lowlight=settings.enable_lowlight if self.enable_lowlight is None else self.enable_lowlight,
            enable_denoise=settings.enable_denoise if self.enable_denoise is None else self.enable_denoise,
            enable_deblur=settings.enable_deblur if self.enable_deblur is None else self.enable_deblur,
            enable_sr=settings.enable_sr if self.enable_sr is None else self.enable_sr,
            fps=self.fps or settings.output_fps,
        )


@dataclass
class ResolvedCamera:
    """全局配置与单路配置合并后的最终参数。"""

    camera: CameraConfig
    scale: int
    enable_lowlight: bool
    enable_denoise: bool
    enable_deblur: bool
    enable_sr: bool
    fps: int
