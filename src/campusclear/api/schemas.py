"""API 请求/响应模型。

全部使用 Pydantic v2，接口层严禁直接暴露摄像头源地址的凭证信息。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    username: str


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=256)


class CameraBase(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    source_rtsp: str = Field(min_length=8, max_length=512)
    location: str = Field(default="", max_length=128)
    enabled: bool = True
    width: int | None = Field(default=None, ge=64, le=8192)
    height: int | None = Field(default=None, ge=64, le=8192)
    fps: int | None = Field(default=None, ge=1, le=60)
    scale: int | None = Field(default=None, ge=1, le=4)
    enable_lowlight: bool | None = None
    enable_denoise: bool | None = None
    enable_deblur: bool | None = None
    enable_sr: bool | None = None
    roi_enabled: bool = True
    notes: str = Field(default="", max_length=512)
    tags: list[str] = Field(default_factory=list)

    @field_validator("source_rtsp")
    @classmethod
    def check_scheme(cls, value: str) -> str:
        """只允许 RTSP/RTMP 等视频协议，防止有人把 HTTP 地址填进来当视频源。"""
        allowed = ("rtsp://", "rtsps://", "rtmp://")
        if not value.lower().startswith(allowed):
            raise ValueError(f"视频源必须以下列协议开头: {', '.join(allowed)}")
        return value


class CameraCreate(CameraBase):
    id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    source_rtsp: str | None = Field(default=None, min_length=8, max_length=512)
    location: str | None = None
    enabled: bool | None = None
    width: int | None = Field(default=None, ge=64, le=8192)
    height: int | None = Field(default=None, ge=64, le=8192)
    fps: int | None = Field(default=None, ge=1, le=60)
    scale: int | None = Field(default=None, ge=1, le=4)
    enable_lowlight: bool | None = None
    enable_denoise: bool | None = None
    enable_deblur: bool | None = None
    enable_sr: bool | None = None
    roi_enabled: bool | None = None
    notes: str | None = None
    tags: list[str] | None = None

    @field_validator("source_rtsp")
    @classmethod
    def check_scheme(cls, value: str | None) -> str | None:
        """PATCH 与 POST 必须遵守同一套取流协议白名单。"""
        if value is None:
            return None
        allowed = ("rtsp://", "rtsps://", "rtmp://")
        if not value.lower().startswith(allowed):
            raise ValueError(f"视频源必须以下列协议开头: {', '.join(allowed)}")
        return value


class CameraView(BaseModel):
    """对外的通道视图。``source_rtsp`` 只回主机名，不回凭证。"""

    id: str
    name: str
    location: str
    enabled: bool
    source_host: str
    output_name: str
    output_url: str
    scale: int | None
    roi_enabled: bool
    running: bool
    connected: bool
    notes: str
    tags: list[str]


class AckResponse(BaseModel):
    ok: bool = True
    message: str = ""
