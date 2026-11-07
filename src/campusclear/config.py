"""全局配置。

设计原则：
1. 所有可调参数集中在本文件，便于现场部署时按硬件条件裁剪；
2. 每个参数都可通过 ``CAMPUSCLEAR_`` 前缀的环境变量覆盖，无需改代码；
3. 默认值面向「纯内网 + 可物理断网 + 零公网依赖」场景，
   不包含任何需要联网回调、遥测上报、在线拉取模型的默认行为。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .compat import cpu_only, find_ffmpeg, resolve_providers

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 环境变量读取助手
# --------------------------------------------------------------------------- #
def _str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _int(key: str, default: int) -> int:
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _str(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _providers(key: str, default: list[str]) -> list[str]:
    raw = _str(key, ",".join(default))
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    # 收敛到本机真正可用的提供器：Windows 内网机常见「配置写着 CUDA/TensorRT，
    # 实际只装了 CPU 版 onnxruntime」，不收敛会在建会话时反复告警甚至报错。
    return resolve_providers(requested)


DEFAULT_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class Settings:
    """运行期配置快照。实例化时机决定读取到的环境变量值。"""

    # ---------------- 服务 ----------------
    app_name: str = field(default_factory=lambda: _str("CAMPUSCLEAR_APP_NAME", "CampusClear"))
    api_prefix: str = field(default_factory=lambda: _str("CAMPUSCLEAR_API_PREFIX", "/api/v1"))
    host: str = field(default_factory=lambda: _str("CAMPUSCLEAR_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("CAMPUSCLEAR_PORT", 8000))
    # 浏览器**额外查看端口**数量。MJPEG 是不关闭的长连接，而浏览器对同一
    # 「协议+主机+端口」只允许约 6 条并发连接——所有路挤一个端口时，在线路
    # 一超过 6 条，后面的画面就永远排队打不开。多开几个本地端口把观看流量
    # 分散开（每个端口 6 个名额）。0 = 关闭（退回单端口）。
    view_port_count: int = field(default_factory=lambda: _int("CAMPUSCLEAR_VIEW_PORTS", 4))
    # 实际启动成功的额外查看端口（app.py 启动后回填；list_streams 用它做轮转分配）
    active_view_ports: list[int] = field(default_factory=list)
    tls_cert: str = field(default_factory=lambda: _str("CAMPUSCLEAR_TLS_CERT", ""))
    tls_key: str = field(default_factory=lambda: _str("CAMPUSCLEAR_TLS_KEY", ""))
    # 管理面是否强制走 HTTPS（内网自签 CA）。生产环境建议开启。
    require_tls: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_REQUIRE_TLS", False))

    # ---------------- 安全 ----------------
    jwt_secret: str = field(default_factory=lambda: _str("CAMPUSCLEAR_JWT_SECRET", ""))
    jwt_algorithm: str = field(default_factory=lambda: _str("CAMPUSCLEAR_JWT_ALG", "HS256"))
    jwt_expire_minutes: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_JWT_EXPIRE_MINUTES", 480)
    )
    bootstrap_admin: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_BOOTSTRAP_ADMIN", "admin")
    )
    bootstrap_password: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_BOOTSTRAP_PASSWORD", "")
    )
    # 登录失败锁定：连续失败达到阈值即锁定该账号，抵御内网暴力破解
    login_max_failures: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_LOGIN_MAX_FAILURES", 5)
    )
    login_lock_seconds: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_LOGIN_LOCK_SECONDS", 300)
    )

    # ---------------- 目录 ----------------
    data_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_DATA_DIR", "./data")).resolve()
    )
    models_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_MODELS_DIR", "./models")).resolve()
    )
    log_dir: Path = field(
        default_factory=lambda: Path(_str("CAMPUSCLEAR_LOG_DIR", "./data/logs")).resolve()
    )
    snapshots_dir: Path = field(
        default_factory=lambda: Path(
            _str("CAMPUSCLEAR_SNAPSHOT_DIR", "./data/snapshots")
        ).resolve()
    )

    # ---------------- 推理 ----------------
    providers: list[str] = field(default_factory=lambda: _providers("CAMPUSCLEAR_PROVIDERS", DEFAULT_PROVIDERS))
    device_id: int = field(default_factory=lambda: _int("CAMPUSCLEAR_DEVICE_ID", 0))
    use_fp16: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_USE_FP16", True))
    tile_max_pixels: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_TILE_MAX_PIXELS", 1920 * 1080 // 2)
    )
    tile_overlap: int = field(default_factory=lambda: _int("CAMPUSCLEAR_TILE_OVERLAP", 16))
    # 首次加载后缓存 TensorRT engine 的目录（离线预构建，避免启动期重复编译）
    trt_engine_cache_dir: Path = field(
        default_factory=lambda: Path(
            _str("CAMPUSCLEAR_TRT_CACHE", "./data/trt_cache")
        ).resolve()
    )
    # 模型清单与权重文件必须校验 SHA256，防止内网传输/拷贝过程被篡改
    verify_model_hash: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_VERIFY_MODEL_HASH", True)
    )

    # ---------------- 流与调度 ----------------
    max_streams: int = field(default_factory=lambda: _int("CAMPUSCLEAR_MAX_STREAMS", 8))
    queue_size: int = field(default_factory=lambda: _int("CAMPUSCLEAR_QUEUE_SIZE", 8))
    infer_fps: int = field(default_factory=lambda: _int("CAMPUSCLEAR_INFER_FPS", 12))
    output_fps: int = field(default_factory=lambda: _int("CAMPUSCLEAR_OUTPUT_FPS", 25))
    rtsp_output_base: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_BASE", "rtsp://127.0.0.1:8554")
    )
    rtsp_output_user: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_USER", "campusclear")
    )
    rtsp_output_password: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_OUTPUT_PASSWORD", "")
    )
    ffmpeg_bin: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_FFMPEG", find_ffmpeg())
    )
    # 是否把增强结果回推成 RTSP 流。
    # 服务器形态下配内网 MediaMTX，让「既有安防平台零改造」；
    # 单机客户端形态下内网并没有 RTSP 网关，开着只会不停重建必然失败的推流进程。
    output_enabled: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_OUTPUT_ENABLED", True)
    )
    # 解码端强制缩放到 reader 的宽高，保证 rawvideo 管道帧长恒定 ——
    # 这是「探测不到流参数时画面撕裂成雪花」的根治手段。
    force_decode_size: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_FORCE_DECODE_SIZE", True)
    )
    # 解码分辨率上限（0 = 不限制）。增强耗时大致随像素数线性增长：
    # 纯 CPU 机器上 720p 全流程约 3 fps，4MP（2560x1440）会掉到 1 fps 以下。
    # 把上限设成 1280（宽）能让画面「动起来」，代价是增强后的分辨率也止步于此。
    max_decode_width: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_MAX_DECODE_WIDTH", 0)
    )
    max_decode_height: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_MAX_DECODE_HEIGHT", 0)
    )
    # RTSP 接收超时（微秒）。内网交换机瞬断时靠它触发重连。
    rtsp_timeout_us: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_RTSP_TIMEOUT_US", 5_000_000)
    )
    # RTSP 传输方式：tcp / udp。绝大多数内网摄像头走 tcp 更稳；
    # 某些老 NVR 或组播环境只支持 udp，现场可用环境变量切换。
    rtsp_transport: str = field(
        default_factory=lambda: _str("CAMPUSCLEAR_RTSP_TRANSPORT", "tcp").lower()
    )
    reconnect_backoff_seconds: list[int] = field(
        default_factory=lambda: [
            _int("CAMPUSCLEAR_RECONNECT_BASE_SECONDS", 1),
            _int("CAMPUSCLEAR_RECONNECT_MAX_SECONDS", 30),
        ]
    )
    # 多通道**错峰启动**间隔（秒）。同时接入多路时如果一起发起 probe+connect，
    # 交换机/NVR/流媒体服务器的并发会话会被瞬间打满，表现为「所有路都打不开」。
    # 每启动一路停一小会儿，让前一路完成 RTSP 握手再上下一路。
    start_stagger_seconds: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_START_STAGGER_SECONDS", 0.5)
    )

    # ---------------- 增强策略 ----------------
    enable_lowlight: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_LOWLIGHT", True))
    enable_denoise: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_DENOISE", True))
    enable_deblur: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_DEBLUR", True))
    enable_sr: bool = field(default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_SR", True))
    sr_scale: int = field(default_factory=lambda: _int("CAMPUSCLEAR_SR_SCALE", 2))
    enable_static_reuse: bool = field(
        default_factory=lambda: _bool("CAMPUSCLEAR_ENABLE_STATIC_REUSE", True)
    )

    # ---------------- 质量判据阈值 ----------------
    low_light_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_LOW_LIGHT_THRESHOLD", 60.0)
    )
    noise_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_NOISE_THRESHOLD", 6.0)
    )
    blur_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_BLUR_THRESHOLD", 90.0)
    )
    motion_threshold: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_MOTION_THRESHOLD", 0.004)
    )
    # 降采样后的噪声超过此值时，认为背景建模不可信，关闭静止帧复用。
    # 代价是算力（所有帧都真实处理），收益是不出现「画面在动却复用旧帧」的卡顿拖影。
    static_reuse_max_noise: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_STATIC_REUSE_MAX_NOISE", 3.0)
    )
    roi_padding: int = field(default_factory=lambda: _int("CAMPUSCLEAR_ROI_PADDING", 24))
    roi_max_boxes: int = field(default_factory=lambda: _int("CAMPUSCLEAR_ROI_MAX_BOXES", 6))

    # ---------------- 负载治理 ----------------
    gpu_high_watermark: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_GPU_HIGH_WATERMARK", 85.0)
    )
    gpu_low_watermark: float = field(
        default_factory=lambda: _float("CAMPUSCLEAR_GPU_LOW_WATERMARK", 60.0)
    )

    # ---------------- 健康检查 ----------------
    health_interval_seconds: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_HEALTH_INTERVAL", 20)
    )
    # 连续 N 次判定为「画面冻结」即触发重启
    # 默认 150 帧：25fps 下约 6 秒。原先的 3 帧（120ms）对静态场景的安防摄像头
    # 会频繁误触发，且 worker 线程直接调 reconnect 会导致与 reader 线程竞态。
    frozen_frame_restart_times: int = field(
        default_factory=lambda: _int("CAMPUSCLEAR_FROZEN_RESTART_TIMES", 150)
    )

    def __post_init__(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.log_dir,
            self.snapshots_dir,
            self.trt_engine_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if self.rtsp_transport not in ("tcp", "udp"):
            logger.warning(
                "CAMPUSCLEAR_RTSP_TRANSPORT=%r 无效，已回退 tcp（可选 tcp/udp）",
                self.rtsp_transport,
            )
            self.rtsp_transport = "tcp"

        # CPU 执行提供器不支持 fp16：与其让 onnxruntime 在建会话时反复报错，
        # 不如在这里直接收敛，把「配置矛盾」降级成一条日志。
        if self.use_fp16 and cpu_only(self.providers):
            logger.info("本机仅有 CPU 执行提供器，已自动关闭 fp16（CAMPUSCLEAR_USE_FP16 被覆盖）")
            self.use_fp16 = False

        # 纯 CPU 部署且未设解码上限时，自动把解码尺寸限制到 1280 宽。
        # 不限的话接入主码流（如海康 2560x1440 H.265）后，单帧增强要数分钟，
        # 预览会「冻住」、队列长期满载。1280 宽下全流程仍能跑动，且
        # 「截图并增强」走独立的按需抓帧，依旧拿到主码流原始分辨率。
        # 如需解除限制，把 CAMPUSCLEAR_MAX_DECODE_WIDTH 设成一个大值（如 7680）。
        if (
            cpu_only(self.providers)
            and self.max_decode_width == 0
            and self.max_decode_height == 0
        ):
            logger.info(
                "纯 CPU 部署且未设置解码上限，自动限制解码宽度为 1280"
                "（接入主码流时保护预览流畅；解除限制可设 CAMPUSCLEAR_MAX_DECODE_WIDTH=7680）"
            )
            self.max_decode_width = 1280

    # ---------------- 派生属性 ----------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "campusclear.db"

    @property
    def secret_path(self) -> Path:
        """JWT 密钥落盘位置。落盘后即使重启也保持同一密钥，避免已签发 token 全部失效。"""
        return self.data_dir / "secret.key"

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert and self.tls_key)

    def resolve_jwt_secret(self) -> str:
        """解析 JWT 密钥：优先环境变量，其次本地文件，最后随机生成并持久化。

        之所以落盘而不是每次随机，是为了让内网服务重启后已签发的令牌仍然有效；
        文件权限设为 0600，仅服务账户可读。
        """
        if self.jwt_secret:
            return self.jwt_secret
        if self.secret_path.exists():
            return self.secret_path.read_text(encoding="utf-8").strip()
        import secrets

        generated = secrets.token_urlsafe(48)
        self.secret_path.write_text(generated, encoding="utf-8")
        try:
            os.chmod(self.secret_path, 0o600)
        except OSError:
            # Windows 上 chmod 语义有限，忽略即可
            pass
        return generated

    def provider_profile(self) -> dict:
        """给 /system/health 用的脱敏摘要。"""
        return {
            "providers": self.providers,
            "device_id": self.device_id,
            "fp16": self.use_fp16,
            "max_streams": self.max_streams,
            "sr_scale": self.sr_scale,
            "tls_enabled": self.tls_enabled,
            "offline_mode": True,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """进程内单例。``refresh=True`` 供测试重置使用。"""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
