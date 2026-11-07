"""FastAPI 应用入口。

启动顺序刻意设计为「先可用，再优雅」：
1. 建目录、初始化 SQLite；
2. 校验模型完整性（权重被改动时必须显式失败，不能带病上线）；
3. 创建初始管理员（无用户时）；
4. 拉起通道管线与监控线程；
5. 才开始接收请求。

这样即便某个通道拉流失败，管理面仍然可用 —— 运维人员能看到失败原因，
而不是对着一个起不来的服务猜。
"""

from __future__ import annotations

import logging
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .api.client import router as client_router
from .api.routes import router
from .auth.security import hash_password
from .config import get_settings
from .core.gateway import ensure_local_gateway
from .core.scheduler import StreamSupervisor
from .core.store import CameraStore
from .models.registry import ModelIntegrityError, ModelRegistry

logger = logging.getLogger("campusclear")

STATIC_DIR = Path(__file__).parent / "static"


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """同时输出到控制台与本地文件。

    离线部署下没有集中日志服务，本地滚动文件是唯一的事故追溯依据，
    因此默认就开启文件日志，不需要运维额外配置。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_dir / "campusclear.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _seed_admin(settings, store: CameraStore) -> None:
    """首次启动创建管理员账号。"""
    if store.list_users():
        return
    password = settings.bootstrap_password
    generated = False
    if not password:
        # 不设默认弱口令。随机生成并打印，强制首次登录后修改。
        password = secrets.token_urlsafe(12)
        generated = True
    store.create_user(settings.bootstrap_admin, hash_password(password), role="admin")
    if generated:
        logger.warning("=" * 68)
        logger.warning("已创建初始管理员账号：%s", settings.bootstrap_admin)
        logger.warning("初始密码：%s", password)
        logger.warning("请在首次登录后立即修改密码（管理面板 → 修改密码）。")
        logger.warning("=" * 68)
    else:
        logger.info("已按环境变量创建管理员账号：%s", settings.bootstrap_admin)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_dir)
    settings.jwt_secret = settings.resolve_jwt_secret()

    logger.info("CampusClear %s 启动中（离线模式，无任何公网依赖）", __version__)
    logger.info("数据目录: %s", settings.data_dir)
    logger.info("模型目录: %s", settings.models_dir)

    store = CameraStore(settings.db_path)

    registry = ModelRegistry(settings.models_dir, verify_hash=settings.verify_model_hash)
    report = registry.verify()
    for stage, state in report.items():
        marker = "✓" if state.startswith("ok") else "✗"
        logger.info("模型校验 %s %-9s %s", marker, stage, state)
    if settings.verify_model_hash:
        try:
            registry.verify_or_raise()
        except ModelIntegrityError as exc:
            # 权重被替换属于安全事件，宁可拒绝启动也不能带着可疑权重跑
            logger.error("模型完整性校验未通过：%s", exc)
            raise

    _seed_admin(settings, store)

    # 回推需要一个 RTSP 网关来接收。单机形态下没有网关时这里会明确说明并跳过，
    # 而不是让每条通道各自去撞墙（以前表现为「日志一直报回推失败、播放器打不开」）。
    gateway = ensure_local_gateway(settings)
    if gateway.reason:
        logger.info("回推网关: %s", gateway.reason)

    supervisor = StreamSupervisor(settings, store, registry)
    supervisor.start()

    app.state.settings = settings
    app.state.store = store
    app.state.supervisor = supervisor
    logger.info("服务就绪，监听 %s:%d，通道上限 %d", settings.host, settings.port, settings.max_streams)

    try:
        yield
    finally:
        supervisor.shutdown()
        gateway.stop()
        logger.info("CampusClear 已停止")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} 明眸 · 校园监控离线实时清晰化系统",
        version=__version__,
        description=(
            "100% 内网离线运行的监控视频实时增强服务：AI 超分 / 降噪 / 去模糊 / 低光增强。\n\n"
            "全部推理在本地 GPU 完成，不接入公网，画面与数据不出校园内网。"
        ),
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.include_router(router, prefix=settings.api_prefix)
    app.include_router(client_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """运维面板。纯本地静态页，不引入任何 CDN 资源。"""
        return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")

    @app.get("/client", include_in_schema=False)
    def client_page() -> FileResponse:
        """老师专用客户端。实时看 + 回放查，全部流由服务器提供。

        加禁缓存头：客户端程序更新后，浏览器必须拿到新页面，
        否则会一直显示缓存的旧界面（曾真实踩过这个坑）。
        """
        return FileResponse(
            STATIC_DIR / "client.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> JSONResponse:
        return JSONResponse({}, status_code=204)

    return app


app = create_app()


def run() -> None:
    """命令行入口。"""
    settings = get_settings()
    kwargs: dict = {
        "host": settings.host,
        "port": settings.port,
        "log_level": "info",
        "access_log": False,  # 访问日志由应用自身记录，避免重复
    }
    if settings.tls_enabled:
        kwargs["ssl_certfile"] = settings.tls_cert
        kwargs["ssl_keyfile"] = settings.tls_key
        logger.info("已启用内网自签证书 HTTPS")
    else:
        logger.warning("未配置 TLS 证书，管理面以 HTTP 运行，建议仅绑定内网管理网段")
    uvicorn.run("campusclear.main:app", **kwargs)


if __name__ == "__main__":
    run()
