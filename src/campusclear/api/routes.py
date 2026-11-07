"""HTTP 接口层。

安全约定：
- 除 ``/healthz``（仅返回存活状态，供内网负载均衡/监控探活）与登录接口外，
  所有接口都需要 Bearer 令牌；
- 写操作按角色鉴权，并把操作写进审计日志；
- 任何响应都不得回传摄像头源地址中的用户名密码。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from ..auth.security import (
    create_access_token,
    get_current_user,
    hash_password,
    is_locked,
    needs_rehash,
    require_role,
    verify_password,
)
from ..core.camera import CameraConfig
from .schemas import (
    AckResponse,
    CameraCreate,
    CameraUpdate,
    CameraView,
    ChangePasswordRequest,
    LoginRequest,
    TokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# 健康检查（无需鉴权，只暴露存活状态）
# --------------------------------------------------------------------------- #
@router.get("/healthz", tags=["system"])
def healthz() -> dict:
    return {"status": "ok", "mode": "offline"}


# --------------------------------------------------------------------------- #
# 认证
# --------------------------------------------------------------------------- #
def _settings(request: Request):
    return request.app.state.settings


def _store(request: Request):
    return request.app.state.store


def _supervisor(request: Request):
    return request.app.state.supervisor


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(payload: LoginRequest, request: Request) -> TokenResponse:
    settings = _settings(request)
    store = _store(request)
    client_ip = request.client.host if request.client else ""
    user = store.get_user(payload.username)

    # 统一的失败提示，不透露「账号是否存在」，避免内网枚举账号
    def _fail(reason: str) -> HTTPException:
        store.add_audit("login_failed", username=payload.username, detail=reason, client_ip=client_ip)
        return HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    if user is None:
        # 仍然执行一次哈希校验，抹平「账号不存在」与「密码错误」的响应时间差
        verify_password(payload.password, hash_password("dummy-password-for-timing"))
        raise _fail("账号不存在")

    locked, remaining = is_locked(user)
    if locked:
        store.add_audit(
            "login_blocked", username=payload.username,
            detail=f"账号锁定中，剩余 {remaining}s", client_ip=client_ip,
        )
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"账号已锁定，请 {remaining} 秒后重试",
        )

    if user.get("disabled"):
        raise _fail("账号已停用")

    if not verify_password(payload.password, user["password_hash"]):
        attempts = store.register_login_failure(
            payload.username, settings.login_max_failures, settings.login_lock_seconds
        )
        raise _fail(f"第 {attempts} 次密码错误")

    # 参数升级后平滑迁移哈希
    if needs_rehash(user["password_hash"]):
        store.update_password(payload.username, hash_password(payload.password))

    store.register_login_success(payload.username)
    store.add_audit("login", username=payload.username, client_ip=client_ip)
    return TokenResponse(
        **create_access_token(
            payload.username,
            user["role"],
            settings.jwt_secret,
            settings.jwt_algorithm,
            settings.jwt_expire_minutes,
        )
    )


@router.get("/auth/me", tags=["auth"])
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"username": user["username"], "role": user["role"]}


@router.post("/auth/change-password", response_model=AckResponse, tags=["auth"])
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: dict = Depends(get_current_user),
) -> AckResponse:
    store = _store(request)
    record = store.get_user(user["username"])
    if record is None or not verify_password(payload.old_password, record["password_hash"]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "原密码不正确")
    try:
        store.update_password(user["username"], hash_password(payload.new_password))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    store.add_audit(
        "change_password", username=user["username"], client_ip=user["client_ip"]
    )
    return AckResponse(message="密码已更新，请重新登录")


# --------------------------------------------------------------------------- #
# 用户管理
# --------------------------------------------------------------------------- #
@router.get("/users", tags=["admin"])
def list_users(request: Request, user: dict = Depends(require_role("admin"))) -> list[dict]:
    return _store(request).list_users()


@router.post("/users/{username}/role", response_model=AckResponse, tags=["admin"])
def set_role(
    username: str,
    request: Request,
    role: str = Query(pattern="^(viewer|operator|admin)$"),
    user: dict = Depends(require_role("admin")),
) -> AckResponse:
    store = _store(request)
    if not store.set_user_role(username, role):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")
    store.add_audit(
        "set_role", username=user["username"], target=username, detail=role,
        client_ip=user["client_ip"],
    )
    return AckResponse(message=f"{username} 角色已设为 {role}")


# --------------------------------------------------------------------------- #
# 通道
# --------------------------------------------------------------------------- #
def _to_view(camera: CameraConfig, pipeline) -> CameraView:
    """把存储模型转成对外视图。源流只回主机，不回凭证。"""
    from ..core.pipeline import _host_of  # 局部导入，避免与 pipeline 形成循环引用

    output_url = ""
    if pipeline is not None:
        output_url = pipeline.snapshot()["output_url"]
    return CameraView(
        id=camera.id,
        name=camera.name,
        location=camera.location,
        enabled=camera.enabled,
        source_host=_host_of(camera.source_rtsp),
        output_name=camera.output_name,
        output_url=output_url,
        scale=camera.scale,
        roi_enabled=camera.roi_enabled,
        running=bool(pipeline and pipeline.state.running),
        connected=bool(pipeline and pipeline.state.connected),
        notes=camera.notes,
        tags=list(camera.tags),
    )


@router.get("/cameras", tags=["cameras"])
def list_cameras(request: Request, user: dict = Depends(get_current_user)) -> list[dict]:
    store = _store(request)
    supervisor = _supervisor(request)
    return [
        _to_view(camera, supervisor.get_pipeline(camera.id)).model_dump()
        for camera in store.list_cameras()
    ]


@router.post("/cameras", response_model=CameraView, status_code=status.HTTP_201_CREATED, tags=["cameras"])
def create_camera(
    payload: CameraCreate, request: Request, user: dict = Depends(require_role("admin"))
) -> CameraView:
    store = _store(request)
    supervisor = _supervisor(request)
    if store.get_camera(payload.id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"通道 {payload.id} 已存在")

    camera = CameraConfig(**payload.model_dump())
    store.upsert_camera(camera)
    store.add_audit(
        "camera_create", username=user["username"], target=camera.id,
        detail=camera.name, client_ip=user["client_ip"],
    )
    if camera.enabled:
        try:
            supervisor.start_camera(camera)
        except RuntimeError as exc:
            # 配置已经入库，启动失败只做记录：运维可以在面板上看到「已保存但未运行」，
            # 修正摄像头地址或网络后再点重启即可，不必重新填写表单。
            logger.warning("通道 %s 启动受阻: %s", camera.id, exc)
            store.add_audit(
                "camera_start_failed", username=user["username"], target=camera.id,
                detail=str(exc), client_ip=user["client_ip"],
            )
    return _to_view(camera, supervisor.get_pipeline(camera.id))


@router.patch("/cameras/{camera_id}", response_model=CameraView, tags=["cameras"])
def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    request: Request,
    user: dict = Depends(require_role("admin")),
) -> CameraView:
    store = _store(request)
    supervisor = _supervisor(request)
    camera = store.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通道不存在")

    changes = payload.model_dump(exclude_unset=True)
    for key, value in changes.items():
        setattr(camera, key, value)
    store.upsert_camera(camera)

    # 变更后重建该通道，确保编码参数与新配置一致
    if camera.enabled:
        try:
            supervisor.start_camera(camera)
        except RuntimeError as exc:
            logger.warning("通道 %s 重建受阻: %s", camera_id, exc)
    else:
        supervisor.stop_camera(camera_id)

    store.add_audit(
        "camera_update", username=user["username"], target=camera_id,
        detail=",".join(changes.keys()), client_ip=user["client_ip"],
    )
    return _to_view(camera, supervisor.get_pipeline(camera_id))


@router.delete("/cameras/{camera_id}", response_model=AckResponse, tags=["cameras"])
def delete_camera(
    camera_id: str, request: Request, user: dict = Depends(require_role("admin"))
) -> AckResponse:
    store = _store(request)
    supervisor = _supervisor(request)
    supervisor.stop_camera(camera_id)
    if not store.delete_camera(camera_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通道不存在")
    store.add_audit("camera_delete", username=user["username"], target=camera_id, client_ip=user["client_ip"])
    return AckResponse(message=f"通道 {camera_id} 已删除")


@router.post("/cameras/{camera_id}/restart", response_model=AckResponse, tags=["cameras"])
def restart_camera(
    camera_id: str, request: Request, user: dict = Depends(require_role("operator"))
) -> AckResponse:
    supervisor = _supervisor(request)
    if not supervisor.restart_camera(camera_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通道未在运行")
    _store(request).add_audit(
        "camera_restart", username=user["username"], target=camera_id, client_ip=user["client_ip"]
    )
    return AckResponse(message=f"通道 {camera_id} 已重启")


@router.get("/cameras/{camera_id}/stats", tags=["cameras"])
def camera_stats(
    camera_id: str, request: Request, user: dict = Depends(get_current_user)
) -> dict:
    pipeline = _supervisor(request).get_pipeline(camera_id)
    if pipeline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通道未在运行")
    return pipeline.snapshot()


@router.get("/cameras/{camera_id}/snapshot.jpg", tags=["cameras"])
def camera_snapshot(
    camera_id: str, request: Request, user: dict = Depends(require_role("operator"))
) -> Response:
    pipeline = _supervisor(request).get_pipeline(camera_id)
    if pipeline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "通道未在运行")
    jpeg = pipeline.snapshot_jpeg()
    if jpeg is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "尚无可用增强帧")
    _store(request).add_audit(
        "snapshot", username=user["username"], target=camera_id, client_ip=user["client_ip"]
    )
    return Response(content=jpeg, media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# 系统
# --------------------------------------------------------------------------- #
@router.get("/system/status", tags=["system"])
def system_status(request: Request, user: dict = Depends(get_current_user)) -> dict:
    settings = _settings(request)
    supervisor = _supervisor(request)
    data = supervisor.snapshot()
    data["settings"] = settings.provider_profile()
    return data


@router.get("/system/metrics", tags=["system"])
def system_metrics(request: Request, user: dict = Depends(get_current_user)) -> dict:
    return _supervisor(request).metrics_snapshot()


@router.get("/system/models", tags=["system"])
def system_models(request: Request, user: dict = Depends(get_current_user)) -> dict:
    registry = _supervisor(request).registry
    return {
        "models_dir": str(registry.models_dir),
        "verify_hash": registry.verify_hash,
        "models": registry.describe(),
    }


@router.post("/system/reload", response_model=AckResponse, tags=["system"])
def system_reload(request: Request, user: dict = Depends(require_role("admin"))) -> AckResponse:
    _supervisor(request).reload()
    _store(request).add_audit("system_reload", username=user["username"], client_ip=user["client_ip"])
    return AckResponse(message="通道配置已重新加载")


@router.get("/audit", tags=["admin"])
def list_audit(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    username: str | None = None,
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    return _store(request).list_audit(limit=limit, username=username)
