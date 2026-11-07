"""内网安全认证。

设计要点（针对校园内网这一特定威胁模型）：

1. **密码**：Argon2id 哈希，参数按下限内最高强度设置。内网不等于可信网，
   学生宿舍网段与办公网段常能互通，弱口令一定会被扫。
2. **令牌**：JWT（HS256），密钥落盘持久化，服务重启后旧令牌依然有效，
   避免值班人员被频繁登出。有效期默认 8 小时，覆盖一个完整值班班次。
3. **角色**：admin（可改配置/管理用户）、operator（可操作通道/取快照）、
   viewer（只读看指标）。最小权限原则，实习值班同学拿 viewer 就够。
4. **暴力破解防护**：连续失败达阈值锁定账号一段时间，同时写审计日志。
5. **传输**：管理面支持内网自签证书 HTTPS；输出 RTSP 流单独鉴权。

注意：本模块不依赖任何在线身份服务，不联 LDAP/AD（部分学校有此需求，
但会引入跨网段依赖，与「可断网运行」冲突，故列为可选扩展）。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Argon2id：内存 64MB、迭代 3 次、并行 4 路。在校园服务器上单次校验约 40-80ms，
# 对登录接口完全够用，对离线暴力破解则是实质门槛。
_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

ROLE_LEVELS = {"viewer": 1, "operator": 2, "admin": 3}

bearer_scheme = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# 密码
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("密码长度至少 8 位")
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:  # noqa: BLE001 - 损坏的哈希记录不应导致 500
        return False


def needs_rehash(password_hash: str) -> bool:
    """参数升级后自动重算哈希，保证老账号平滑过渡。"""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# 令牌
# --------------------------------------------------------------------------- #
def create_access_token(username: str, role: str, secret: str, algorithm: str, expire_minutes: int) -> dict:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expire_minutes)
    payload = {
        "sub": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "iss": "campusclear",
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expire_minutes * 60,
        "role": role,
        "username": username,
    }


def decode_token(token: str, secret: str, algorithm: str) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=[algorithm], issuer="campusclear")
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌无效") from exc


# --------------------------------------------------------------------------- #
# FastAPI 依赖
# --------------------------------------------------------------------------- #
def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "缺少访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = request.app.state.settings
    payload = decode_token(credentials.credentials, settings.jwt_secret, settings.jwt_algorithm)

    store = request.app.state.store
    user = store.get_user(payload.get("sub", ""))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号不存在")
    if user.get("disabled"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "账号已被停用")
    return {
        "username": user["username"],
        "role": user["role"],
        "client_ip": _client_ip(request),
    }


def require_role(minimum: str):
    """生成一个「至少具备 minimum 角色」的依赖。"""

    def dependency(user: dict = Depends(get_current_user)) -> dict:
        if ROLE_LEVELS.get(user["role"], 0) < ROLE_LEVELS.get(minimum, 99):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"需要 {minimum} 及以上权限",
            )
        return user

    return dependency


def _client_ip(request: Request) -> str:
    # 内网部署一般在反向代理之后，优先取 X-Forwarded-For 的首个地址；
    # 未经过代理时直接使用连接地址。审计日志需要真实来源以便追责。
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def is_locked(user: dict) -> tuple[bool, int]:
    """账号是否处于锁定状态，返回 (是否锁定, 剩余秒数)。"""
    locked_until = float(user.get("locked_until") or 0)
    remaining = int(locked_until - time.time())
    return remaining > 0, max(0, remaining)
