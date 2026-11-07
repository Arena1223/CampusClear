"""内网认证测试。

威胁模型：校园内网并不等于可信网络（宿舍网段与办公网段常互通），
因此这里的每条断言都对应一个具体的内网攻击面。
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from campusclear.auth.security import (
    ROLE_LEVELS,
    create_access_token,
    decode_token,
    hash_password,
    is_locked,
    needs_rehash,
    verify_password,
)

SECRET = "unit-test-secret-key-at-least-32-bytes-long"
ALGO = "HS256"


# --------------------------------------------------------------------------- #
# 口令
# --------------------------------------------------------------------------- #
def test_password_hash_verifies():
    digest = hash_password("CorrectHorse1")
    assert digest.startswith("$argon2")
    assert verify_password("CorrectHorse1", digest) is True


def test_wrong_password_rejected():
    digest = hash_password("CorrectHorse1")
    assert verify_password("correcthorse1", digest) is False
    assert verify_password("", digest) is False


def test_short_password_rejected():
    with pytest.raises(ValueError):
        hash_password("short")


def test_same_password_produces_different_hashes():
    """必须加盐：否则同一密码在两个账号上哈希相同，内网撞库一目了然。"""
    assert hash_password("SamePassword1") != hash_password("SamePassword1")


def test_corrupted_hash_does_not_raise():
    """数据库里出现损坏记录时不能 500，只能判定为验证失败。"""
    assert verify_password("whatever", "not-a-valid-hash") is False
    assert verify_password("whatever", "") is False
    assert needs_rehash("not-a-valid-hash") is False


# --------------------------------------------------------------------------- #
# 令牌
# --------------------------------------------------------------------------- #
def test_token_roundtrip():
    payload = create_access_token("admin", "admin", SECRET, ALGO, 60)
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] == 3600
    decoded = decode_token(payload["access_token"], SECRET, ALGO)
    assert decoded["sub"] == "admin"
    assert decoded["role"] == "admin"
    assert decoded["iss"] == "campusclear"


def test_token_signed_with_other_secret_rejected():
    """密钥必须落盘持久化且不可猜测：换密钥后旧令牌立即失效。"""
    token = create_access_token("admin", "admin", SECRET, ALGO, 60)["access_token"]
    with pytest.raises(HTTPException) as excinfo:
        decode_token(token, "another-secret", ALGO)
    assert excinfo.value.status_code == 401


def test_expired_token_rejected():
    token = create_access_token("admin", "admin", SECRET, ALGO, -1)["access_token"]
    with pytest.raises(HTTPException) as excinfo:
        decode_token(token, SECRET, ALGO)
    assert excinfo.value.status_code == 401
    assert "过期" in excinfo.value.detail


def test_garbage_token_rejected():
    with pytest.raises(HTTPException) as excinfo:
        decode_token("not.a.token", SECRET, ALGO)
    assert excinfo.value.status_code == 401


# --------------------------------------------------------------------------- #
# 角色与锁定
# --------------------------------------------------------------------------- #
def test_role_hierarchy_is_strictly_ordered():
    assert ROLE_LEVELS["admin"] > ROLE_LEVELS["operator"] > ROLE_LEVELS["viewer"]


def test_is_locked_reports_remaining_seconds():
    locked, remaining = is_locked({"locked_until": time.time() + 120})
    assert locked is True
    assert 100 <= remaining <= 121

    locked, remaining = is_locked({"locked_until": time.time() - 5})
    assert locked is False
    assert remaining == 0

    locked, _ = is_locked({})
    assert locked is False
