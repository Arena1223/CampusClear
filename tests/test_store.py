"""本地存储测试。"""

from __future__ import annotations

import pytest

from campusclear.core.camera import CameraConfig, ResolvedCamera


def make_camera(camera_id: str = "gate_01", **overrides) -> CameraConfig:
    data = dict(
        id=camera_id,
        name="校门口",
        source_rtsp="rtsp://10.20.30.51:554/Streaming/Channels/101",
        location="东门",
        tags=["出入口", "重点"],
    )
    data.update(overrides)
    return CameraConfig(**data)


# --------------------------------------------------------------------------- #
# 配置模型
# --------------------------------------------------------------------------- #
def test_camera_output_name_defaults_to_prefixed_id():
    camera = make_camera("lab_02")
    assert camera.output_name == "cc_lab_02"


def test_camera_rejects_empty_source():
    with pytest.raises(ValueError):
        CameraConfig(id="x", name="x", source_rtsp="")


def test_camera_rejects_empty_id():
    with pytest.raises(ValueError):
        CameraConfig(id="", name="x", source_rtsp="rtsp://a")


def test_camera_resolution_inherits_global_defaults(settings):
    camera = make_camera(scale=None, enable_sr=None)
    resolved = camera.resolve(settings)
    assert isinstance(resolved, ResolvedCamera)
    assert resolved.scale == settings.sr_scale
    assert resolved.enable_sr == settings.enable_sr


def test_camera_resolution_overrides_global_defaults(settings):
    camera = make_camera(scale=4, enable_sr=False, fps=15)
    resolved = camera.resolve(settings)
    assert resolved.scale == 4
    assert resolved.enable_sr is False
    assert resolved.fps == 15


# --------------------------------------------------------------------------- #
# 存储
# --------------------------------------------------------------------------- #
def test_store_roundtrip_preserves_tristate_fields(store):
    store.upsert_camera(make_camera(scale=4, enable_deblur=None, enable_sr=True))
    loaded = store.get_camera("gate_01")
    assert loaded is not None
    assert loaded.scale == 4
    assert loaded.enable_deblur is None, "None 必须原样保留，它表示「继承全局配置」"
    assert loaded.enable_sr is True
    assert loaded.tags == ["出入口", "重点"]
    assert loaded.roi_enabled is True


def test_store_upsert_updates_existing_row(store):
    store.upsert_camera(make_camera(name="旧名"))
    store.upsert_camera(make_camera(name="新名", location="西门"))
    assert store.count_cameras() == 1
    assert store.get_camera("gate_01").name == "新名"


def test_store_delete(store):
    store.upsert_camera(make_camera())
    assert store.delete_camera("gate_01") is True
    assert store.delete_camera("gate_01") is False
    assert store.get_camera("gate_01") is None


def test_store_list_is_sorted_by_id(store):
    for camera_id in ("c_03", "a_01", "b_02"):
        store.upsert_camera(make_camera(camera_id))
    assert [c.id for c in store.list_cameras()] == ["a_01", "b_02", "c_03"]


# --------------------------------------------------------------------------- #
# 用户与审计
# --------------------------------------------------------------------------- #
def test_store_user_lifecycle(store):
    store.create_user("operator1", "hash-not-used-here", role="operator")
    user = store.get_user("operator1")
    assert user["role"] == "operator"
    assert store.set_user_role("operator1", "admin") is True
    assert store.get_user("operator1")["role"] == "admin"
    assert store.set_user_role("nobody", "admin") is False


def test_store_login_failure_locks_account(store):
    store.create_user("victim", "hash", role="viewer")
    for _ in range(5):
        store.register_login_failure("victim", max_failures=5, lock_seconds=300)
    locked_until = store.get_user("victim")["locked_until"]
    assert locked_until > 0, "连续 5 次失败后应锁定账号"


def test_store_login_success_clears_failures(store):
    store.create_user("victim2", "hash", role="viewer")
    store.register_login_failure("victim2", max_failures=5, lock_seconds=300)
    store.register_login_success("victim2")
    user = store.get_user("victim2")
    assert user["failed_attempts"] == 0
    assert user["locked_until"] == 0
    assert user["last_login_at"]


def test_store_audit_and_export(tmp_path):
    """用独立数据库，避免与其他用例共用同一份审计表导致计数互相污染。"""
    from campusclear.core.store import CameraStore

    isolated = CameraStore(tmp_path / "audit-only.db")
    isolated.add_audit("login", username="admin", client_ip="10.20.1.5")
    isolated.add_audit("camera_create", username="admin", target="gate_01", detail="东门")

    logs = isolated.list_audit(limit=10)
    assert len(logs) == 2
    assert logs[0]["action"] == "camera_create", "审计日志应按时间倒序返回"
    assert isolated.list_audit(limit=10, username="admin")
    assert isolated.list_audit(limit=10, username="nobody") == []

    target = tmp_path / "audit.jsonl"
    count = isolated.export_audit_jsonl(target)
    assert count == 2
    assert target.read_text(encoding="utf-8").count("\n") == 2


def test_store_audit_detail_is_truncated(store):
    store.add_audit("x", username="admin", detail="a" * 5000)
    assert len(store.list_audit(limit=1)[0]["detail"]) <= 2000
