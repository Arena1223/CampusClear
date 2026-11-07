"""HTTP 接口测试。

覆盖三件在校园内网部署里最容易出问题的事：
1. 未授权访问必须被拦住（内网不等于免鉴权）；
2. 权限分级必须真的生效（viewer 不能改配置）；
3. 任何响应都不能把摄像头源地址的凭证回吐出去。
"""

from __future__ import annotations

import json

import pytest

from campusclear.auth.security import hash_password

VALID_CAMERA = {
    "id": "gate_01",
    "name": "东门主通道",
    "source_rtsp": "rtsp://camera-user:camera-pass@10.20.30.51:554/Streaming/Channels/101",
    "location": "东门",
    "enabled": False,          # 测试中不真的启动拉流
    "tags": ["出入口"],
}


@pytest.fixture(autouse=True)
def _clean_cameras(client):
    """每个用例开始前清空通道表。

    整个测试会话共用同一个 SQLite 文件，若某个用例中途失败留下残留记录，
    后续用例的「新增」就会撞上 409 冲突，制造出与被测代码无关的假失败。
    """
    store = client.app.state.store
    for camera in store.list_cameras():
        store.delete_camera(camera.id)
    yield


def login(client, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def make_viewer(client, username: str = "student_viewer") -> str:
    """直接在存储层造一个只读账号，避免依赖用户创建接口。"""
    store = client.app.state.store
    if store.get_user(username) is None:
        store.create_user(username, hash_password("ViewerPass1"), role="viewer")
    return login(client, username, "ViewerPass1")


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def test_healthz_is_public_and_reports_offline(client):
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    assert response.json()["mode"] == "offline"


def test_dashboard_is_served_locally(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "CampusClear" in body
    # 离线部署硬约束：页面不得引用任何外部资源
    assert "http://cdn" not in body
    assert "https://cdn" not in body
    assert "unpkg.com" not in body
    assert "googleapis" not in body


# --------------------------------------------------------------------------- #
# 认证
# --------------------------------------------------------------------------- #
def test_login_success_returns_token(client):
    token = login(client, "admin", "TestPassw0rd!")
    assert token


def test_login_with_wrong_password_is_rejected(client):
    response = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong-pass"})
    assert response.status_code == 401


def test_login_with_unknown_user_gives_same_message(client):
    """不透露账号是否存在，避免内网枚举。"""
    a = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong-pass"})
    b = client.post("/api/v1/auth/login", json={"username": "ghost", "password": "wrong-pass"})
    assert a.json()["detail"] == b.json()["detail"]


def test_protected_endpoint_requires_token(client):
    assert client.get("/api/v1/cameras").status_code == 401
    assert client.get("/api/v1/system/status").status_code == 401


def test_me_returns_role(client, auth_headers):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"username": "admin", "role": "admin"}


def test_change_password_then_relogin(client, auth_headers):
    response = client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={"old_password": "TestPassw0rd!", "new_password": "NewPassw0rd!"},
    )
    assert response.status_code == 200
    assert login(client, "admin", "NewPassw0rd!")

    # 改回去，避免影响同会话内的其他用例
    token = login(client, "admin", "NewPassw0rd!")
    client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"old_password": "NewPassw0rd!", "new_password": "TestPassw0rd!"},
    )


def test_change_password_with_wrong_old_password(client, auth_headers):
    response = client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={"old_password": "nope", "new_password": "NewPassw0rd!"},
    )
    assert response.status_code == 400


def test_weak_new_password_rejected(client, auth_headers):
    """长度不足的口令由请求模型直接拦截（422），不会走到哈希环节。"""
    response = client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={"old_password": "TestPassw0rd!", "new_password": "short"},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 通道管理
# --------------------------------------------------------------------------- #
def test_create_list_and_delete_camera(client, auth_headers):
    created = client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["id"] == "gate_01"
    assert body["output_name"] == "cc_gate_01"
    assert body["running"] is False

    listed = client.get("/api/v1/cameras", headers=auth_headers)
    assert listed.status_code == 200
    assert any(item["id"] == "gate_01" for item in listed.json())

    deleted = client.delete("/api/v1/cameras/gate_01", headers=auth_headers)
    assert deleted.status_code == 200
    assert client.delete("/api/v1/cameras/gate_01", headers=auth_headers).status_code == 404


def test_camera_response_never_leaks_credentials(client, auth_headers):
    """源流里的用户名密码绝不能出现在任何接口响应中。"""
    client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    listed = client.get("/api/v1/cameras", headers=auth_headers).json()
    target = next(item for item in listed if item["id"] == "gate_01")
    serialised = str(target)
    assert "camera-pass" not in serialised
    assert "camera-user" not in serialised
    assert target["source_host"] == "10.20.30.51:554"
    client.delete("/api/v1/cameras/gate_01", headers=auth_headers)


def test_duplicate_camera_id_conflicts(client, auth_headers):
    client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    again = client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    assert again.status_code == 409
    client.delete("/api/v1/cameras/gate_01", headers=auth_headers)


def test_camera_source_must_be_video_scheme(client, auth_headers):
    bad = dict(VALID_CAMERA, id="bad_01", source_rtsp="http://10.20.30.51/live.m3u8")
    response = client.post("/api/v1/cameras", headers=auth_headers, json=bad)
    assert response.status_code == 422


def test_camera_id_pattern_is_enforced(client, auth_headers):
    bad = dict(VALID_CAMERA, id="东门 01")
    assert client.post("/api/v1/cameras", headers=auth_headers, json=bad).status_code == 422


def test_update_camera(client, auth_headers):
    client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    response = client.patch(
        "/api/v1/cameras/gate_01",
        headers=auth_headers,
        json={"name": "东门（改造后）", "scale": 4},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "东门（改造后）"
    assert response.json()["scale"] == 4
    client.delete("/api/v1/cameras/gate_01", headers=auth_headers)


def test_update_missing_camera_is_404(client, auth_headers):
    response = client.patch("/api/v1/cameras/nope", headers=auth_headers, json={"name": "x"})
    assert response.status_code == 404


def test_update_camera_also_validates_source_scheme(client, auth_headers):
    client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    response = client.patch(
        "/api/v1/cameras/gate_01",
        headers=auth_headers,
        json={"source_rtsp": "http://10.0.0.1/live.m3u8"},
    )
    assert response.status_code == 422
    client.delete("/api/v1/cameras/gate_01", headers=auth_headers)


def test_stats_for_non_running_camera_is_404(client, auth_headers):
    client.post("/api/v1/cameras", headers=auth_headers, json=VALID_CAMERA)
    response = client.get("/api/v1/cameras/gate_01/stats", headers=auth_headers)
    assert response.status_code == 404
    client.delete("/api/v1/cameras/gate_01", headers=auth_headers)


# --------------------------------------------------------------------------- #
# 权限分级
# --------------------------------------------------------------------------- #
def test_viewer_cannot_create_camera(client):
    token = make_viewer(client)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/cameras", headers=headers, json=VALID_CAMERA)
    assert response.status_code == 403
    assert client.get("/api/v1/cameras", headers=headers).status_code == 200


def test_viewer_cannot_read_audit_log(client):
    token = make_viewer(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/audit", headers=headers).status_code == 403


def test_viewer_cannot_reload_system(client):
    token = make_viewer(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/system/reload", headers=headers).status_code == 403


def test_viewer_cannot_take_snapshot(client):
    """快照涉及画面内容，属于 operator 及以上权限。"""
    token = make_viewer(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/cameras/gate_01/snapshot.jpg", headers=headers).status_code == 403


# --------------------------------------------------------------------------- #
# 系统信息
# --------------------------------------------------------------------------- #
def test_system_status_shape(client, auth_headers):
    response = client.get("/api/v1/system/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["settings"]["offline_mode"] is True
    assert "load" in data
    assert set(data["load"]) == {"level", "gpu_util_smoothed", "gpu_util_latest", "reason"}


def test_system_models_reports_integrity(client, auth_headers):
    response = client.get("/api/v1/system/models", headers=auth_headers)
    assert response.status_code == 200
    stages = [item["stage"] for item in response.json()["models"]]
    assert stages == ["lowlight", "denoise", "deblur", "sr"]


def test_audit_records_login_and_admin_actions(client, auth_headers):
    logs = client.get("/api/v1/audit?limit=50", headers=auth_headers).json()
    actions = {row["action"] for row in logs}
    assert "login" in actions


def test_streams_round_robin_view_ports(client, monkeypatch):
    """多端口观看：view_port 必须在可用端口间轮转，避免全部挤主端口。"""
    from campusclear.core.camera import CameraConfig

    settings = client.app.state.settings
    monkeypatch.setattr(settings, "active_view_ports", [8001, 8002, 8003])
    store = client.app.state.store
    for i in range(4):
        store.upsert_camera(CameraConfig(
            id=f"vp_{i}", name=f"vp{i}",
            source_rtsp=f"rtsp://u:p@10.0.0.1:554/ch{i}", enabled=False,
        ))
    try:
        data = client.get("/api/v1/client/streams").json()
        by_id = {s["id"]: s["view_port"] for s in data["streams"]}
        # 4 路摄像头 + demo，对 3 个端口轮转：8001,8002,8003,8001,(8002)
        assert [by_id[f"vp_{i}"] for i in range(4)] == [8001, 8002, 8003, 8001]
        assert all(v is not None for v in by_id.values())
    finally:
        for i in range(4):
            store.delete_camera(f"vp_{i}")


def test_streams_view_port_null_when_single_port(client, monkeypatch):
    """未启用多端口（active_view_ports 为空）时 view_port 为 null，前端退回主端口。"""
    settings = client.app.state.settings
    monkeypatch.setattr(settings, "active_view_ports", [])
    data = client.get("/api/v1/client/streams").json()
    assert all(s["view_port"] is None for s in data["streams"])


# ---------------------------------------------------------------------------
# 批量导入（/client/sources/batch）：逐行保留各自端口、错峰启动
# ---------------------------------------------------------------------------

def test_batch_import_preserves_per_line_ports(client, monkeypatch):
    """多端口批量导入：每行的端口必须原样保留，不能被压成同一个端口。"""
    supervisor = client.app.state.supervisor
    monkeypatch.setattr(supervisor, "start_camera", lambda cam: None)
    monkeypatch.setattr("campusclear.api.client.time.sleep", lambda s: None)
    urls = "\n".join([
        "rtsp://u:p@10.0.0.5:8001/Streaming/Channels/101 # 教学楼-101",
        "rtsp://u:p@10.0.0.5:8002/Streaming/Channels/101 # 教学楼-102",
        "rtsp://u:p@10.0.0.6:554/cam/realmonitor?channel=1&subtype=0 # 宿舍-大门",
    ])
    store = client.app.state.store
    try:
        r = client.post("/api/v1/client/sources/batch", json={"urls": urls})
        assert r.status_code == 201
        data = r.json()
        assert data["added"] == 3 and data["started"] == 3 and data["failed"] == 0
        ports = set()
        for cam in store.list_cameras():
            hostport = cam.source_rtsp.split("@", 1)[1].split("/", 1)[0]
            ports.add(hostport.rsplit(":", 1)[1])
        assert ports == {"8001", "8002", "554"}
        # 响应里不得出现明文凭证
        assert "u:p@" not in json.dumps(data, ensure_ascii=False)
        assert any(c.name == "教学楼-101" for c in store.list_cameras())
    finally:
        for c in list(store.list_cameras()):
            store.delete_camera(c.id)


def test_batch_import_empty_and_oversize_rejected(client):
    """空输入与超上限（>1000 行）都必须 400，不得静默入库。"""
    assert client.post(
        "/api/v1/client/sources/batch", json={"urls": "  \n # x\n"}
    ).status_code == 400
    r = client.post("/api/v1/client/sources/batch", json={"urls": "rtsp://x\n" * 1001})
    assert r.status_code == 400


def test_batch_import_no_start_leaves_pipelines_down(client):
    """start=False 只入库不启动：不应有任何真实拉流被拉起。"""
    urls = "rtsp://u:p@10.0.0.9:9001/live/ch # 离线一路"
    store = client.app.state.store
    try:
        r = client.post("/api/v1/client/sources/batch", json={"urls": urls, "start": False})
        assert r.status_code == 201
        data = r.json()
        assert data["added"] == 1 and data["started"] == 0
        assert data["items"][0]["started"] is None
        streams = client.get("/api/v1/client/streams").json()["streams"]
        assert all(s["online"] is False for s in streams if s["id"] != "demo")
    finally:
        for c in list(store.list_cameras()):
            store.delete_camera(c.id)
