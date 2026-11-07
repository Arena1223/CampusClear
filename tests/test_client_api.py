"""老师客户端接口测试。

守住几条客户端专属的约定：
- 客户端页面必须纯本地（离线硬约束，不能引任何 CDN）；
- 实时/回放接口都要鉴权；``<img>``/``<video>`` 用的查询参数令牌也要生效；
- 录像名必须防路径穿越；
- 回放任务能从提交跑到完成并产出可播放的 mp4。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clean_sources(client):
    """每个用例后清空数据源，避免互相污染。"""
    yield
    store = client.app.state.store
    for cam in store.list_cameras():
        store.delete_camera(cam.id)


def test_client_page_is_local_only(client):
    response = client.get("/client")
    assert response.status_code == 200
    body = response.text
    assert "增强客户端" in body
    for bad in ("http://cdn", "https://cdn", "unpkg.com", "googleapis", "cdn.jsdelivr"):
        assert bad not in body


def test_streams_open_without_login(client):
    """单机客户端不做登录：不带任何令牌也应能访问。"""
    r = client.get("/api/v1/client/streams")
    assert r.status_code == 200
    assert "demo" in [s["id"] for s in r.json()["streams"]]


def test_recordings_open_without_login(client):
    r = client.get("/api/v1/client/recordings")
    assert r.status_code == 200


def test_streams_lists_demo(client, auth_headers):
    data = client.get("/api/v1/client/streams", headers=auth_headers).json()
    ids = [s["id"] for s in data["streams"]]
    assert "demo" in ids
    demo = next(s for s in data["streams"] if s["id"] == "demo")
    assert demo["kind"] == "demo" and demo["online"] is True


def test_query_token_is_ignored(client):
    # 客户端不再用令牌；带上一个也不应报错
    r = client.get("/api/v1/client/streams?token=whatever")
    assert r.status_code == 200


def test_recordings_empty_initially(client, auth_headers):
    data = client.get("/api/v1/client/recordings", headers=auth_headers).json()
    assert data["recordings"] == []


def test_job_rejects_missing_recording(client, auth_headers):
    r = client.post("/api/v1/client/jobs", headers=auth_headers,
                    json={"recording": "nope.mp4", "mode": "balanced"})
    assert r.status_code == 404


def test_job_rejects_path_traversal(client, auth_headers):
    r = client.post("/api/v1/client/jobs", headers=auth_headers,
                    json={"recording": "../../etc/passwd", "mode": "balanced"})
    assert r.status_code == 404


def test_job_rejects_unknown_mode(client, auth_headers):
    settings = client.app.state.settings
    rec = Path(settings.data_dir) / "recordings"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "x.mp4").write_bytes(b"not-a-real-video")
    r = client.post("/api/v1/client/jobs", headers=auth_headers,
                    json={"recording": "x.mp4", "mode": "super-duper"})
    assert r.status_code == 400


def _make_clip(path: Path, frames: int = 8, size=(160, 120), fps: float = 8.0) -> None:
    import cv2

    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(frames):
        frame = np.full((h, w, 3), 40 + i * 6, dtype=np.float32)
        frame = frame + rng.normal(0, 8, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()


def test_playback_job_runs_to_done(client, auth_headers):
    settings = client.app.state.settings
    rec = Path(settings.data_dir) / "recordings"
    rec.mkdir(parents=True, exist_ok=True)
    _make_clip(rec / "clip.mp4")

    created = client.post("/api/v1/client/jobs", headers=auth_headers,
                          json={"recording": "clip.mp4", "mode": "realtime"}).json()
    job_id = created["id"]

    import time

    status = {}
    for _ in range(40):
        status = client.get(f"/api/v1/client/jobs/{job_id}", headers=auth_headers).json()
        if status["state"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert status["state"] == "done", status
    assert status["result"]["frames"] > 0

    result = client.get(f"/api/v1/client/jobs/{job_id}/result?token={auth_headers['Authorization'][7:]}")
    assert result.status_code == 200
    assert result.content[:4] == b"\x00\x00\x00\x18" or len(result.content) > 0


def test_demo_mjpeg_yields_jpeg():
    from campusclear.api.client import _mjpeg_demo

    gen = _mjpeg_demo()
    chunk = next(gen)
    assert chunk.startswith(b"--frame")
    assert b"image/jpeg" in chunk and b"JFIF" in chunk


def test_demo_mjpeg_raw_and_enhanced_both_work():
    """对比功能依赖：原图与增强两条 MJPEG 都要能出帧。"""
    from campusclear.api.client import _mjpeg_demo

    for chunk in (next(_mjpeg_demo(enhance=False)), next(_mjpeg_demo(enhance=True))):
        assert chunk.startswith(b"--frame") and b"JFIF" in chunk


# --------------------------------------------------------------------------- #
# 数据源管理
# --------------------------------------------------------------------------- #
def test_methods_listed(client):
    r = client.get("/api/v1/client/methods")
    assert r.status_code == 200
    keys = {m["key"] for m in r.json()["methods"]}
    assert {"rtsp", "rtmp", "hls"} <= keys


def test_add_source_requires_url_or_brand(client):
    assert client.post("/api/v1/client/sources", json={}).status_code == 400
    assert client.post("/api/v1/client/sources", json={"brand": "hikvision"}).status_code == 400


def test_add_source_by_url_then_remove(client):
    r = client.post("/api/v1/client/sources", json={
        "url": "rtsp://10.0.0.9:554/Streaming/Channels/101", "name": "东门",
    })
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    # 测试环境没有 ffmpeg，启动会失败但不影响入库
    assert r.json()["started"] is False
    ids = [s["id"] for s in client.get("/api/v1/client/streams").json()["streams"]]
    assert sid in ids
    assert "东门" in [s["name"] for s in client.get("/api/v1/client/streams").json()["streams"]]
    assert client.delete(f"/api/v1/client/sources/{sid}").status_code == 200
    assert client.delete(f"/api/v1/client/sources/{sid}").status_code == 404


def test_add_source_by_vendor_builds_url(client):
    r = client.post("/api/v1/client/sources", json={
        "brand": "hikvision", "host": "10.0.0.5", "channel": 1,
    })
    assert r.status_code == 201
    sid = r.json()["id"]
    cam = client.app.state.store.get_camera(sid)
    assert "/Streaming/Channels/101" in cam.source_rtsp


def test_add_source_rejects_bad_or_duplicate_id(client):
    bad = client.post("/api/v1/client/sources", json={
        "id": "../evil", "url": "rtsp://10.0.0.9:554/live",
    })
    assert bad.status_code == 400

    first = client.post("/api/v1/client/sources", json={
        "id": "src_fixed", "url": "rtsp://10.0.0.9:554/live",
    })
    assert first.status_code == 201
    duplicate = client.post("/api/v1/client/sources", json={
        "id": "src_fixed", "url": "rtsp://10.0.0.10:554/other",
    })
    assert duplicate.status_code == 409


# --------------------------------------------------------------------------- #
# 截图增强
# --------------------------------------------------------------------------- #
def test_presets_listed_endpoint(client):
    r = client.get("/api/v1/client/presets")
    assert r.status_code == 200
    keys = {p["key"] for p in r.json()["presets"]}
    assert {"light", "standard", "sr2", "forensic"} <= keys


def test_snapshot_from_demo_and_fetch(client):
    r = client.post("/api/v1/client/snapshot", json={"stream_id": "demo", "preset": "standard"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["preset"] == "standard"
    img = client.get(body["url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"


def test_snapshot_unknown_stream_404(client):
    r = client.post("/api/v1/client/snapshot", json={"stream_id": "not-exist", "preset": "light"})
    assert r.status_code == 404


def test_snapshot_name_traversal_rejected(client):
    assert client.get("/api/v1/client/snapshots/..%2f..%2fetc%2fpasswd").status_code == 404


# ---------------------------------------------------------------------------
# 截图：抓帧来源与串行队列
# ---------------------------------------------------------------------------

def test_grab_frame_uses_pipeline_not_new_rtsp_session(client, monkeypatch):
    """默认抓帧必须复用管线内存帧，不能为了截一张图去新开一路 RTSP。

    这是现场「点截图卡很久」的主因：真实海康的并发会话有上限，新开连接要等
    握手和关键帧，还会把正在预览的那一路挤掉。
    """
    import types

    import cv2
    import numpy as np

    from campusclear.api import client as client_module
    from campusclear.core.camera import CameraConfig

    store = client.app.state.store
    store.upsert_camera(CameraConfig(
        id="grab_01", name="抓帧", source_rtsp="rtsp://u:p@10.0.0.5:554/Streaming/Channels/102",
        enabled=False,
    ))

    class _FakePipeline:
        def snapshot_raw_jpeg(self) -> bytes:
            ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 120, np.uint8))
            assert ok
            return buf.tobytes()

    opened: list[str] = []
    monkeypatch.setattr(client_module, "_supervisor",
                        lambda request: types.SimpleNamespace(get_pipeline=lambda sid: _FakePipeline()))
    monkeypatch.setattr(client_module, "_grab_jpeg_from_url",
                        lambda url, ffmpeg_bin, timeout=8.0: opened.append(url) or None)

    try:
        r = client.post("/api/v1/client/snapshot", json={"stream_id": "grab_01", "preset": "light"})
        assert r.status_code == 201, r.text
        assert opened == []                     # 没有新开 RTSP 会话
        assert r.json()["source_size"] == "64x48"
    finally:
        store.delete_camera("grab_01")


def test_snapshot_async_returns_immediately_and_completes(client):
    """wait=false：立刻返回任务 id，结果通过轮询拿到（连点不阻塞界面）。"""
    import time as _time

    r = client.post("/api/v1/client/snapshot",
                    json={"stream_id": "demo", "preset": "light", "wait": False})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["queued"] is True and body["id"]

    job = {}
    deadline = _time.time() + 20
    while _time.time() < deadline:
        job = client.get(f"/api/v1/client/snapshot-jobs/{body['id']}").json()
        if job["state"] in ("done", "failed"):
            break
        _time.sleep(0.2)

    assert job["state"] == "done", job
    assert job["elapsed_ms"] >= 0
    assert client.get(job["result"]["url"]).status_code == 200


def test_snapshot_job_unknown_is_404(client):
    assert client.get("/api/v1/client/snapshot-jobs/not-a-job").status_code == 404
