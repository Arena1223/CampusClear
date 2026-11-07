"""多端口观看的启动约束。

背景：MJPEG 是长连接，浏览器对同一「协议+主机+端口」只允许约 6 条并发，
所以客户端会额外监听几个本地端口把观看流量摊开。但这些端口只是**同一个
应用的观看入口**，绝不能各自再跑一遍 FastAPI lifespan —— 那会重新建
CameraStore / StreamSupervisor 并重新拉起所有摄像头，导致通道被重复启动。
"""

from __future__ import annotations

import uvicorn

import campusclear.app as app_module
from campusclear.config import get_settings


def test_view_servers_disable_lifespan(monkeypatch):
    captured: list[dict] = []

    class FakeConfig:
        def __init__(self, app, **kwargs):
            captured.append(kwargs)

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            return None

    settings = get_settings()
    monkeypatch.setattr(settings, "view_port_count", 2)
    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(app_module, "_port_free", lambda host, port: True)

    original_ports = list(settings.active_view_ports)
    try:
        started = app_module._start_view_servers(object(), "127.0.0.1", 8000)
    finally:
        settings.active_view_ports = original_ports

    assert started == [8001, 8002]
    assert len(captured) == 2
    assert all(config.get("lifespan") == "off" for config in captured), (
        "查看端口必须关闭 lifespan，否则每个端口都会重复启动一套通道管线"
    )
