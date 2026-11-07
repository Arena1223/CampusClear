"""回推网关的探测与「没有网关时优雅跳过」。

这两条约束来自一次真实的现场困惑：用户按默认地址 ``rtsp://127.0.0.1:8554``
去播放器里打开回推流，结果什么也打不开——因为项目本身不带 RTSP 服务端。
代码这边能做的是：**在批量建流前先把「有没有网关」问清楚**，没有就明确跳过，
而不是让每条通道每 60 秒拉起一个注定失败的 ffmpeg，把日志刷满、把 CPU 占住。
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from campusclear.core import gateway
from campusclear.core.camera import CameraConfig
from campusclear.core.enhancer import ClassicalEnhancer
from campusclear.core.gateway import (
    ensure_local_gateway,
    find_gateway_binary,
    is_local_endpoint,
    rtsp_endpoint_reachable,
)
from campusclear.core.pipeline import StreamPipeline
from campusclear.core.planner import LoadGovernor

RTSP = "rtsp://192.0.2.10:554/Streaming/Channels/101"


def _listening_socket() -> socket.socket:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def test_reachable_false_when_nothing_listens():
    sock = _listening_socket()
    port = sock.getsockname()[1]
    sock.close()
    assert rtsp_endpoint_reachable(f"rtsp://127.0.0.1:{port}/x", timeout=0.3) is False


def test_reachable_true_when_port_open():
    sock = _listening_socket()
    try:
        port = sock.getsockname()[1]
        assert rtsp_endpoint_reachable(f"rtsp://127.0.0.1:{port}/x", timeout=0.5) is True
    finally:
        sock.close()


def test_is_local_endpoint():
    assert is_local_endpoint("rtsp://127.0.0.1:8554") is True
    assert is_local_endpoint("rtsp://localhost:8554") is True
    assert is_local_endpoint("rtsp://10.0.0.9:8554") is False


def test_find_gateway_binary_absent(tmp_path):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"stub")
    assert find_gateway_binary(str(ffmpeg)) is None
    assert find_gateway_binary(None) is None


def test_find_gateway_binary_prefers_bundled(tmp_path):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"stub")
    binary = tmp_path / "mediamtx.exe"
    binary.write_bytes(b"stub")
    assert find_gateway_binary(str(ffmpeg)) == binary


def test_ensure_gateway_skips_when_disabled(settings):
    settings.output_enabled = False
    gateway_state = ensure_local_gateway(settings)
    assert gateway_state.process is None
    assert "关闭" in gateway_state.reason


def test_ensure_gateway_never_spawns_for_remote_base(settings):
    settings.output_enabled = True
    settings.rtsp_output_base = "rtsp://10.0.0.9:8554"
    gateway_state = ensure_local_gateway(settings)
    assert gateway_state.process is None
    assert "外部网关" in gateway_state.reason


def test_ensure_gateway_reports_missing_binary(settings, tmp_path):
    sock = _listening_socket()
    port = sock.getsockname()[1]
    sock.close()  # 端口空着，且 bin/ 里没有 mediamtx
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"stub")
    settings.output_enabled = True
    settings.ffmpeg_bin = str(ffmpeg)
    settings.rtsp_output_base = f"rtsp://127.0.0.1:{port}"
    gateway_state = ensure_local_gateway(settings)
    assert gateway_state.process is None
    assert "未提供 mediamtx" in gateway_state.reason


# ---------------------------------------------------------------------------
# 管线侧：没有网关时跳过，而不是按退避节奏反复重连
# ---------------------------------------------------------------------------

def _pipeline(settings) -> StreamPipeline:
    camera = CameraConfig(
        id="gw_01", name="GW", source_rtsp=RTSP,
    )
    return StreamPipeline(camera.resolve(settings), settings, ClassicalEnhancer(), LoadGovernor())


def test_pipeline_skips_push_without_gateway(settings, monkeypatch):
    """目标端口没人监听：不建推流器、状态标为未就绪，并告诉调用方「这是跳过不是故障」。"""
    from campusclear.core import pipeline as pipeline_module

    sock = _listening_socket()
    port = sock.getsockname()[1]
    sock.close()
    settings.output_enabled = True
    settings.rtsp_output_base = f"rtsp://127.0.0.1:{port}"
    monkeypatch.setattr(pipeline_module, "rtsp_endpoint_reachable", lambda url, timeout=0.6: False)
    pipeline = _pipeline(settings)
    try:
        assert pipeline._configure_output() is True          # noqa: SLF001 - 跳过应由返回值表达
        assert pipeline.pusher is None
        assert pipeline.state.output_ready is False
    finally:
        pipeline.stop()


def test_pipeline_builds_pusher_when_gateway_present(settings, monkeypatch):
    """网关在线：照常建立推流器，目标地址带通道名。"""
    from campusclear.core import output as output_module
    from campusclear.core import pipeline as pipeline_module

    settings.output_enabled = True
    settings.rtsp_output_base = "rtsp://127.0.0.1:8554"
    monkeypatch.setattr(pipeline_module, "rtsp_endpoint_reachable", lambda url, timeout=0.6: True)
    started: list[str] = []
    monkeypatch.setattr(output_module.RtspPusher, "start", lambda self: started.append(self.target_url))
    pipeline = _pipeline(settings)
    try:
        assert pipeline._configure_output() is False         # noqa: SLF001
        assert pipeline.pusher is not None
        assert pipeline.state.output_ready is True
        assert started and started[0].endswith(f"/{pipeline.camera.output_name}")
    finally:
        pipeline.stop()
