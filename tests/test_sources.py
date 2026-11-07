"""取流来源（多协议 + 厂商地址）测试。"""
from __future__ import annotations

import pytest

from campusclear.core.sources import (
    build_input_args,
    build_playback_url,
    describe_methods,
    detect_scheme,
    vendor_live_url,
)


def test_detect_scheme():
    assert detect_scheme("rtsp://h/x") == "rtsp"
    assert detect_scheme("http://h/x.flv") == "http"
    assert detect_scheme("/data/clip.mp4") == ""
    assert detect_scheme("file:///data/clip.mp4") == "file"


def test_methods_listed():
    keys = {m["key"] for m in describe_methods()}
    assert {"rtsp", "rtmp", "http_flv", "hls", "rtp", "file"} <= keys


def test_rtsp_args_use_transport_and_timeout():
    args = build_input_args("rtsp://h/x", rtsp_transport="udp", timeout_us=1234)
    assert "-rtsp_transport" in args and "udp" in args
    assert "-timeout" in args and "1234" in args


def test_hls_args_seek_live_edge():
    args = build_input_args("http://h/live/x.m3u8")
    assert "-live_start_index" in args


def test_http_flv_args_no_rtsp_flags():
    args = build_input_args("http://h/live/x.flv")
    assert "-rtsp_transport" not in args
    assert "-fflags" in args


def test_file_args_empty():
    assert build_input_args("/data/a.mp4") == []


def test_vendor_live_urls():
    hik = vendor_live_url("hikvision", "10.0.0.5", channel=1, username="a", password="b")
    assert hik.startswith("rtsp://a:b@10.0.0.5:554/Streaming/Channels/101")
    hik_sub = vendor_live_url("海康", "10.0.0.5", channel=2, sub=True)
    assert hik_sub.endswith("/Streaming/Channels/202")
    dahua = vendor_live_url("dahua", "10.0.0.6")
    assert "realmonitor" in dahua and "subtype=0" in dahua
    unv = vendor_live_url("uniview", "10.0.0.7", sub=True)
    assert unv.endswith("/media/video2")


def test_build_playback_url_hikvision():
    url = build_playback_url(
        "hikvision", "10.0.0.5", channel=1,
        start="2026-10-10 21:00:00", end="2026-10-10 21:05:00",
        username="admin", password="pw",
    )
    assert "/Streaming/tracks/101" in url
    assert "starttime=20261010T210000Z" in url
    assert "endtime=20261010T210500Z" in url


def test_build_playback_url_unknown_brand_raises():
    with pytest.raises(ValueError):
        build_playback_url("sony", "h", start="20261010210000", end="20261010210500")
