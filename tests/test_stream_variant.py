"""主/子码流切换：带查询串、大华、宇视都要能识别。"""

from __future__ import annotations

from campusclear.api.client import _is_sub_stream, _stream_url_variant


def test_hikvision_variant_without_query():
    main = "rtsp://u:p@10.0.0.5:554/Streaming/Channels/101"
    sub = "rtsp://u:p@10.0.0.5:554/Streaming/Channels/102"
    assert _stream_url_variant(main, sub=True) == sub
    assert _stream_url_variant(sub, sub=False) == main


def test_hikvision_variant_keeps_query_string():
    url = "rtsp://10.0.0.5/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
    expected = "rtsp://10.0.0.5/Streaming/Channels/102?transportmode=unicast&profile=Profile_1"
    assert _stream_url_variant(url, sub=True) == expected
    assert _is_sub_stream(expected) is True


def test_dahua_variant():
    main = "rtsp://10.0.0.6/cam/realmonitor?channel=1&subtype=0"
    sub = "rtsp://10.0.0.6/cam/realmonitor?channel=1&subtype=1"
    assert _stream_url_variant(main, sub=True) == sub
    assert _stream_url_variant(sub, sub=False) == main
    assert _is_sub_stream(sub) is True
    assert _is_sub_stream(main) is False


def test_uniview_variant():
    assert (
        _stream_url_variant("rtsp://10.0.0.7/media/video1", sub=True)
        == "rtsp://10.0.0.7/media/video2"
    )
    assert _is_sub_stream("rtsp://10.0.0.7/media/video2?foo=1") is True


def test_unknown_url_is_left_alone():
    url = "http://10.0.0.8/live/ch.flv"
    assert _stream_url_variant(url, sub=True) == url
    assert _is_sub_stream(url) is False
