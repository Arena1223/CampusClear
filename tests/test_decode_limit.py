"""解码分辨率上限（CAMPUSCLEAR_MAX_DECODE_WIDTH / HEIGHT）的行为约束。

为什么值得单独测：这个开关是纯 CPU 机器上「画面能不能动起来」的唯一手段，
而它做的是**算术 + 取偶**——算错不会报错，只会让 ffmpeg 的 ``scale``
以 "width not divisible by 2" 拒跑，或者在用户毫无察觉的情况下把 4MP 悄悄
砍成 1/4。所以把边界一条条钉死。
"""

from __future__ import annotations

import pytest

from campusclear.core.ingest import RtspReader, StreamProbe

RTSP = "rtsp://192.0.2.10:554/Streaming/Channels/101"


def _reader(**kwargs) -> RtspReader:
    return RtspReader(RTSP, **kwargs)


def test_no_limit_keeps_source_size():
    """默认不设上限：4MP 摄像头就该按 2560x1440 解，不做任何隐式降采样。"""
    reader = _reader()
    assert reader.max_width == 0 and reader.max_height == 0
    reader.configure(StreamProbe(width=2560, height=1440, fps=25))
    assert (reader.width, reader.height) == (2560, 1440)
    assert reader._frame_bytes == 2560 * 1440 * 3


def test_max_width_scales_keeping_aspect():
    reader = _reader(max_width=1280)
    reader.configure(StreamProbe(width=2560, height=1440, fps=25))
    assert (reader.width, reader.height) == (1280, 720)


def test_never_upscales():
    """上限是「上限」：源本身比它还小时必须原样放行，不能放大。"""
    reader = _reader(max_width=1920, max_height=1080)
    reader.configure(StreamProbe(width=1280, height=720, fps=25))
    assert (reader.width, reader.height) == (1280, 720)


def test_max_height_alone_is_enough():
    reader = _reader(max_height=720)
    reader.configure(StreamProbe(width=3840, height=2160, fps=25))
    assert (reader.width, reader.height) == (1280, 720)


def test_tighter_of_the_two_wins():
    reader = _reader(max_width=1000, max_height=1000)
    reader.configure(StreamProbe(width=2560, height=1440, fps=25))
    assert (reader.width, reader.height) == (1000, 562)


@pytest.mark.parametrize(
    ("src", "limit", "expect"),
    [
        ((1920, 1080), 1000, (1000, 562)),
        ((1920, 1080), 1281, (1280, 720)),   # 1281/1920 -> 1281x720.19，高度取偶
        ((2560, 1440), 853, (852, 478)),
        ((704, 576), 704, (704, 576)),
    ],
)
def test_output_is_always_even(src, limit, expect):
    """yuv420p 的色度平面是 2x2 下采样，奇数宽高会让 ffmpeg 直接拒跑。"""
    reader = _reader(max_width=limit)
    reader.configure(StreamProbe(width=src[0], height=src[1], fps=25))
    assert (reader.width, reader.height) == expect
    assert reader.width % 2 == 0 and reader.height % 2 == 0


def test_explicit_size_is_not_overwritten_by_probe():
    """调用方显式钉死尺寸时，probe 只用来校准 fps。

    原实现靠「当前值恰好等于 1920x1080 就当作占位符」来判断，遇到真的 1080p
    流会误判成「还没探测过」从而覆盖调用方的显式设定。
    """
    reader = _reader(width=1024, height=768)
    reader.configure(StreamProbe(width=2560, height=1440, fps=30))
    assert (reader.width, reader.height) == (1024, 768)
    assert reader.fps == 30


def test_limit_also_applies_to_explicit_size():
    """显式尺寸同样受上限约束——「max」是上界，不是「仅在自动探测时生效」。"""
    reader = _reader(width=2560, height=1440, max_width=1280)
    reader.configure(StreamProbe(width=2560, height=1440, fps=25))
    assert (reader.width, reader.height) == (1280, 720)


def test_degenerate_probe_is_ignored():
    """探测到 0x0（解析失败）时不能把 reader 的尺寸清成 0，否则帧长算成 0。"""
    reader = _reader()
    reader.configure(StreamProbe(width=0, height=0, fps=0))
    assert (reader.width, reader.height) == (1920, 1080)
    assert reader._frame_bytes == 1920 * 1080 * 3


def test_frame_bytes_follows_capped_size():
    """帧长必须与最终尺寸一致：rawvideo 管道是按它读的，错了就是满屏雪花。"""
    reader = _reader(max_width=1280)
    reader.configure(StreamProbe(width=2560, height=1440, fps=25))
    assert reader._frame_bytes == 1280 * 720 * 3
