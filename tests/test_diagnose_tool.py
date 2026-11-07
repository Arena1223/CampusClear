"""服务器诊断工具的候选地址与时间换算。

重点防回归：回放时间必须同时给出 UTC 和本地两种写法；品牌 auto 时
实时/回放候选都要覆盖；日志脱敏不能在候选地址里泄露凭证。
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    path = ROOT / "tools" / "diagnose_server.py"
    spec = importlib.util.spec_from_file_location("campusclear_diagnose_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass 在解析注解时会从 sys.modules 找模块，必须先登记再 exec
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_time_variants_include_utc_and_local():
    tool = _load_tool()
    start = tool.parse_dt("2026-10-29T16:00:00+08:00")
    end = tool.parse_dt("2026-10-29T16:05:00+08:00")
    variants = tool.time_variants(start, end)

    assert variants["utc_z"][0] == "20261029T080000Z"
    assert variants["utc_z"][1] == "20261029T080500Z"
    assert variants["local_space"][0] == "2026-10-29 16:00:00"
    assert variants["local_compact"][0] == "20261029T160000"


def test_auto_brand_builds_live_and_playback_candidates():
    tool = _load_tool()
    args = SimpleNamespace(
        url="", host="10.0.0.9", port=554, user="admin", password="Secret123",
        brand="auto", channel=1, mode="all", max_candidates=40,
    )
    start = datetime.fromisoformat("2026-10-29T16:00:00+08:00")
    end = datetime.fromisoformat("2026-10-29T16:05:00+08:00")
    candidates = tool.build_candidates(args, start, end)

    modes = {item.mode for item in candidates}
    assert modes == {"live", "playback"}
    assert any(
        item.mode == "playback" and "/Streaming/tracks/101" in item.url
        and "starttime=20261029T080000Z" in item.url
        for item in candidates
    )
    assert any("cam/playback" in item.url for item in candidates)
    assert any("unicast/c1/s0/playback" in item.url for item in candidates)
    # 候选地址可以带凭证给 ffprobe，但日志展示必须脱敏
    for item in candidates:
        assert "Secret123" not in tool.mask_url(item.url)
