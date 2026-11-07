"""pytest 全局夹具。

设计原则：**测试进程绝不接触真实摄像头、真实 GPU、真实数据目录**。
所有路径都指向临时目录，所有模型校验在无权重环境下也应正常退化。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 必须在导入 campusclear 之前设置：Settings 在实例化时读取环境变量
_TMP = Path(tempfile.mkdtemp(prefix="campusclear-test-"))
os.environ["CAMPUSCLEAR_DATA_DIR"] = str(_TMP / "data")
os.environ["CAMPUSCLEAR_MODELS_DIR"] = str(_TMP / "models")
os.environ["CAMPUSCLEAR_LOG_DIR"] = str(_TMP / "logs")
os.environ["CAMPUSCLEAR_SNAPSHOT_DIR"] = str(_TMP / "snapshots")
os.environ["CAMPUSCLEAR_TRT_CACHE"] = str(_TMP / "trt")
os.environ["CAMPUSCLEAR_JWT_SECRET"] = "unit-test-secret-not-for-production"
os.environ["CAMPUSCLEAR_BOOTSTRAP_ADMIN"] = "admin"
os.environ["CAMPUSCLEAR_BOOTSTRAP_PASSWORD"] = "TestPassw0rd!"
os.environ["CAMPUSCLEAR_VERIFY_MODEL_HASH"] = "false"
# 指向一个不存在的二进制：让「启用了通道」的测试用例快速失败而不是真的去拉流，
# 这样测试进程不会残留 FFmpeg 子进程，也不会因为等待网络超时而变慢。
os.environ["CAMPUSCLEAR_FFMPEG"] = "ffmpeg-not-present-in-test-env"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def settings():
    from campusclear.config import Settings

    return Settings()


@pytest.fixture()
def frame_factory():
    """生成可控退化程度的合成帧。"""

    def _make(width: int = 320, height: int = 240, mode: str = "normal") -> np.ndarray:
        rng = np.random.default_rng(42)
        base = np.full((height, width, 3), 110, dtype=np.float32)
        # 低频结构：模拟墙面/地面的明暗过渡
        base += 30 * np.sin(np.linspace(0, 6, height, dtype=np.float32))[:, None, None]
        # 高频结构：模拟栏杆、瓷砖缝隙一类的细纹理。
        # 这一项是必需的：没有高频细节时，高斯模糊几乎不改变拉普拉斯方差，
        # 模糊度判据就失去了可测的区分度。
        base += 22 * np.sin(np.linspace(0, 125, width, dtype=np.float32))[None, :, None]
        if mode == "dark":
            base = np.clip(base * 0.12, 0, 255)
        elif mode == "noisy":
            base = base + rng.normal(0, 20, base.shape)
        elif mode == "blurry":
            import cv2

            base = cv2.GaussianBlur(base, (0, 0), 4.0)
        return np.clip(base, 0, 255).astype(np.uint8)

    return _make


@pytest.fixture()
def store(settings):
    from campusclear.core.store import CameraStore

    return CameraStore(settings.db_path)


@pytest.fixture()
def client():
    """FastAPI 测试客户端。使用无通道配置的干净环境，不启动任何真实拉流。"""
    from fastapi.testclient import TestClient

    from campusclear.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def admin_token(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "TestPassw0rd!"},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture()
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
