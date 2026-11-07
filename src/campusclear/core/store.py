"""本地配置与审计存储（SQLite）。

为什么用 SQLite 而不是 PostgreSQL：
- 单机内网部署，数据量小（通道配置几十条、审计日志每天几千条），
  引入独立数据库服务只会增加运维负担和故障面；
- SQLite 单文件、无守护进程、无端口暴露，天然满足「数据不出内网」；
- 备份就是把一个文件拷走，学校信息中心的老师也能自己操作。

并发说明：启用 WAL 模式，读多写少场景下足以支撑本系统的写入频率。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .camera import CameraConfig

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cameras (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    source_rtsp       TEXT NOT NULL,
    location          TEXT DEFAULT '',
    enabled           INTEGER DEFAULT 1,
    width             INTEGER,
    height            INTEGER,
    fps               INTEGER,
    scale             INTEGER,
    enable_lowlight   INTEGER,
    enable_denoise    INTEGER,
    enable_deblur     INTEGER,
    enable_sr         INTEGER,
    roi_enabled       INTEGER DEFAULT 1,
    output_name       TEXT DEFAULT '',
    notes             TEXT DEFAULT '',
    tags              TEXT DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    disabled        INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    locked_until    REAL DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    username  TEXT,
    action    TEXT NOT NULL,
    target    TEXT DEFAULT '',
    detail    TEXT DEFAULT '',
    client_ip TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CameraStore:
    """通道配置 + 审计日志的本地存储。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_schema()

    # ------------------------------------------------------------------ #
    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def init_schema(self) -> None:
        with self._lock, self.connect() as connection:
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------ #
    # 通道
    # ------------------------------------------------------------------ #
    def list_cameras(self) -> list[CameraConfig]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM cameras ORDER BY id").fetchall()
        return [_row_to_camera(row) for row in rows]

    def get_camera(self, camera_id: str) -> CameraConfig | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return _row_to_camera(row) if row else None

    def upsert_camera(self, camera: CameraConfig) -> CameraConfig:
        now = utc_now()
        row = camera.to_row()
        with self._lock, self.connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM cameras WHERE id = ?", (camera.id,)
            ).fetchone()
            created = existing["created_at"] if existing else now
            connection.execute(
                """
                INSERT INTO cameras (
                    id, name, source_rtsp, location, enabled, width, height, fps, scale,
                    enable_lowlight, enable_denoise, enable_deblur, enable_sr,
                    roi_enabled, output_name, notes, tags, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    source_rtsp=excluded.source_rtsp,
                    location=excluded.location,
                    enabled=excluded.enabled,
                    width=excluded.width,
                    height=excluded.height,
                    fps=excluded.fps,
                    scale=excluded.scale,
                    enable_lowlight=excluded.enable_lowlight,
                    enable_denoise=excluded.enable_denoise,
                    enable_deblur=excluded.enable_deblur,
                    enable_sr=excluded.enable_sr,
                    roi_enabled=excluded.roi_enabled,
                    output_name=excluded.output_name,
                    notes=excluded.notes,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    row["id"], row["name"], row["source_rtsp"], row["location"],
                    int(bool(row["enabled"])), row["width"], row["height"], row["fps"], row["scale"],
                    _opt_bool(row["enable_lowlight"]), _opt_bool(row["enable_denoise"]),
                    _opt_bool(row["enable_deblur"]), _opt_bool(row["enable_sr"]),
                    int(bool(row["roi_enabled"])), row["output_name"], row["notes"], row["tags"],
                    created, now,
                ),
            )
        return camera

    def delete_camera(self, camera_id: str) -> bool:
        with self._lock, self.connect() as connection:
            cursor = connection.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        return cursor.rowcount > 0

    def count_cameras(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) AS c FROM cameras").fetchone()["c"])

    # ------------------------------------------------------------------ #
    # 用户
    # ------------------------------------------------------------------ #
    def get_user(self, username: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT username, role, disabled, last_login_at, created_at FROM users ORDER BY username"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_user(self, username: str, password_hash: str, role: str = "viewer") -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username, password_hash, role, utc_now()),
            )

    def update_password(self, username: str, password_hash: str) -> bool:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = 0 WHERE username = ?",
                (password_hash, username),
            )
        return cursor.rowcount > 0

    def set_user_role(self, username: str, role: str) -> bool:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET role = ? WHERE username = ?", (role, username)
            )
        return cursor.rowcount > 0

    def register_login_failure(self, username: str, max_failures: int, lock_seconds: int) -> int:
        """记录一次登录失败，达到阈值则锁定账号。返回当前失败次数。"""
        import time

        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT failed_attempts FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                return 0
            attempts = int(row["failed_attempts"]) + 1
            locked_until = time.time() + lock_seconds if attempts >= max_failures else 0
            connection.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                (attempts, locked_until, username),
            )
        return attempts

    def register_login_success(self, username: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = 0, last_login_at = ? WHERE username = ?",
                (utc_now(), username),
            )

    # ------------------------------------------------------------------ #
    # 审计
    # ------------------------------------------------------------------ #
    def add_audit(
        self,
        action: str,
        *,
        username: str = "",
        target: str = "",
        detail: str = "",
        client_ip: str = "",
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO audit_log (ts, username, action, target, detail, client_ip) VALUES (?,?,?,?,?,?)",
                (utc_now(), username, action, target, detail[:2000], client_ip),
            )

    def list_audit(self, limit: int = 100, username: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        with self.connect() as connection:
            if username:
                rows = connection.execute(
                    "SELECT * FROM audit_log WHERE username = ? ORDER BY id DESC LIMIT ?",
                    (username, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(row) for row in rows]

    def export_audit_jsonl(self, path: Path, limit: int = 100000) -> int:
        """导出审计日志为 JSONL，便于离线归档或交给信息中心留存。"""
        rows = self.list_audit(limit=limit)
        with Path(path).open("w", encoding="utf-8") as handle:
            for row in reversed(rows):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


# --------------------------------------------------------------------------- #
def _row_to_camera(row: sqlite3.Row) -> CameraConfig:
    tags_raw = row["tags"] or ""
    return CameraConfig(
        id=row["id"],
        name=row["name"],
        source_rtsp=row["source_rtsp"],
        location=row["location"] or "",
        enabled=bool(row["enabled"]),
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        scale=row["scale"],
        enable_lowlight=_from_opt_bool(row["enable_lowlight"]),
        enable_denoise=_from_opt_bool(row["enable_denoise"]),
        enable_deblur=_from_opt_bool(row["enable_deblur"]),
        enable_sr=_from_opt_bool(row["enable_sr"]),
        roi_enabled=bool(row["roi_enabled"]),
        output_name=row["output_name"] or "",
        notes=row["notes"] or "",
        tags=[t for t in tags_raw.split(",") if t],
    )


def _opt_bool(value) -> int | None:
    """三态布尔入库：None 表示继承全局配置。"""
    if value is None:
        return None
    return int(bool(value))


def _from_opt_bool(value) -> bool | None:
    if value is None:
        return None
    return bool(value)
