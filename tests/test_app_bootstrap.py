"""初始管理员密码的落盘与找回。

背景：``main._seed_admin`` 在没配 ``CAMPUSCLEAR_BOOTSTRAP_PASSWORD`` 时会随机
生成密码，且只在控制台打印一次；库里存的是 Argon2 哈希，不可逆。用户双击启动
后关掉控制台就彻底进不去了。``app._ensure_bootstrap_password`` 负责把它落一份
到数据目录兜底——这里把这个机制的各种分支钉死。

判据的关键点：**「有没有管理员账号」和「库文件在不在」是两件事**。
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from campusclear.app import PASSWORD_FILENAME, _admin_exists, _ensure_bootstrap_password

ENV_KEY = "CAMPUSCLEAR_BOOTSTRAP_PASSWORD"
DB_NAME = "campusclear.db"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """每个用例都从「没配过初始密码」的状态出发。

    monkeypatch 会在用例结束后把直接写进 os.environ 的值也一并还原，
    否则第一个用例的随机密码会污染后面所有用例。
    """
    monkeypatch.delenv(ENV_KEY, raising=False)


def _make_db(path, users: int = 0) -> None:
    """造一个带 users 表的库，表里放 ``users`` 行。"""
    con = sqlite3.connect(path)
    con.execute("create table users (username text primary key, password text)")
    for i in range(users):
        con.execute("insert into users values (?, ?)", (f"user{i}", "hash"))
    con.commit()
    con.close()


# --------------------------------------------------------------------------- #
# _admin_exists
# --------------------------------------------------------------------------- #
def test_admin_exists_false_when_no_db(tmp_path):
    assert _admin_exists(tmp_path / DB_NAME) is False


def test_admin_exists_false_when_table_missing(tmp_path):
    """库建了但表还没建（进程在初始化中途被杀）—— 仍算「没账号」。"""
    con = sqlite3.connect(tmp_path / DB_NAME)
    con.execute("create table cameras (id text)")
    con.commit()
    con.close()
    assert _admin_exists(tmp_path / DB_NAME) is False


def test_admin_exists_false_when_table_empty(tmp_path):
    _make_db(tmp_path / DB_NAME, users=0)
    assert _admin_exists(tmp_path / DB_NAME) is False


def test_admin_exists_true_with_a_user(tmp_path):
    _make_db(tmp_path / DB_NAME, users=1)
    assert _admin_exists(tmp_path / DB_NAME) is True


def test_admin_exists_is_conservative_on_corrupt_db(tmp_path):
    """库损坏时返回 True：宁可不写文件，也不要覆盖用户自己设过的密码。"""
    (tmp_path / DB_NAME).write_bytes(b"this is definitely not a sqlite file, just bytes" * 20)
    assert _admin_exists(tmp_path / DB_NAME) is True


def test_admin_exists_does_not_modify_anything(tmp_path):
    """必须是纯读：不能顺手把 -wal 建出来，也不能改动 mtime 之外的东西。"""
    _make_db(tmp_path / DB_NAME, users=1)
    before = set(p.name for p in tmp_path.iterdir())
    _admin_exists(tmp_path / DB_NAME)
    assert set(p.name for p in tmp_path.iterdir()) == before


# --------------------------------------------------------------------------- #
# _ensure_bootstrap_password
# --------------------------------------------------------------------------- #
def test_first_run_writes_retrievable_password(tmp_path):
    marker = _ensure_bootstrap_password(tmp_path)

    assert marker == tmp_path / PASSWORD_FILENAME
    assert marker.is_file()

    password = os.environ.get(ENV_KEY)
    assert password, "必须通过环境变量把密码交给 main._seed_admin"

    text = marker.read_text(encoding="utf-8")
    assert "admin" in text
    assert password in text, "文件里必须是真正会被使用的那一个密码"


def test_existing_admin_suppresses_file(tmp_path):
    """账号已存在时再写文件会给出一个登不上的密码，比没有文件更误导。"""
    _make_db(tmp_path / DB_NAME, users=1)

    assert _ensure_bootstrap_password(tmp_path) is None
    assert not (tmp_path / PASSWORD_FILENAME).exists()
    assert ENV_KEY not in os.environ


def test_db_without_admin_still_writes_file(tmp_path):
    """库在但没账号：必须写文件。这正是「库文件存在」判据会漏掉的情况。"""
    _make_db(tmp_path / DB_NAME, users=0)

    marker = _ensure_bootstrap_password(tmp_path)

    assert marker == tmp_path / PASSWORD_FILENAME
    assert marker.is_file()
    assert os.environ.get(ENV_KEY)


def test_reuses_persisted_password_when_database_is_gone(tmp_path):
    """用户删了库但留着密码文件：应沿用原密码，而不是又生成一个新的。"""
    _ensure_bootstrap_password(tmp_path)
    original = os.environ[ENV_KEY]
    del os.environ[ENV_KEY]

    marker = _ensure_bootstrap_password(tmp_path)

    assert marker == tmp_path / PASSWORD_FILENAME
    assert os.environ[ENV_KEY] == original


def test_explicit_env_var_wins_and_no_file(tmp_path, monkeypatch):
    """显式配置优先，且不产生任何文件——用户自己管密码就该完全不干预。"""
    monkeypatch.setenv(ENV_KEY, "MyOwnPassw0rd!")

    assert _ensure_bootstrap_password(tmp_path) is None
    assert not (tmp_path / PASSWORD_FILENAME).exists()
    assert os.environ[ENV_KEY] == "MyOwnPassw0rd!"


def test_corrupt_file_is_regenerated(tmp_path):
    """文件被改坏（没有密码行）时要重写，而不是拿空密码去建账号。"""
    marker = tmp_path / PASSWORD_FILENAME
    marker.write_text("这文件被人改坏了，没有密码行\n", encoding="utf-8")

    assert _ensure_bootstrap_password(tmp_path) == marker

    assert "密码:" in marker.read_text(encoding="utf-8")
    assert os.environ.get(ENV_KEY)


def test_second_run_with_existing_admin_leaves_env_alone(tmp_path):
    """第二次启动（账号已建）不应重复设置环境变量。"""
    _ensure_bootstrap_password(tmp_path)
    _make_db(tmp_path / DB_NAME, users=1)
    del os.environ[ENV_KEY]

    assert _ensure_bootstrap_password(tmp_path) is None
    assert ENV_KEY not in os.environ
