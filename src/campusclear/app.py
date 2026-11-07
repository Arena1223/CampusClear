"""明眸客户端 · 桌面启动器。

这是打包成「一个程序」后的入口。它做三件事：

1. 在**本机**（默认 127.0.0.1）启动一个内置服务——**只给本机浏览器/窗口看**，
   不对外监听，符合「纯客户端、本地处理」的定位；
2. 数据目录放在用户可写的位置（打包后不能再依赖源码目录）；
3. 打开界面：有内嵌浏览器内核（pywebview）就开独立窗口，否则用系统默认浏览器。

用法:
    python -m campusclear.app                 # 启动并打开界面
    python -m campusclear.app --port 9000     # 指定端口
    python -m campusclear.app --no-window     # 只起服务，不自动开界面
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import sqlite3
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _user_data_dir() -> Path:
    """打包后数据目录：Windows 用 %LOCALAPPDATA%，其它平台用 ~/.campusclear。"""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "CampusClear"
    return Path.home() / ".campusclear"


def _pick_models_dir(data_dir: Path) -> Path:
    """按优先级挑一个权重目录，都不存在就用数据目录下的 models/。

    顺序（同 compat.find_ffmpeg 的思路，两处保持一致的「先外后内」策略）：
      1. exe 旁边 ``models/``  —— 便于「绿色免安装」与现场换权重：把
         CampusClear.exe + _internal + models 一起拷走即可，权重随程序走；
      2. PyInstaller 解包目录里的 ``models/``（打包时随 spec 一起进去），
         兜住「用户只拷了 exe 和 _internal、漏了 models」的情况；
      3. ``<data_dir>/models`` —— 最后兜底，空目录意味着超分走经典插值。

    第 1 顺位之所以在前：现场部署时学校可能想换成自己训的权重，
    放在 exe 旁边是肉眼可见、随时可替换的位置。
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "models")
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle:
            candidates.append(Path(bundle) / "models")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.onnx")):
            return candidate.resolve()
    return data_dir / "models"


PASSWORD_FILENAME = "初始密码.txt"


def _admin_exists(db_path: Path) -> bool:
    """库里是否**已经建过管理员账号**。

    为什么不用「库文件是否存在」来判断首次运行：这两个条件不是一回事。
    库文件可能已经建好但账号还没建（进程恰好在 ``_seed_admin`` 之前被杀），
    这时必须写密码文件；而反过来，只要账号已存在就绝不能写——那会给出一个
    根本登不上的密码，比没有文件更容易误导人。

    读失败（库损坏、被另一个实例锁住）时保守返回 True：宁可不写文件，
    也不要覆盖掉用户可能已经自己设过的那份密码。
    """
    if not db_path.is_file():
        return False
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            has_table = con.execute(
                "select count(*) from sqlite_master where type='table' and name='users'"
            ).fetchone()
            if not has_table or not has_table[0]:
                return False
            return bool(con.execute("select count(*) from users").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error:
        return True


def _ensure_bootstrap_password(data_dir: Path) -> Path | None:
    """保证首次启动有一个**可找回**的初始管理员密码，返回密码文件路径。

    这里解决的是一个很实际的坑：数据库里存的是 Argon2 哈希（不可逆），
    初始密码由 main._seed_admin 随机生成、只在控制台打印一次。用户双击启动后
    随手关掉那个黑色控制台窗口，就再也进不去了——唯一出路是删库重来。

    所以把首次生成的密码额外落一份到数据目录：
    - 控制台仍会打印（保持原有提醒效果）；
    - 文件是兜底，登录后在管理面板改密码，然后删掉它即可。

    两种情况下完全不干预：
    1. 用户显式设了 ``CAMPUSCLEAR_BOOTSTRAP_PASSWORD``；
    2. 库里已有管理员账号（``_admin_exists``）—— 再写文件只会给出一个假密码。
    """
    if os.environ.get("CAMPUSCLEAR_BOOTSTRAP_PASSWORD"):
        return None
    if _admin_exists(data_dir / "campusclear.db"):
        return None

    marker = data_dir / PASSWORD_FILENAME
    if marker.is_file():
        # 库那边没账号但密码文件在（例如用户删库重来、或上次启动中途失败）：
        # 沿用文件里的密码，免得用户手上那份记录突然失效。
        for line in marker.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("密码:"):
                password = line.split(":", 1)[1].strip()
                if password:
                    os.environ["CAMPUSCLEAR_BOOTSTRAP_PASSWORD"] = password
                    return marker
        # 文件被改坏了：重写，免得拿一个空密码去建账号
        marker.unlink(missing_ok=True)

    password = secrets.token_urlsafe(12)
    os.environ["CAMPUSCLEAR_BOOTSTRAP_PASSWORD"] = password
    marker.write_text(
        "明眸客户端 · 初始管理员账号\n"
        + "=" * 34 + "\n"
        "用户名: admin\n"
        f"密码: {password}\n"
        "\n"
        "首次登录后请在管理面板修改密码，然后删除本文件。\n",
        encoding="utf-8",
    )
    return marker


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0


def _pick_port(host: str, want: int) -> int:
    """优先用指定端口；若被占用（可能是上一个没退干净的实例）则自动换一个空闲端口。

    否则会出现：新实例绑不上 8000，而旧实例还在 8000 上监听，
    启动器误判「服务已就绪」，于是把浏览器开到旧实例上——用户看到的就是旧界面。
    """
    if want and _port_free(host, want):
        return want
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_ready(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.15)
    return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="明眸客户端（本地处理，界面只在本机）")
    ap.add_argument("--host", default="127.0.0.1", help="仅本机使用，勿改为对外地址")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=None, help="录像/输出/数据库目录")
    ap.add_argument("--models-dir", default=None, help="ONNX 权重目录（重量增强用）")
    ap.add_argument("--no-window", action="store_true", help="不自动打开界面")
    ap.add_argument("--url-only", action="store_true", help="只打印地址后退出（调试用）")
    return ap


def _start_view_servers(app, host: str, base_port: int) -> list[int]:
    """启动额外的「浏览器查看端口」（同一 ASGI 应用多端口监听）。

    为什么需要：MJPEG 是**永不关闭**的长连接，而浏览器对同一
    「协议+主机+端口」只允许约 6 条并发连接。所有路都从 base_port 出图时，
    在线路一超过 6 条，多出的画面就永远排在浏览器连接队列里打不开——
    现场表现为「网格页只有前 6 路有画面，其余一直转圈」。

    解法：把观看流量分散到 base_port+1..+N 共 N 个本地端口。浏览器的
    并发限制按「主机:端口」计数，N 个端口就是约 6×(N+1) 个名额。
    每个端口都是完整 ASGI 应用、仅绑定 127.0.0.1，不扩大暴露面。
    """
    import uvicorn

    try:
        from .config import get_settings
    except ImportError:  # PyInstaller 打包后以脚本方式启动
        from campusclear.config import get_settings

    settings = get_settings()
    if settings.tls_enabled:
        # TLS 形态下额外端口没有证书配置，不启用，避免给出半残的 https 端口
        return []
    count = max(0, int(settings.view_port_count))
    started: list[int] = []
    for offset in range(1, count + 1):
        port = base_port + offset
        if not _port_free(host, port):
            print(f"查看端口 {port} 已被占用，跳过（可减小 CAMPUSCLEAR_VIEW_PORTS）")
            continue
        # 必须关闭 lifespan：这些端口只是同一应用的「观看入口」，而 FastAPI 的
        # lifespan 会新建 CameraStore / ModelRegistry / StreamSupervisor 并
        # ``supervisor.start()``。若让每个查看端口都跑一遍 lifespan，4 个端口
        # 就会额外拉起 4 套通道管线（4 倍 ffmpeg 进程与上游 RTSP 连接），
        # 且 app.state.supervisor 会被最后启动的那套覆盖，退出时只关一套。
        vconfig = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        vserver = uvicorn.Server(vconfig)
        threading.Thread(target=vserver.run, name=f"cc-view-{port}", daemon=True).start()
        started.append(port)
    settings.active_view_ports = started
    return started


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from .compat import enable_utf8_console, find_ffmpeg
        from .core.gateway import find_gateway_binary
    except ImportError:  # PyInstaller 打包后以脚本方式启动，包上下文丢失
        from campusclear.compat import enable_utf8_console, find_ffmpeg
        from campusclear.core.gateway import find_gateway_binary

    # Windows 控制台默认 cp936（GBK）：中文日志尚可，但「✓/✗」这类符号会抛
    # UnicodeEncodeError，被 logging 吞掉后变成满屏「--- Logging error ---」。
    enable_utf8_console()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else _user_data_dir()
    if args.models_dir:
        models_dir = Path(args.models_dir).resolve()
    else:
        models_dir = _pick_models_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "recordings").mkdir(exist_ok=True)

    # 在导入应用之前把配置写进环境：config 是惰性读取环境变量的
    os.environ.setdefault("CAMPUSCLEAR_DATA_DIR", str(data_dir))
    os.environ.setdefault("CAMPUSCLEAR_MODELS_DIR", str(models_dir))
    os.environ.setdefault("CAMPUSCLEAR_LOG_DIR", str(data_dir / "logs"))
    os.environ.setdefault("CAMPUSCLEAR_HOST", args.host)
    # 客户端形态：无权重时不该拒绝启动，校验哈希放宽（权重到位后可在配置里打开）
    os.environ.setdefault("CAMPUSCLEAR_VERIFY_MODEL_HASH", "false")
    # 回推：RTSP 是「有服务端才能推」的协议。早先单机形态一律关闭，是因为
    # 包里没有网关，开了只会每路去撞一个不存在的地址；现在安装目录里自带
    # MediaMTX（bin/mediamtx.exe），就按「有网关才启用」来决定，仍然允许
    # 用户显式设 CAMPUSCLEAR_OUTPUT_ENABLED 覆盖。
    try:
        _gateway_bin = find_gateway_binary(find_ffmpeg())
    except Exception:  # noqa: BLE001 - 查找失败按「没有网关」处理
        _gateway_bin = None
    os.environ.setdefault(
        "CAMPUSCLEAR_OUTPUT_ENABLED", "true" if _gateway_bin else "false"
    )

    port = _pick_port(args.host, args.port)
    url = f"http://{args.host}:{port}/client"

    # 把**实际选中**的端口回写进环境：否则 main.py 的「服务就绪，监听 X」会打印
    # 配置里的默认值（8000），与启动器真正使用的端口不一致 —— 排查时会误导人。
    os.environ["CAMPUSCLEAR_PORT"] = str(port)

    if args.url_only:
        print(url)
        return 0

    # 初始密码要在启动服务之前定下来：config 惰性读取环境变量，而 _seed_admin
    # 会在 lifespan 启动时立刻拿它建账号。放在 url_only 之后是为了不产生副作用
    # ——调试开关不该往用户数据目录里写文件。
    password_file = _ensure_bootstrap_password(data_dir)

    import uvicorn

    try:
        from .main import app
    except ImportError:
        # PyInstaller 打包后以「脚本」方式启动入口，包上下文丢失，需要绝对导入
        from campusclear.main import app

    config = uvicorn.Config(app, host=args.host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="cc-app", daemon=True)
    thread.start()

    # 只认「我们自己这个 uvicorn 是否真的起来了」，而不是「端口是否有人应答」——
    # 后者在旧实例占用端口时会误判。
    deadline = time.time() + 20
    while time.time() < deadline and not getattr(server, "started", False):
        if not thread.is_alive():
            break
        time.sleep(0.1)
    if not getattr(server, "started", False):
        print("服务启动失败（端口被占用或启动异常）", file=sys.stderr)
        return 1
    view_ports = _start_view_servers(app, args.host, port)
    print(f"明眸客户端已就绪： {url}")
    print(f"数据目录： {data_dir}")
    print(f"ffmpeg ： {find_ffmpeg()}")
    if view_ports:
        print(f"多端口观看： {', '.join(str(p) for p in view_ports)}"
              f"（浏览器并发名额 ~6×{len(view_ports) + 1}）")
    if password_file is not None:
        print()
        print("=" * 62)
        print(" 首次启动，已生成初始管理员账号（用户名 admin）")
        print(f" 初始密码同时保存在： {password_file}")
        print(" 登录后请在管理面板修改密码，然后删除该文件。")
        print("=" * 62)

    if args.no_window:
        thread.join()
        return 0

    if not _open_window(url, data_dir):
        # 窗口内核不可用时退回默认浏览器
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        thread.join()
    return 0


def _open_window(url: str, data_dir: Path) -> bool:
    """尝试用 pywebview 开独立窗口；不可用返回 False 交给浏览器兜底。"""
    try:
        import webview  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        webview.create_window("明眸 · 校园监控增强客户端", url, width=1280, height=800)
        webview.start()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
