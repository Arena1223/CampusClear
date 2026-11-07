#!/usr/bin/env python
"""把 ffmpeg 就位到项目的 bin/ffmpeg.exe（Windows 打包前置步骤）。

为什么需要这一步：本项目的全部前提是「无公网依赖」，运行时绝不能去
GitHub 下载任何东西。但 ffmpeg 又不是 pip 能直接给的可执行文件——
imageio-ffmpeg 这个包恰好自带一份静态构建，于是流程变成：

    pip 装 imageio-ffmpeg  →  把它的二进制复制成 bin/ffmpeg.exe
                          →  打包时由 spec 打进程序的 _internal/bin/

注意**不要**用 ``imageio_ffmpeg.get_ffmpeg_exe()`` 去拿路径：二进制缺失时
它会静默去 GitHub 下载。这里只扫包内 binaries/ 目录，找不到就老实报错。

用法：
    python tools/prepare_ffmpeg.py            # 缺才复制
    python tools/prepare_ffmpeg.py --force    # 强制覆盖
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "bin" / "ffmpeg.exe"


def _imageio_binaries() -> list[Path]:
    """列出 imageio-ffmpeg 自带的静态二进制（按文件名前缀，带平台后缀）。"""
    try:
        import imageio_ffmpeg
    except ImportError:
        return []
    binaries = Path(imageio_ffmpeg.__file__).resolve().parent / "binaries"
    if not binaries.is_dir():
        return []
    return [
        item for item in sorted(binaries.iterdir())
        if item.is_file() and item.name.startswith("ffmpeg")
    ]


def _version_of(path: Path) -> str | None:
    """跑一次 ``-version``：既验证「能执行」，也顺手拿到版本串。

    返回版本行，失败返回 None。之所以必须真跑一次——一个截断的/被安全软件
    拦下的 exe 看起来也是普通文件，只有执行才知道是死是活。
    """
    try:
        result = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = (result.stdout or "").splitlines()
    return lines[0] if lines else None


def _report(path: Path) -> bool:
    """打印 ffmpeg 体积与版本；不可执行时给出可操作的建议。"""
    size_mb = path.stat().st_size / 1024 / 1024
    version = _version_of(path)
    if version is None:
        print(
            f"[fail] {path} 无法执行（{size_mb:.1f} MB）。\n"
            "       可能是复制不完整，或被安全软件拦截。\n"
            "       试试：python tools/prepare_ffmpeg.py --force\n"
            "       或设置 CAMPUSCLEAR_FFMPEG 指向一个能用的 ffmpeg",
            file=sys.stderr,
        )
        return False
    print(f"     {path}（{size_mb:.1f} MB）")
    print(f"     {version}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备项目自带的 ffmpeg.exe")
    parser.add_argument("--force", action="store_true", help="已有也重新复制")
    args = parser.parse_args(argv)

    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.is_file() and not args.force:
        print("[skip] 已存在，跳过复制：")
        return 0 if _report(TARGET) else 1

    candidates = _imageio_binaries()
    if not candidates:
        print(
            "       请任选一种方式后重试：\n"
            "       1) pip install imageio-ffmpeg   （本机自带静态 ffmpeg）\n"
            f"       2) 手动下载 Windows 版 ffmpeg.exe 放到 {TARGET}\n"
            "       3) 设置环境变量 CAMPUSCLEAR_FFMPEG 指向已有的 ffmpeg",
            file=sys.stderr,
        )
        return 1

    source = candidates[0]
    shutil.copy2(source, TARGET)
    print(f"[copy] {source.name} -> {TARGET.name}")
    return 0 if _report(TARGET) else 1


if __name__ == "__main__":
    raise SystemExit(main())
