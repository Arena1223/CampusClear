#!/usr/bin/env bash
# 一键打包「明眸客户端」为可执行程序。
#
#   ./packaging/build.sh
#
# 产物在 dist/CampusClear/。Windows 上请在 Windows 环境执行同样的命令
# （PyInstaller 不能跨平台交叉编译）。
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi

echo "== 确保依赖 =="
"$PY" -m pip install -q --upgrade pyinstaller
"$PY" -m pip install -q -r requirements.txt 2>/dev/null || true

echo "== 开始打包（首次较慢，因为要收集 onnxruntime/opencv）=="
"$PY" -m PyInstaller packaging/campusclear.spec --noconfirm --clean

echo
echo "== 完成 =="
echo "可执行文件： $(pwd)/dist/CampusClear/CampusClear"
echo "运行即可，界面默认打开 http://127.0.0.1:8000/client"
