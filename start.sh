#!/usr/bin/env bash
# 明眸客户端 · 快速启动（源码目录版）
#
# 特点：
#   - 自动寻找 pylibs（项目内 / MOKOUSPACE / 环境变量指定）；
#   - 自动带上项目 models/ 权重，避免源码运行时静默退化成经典插值；
#   - 只绑定 127.0.0.1，不暴露端口；拒绝 --host 0.0.0.0 之类的参数；
#   - 其余参数原样传给 campusclear.app（--port / --no-window / --data-dir）。
#
# 用法：
#   ./start.sh
#   ./start.sh --no-window
#   ./start.sh --port 9000
#   CAMPUSCLEAR_PYLIBS=/path/to/pylibs ./start.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "错误：找不到 $PYTHON。请安装 Python 3.10+ 或设置 PYTHON=/path/to/python3" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. 找离线依赖目录
# ---------------------------------------------------------------------------
PYLIBS="${CAMPUSCLEAR_PYLIBS:-}"
if [ -z "$PYLIBS" ]; then
    for candidate in \
        "$HERE/pylibs" \
        "$HERE/../pylibs" \
        "/run/media/mokou/MOKOUSPACE/PROGRAM/CampusClear/pylibs" \
        "$HOME/.campusclear/pylibs"
    do
        if [ -d "$candidate/fastapi" ]; then
            PYLIBS="$candidate"
            break
        fi
    done
fi

if [ -z "$PYLIBS" ] || [ ! -d "$PYLIBS/fastapi" ]; then
    echo "错误：找不到 pylibs（至少应包含 fastapi/）。" >&2
    echo "可显式指定：CAMPUSCLEAR_PYLIBS=/path/to/pylibs ./start.sh" >&2
    exit 1
fi

export PYTHONPATH="$HERE/src:$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON" -c "import fastapi, uvicorn, cv2, numpy, pydantic, argon2, jwt" 2>/dev/null; then
    echo "错误：依赖不完整（$PYLIBS）。请检查 pylibs 是否完整。" >&2
    "$PYTHON" -c "import fastapi, uvicorn, cv2, numpy, pydantic, argon2, jwt" || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. 解析参数：只允许本机监听
# ---------------------------------------------------------------------------
args=()
host_seen=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --host)
            host="${2:-}"
            if [ "$host" != "127.0.0.1" ] && [ "$host" != "localhost" ] && [ "$host" != "::1" ]; then
                echo "错误：本启动脚本只允许本机访问，不接受 --host $host" >&2
                echo "      鉴权由后续本地软件接管；当前请勿暴露端口。" >&2
                exit 2
            fi
            args+=("$1" "$2")
            host_seen=1
            shift 2
            ;;
        --host=*)
            host="${1#--host=}"
            if [ "$host" != "127.0.0.1" ] && [ "$host" != "localhost" ] && [ "$host" != "::1" ]; then
                echo "错误：本启动脚本只允许本机访问，不接受 --host=$host" >&2
                exit 2
            fi
            args+=("$1")
            host_seen=1
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if [ "$host_seen" -eq 0 ]; then
    args+=(--host 127.0.0.1)
fi

# ---------------------------------------------------------------------------
# 3. 权重目录：源码运行时 app.py 不会自动找项目根目录的 models/
# ---------------------------------------------------------------------------
has_models_arg=0
for arg in "${args[@]}"; do
    case "$arg" in
        --models-dir|--models-dir=*) has_models_arg=1 ;;
    esac
done
if [ "$has_models_arg" -eq 0 ] && [ -d "$HERE/models" ]; then
    if find "$HERE/models" -maxdepth 1 -name '*.onnx' -print -quit 2>/dev/null | grep -q .; then
        args+=(--models-dir "$HERE/models")
    fi
fi

# ---------------------------------------------------------------------------
# 4. ffmpeg：没有就提醒（演示流不需要，接真实摄像头需要）
# ---------------------------------------------------------------------------
if [ -z "${CAMPUSCLEAR_FFMPEG:-}" ]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        export CAMPUSCLEAR_FFMPEG="$(command -v ffmpeg)"
    else
        echo "提醒：PATH 中没有 ffmpeg，接真实摄像头前请安装，或设置 CAMPUSCLEAR_FFMPEG。" >&2
    fi
fi

echo "依赖目录： $PYLIBS"
echo "启动：     $PYTHON -m campusclear.app ${args[*]}"
exec "$PYTHON" -m campusclear.app "${args[@]}"
