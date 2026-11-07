#!/usr/bin/env bash
# 在 MOKOUSPACE（exfat）上跑测试。
# 依赖在 pylibs/（pip --target 安装），不依赖 venv。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PYLIBS="${CAMPUSCLEAR_PYLIBS:-}"
if [ -z "$PYLIBS" ]; then
    for candidate in \
        "$HERE/pylibs" \
        "$HERE/../pylibs" \
        "/run/media/mokou/MOKOUSPACE/PROGRAM/CampusClear/pylibs" \
        "$HOME/.campusclear/pylibs"
    do
        if [ -d "$candidate/pytest" ]; then
            PYLIBS="$candidate"
            break
        fi
    done
fi
if [ -z "$PYLIBS" ] || [ ! -d "$PYLIBS/pytest" ]; then
    echo "找不到 pylibs（需包含 pytest/）。可设 CAMPUSCLEAR_PYLIBS=/path/to/pylibs" >&2
    exit 1
fi

export PYTHONPATH="$HERE/src:$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE"
exec python3 -m pytest "$@"
