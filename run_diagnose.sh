#!/usr/bin/env bash
# 服务器实时/回放推流诊断（源码版）。
#
# 用法：
#   bash run_diagnose.sh --host 192.168.1.64 --user admin --password xxx --brand auto
#   bash run_diagnose.sh --url "http://10.0.0.9:8080/live/cam01.flv"
#   bash run_diagnose.sh --host 10.0.0.9 --brand auto --mode playback --dry-run
#
# 详细日志默认写到 ./logs/diagnose-*.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
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

if [ -n "$PYLIBS" ] && [ -d "$PYLIBS" ]; then
    export PYTHONPATH="$HERE/src:$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "$PYTHON" "$HERE/tools/diagnose_server.py" "$@"
