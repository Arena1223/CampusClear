#!/usr/bin/env bash
# 在 MOKOUSPACE（exfat）上运行「明眸客户端」。
#
# 为什么不用 venv：exfat 不支持符号链接，而 venv 会创建 lib64 符号链接，
# 因此在 exfat 上建 venv 会失败。这里改用 pip --target 把依赖装到 pylibs/，
# 运行时用 PYTHONPATH 指过去，等价于一个「无符号链接的虚拟环境」。
#
# 用法:
#   ./run_client.sh                # 启动客户端并打开界面
#   ./run_client.sh --no-window    # 只起本地服务
#   ./run_client.sh --port 9000
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/src:$HERE/pylibs${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m campusclear.app "$@"
