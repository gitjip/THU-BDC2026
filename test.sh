#!/usr/bin/env sh
set -eu

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

case "${1:-}" in
  debug|fast|--debug|--fast)
    export BDC_FAST_DEV="${BDC_FAST_DEV:-1}"
    shift
    ;;
esac

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python code/src/predict.py "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python code/src/predict.py "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python code/src/predict.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 code/src/predict.py "$@"
fi

echo "未找到可用的 Python。请先运行 uv sync，或激活项目虚拟环境。" >&2
exit 1
