#!/usr/bin/env sh
set -eu

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python code/src/train.py
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python code/src/train.py
fi

if command -v python >/dev/null 2>&1; then
  exec python code/src/train.py
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 code/src/train.py
fi

echo "未找到可用的 Python。请先运行 uv sync，或激活项目虚拟环境。" >&2
exit 1
