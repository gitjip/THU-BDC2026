#!/usr/bin/env sh
set -eu

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

debug_arg=""
case "${1:-}" in
  debug|fast|--debug|--fast)
    export BDC_FAST_DEV="${BDC_FAST_DEV:-1}"
    debug_arg="--debug"
    shift
    ;;
esac

if [ -x ".venv/bin/python" ]; then
  python_bin=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  python_bin="uv run python"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  echo "未找到可用的 Python。请先运行 uv sync，或激活项目虚拟环境。" >&2
  exit 1
fi

submission_mode="${BDC_SUBMISSION_MODE:-ensemble}"
case "${1:-}" in
  ensemble|--ensemble)
    submission_mode="ensemble"
    shift
    ;;
  single|--single)
    submission_mode="single"
    shift
    ;;
esac

if [ "$submission_mode" = "single" ]; then
  # shellcheck disable=SC2086
  exec $python_bin code/src/predict.py "$@"
fi

if [ -n "$debug_arg" ]; then
  # shellcheck disable=SC2086
  exec $python_bin code/src/ensemble_predict.py "$debug_arg" "$@"
fi

# shellcheck disable=SC2086
exec $python_bin code/src/ensemble_predict.py "$@"
