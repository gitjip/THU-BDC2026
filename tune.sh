#!/usr/bin/env sh
set -eu

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

set_default() {
  name="$1"
  value="$2"
  eval "current=\${$name:-}"
  if [ -z "$current" ]; then
    export "$name=$value"
  fi
}

version="$(cat VERSION 2>/dev/null || printf 'v1.0.0')"
case "${1:-}" in
  v[0-9]*.[0-9]*.[0-9]*)
    version="$1"
    shift
    ;;
esac

profile="${BDC_TUNE_PROFILE:-balanced}"
case "${1:-}" in
  debug|fast|quick|lite|--debug|--fast|--quick)
    profile="quick"
    shift
    ;;
  balanced|--balanced)
    profile="balanced"
    shift
    ;;
  full|--full)
    profile="full"
    shift
    ;;
esac
export BDC_TUNE_PROFILE="$profile"

case "$profile" in
  quick)
    set_default BDC_FAST_DEV 1
    set_default BDC_WF_WINDOWS 1
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 30
    set_default BDC_TRAIN_TARGET_DAYS 24
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 60
    set_default BDC_D_MODEL 64
    set_default BDC_NHEAD 2
    set_default BDC_NUM_LAYERS 1
    set_default BDC_DIM_FEEDFORWARD 128
    set_default BDC_BATCH_SIZE 8
    set_default BDC_NUM_EPOCHS 3
    set_default BDC_NUM_PROCESSES 4
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  balanced)
    set_default BDC_WF_WINDOWS 2
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 10
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 3
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  full)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_NUM_EPOCHS 3
    ;;
  *)
    echo "未知 tune profile: $profile，可选 quick、balanced、full。" >&2
    exit 2
    ;;
esac

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python code/src/walk_forward.py "$version" "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python code/src/walk_forward.py "$version" "$@"
fi

if command -v python >/dev/null 2>&1; then
  exec python code/src/walk_forward.py "$version" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 code/src/walk_forward.py "$version" "$@"
fi

echo "未找到可用的 Python。请先运行 uv sync，或激活项目虚拟环境。" >&2
exit 1
