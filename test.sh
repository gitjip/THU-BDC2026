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

apply_rank_replace_defaults() {
  model_dir="./model/rank_replace"
  if [ -n "$debug_arg" ]; then
    model_dir="./model/rank_replace_debug"
    set_default BDC_FAST_DEV 1
    set_default BDC_SEQUENCE_LENGTH 30
    set_default BDC_D_MODEL 64
    set_default BDC_NHEAD 2
    set_default BDC_NUM_LAYERS 1
    set_default BDC_DIM_FEEDFORWARD 128
    set_default BDC_BATCH_SIZE 8
    set_default BDC_NUM_PROCESSES 4
  else
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_PROCESSES 6
  fi
  set_default BDC_OUTPUT_DIR "$model_dir"
  set_default BDC_FEATURE_NUM 39
  set_default BDC_VAL_DAYS 5
  set_default BDC_LEARNING_RATE 3e-5
  set_default BDC_WEIGHT_DECAY 1e-5
  set_default BDC_DROPOUT 0.1
  set_default BDC_USE_INSTRUMENT_FEATURE 0
  set_default BDC_USE_MARKET_RELATIVE_FEATURES 0
  set_default BDC_USE_MARKET_BREADTH_FEATURES 0
  set_default BDC_USE_RANK_MOMENTUM_FEATURES 0
  set_default BDC_USE_RANK_RISKADJ_FEATURES 0
  set_default BDC_USE_TREND_QUALITY_FEATURES 0
  set_default BDC_USE_CLEAN_RISK_FEATURES 0
  set_default BDC_USE_MULTI_PERIOD_FEATURES 0
  set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
  set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
  set_default BDC_OPTIMIZER adamw
  set_default BDC_LR_SCHEDULER plateau
  set_default BDC_TORCH_NUM_THREADS 4
  set_default BDC_TENSORBOARD 0
}

dry_run_requested() {
  for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
      return 0
    fi
  done
  return 1
}

print_rank_replace_dry_run() {
  output_path="./output/result.csv"
  previous_arg=""
  for arg in "$@"; do
    if [ "$previous_arg" = "--output" ]; then
      output_path="$arg"
    fi
    previous_arg="$arg"
  done
  echo "提交预测模式: rank-replace"
  echo "最终输出: $output_path"
  for key in \
    BDC_OUTPUT_DIR BDC_FEATURE_NUM BDC_SEQUENCE_LENGTH BDC_D_MODEL BDC_NHEAD BDC_NUM_LAYERS \
    BDC_DIM_FEEDFORWARD BDC_BATCH_SIZE BDC_USE_INSTRUMENT_FEATURE BDC_USE_MARKET_RELATIVE_FEATURES \
    BDC_USE_MARKET_BREADTH_FEATURES BDC_USE_RANK_MOMENTUM_FEATURES BDC_USE_RANK_RISKADJ_FEATURES \
    BDC_USE_TREND_QUALITY_FEATURES \
    BDC_USE_CLEAN_RISK_FEATURES BDC_USE_MULTI_PERIOD_FEATURES \
    BDC_USE_CROSS_SECTIONAL_RANKS BDC_CROSS_SECTIONAL_RANK_MODE \
    BDC_NUM_PROCESSES BDC_TORCH_NUM_THREADS
  do
    eval "value=\${$key:-}"
    echo "  $key=$value"
  done
}

print_single_dry_run() {
  output_path="./output/result.csv"
  previous_arg=""
  for arg in "$@"; do
    if [ "$previous_arg" = "--output" ]; then
      output_path="$arg"
    fi
    previous_arg="$arg"
  done
  echo "提交预测模式: single"
  echo "说明: 使用 code/src/predict.py 的原始单模型配置，不套用 rank-replace 默认参数"
  echo "最终输出: $output_path"
  if [ -n "$debug_arg" ]; then
    echo "调试模式: 开启"
  fi
}

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

submission_mode="${BDC_SUBMISSION_MODE:-rank-replace}"
case "${1:-}" in
  ensemble|--ensemble)
    submission_mode="ensemble"
    shift
    ;;
  single|--single)
    submission_mode="single"
    shift
    ;;
  rank|rank-replace|rank_replace|--rank|--rank-replace|--rank_replace)
    submission_mode="rank-replace"
    shift
    ;;
esac

if [ "$submission_mode" = "rank-replace" ]; then
  apply_rank_replace_defaults
  if dry_run_requested "$@"; then
    print_rank_replace_dry_run "$@"
    exit 0
  fi
  # shellcheck disable=SC2086
  exec $python_bin code/src/predict.py "$@"
fi

if [ "$submission_mode" = "single" ]; then
  if dry_run_requested "$@"; then
    print_single_dry_run "$@"
    exit 0
  fi
  # shellcheck disable=SC2086
  exec $python_bin code/src/predict.py "$@"
fi

if [ -n "$debug_arg" ]; then
  # shellcheck disable=SC2086
  exec $python_bin code/src/ensemble_predict.py "$debug_arg" "$@"
fi

# shellcheck disable=SC2086
exec $python_bin code/src/ensemble_predict.py "$@"
