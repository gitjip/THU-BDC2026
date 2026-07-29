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

sync_cli_window_env() {
  previous_arg=""
  for arg in "$@"; do
    if [ "$previous_arg" = "--windows" ]; then
      export BDC_WF_WINDOWS="$arg"
      return
    fi
    case "$arg" in
      --windows=*)
        export BDC_WF_WINDOWS="${arg#--windows=}"
        return
        ;;
    esac
    previous_arg="$arg"
  done
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
  "")
    ;;
  debug|fast|quick|lite|--debug|--fast|--quick)
    profile="quick"
    shift
    ;;
  balanced|--balanced)
    profile="balanced"
    shift
    ;;
  noid|no-id|no_instrument|--noid|--no-id|--no-instrument)
    profile="noid"
    shift
    ;;
  noid-lowvol|lowvol|low-vol|--noid-lowvol|--lowvol|--low-vol)
    profile="noid-lowvol"
    shift
    ;;
  ensemble-lowvol|ensemble|--ensemble-lowvol|--ensemble)
    profile="ensemble-lowvol"
    shift
    ;;
  ensemble-lowoverheat|ensemble-low-overheat|ensemble-overheat|--ensemble-lowoverheat|--ensemble-low-overheat|--ensemble-overheat)
    profile="ensemble-lowoverheat"
    shift
    ;;
  noid-marketrel|marketrel|--noid-marketrel|--marketrel)
    profile="noid-marketrel"
    shift
    ;;
  noid-rank|--noid-rank)
    profile="noid-rank"
    shift
    ;;
  noid-rank-lite|rank-lite|--noid-rank-lite|--rank-lite)
    profile="noid-rank-lite"
    shift
    ;;
  noid-rank-replace|--noid-rank-replace)
    profile="noid-rank-replace"
    shift
    ;;
  noid-rank-lgbm|rank-lgbm|lgbm|--noid-rank-lgbm|--rank-lgbm|--lgbm)
    profile="noid-rank-lgbm"
    shift
    ;;
  noid-rank-sharp|rank-sharp|--noid-rank-sharp|--rank-sharp)
    profile="noid-rank-sharp"
    shift
    ;;
  noid-rank-trendq|rank-trendq|trendq|--noid-rank-trendq|--rank-trendq|--trendq)
    profile="noid-rank-trendq"
    shift
    ;;
  noid-rank-cleanrisk|rank-cleanrisk|cleanrisk|--noid-rank-cleanrisk|--rank-cleanrisk|--cleanrisk)
    profile="noid-rank-cleanrisk"
    shift
    ;;
  noid-rank-multiperiod|rank-multiperiod|multiperiod|multi-period|--noid-rank-multiperiod|--rank-multiperiod|--multiperiod|--multi-period)
    profile="noid-rank-multiperiod"
    shift
    ;;
  noid-rank-breadth|rank-breadth|breadth|market-breadth|--noid-rank-breadth|--rank-breadth|--breadth|--market-breadth)
    profile="noid-rank-breadth"
    shift
    ;;
  noid-rank-marketenv|rank-marketenv|marketenv|market-env|--noid-rank-marketenv|--rank-marketenv|--marketenv|--market-env)
    profile="noid-rank-marketenv"
    shift
    ;;
  noid-rank-marketenv-lite|rank-marketenv-lite|marketenv-lite|market-env-lite|--noid-rank-marketenv-lite|--rank-marketenv-lite|--marketenv-lite|--market-env-lite)
    profile="noid-rank-marketenv-lite"
    shift
    ;;
  noid-rank-marketenv-roll|rank-marketenv-roll|marketenv-roll|market-env-roll|marketenv-rolling|market-env-rolling|--noid-rank-marketenv-roll|--rank-marketenv-roll|--marketenv-roll|--market-env-roll|--marketenv-rolling|--market-env-rolling)
    profile="noid-rank-marketenv-roll"
    shift
    ;;
  noid-rank-momdelta|rank-momdelta|momdelta|rank-momentum|--noid-rank-momdelta|--rank-momdelta|--momdelta|--rank-momentum)
    profile="noid-rank-momdelta"
    shift
    ;;
  noid-rank-riskadj|rank-riskadj|riskadj|rank-risk|--noid-rank-riskadj|--rank-riskadj|--riskadj|--rank-risk)
    profile="noid-rank-riskadj"
    shift
    ;;
  noid-rank-ret5rank|rank-ret5rank|ret5rank|ret5-rank|--noid-rank-ret5rank|--rank-ret5rank|--ret5rank|--ret5-rank)
    profile="noid-rank-ret5rank"
    shift
    ;;
  noid-rank-overheatguard|rank-overheatguard|overheatguard|overheat-guard|--noid-rank-overheatguard|--rank-overheatguard|--overheatguard|--overheat-guard)
    profile="noid-rank-overheatguard"
    shift
    ;;
  smooth|lookahead|--smooth|--lookahead)
    profile="smooth"
    shift
    ;;
  stable|regularized|--stable|--regularized)
    profile="stable"
    shift
    ;;
  large|slow|candidate|--large|--slow|--candidate)
    profile="large"
    shift
    ;;
  full|--full)
    profile="full"
    shift
    ;;
  --*)
    ;;
  *)
    profile="$1"
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
    set_default BDC_NUM_EPOCHS 4
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 2
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 4
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  balanced)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 15
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE append
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-lite)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_CROSS_SECTIONAL_RANK_REPLACE_SET lite
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-replace)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-lgbm)
    set_default BDC_MODEL_KIND lgbm
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 120
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 0
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    set_default BDC_LGBM_N_ESTIMATORS 300
    set_default BDC_LGBM_LEARNING_RATE 0.03
    set_default BDC_LGBM_NUM_LEAVES 31
    set_default BDC_LGBM_MIN_CHILD_SAMPLES 20
    set_default BDC_LGBM_SUBSAMPLE 0.9
    set_default BDC_LGBM_COLSAMPLE_BYTREE 0.9
    set_default BDC_LGBM_REG_ALPHA 0.0
    set_default BDC_LGBM_REG_LAMBDA 1.0
    set_default BDC_LGBM_NUM_THREADS 8
    ;;
  noid-rank-sharp)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_LOSS_TARGET_TEMPERATURE 0.05
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-trendq)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_TREND_QUALITY_FEATURES 1
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-cleanrisk)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_CLEAN_RISK_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-multiperiod)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_MULTI_PERIOD_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-breadth)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_MARKET_BREADTH_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-marketenv)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_MARKET_ENV_FEATURES 1
    set_default BDC_MARKET_ENV_FEATURE_SET full
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-marketenv-lite)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_MARKET_ENV_FEATURES 1
    set_default BDC_MARKET_ENV_FEATURE_SET lite
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-marketenv-roll)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_MARKET_ENV_FEATURES 1
    set_default BDC_MARKET_ENV_FEATURE_SET rolling
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-momdelta)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_RANK_MOMENTUM_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-riskadj)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_RANK_RISKADJ_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-ret5rank)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_RET5_RANK_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-rank-overheatguard)
    set_default BDC_WF_WINDOWS 6
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_CROSS_SECTIONAL_RANKS 1
    set_default BDC_CROSS_SECTIONAL_RANK_MODE replace
    set_default BDC_USE_SHORT_OVERHEAT_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-marketrel)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_USE_MARKET_RELATIVE_FEATURES 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-stable)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-4
    set_default BDC_DROPOUT 0.2
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-full)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 0
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 0
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  noid-lowvol)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 30
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 0
    set_default BDC_SELECTION_STRATEGY low_vol_then_rank_top5
    set_default BDC_STAGE2_POOL_SIZE 10
    set_default BDC_STAGE2_VOL_WINDOW 20
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  ensemble-lowvol)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_ENSEMBLE_SOURCES v1.2.13,v1.3.2
    set_default BDC_SELECTION_STRATEGY ensemble_low_vol_top5
    set_default BDC_STAGE2_VOL_WINDOW 20
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  ensemble-lowoverheat)
    set_default BDC_WF_WINDOWS 24
    set_default BDC_ENSEMBLE_SOURCES v1.4.5,v1.12.0
    set_default BDC_SELECTION_STRATEGY ensemble_low_overheat_top5
    set_default BDC_STAGE2_VOL_WINDOW 20
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  smooth)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 15
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER lookahead
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  stable)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 2
    set_default BDC_DIM_FEEDFORWARD 192
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 15
    set_default BDC_LEARNING_RATE 3e-5
    set_default BDC_WEIGHT_DECAY 1e-4
    set_default BDC_DROPOUT 0.2
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  large)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_FEATURE_NUM 39
    set_default BDC_SEQUENCE_LENGTH 45
    set_default BDC_TRAIN_TARGET_DAYS 60
    set_default BDC_VAL_DAYS 5
    set_default BDC_MAX_STOCKS_PER_DAY 120
    set_default BDC_D_MODEL 96
    set_default BDC_NHEAD 4
    set_default BDC_NUM_LAYERS 3
    set_default BDC_DIM_FEEDFORWARD 512
    set_default BDC_BATCH_SIZE 4
    set_default BDC_NUM_EPOCHS 20
    set_default BDC_LEARNING_RATE 5e-6
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-7
    set_default BDC_EARLY_STOPPING_PATIENCE 5
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 14
    set_default BDC_TENSORBOARD 0
    ;;
  full)
    set_default BDC_WF_WINDOWS 3
    set_default BDC_VAL_DAYS 5
    set_default BDC_NUM_EPOCHS 6
    set_default BDC_LEARNING_RATE 2e-5
    set_default BDC_WEIGHT_DECAY 1e-5
    set_default BDC_DROPOUT 0.1
    set_default BDC_USE_INSTRUMENT_FEATURE 1
    set_default BDC_OPTIMIZER adamw
    set_default BDC_LOOKAHEAD_K 5
    set_default BDC_LOOKAHEAD_ALPHA 0.5
    set_default BDC_LR_SCHEDULER plateau
    set_default BDC_LR_PATIENCE 1
    set_default BDC_LR_FACTOR 0.5
    set_default BDC_LR_THRESHOLD 1e-4
    set_default BDC_MIN_LR 1e-6
    set_default BDC_EARLY_STOPPING_PATIENCE 3
    set_default BDC_EARLY_STOPPING_MIN_DELTA 1e-4
    set_default BDC_GRAD_CLIP 1
    set_default BDC_NUM_PROCESSES 6
    set_default BDC_TORCH_NUM_THREADS 4
    set_default BDC_TENSORBOARD 0
    ;;
  *)
    echo "未知 tune profile: $profile，可选 quick、balanced、noid、noid-lowvol、ensemble-lowvol、ensemble-lowoverheat、noid-marketrel、noid-rank、noid-rank-lite、noid-rank-replace、noid-rank-lgbm、noid-rank-sharp、noid-rank-trendq、noid-rank-cleanrisk、noid-rank-multiperiod、noid-rank-breadth、noid-rank-marketenv、noid-rank-marketenv-lite、noid-rank-marketenv-roll、noid-rank-momdelta、noid-rank-riskadj、noid-rank-ret5rank、noid-rank-overheatguard、noid-stable、noid-full、smooth、stable、large、full。" >&2
    exit 2
    ;;
esac

set_default BDC_USE_CROSS_SECTIONAL_RANKS 0
set_default BDC_USE_MARKET_RELATIVE_FEATURES 0
set_default BDC_USE_MARKET_BREADTH_FEATURES 0
set_default BDC_USE_MARKET_ENV_FEATURES 0
set_default BDC_USE_RANK_MOMENTUM_FEATURES 0
set_default BDC_USE_RANK_RISKADJ_FEATURES 0
set_default BDC_USE_RET5_RANK_FEATURES 0
set_default BDC_USE_SHORT_OVERHEAT_FEATURES 0
set_default BDC_USE_TREND_QUALITY_FEATURES 0
set_default BDC_USE_CLEAN_RISK_FEATURES 0
set_default BDC_USE_MULTI_PERIOD_FEATURES 0

sync_cli_window_env "$@"

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
