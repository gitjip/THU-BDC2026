import os


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


# 配置参数
fast_dev_mode = _env_bool("BDC_FAST_DEV", False)
sequence_length = _env_int("BDC_SEQUENCE_LENGTH", 30 if fast_dev_mode else 60)
feature_num = os.environ.get("BDC_FEATURE_NUM", "39" if fast_dev_mode else "158+39")
use_market_relative_features = _env_bool("BDC_USE_MARKET_RELATIVE_FEATURES", False)
use_market_breadth_features = _env_bool("BDC_USE_MARKET_BREADTH_FEATURES", False)
use_market_env_features = _env_bool("BDC_USE_MARKET_ENV_FEATURES", False)
market_env_feature_set = (
    os.environ.get("BDC_MARKET_ENV_FEATURE_SET", "full").strip().lower()
)
if market_env_feature_set in {"", "default"}:
    market_env_feature_set = "full"
if market_env_feature_set not in {"full", "lite", "rolling"}:
    raise ValueError(
        f"Unsupported BDC_MARKET_ENV_FEATURE_SET: {market_env_feature_set}"
    )
model_kind = os.environ.get("BDC_MODEL_KIND", "transformer").strip().lower()
if model_kind in {"", "torch"}:
    model_kind = "transformer"
if model_kind not in {"transformer", "lgbm"}:
    raise ValueError(f"Unsupported BDC_MODEL_KIND: {model_kind}")
use_rank_momentum_features = _env_bool("BDC_USE_RANK_MOMENTUM_FEATURES", False)
use_rank_riskadj_features = _env_bool("BDC_USE_RANK_RISKADJ_FEATURES", False)
use_ret5_rank_features = _env_bool("BDC_USE_RET5_RANK_FEATURES", False)
use_short_overheat_features = _env_bool("BDC_USE_SHORT_OVERHEAT_FEATURES", False)
use_trend_quality_features = _env_bool("BDC_USE_TREND_QUALITY_FEATURES", False)
use_clean_risk_features = _env_bool("BDC_USE_CLEAN_RISK_FEATURES", False)
use_multi_period_features = _env_bool("BDC_USE_MULTI_PERIOD_FEATURES", False)
selection_strategy = (
    os.environ.get("BDC_SELECTION_STRATEGY", "model_top5").strip().lower()
)
top_k = _env_int("BDC_TOP_K", 5)
if not 1 <= top_k <= 5:
    raise ValueError(f"BDC_TOP_K must be between 1 and 5, current: {top_k}")
total_exposure = _env_float("BDC_TOTAL_EXPOSURE", 1.0)
if not 0.0 <= total_exposure <= 1.0:
    raise ValueError(
        f"BDC_TOTAL_EXPOSURE must be between 0 and 1, current: {total_exposure}"
    )
loss_temperature = _env_float("BDC_LOSS_TEMPERATURE", 1.0)
if loss_temperature <= 0:
    raise ValueError(
        f"BDC_LOSS_TEMPERATURE must be positive, current: {loss_temperature}"
    )
loss_target_temperature = _env_float("BDC_LOSS_TARGET_TEMPERATURE", loss_temperature)
if loss_target_temperature <= 0:
    raise ValueError(
        f"BDC_LOSS_TARGET_TEMPERATURE must be positive, current: {loss_target_temperature}"
    )
use_cross_sectional_rank_features_flag = _env_bool(
    "BDC_USE_CROSS_SECTIONAL_RANKS", False
)
rank_mode_value = os.environ.get("BDC_CROSS_SECTIONAL_RANK_MODE")
if rank_mode_value in (None, ""):
    cross_sectional_rank_mode = (
        "append" if use_cross_sectional_rank_features_flag else "off"
    )
else:
    cross_sectional_rank_mode = rank_mode_value.strip().lower()
if cross_sectional_rank_mode in {"0", "false", "none"}:
    cross_sectional_rank_mode = "off"
if cross_sectional_rank_mode == "add":
    cross_sectional_rank_mode = "append"
if cross_sectional_rank_mode == "substitute":
    cross_sectional_rank_mode = "replace"
if cross_sectional_rank_mode not in {"off", "append", "replace"}:
    raise ValueError(
        f"Unsupported BDC_CROSS_SECTIONAL_RANK_MODE: {cross_sectional_rank_mode}"
    )
cross_sectional_rank_replace_set = (
    os.environ.get("BDC_CROSS_SECTIONAL_RANK_REPLACE_SET", "default").strip().lower()
)
if cross_sectional_rank_replace_set in {"", "full"}:
    cross_sectional_rank_replace_set = "default"
if cross_sectional_rank_replace_set not in {"default", "lite"}:
    raise ValueError(
        f"Unsupported BDC_CROSS_SECTIONAL_RANK_REPLACE_SET: {cross_sectional_rank_replace_set}"
    )
use_cross_sectional_rank_features = cross_sectional_rank_mode != "off"
feature_dir_label = feature_num
if use_cross_sectional_rank_features:
    rank_label = (
        "rank"
        if cross_sectional_rank_mode == "append"
        else f"rank_{cross_sectional_rank_mode}"
    )
    if (
        cross_sectional_rank_mode == "replace"
        and cross_sectional_rank_replace_set != "default"
    ):
        rank_label = f"{rank_label}_{cross_sectional_rank_replace_set}"
    feature_dir_label = f"{feature_dir_label}_{rank_label}"
if use_market_relative_features:
    feature_dir_label = f"{feature_dir_label}_mktrel"
if use_market_breadth_features:
    feature_dir_label = f"{feature_dir_label}_breadth"
if use_market_env_features:
    env_label = (
        "marketenv"
        if market_env_feature_set == "full"
        else f"marketenv_{market_env_feature_set}"
    )
    feature_dir_label = f"{feature_dir_label}_{env_label}"
if use_rank_momentum_features:
    feature_dir_label = f"{feature_dir_label}_rankmom"
if use_rank_riskadj_features:
    feature_dir_label = f"{feature_dir_label}_riskadj"
if use_ret5_rank_features:
    feature_dir_label = f"{feature_dir_label}_ret5rank"
if use_short_overheat_features:
    feature_dir_label = f"{feature_dir_label}_overheatguard"
if use_trend_quality_features:
    feature_dir_label = f"{feature_dir_label}_trendq"
if use_clean_risk_features:
    feature_dir_label = f"{feature_dir_label}_cleanrisk"
if use_multi_period_features:
    feature_dir_label = f"{feature_dir_label}_multiperiod"
output_dir_prefix = "debug_" if fast_dev_mode else ""
if model_kind != "transformer":
    feature_dir_label = f"{feature_dir_label}_{model_kind}"
config = {
    "model_kind": model_kind,
    "sequence_length": sequence_length,  # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    "label_horizon": 5,  # 标签为未来第5个交易日相对未来第1个交易日的收益
    "prediction_horizon": 5,
    "submission_deadline_date": os.environ.get("BDC_SUBMISSION_DATE", "2026-08-02"),
    "market_holidays": os.environ.get("BDC_MARKET_HOLIDAYS", ""),
    "val_days": _env_int("BDC_VAL_DAYS", 5),
    "train_target_days": _env_int("BDC_TRAIN_TARGET_DAYS", 24 if fast_dev_mode else 0),
    "max_stocks_per_day": _env_int(
        "BDC_MAX_STOCKS_PER_DAY", 60 if fast_dev_mode else 0
    ),
    "fast_dev_mode": fast_dev_mode,
    "d_model": _env_int(
        "BDC_D_MODEL", 64 if fast_dev_mode else 256
    ),  # Transformer输入维度
    "nhead": _env_int("BDC_NHEAD", 2 if fast_dev_mode else 4),  # 注意力头数量
    "num_layers": _env_int(
        "BDC_NUM_LAYERS", 1 if fast_dev_mode else 3
    ),  # Transformer层数
    "dim_feedforward": _env_int(
        "BDC_DIM_FEEDFORWARD", 128 if fast_dev_mode else 512
    ),  # 前馈网络维度
    "batch_size": _env_int(
        "BDC_BATCH_SIZE", 8 if fast_dev_mode else 4
    ),  # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    "num_epochs": _env_int(
        "BDC_NUM_EPOCHS", 4 if fast_dev_mode else 6
    ),  # 最大epoch数，早停可能提前结束
    "learning_rate": _env_float("BDC_LEARNING_RATE", 3e-5 if fast_dev_mode else 2e-5),
    "weight_decay": _env_float("BDC_WEIGHT_DECAY", 1e-5),
    "optimizer": os.environ.get("BDC_OPTIMIZER", "adamw"),
    "lookahead_k": _env_int("BDC_LOOKAHEAD_K", 5),
    "lookahead_alpha": _env_float("BDC_LOOKAHEAD_ALPHA", 0.5),
    "lr_scheduler": os.environ.get("BDC_LR_SCHEDULER", "plateau"),
    "lr_factor": _env_float("BDC_LR_FACTOR", 0.5),
    "lr_patience": _env_int("BDC_LR_PATIENCE", 1),
    "lr_threshold": _env_float("BDC_LR_THRESHOLD", 1e-4),
    "min_learning_rate": _env_float("BDC_MIN_LR", 1e-6),
    "early_stopping_patience": _env_int(
        "BDC_EARLY_STOPPING_PATIENCE", 2 if fast_dev_mode else 3
    ),
    "early_stopping_min_delta": _env_float("BDC_EARLY_STOPPING_MIN_DELTA", 1e-4),
    "dropout": _env_float("BDC_DROPOUT", 0.1),
    "use_instrument_feature": _env_bool("BDC_USE_INSTRUMENT_FEATURE", True),
    "use_market_relative_features": use_market_relative_features,
    "use_market_breadth_features": use_market_breadth_features,
    "use_market_env_features": use_market_env_features,
    "market_env_feature_set": market_env_feature_set,
    "use_rank_momentum_features": use_rank_momentum_features,
    "use_rank_riskadj_features": use_rank_riskadj_features,
    "use_ret5_rank_features": use_ret5_rank_features,
    "use_short_overheat_features": use_short_overheat_features,
    "use_trend_quality_features": use_trend_quality_features,
    "use_clean_risk_features": use_clean_risk_features,
    "use_multi_period_features": use_multi_period_features,
    "selection_strategy": selection_strategy,
    "top_k": top_k,
    "total_exposure": total_exposure,
    "stage2_pool_size": _env_int("BDC_STAGE2_POOL_SIZE", 10),
    "stage2_vol_window": _env_int("BDC_STAGE2_VOL_WINDOW", 20),
    "use_cross_sectional_rank_features": use_cross_sectional_rank_features,
    "cross_sectional_rank_mode": cross_sectional_rank_mode,
    "cross_sectional_rank_replace_set": cross_sectional_rank_replace_set,
    "feature_num": feature_num,
    "max_grad_norm": 5.0,
    "enable_grad_clip": _env_bool("BDC_GRAD_CLIP", True),
    "seed": 42,
    "num_processes": _env_int("BDC_NUM_PROCESSES", 4 if fast_dev_mode else 6),
    "torch_num_threads": _env_int("BDC_TORCH_NUM_THREADS", 0),
    "enable_tensorboard": _env_bool("BDC_TENSORBOARD", not fast_dev_mode),
    "pairwise_weight": 1,  # 配对损失权重
    "base_weight": 1.0,  # 非top-k样本权重
    "top5_weight": 2.0,  # top-5样本权重（应大于base_weight）
    "loss_temperature": loss_temperature,
    "loss_target_temperature": loss_target_temperature,
    "lgbm_n_estimators": _env_int("BDC_LGBM_N_ESTIMATORS", 300),
    "lgbm_learning_rate": _env_float("BDC_LGBM_LEARNING_RATE", 0.03),
    "lgbm_num_leaves": _env_int("BDC_LGBM_NUM_LEAVES", 31),
    "lgbm_min_child_samples": _env_int("BDC_LGBM_MIN_CHILD_SAMPLES", 20),
    "lgbm_subsample": _env_float("BDC_LGBM_SUBSAMPLE", 0.9),
    "lgbm_colsample_bytree": _env_float("BDC_LGBM_COLSAMPLE_BYTREE", 0.9),
    "lgbm_reg_alpha": _env_float("BDC_LGBM_REG_ALPHA", 0.0),
    "lgbm_reg_lambda": _env_float("BDC_LGBM_REG_LAMBDA", 1.0),
    "lgbm_num_threads": _env_int("BDC_LGBM_NUM_THREADS", 8),
    "output_dir": os.environ.get(
        "BDC_OUTPUT_DIR",
        f"./model/{output_dir_prefix}{sequence_length}_{feature_dir_label}",
    ),
    "data_path": "./data",
    "stock_data_file": None,  # 默认自动寻找 data/stock_data.csv 或 data/stock_data
    "prediction_output_path": "./output/result.csv",
}
