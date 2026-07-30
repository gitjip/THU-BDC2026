import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleSource:
    label: str
    output_dir: str
    env: dict[str, str]


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value in {"", "default"}:
        value = default
    if value not in allowed:
        raise ValueError(f"Unsupported {name}: {value}. Allowed: {sorted(allowed)}")
    return value


SUBMISSION_STRENGTH = _env_choice(
    "BDC_SUBMISSION_STRENGTH",
    "strong",
    {"validated", "strong", "max"},
)

OFFICIAL_BASE_ENV = {
    "BDC_FEATURE_NUM": "39",
    "BDC_SEQUENCE_LENGTH": "45",
    "BDC_VAL_DAYS": "5",
    "BDC_USE_INSTRUMENT_FEATURE": "0",
    "BDC_USE_MARKET_RELATIVE_FEATURES": "0",
    "BDC_USE_MARKET_BREADTH_FEATURES": "0",
    "BDC_USE_MARKET_ENV_FEATURES": "0",
    "BDC_USE_RANK_MOMENTUM_FEATURES": "0",
    "BDC_USE_RANK_RISKADJ_FEATURES": "0",
    "BDC_USE_RET5_RANK_FEATURES": "0",
    "BDC_USE_SHORT_OVERHEAT_FEATURES": "0",
    "BDC_USE_TREND_QUALITY_FEATURES": "0",
    "BDC_USE_CLEAN_RISK_FEATURES": "0",
    "BDC_USE_MULTI_PERIOD_FEATURES": "0",
    "BDC_USE_CROSS_SECTIONAL_RANKS": "1",
    "BDC_CROSS_SECTIONAL_RANK_MODE": "replace",
    "BDC_CROSS_SECTIONAL_RANK_REPLACE_SET": "default",
    "BDC_NUM_PROCESSES": "6",
    "BDC_TORCH_NUM_THREADS": "4",
    "BDC_TENSORBOARD": "0",
    "BDC_TOP_K": "5",
    "BDC_TOTAL_EXPOSURE": "1.0",
}

TRANSFORMER_ENV_BY_STRENGTH = {
    # v1.13.0/v1.13.4 同口径源模型配置，保留给最终复核或紧急回退。
    "validated": {
        "BDC_TRAIN_TARGET_DAYS": "60",
        "BDC_MAX_STOCKS_PER_DAY": "120",
        "BDC_D_MODEL": "96",
        "BDC_NHEAD": "4",
        "BDC_NUM_LAYERS": "2",
        "BDC_DIM_FEEDFORWARD": "192",
        "BDC_BATCH_SIZE": "4",
        "BDC_NUM_EPOCHS": "30",
        "BDC_EARLY_STOPPING_PATIENCE": "5",
    },
    # 正式默认：主要增加训练覆盖面，不大幅改架构，避免 8GB 显存和 8 小时复现风险。
    "strong": {
        "BDC_TRAIN_TARGET_DAYS": "0",
        "BDC_MAX_STOCKS_PER_DAY": "0",
        "BDC_D_MODEL": "96",
        "BDC_NHEAD": "4",
        "BDC_NUM_LAYERS": "2",
        "BDC_DIM_FEEDFORWARD": "192",
        "BDC_BATCH_SIZE": "4",
        "BDC_NUM_EPOCHS": "40",
        "BDC_EARLY_STOPPING_PATIENCE": "8",
    },
    # 只建议在正式前本地计时确认后启用。
    "max": {
        "BDC_TRAIN_TARGET_DAYS": "0",
        "BDC_MAX_STOCKS_PER_DAY": "0",
        "BDC_D_MODEL": "128",
        "BDC_NHEAD": "4",
        "BDC_NUM_LAYERS": "2",
        "BDC_DIM_FEEDFORWARD": "256",
        "BDC_BATCH_SIZE": "4",
        "BDC_NUM_EPOCHS": "60",
        "BDC_EARLY_STOPPING_PATIENCE": "10",
    },
}

TRANSFORMER_OPTIM_ENV = {
    "BDC_MODEL_KIND": "transformer",
    "BDC_LEARNING_RATE": "3e-5",
    "BDC_WEIGHT_DECAY": "1e-5",
    "BDC_DROPOUT": "0.1",
    "BDC_OPTIMIZER": "adamw",
    "BDC_LR_SCHEDULER": "plateau",
    "BDC_LR_PATIENCE": "1",
    "BDC_LR_FACTOR": "0.5",
    "BDC_LR_THRESHOLD": "1e-4",
    "BDC_MIN_LR": "1e-6",
    "BDC_EARLY_STOPPING_MIN_DELTA": "1e-4",
    "BDC_LOSS_TEMPERATURE": "1.0",
    "BDC_LOSS_TARGET_TEMPERATURE": "1.0",
    "BDC_GRAD_CLIP": "1",
}

LGBM_ENV_BY_STRENGTH = {
    "validated": {
        "BDC_TRAIN_TARGET_DAYS": "120",
        "BDC_LGBM_N_ESTIMATORS": "300",
    },
    "strong": {
        "BDC_TRAIN_TARGET_DAYS": "240",
        "BDC_LGBM_N_ESTIMATORS": "600",
    },
    "max": {
        "BDC_TRAIN_TARGET_DAYS": "0",
        "BDC_LGBM_N_ESTIMATORS": "900",
    },
}

LGBM_BASE_ENV = {
    "BDC_MODEL_KIND": "lgbm",
    "BDC_MAX_STOCKS_PER_DAY": "0",
    "BDC_LGBM_LEARNING_RATE": "0.03",
    "BDC_LGBM_NUM_LEAVES": "31",
    "BDC_LGBM_MIN_CHILD_SAMPLES": "20",
    "BDC_LGBM_SUBSAMPLE": "0.9",
    "BDC_LGBM_COLSAMPLE_BYTREE": "0.9",
    "BDC_LGBM_REG_ALPHA": "0.0",
    "BDC_LGBM_REG_LAMBDA": "1.0",
    "BDC_LGBM_NUM_THREADS": "8",
}

DEBUG_BASE_ENV = {
    **OFFICIAL_BASE_ENV,
    "BDC_FAST_DEV": "1",
    "BDC_SEQUENCE_LENGTH": "30",
    "BDC_TRAIN_TARGET_DAYS": "24",
    "BDC_VAL_DAYS": "5",
    "BDC_MAX_STOCKS_PER_DAY": "60",
    "BDC_D_MODEL": "64",
    "BDC_NHEAD": "2",
    "BDC_NUM_LAYERS": "1",
    "BDC_DIM_FEEDFORWARD": "128",
    "BDC_BATCH_SIZE": "8",
    "BDC_NUM_EPOCHS": "4",
    "BDC_EARLY_STOPPING_PATIENCE": "2",
    "BDC_NUM_PROCESSES": "4",
    "BDC_LGBM_N_ESTIMATORS": "120",
}


def build_source_env(
    label: str,
    output_dir: str,
    base_env: dict[str, str],
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = {
        **base_env,
        "BDC_OUTPUT_DIR": output_dir,
        "BDC_SELECTION_STRATEGY": "model_top5",
        "BDC_STAGE2_POOL_SIZE": "5",
        "BDC_TUNE_PROFILE": label,
    }
    if extra_env:
        env.update(extra_env)
    return env


def get_submission_ensemble_sources(debug: bool = False) -> list[EnsembleSource]:
    model_root = os.environ.get(
        "BDC_ENSEMBLE_MODEL_ROOT",
        "./model/ensemble_debug" if debug else "./model/submission",
    )

    if debug:
        primary_env = {
            **DEBUG_BASE_ENV,
            **TRANSFORMER_OPTIM_ENV,
        }
        lgbm_env = {
            **DEBUG_BASE_ENV,
            **LGBM_BASE_ENV,
            "BDC_TRAIN_TARGET_DAYS": "24",
        }
    else:
        primary_env = {
            **OFFICIAL_BASE_ENV,
            **TRANSFORMER_OPTIM_ENV,
            **TRANSFORMER_ENV_BY_STRENGTH[SUBMISSION_STRENGTH],
        }
        lgbm_env = {
            **OFFICIAL_BASE_ENV,
            **LGBM_BASE_ENV,
            **LGBM_ENV_BY_STRENGTH[SUBMISSION_STRENGTH],
        }

    return [
        EnsembleSource(
            label="primary_rank_replace",
            output_dir=f"{model_root}/primary_rank_replace",
            env=build_source_env(
                "submission-primary-rank-replace",
                f"{model_root}/primary_rank_replace",
                primary_env,
            ),
        ),
        EnsembleSource(
            label="lgbm_rank_replace",
            output_dir=f"{model_root}/lgbm_rank_replace",
            env=build_source_env(
                "submission-lgbm-rank-replace",
                f"{model_root}/lgbm_rank_replace",
                lgbm_env,
            ),
        ),
    ]


ENSEMBLE_SELECTION_STRATEGY = "ensemble_gate_overheat_top5"
ENSEMBLE_GATE_OVERHEAT_THRESHOLD = 0.65
ENSEMBLE_VOL_WINDOW = 20
