from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleSource:
    label: str
    output_dir: str
    env: dict[str, str]


OFFICIAL_BASE_ENV = {
    "BDC_FEATURE_NUM": "39",
    "BDC_SEQUENCE_LENGTH": "45",
    "BDC_TRAIN_TARGET_DAYS": "60",
    "BDC_VAL_DAYS": "5",
    "BDC_MAX_STOCKS_PER_DAY": "120",
    "BDC_D_MODEL": "96",
    "BDC_NHEAD": "4",
    "BDC_NUM_LAYERS": "2",
    "BDC_DIM_FEEDFORWARD": "192",
    "BDC_BATCH_SIZE": "4",
    "BDC_LEARNING_RATE": "3e-5",
    "BDC_WEIGHT_DECAY": "1e-5",
    "BDC_DROPOUT": "0.1",
    "BDC_USE_INSTRUMENT_FEATURE": "0",
    "BDC_USE_MARKET_RELATIVE_FEATURES": "0",
    "BDC_USE_MARKET_BREADTH_FEATURES": "0",
    "BDC_USE_RANK_MOMENTUM_FEATURES": "0",
    "BDC_USE_RANK_RISKADJ_FEATURES": "0",
    "BDC_USE_RET5_RANK_FEATURES": "0",
    "BDC_OPTIMIZER": "adamw",
    "BDC_LR_SCHEDULER": "plateau",
    "BDC_LR_PATIENCE": "1",
    "BDC_LR_FACTOR": "0.5",
    "BDC_LR_THRESHOLD": "1e-4",
    "BDC_MIN_LR": "1e-6",
    "BDC_EARLY_STOPPING_PATIENCE": "5",
    "BDC_EARLY_STOPPING_MIN_DELTA": "1e-4",
    "BDC_GRAD_CLIP": "1",
    "BDC_NUM_PROCESSES": "6",
    "BDC_TORCH_NUM_THREADS": "4",
    "BDC_TENSORBOARD": "0",
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
}


def build_source_env(label: str, output_dir: str, base_env: dict[str, str], extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        **base_env,
        "BDC_OUTPUT_DIR": output_dir,
        "BDC_SELECTION_STRATEGY": "model_top5",
        "BDC_STAGE2_POOL_SIZE": "5",
        "BDC_USE_CROSS_SECTIONAL_RANKS": "0",
        "BDC_CROSS_SECTIONAL_RANK_MODE": "off",
        "BDC_TUNE_PROFILE": label,
    }
    if extra_env:
        env.update(extra_env)
    return env


def get_submission_ensemble_sources(debug: bool = False) -> list[EnsembleSource]:
    base_env = DEBUG_BASE_ENV if debug else OFFICIAL_BASE_ENV
    model_root = "./model/ensemble_debug" if debug else "./model/ensemble"
    noid_epochs = "4" if debug else "30"
    rank_epochs = "4" if debug else "30"

    return [
        EnsembleSource(
            label="noid",
            output_dir=f"{model_root}/noid",
            env=build_source_env(
                "submission-noid",
                f"{model_root}/noid",
                base_env,
                {"BDC_NUM_EPOCHS": noid_epochs},
            ),
        ),
        EnsembleSource(
            label="noid_rank_replace",
            output_dir=f"{model_root}/noid_rank_replace",
            env=build_source_env(
                "submission-noid-rank-replace",
                f"{model_root}/noid_rank_replace",
                base_env,
                {
                    "BDC_NUM_EPOCHS": rank_epochs,
                    "BDC_USE_CROSS_SECTIONAL_RANKS": "1",
                    "BDC_CROSS_SECTIONAL_RANK_MODE": "replace",
                    "BDC_CROSS_SECTIONAL_RANK_REPLACE_SET": "default",
                },
            ),
        ),
    ]


ENSEMBLE_SELECTION_STRATEGY = "ensemble_low_vol_top5"
ENSEMBLE_VOL_WINDOW = 20
