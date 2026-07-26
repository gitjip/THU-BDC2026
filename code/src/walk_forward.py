import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_utils import get_trading_dates, load_stock_data, normalize_stock_codes, setup_logging


SEMVER_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
DEFAULT_VERSION = "v1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
VERSION_FILE = REPO_ROOT / "VERSION"
logger = logging.getLogger("bdc.walk_forward")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def log_section(title: str) -> None:
    line = "=" * 72
    logger.info("\n%s\n%s\n%s", line, title, line)


@dataclass
class WalkForwardWindow:
    name: str
    as_of_date: str
    mock_submission_date: str
    target_start_date: str
    target_end_date: str
    target_dates: list[str]


def read_default_version() -> str:
    if VERSION_FILE.exists():
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        if version:
            return version
    return DEFAULT_VERSION


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def parse_args() -> argparse.Namespace:
    fast_dev = parse_bool_env("BDC_FAST_DEV", False)
    default_windows = parse_int_env("BDC_WF_WINDOWS", 2 if fast_dev else 3)

    parser = argparse.ArgumentParser(description="按语义版本运行多窗口 walk-forward 调参流程")
    parser.add_argument(
        "version",
        nargs="?",
        default=read_default_version(),
        help="实验语义版本号，例如 v1.0.0；默认读取 VERSION",
    )
    parser.add_argument("--windows", type=int, default=default_windows, help="walk-forward 窗口数量")
    parser.add_argument("--target-days", type=int, default=parse_int_env("BDC_WF_TARGET_DAYS", 5), help="每个窗口验证的连续交易日数量")
    parser.add_argument("--step-days", type=int, default=parse_int_env("BDC_WF_STEP_DAYS", 5), help="相邻窗口向前滚动的交易日步长")
    parser.add_argument("--data-file", default=os.environ.get("BDC_STOCK_DATA_FILE"), help="可选数据文件；默认按项目数据规则自动寻找")
    parser.add_argument("--skip-final", action="store_true", help="只跑 walk-forward，不训练最终模型")
    parser.add_argument("--publish-final", action="store_true", help="把最终预测复制到 output/result.csv；默认只保存在 experiments 下")
    parser.add_argument("--create-tag", action="store_true", help="流程成功后创建同名本地 Git tag；要求工作区无未提交改动")
    parser.add_argument("--resume", action="store_true", help="复用已有版本目录中已完成的窗口")
    parser.add_argument("--dry-run", action="store_true", help="只打印窗口计划，不执行训练、预测和打分")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not SEMVER_PATTERN.fullmatch(args.version):
        raise ValueError(f"版本号必须是 vMAJOR.MINOR.PATCH 格式，例如 v1.0.0。当前: {args.version}")
    if args.windows <= 0:
        raise ValueError("--windows 必须大于 0")
    if args.target_days != 5:
        raise ValueError("当前比赛提交窗口固定为连续 5 个交易日，请保持 --target-days 5")
    if args.step_days <= 0:
        raise ValueError("--step-days 必须大于 0")


def get_git_info() -> dict:
    def run_git(command: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *command],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def create_git_tag(version: str) -> None:
    git_info = get_git_info()
    if git_info["status_short"]:
        raise ValueError("创建 Git tag 前请先提交当前代码改动，确保 tag 能准确对应代码版本")

    existing = subprocess.run(
        ["git", "tag", "--list", version],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if existing:
        raise ValueError(f"Git tag 已存在: {version}")

    subprocess.run(["git", "tag", "-a", version, "-m", version], cwd=REPO_ROOT, check=True)
    logger.info("已创建本地 Git tag: %s", version)


def format_date_list(dates: pd.DatetimeIndex | list[str]) -> list[str]:
    return [pd.Timestamp(day).normalize().strftime("%Y-%m-%d") for day in dates]


def is_consecutive_trading_dates(
    trading_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex | list[str],
) -> bool:
    trading_date_list = format_date_list(trading_dates)
    target_date_list = format_date_list(target_dates)
    positions = {date: idx for idx, date in enumerate(trading_date_list)}

    if not target_date_list or any(date not in positions for date in target_date_list):
        return False

    target_positions = [positions[date] for date in target_date_list]
    expected_positions = list(range(target_positions[0], target_positions[0] + len(target_positions)))
    return target_positions == expected_positions


def is_consecutive_calendar_dates(target_dates: pd.DatetimeIndex | list[str]) -> bool:
    normalized = pd.DatetimeIndex(pd.to_datetime(target_dates).normalize())
    if len(normalized) == 0:
        return False
    expected = pd.date_range(normalized[0], periods=len(normalized), freq="D")
    return normalized.equals(expected)


def build_window_metadata(window: WalkForwardWindow, trading_dates: pd.DatetimeIndex) -> dict:
    target_dates = format_date_list(window.target_dates)
    first_day = pd.Timestamp(target_dates[0])
    last_day = pd.Timestamp(target_dates[-1])
    calendar_span_days = int((last_day - first_day).days + 1)
    return {
        **asdict(window),
        "target_window_type": "consecutive_calendar_days_with_stock_data",
        "target_trading_dates": target_dates,
        "target_trading_day_count": len(target_dates),
        "target_calendar_span_days": calendar_span_days,
        "is_consecutive_calendar_days": is_consecutive_calendar_dates(target_dates),
        "is_consecutive_trading_days": is_consecutive_trading_dates(trading_dates, target_dates),
        "date_note": "验证窗口要求连续自然日，并且这些自然日都必须在数据中有交易记录；周末或节假日窗口会被跳过。",
    }


def build_windows(dates: pd.DatetimeIndex, window_count: int, target_days: int, step_days: int) -> list[WalkForwardWindow]:
    candidates: list[tuple[int, int, pd.DatetimeIndex]] = []
    for target_start_idx in range(1, len(dates) - target_days + 1):
        target_end_idx = target_start_idx + target_days - 1
        target_dates = pd.DatetimeIndex(dates[target_start_idx : target_end_idx + 1])
        if len(target_dates) != target_days:
            raise ValueError(f"验证窗口交易日数量不等于 {target_days}")
        if not is_consecutive_trading_dates(dates, target_dates):
            raise ValueError("验证窗口不是连续交易日，请检查数据日期")
        if not is_consecutive_calendar_dates(target_dates):
            continue

        candidates.append((target_start_idx, target_end_idx, target_dates))

    windows: list[WalkForwardWindow] = []
    last_selected_end_idx: int | None = None

    for target_start_idx, target_end_idx, target_dates in reversed(candidates):
        if last_selected_end_idx is not None and target_end_idx > last_selected_end_idx - step_days:
            continue

        as_of_idx = target_start_idx - 1
        window = WalkForwardWindow(
            name=f"window_{len(windows) + 1:02d}",
            as_of_date=pd.Timestamp(dates[as_of_idx]).strftime("%Y-%m-%d"),
            mock_submission_date=(pd.Timestamp(target_dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            target_start_date=pd.Timestamp(target_dates[0]).strftime("%Y-%m-%d"),
            target_end_date=pd.Timestamp(target_dates[-1]).strftime("%Y-%m-%d"),
            target_dates=[pd.Timestamp(day).strftime("%Y-%m-%d") for day in target_dates],
        )
        windows.append(window)
        last_selected_end_idx = target_end_idx
        if len(windows) == window_count:
            break

    if len(windows) < window_count:
        raise ValueError(
            f"可用的连续 {target_days} 个自然日且都有交易数据的验证窗口不足: "
            f"需要 {window_count} 个，实际 {len(windows)} 个"
        )

    windows.reverse()
    for idx, window in enumerate(windows, start=1):
        window.name = f"window_{idx:02d}"

    return windows


def prepare_experiment_dir(experiment_dir: Path, resume: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not resume:
        raise FileExistsError(f"版本目录已存在: {experiment_dir}。如需继续旧任务，请加 --resume")
    experiment_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_existing_summary(experiment_dir: Path) -> dict[str, dict]:
    summary_path = experiment_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    if "window" not in summary.columns:
        return {}
    return {
        str(row["window"]): row.to_dict()
        for _, row in summary.iterrows()
    }


def optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    return float(value)


def stage_total_seconds(*values: float | None, fallback: float | None = None) -> float | None:
    if values and all(value is not None for value in values):
        return float(sum(value for value in values if value is not None))
    if fallback is not None:
        return fallback
    known = [value for value in values if value is not None]
    if known:
        return float(sum(known))
    return fallback


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("运行命令: %s", " ".join(command))
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    duration = time.perf_counter() - start_time
    logger.info("命令完成: 耗时=%s", format_duration(duration))
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return duration


def make_child_env(model_dir: Path, stock_data_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["BDC_OUTPUT_DIR"] = str(model_dir)
    env["BDC_STOCK_DATA_FILE"] = str(stock_data_file)
    env.setdefault("BDC_TENSORBOARD", "0")
    return env


def get_tune_env_snapshot() -> dict[str, str | None]:
    keys = [
        "BDC_TUNE_PROFILE",
        "BDC_FAST_DEV",
        "BDC_WF_WINDOWS",
        "BDC_FEATURE_NUM",
        "BDC_SEQUENCE_LENGTH",
        "BDC_TRAIN_TARGET_DAYS",
        "BDC_VAL_DAYS",
        "BDC_MAX_STOCKS_PER_DAY",
        "BDC_D_MODEL",
        "BDC_NHEAD",
        "BDC_NUM_LAYERS",
        "BDC_DIM_FEEDFORWARD",
        "BDC_BATCH_SIZE",
        "BDC_NUM_EPOCHS",
        "BDC_LEARNING_RATE",
        "BDC_WEIGHT_DECAY",
        "BDC_DROPOUT",
        "BDC_USE_INSTRUMENT_FEATURE",
        "BDC_OPTIMIZER",
        "BDC_LOOKAHEAD_K",
        "BDC_LOOKAHEAD_ALPHA",
        "BDC_LR_SCHEDULER",
        "BDC_LR_PATIENCE",
        "BDC_LR_FACTOR",
        "BDC_LR_THRESHOLD",
        "BDC_MIN_LR",
        "BDC_EARLY_STOPPING_PATIENCE",
        "BDC_EARLY_STOPPING_MIN_DELTA",
        "BDC_GRAD_CLIP",
        "BDC_NUM_PROCESSES",
        "BDC_TORCH_NUM_THREADS",
        "BDC_TENSORBOARD",
    ]
    return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def run_train(model_dir: Path, stock_data_file: Path, log_path: Path) -> float:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file)
    return run_logged([sys.executable, "code/src/train.py"], env=env, log_path=log_path)


def run_predict(
    model_dir: Path,
    stock_data_file: Path,
    result_path: Path,
    scores_path: Path | None = None,
    submission_date: str | None = None,
    as_of_date: str | None = None,
    target_start_date: str | None = None,
) -> float:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file)
    command = [sys.executable, "code/src/predict.py", "--output", str(result_path)]
    if scores_path:
        command.extend(["--scores-output", str(scores_path)])
    if submission_date:
        command.extend(["--submission-date", submission_date, "--as-of-date", as_of_date or submission_date])
    if target_start_date:
        command.extend(["--target-start-date", target_start_date])
    return run_logged(command, env=env, log_path=result_path.with_suffix(".log"))


def normalize_prediction(prediction_path: Path) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path, dtype={"stock_id": str, "股票代码": str})
    id_col = "stock_id" if "stock_id" in prediction.columns else "股票代码" if "股票代码" in prediction.columns else None
    weight_col = "weight" if "weight" in prediction.columns else "权重" if "权重" in prediction.columns else None
    if id_col is None or weight_col is None:
        raise ValueError(f"{prediction_path} 缺少 stock_id/股票代码 或 weight/权重 字段")

    prediction = prediction.rename(columns={id_col: "stock_id", weight_col: "weight"})[["stock_id", "weight"]].copy()
    prediction["stock_id"] = normalize_stock_codes(prediction["stock_id"])
    prediction["weight"] = pd.to_numeric(prediction["weight"], errors="coerce")

    if prediction["weight"].isna().any():
        raise ValueError(f"{prediction_path} 存在无法解析的权重")
    if len(prediction) > 5:
        raise ValueError(f"{prediction_path} 最多只能包含 5 只股票")
    weight_sum = float(prediction["weight"].sum())
    if weight_sum < 0 or weight_sum > 1.0:
        raise ValueError(f"{prediction_path} 权重和必须在 0 到 1 之间，当前为 {weight_sum}")
    return prediction


def calculate_window_score(full_df: pd.DataFrame, prediction_path: Path, target_dates: list[str]) -> dict:
    prediction = normalize_prediction(prediction_path)
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates).normalize())
    first_day = target_index[0]
    last_day = target_index[-1]

    window_data = full_df[full_df["日期"].isin(target_index)].copy()
    first_open = window_data[window_data["日期"] == first_day][["股票代码", "开盘"]].rename(columns={"开盘": "start_open"})
    last_open = window_data[window_data["日期"] == last_day][["股票代码", "开盘"]].rename(columns={"开盘": "end_open"})
    returns = first_open.merge(last_open, on="股票代码", how="inner")
    returns["return"] = (returns["end_open"] - returns["start_open"]) / (returns["start_open"] + 1e-12)

    selected = prediction.merge(returns, left_on="stock_id", right_on="股票代码", how="left")
    if selected["return"].isna().any():
        missing = selected.loc[selected["return"].isna(), "stock_id"].tolist()
        raise ValueError(f"目标窗口首尾交易日缺少这些预测股票的数据: {missing}")

    selected["weighted_return"] = selected["return"] * selected["weight"]
    score = float(selected["weighted_return"].sum())
    oracle_equal_weight_top5 = float(returns["return"].nlargest(5).mean()) if len(returns) >= 5 else 0.0
    market_equal_weight = float(returns["return"].mean()) if len(returns) else 0.0

    return {
        "score": score,
        "oracle_equal_weight_top5": oracle_equal_weight_top5,
        "market_equal_weight": market_equal_weight,
        "target_window_type": "consecutive_calendar_days_with_stock_data",
        "target_dates": target_dates,
        "target_trading_dates": target_dates,
        "target_trading_day_count": len(target_dates),
        "target_calendar_span_days": int((last_day - first_day).days + 1),
        "is_consecutive_calendar_days": is_consecutive_calendar_dates(target_dates),
        "selected": [
            {
                "stock_id": row.stock_id,
                "weight": float(row.weight),
                "return": float(row.return_),
                "weighted_return": float(row.weighted_return),
            }
            for row in selected.rename(columns={"return": "return_"}).itertuples(index=False)
        ],
    }


def write_window_data(full_df: pd.DataFrame, window: WalkForwardWindow, window_dir: Path) -> tuple[Path, Path]:
    data_dir = window_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(window.as_of_date)
    target_dates = pd.DatetimeIndex(pd.to_datetime(window.target_dates).normalize())

    train_data_path = data_dir / "train_until_as_of.csv"
    target_data_path = data_dir / "target_window.csv"
    full_df[full_df["日期"] <= as_of].to_csv(train_data_path, index=False)
    full_df[full_df["日期"].isin(target_dates)].to_csv(target_data_path, index=False)
    return train_data_path, target_data_path


def run_window(
    full_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    experiment_dir: Path,
    version: str,
    window: WalkForwardWindow,
    resume: bool,
    existing_row: dict | None = None,
) -> dict:
    window_dir = experiment_dir / "windows" / window.name
    model_dir = window_dir / "model"
    prediction_path = window_dir / "prediction.csv"
    prediction_scores_path = window_dir / "prediction_scores.csv"
    score_path = window_dir / "score.json"
    metadata_path = window_dir / "metadata.json"
    window_start_time = time.perf_counter()
    train_seconds = optional_float(existing_row.get("train_seconds")) if existing_row else None
    predict_seconds = optional_float(existing_row.get("predict_seconds")) if existing_row else None
    existing_window_seconds = optional_float(existing_row.get("window_seconds")) if existing_row else None

    log_section(
        f"{window.name} | as_of={window.as_of_date} | target={window.target_start_date} ~ {window.target_end_date}"
    )

    train_data_path, target_data_path = write_window_data(full_df, window, window_dir)
    write_json(
        metadata_path,
        {
            "version": version,
            "window": build_window_metadata(window, trading_dates),
            "train_data": str(train_data_path.relative_to(REPO_ROOT)),
            "target_data": str(target_data_path.relative_to(REPO_ROOT)),
            "model_dir": str(model_dir.relative_to(REPO_ROOT)),
            "prediction": str(prediction_path.relative_to(REPO_ROOT)),
            "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
            "score": str(score_path.relative_to(REPO_ROOT)),
        },
    )

    if resume and (model_dir / "best_model.pth").exists():
        logger.info("%s 已有模型，跳过训练", window.name)
    else:
        logger.info("%s 训练: as_of=%s", window.name, window.as_of_date)
        train_seconds = run_train(model_dir=model_dir, stock_data_file=train_data_path, log_path=window_dir / "logs" / "train_command.log")

    if resume and prediction_path.exists() and prediction_scores_path.exists():
        logger.info("%s 已有预测结果，跳过预测", window.name)
    else:
        logger.info("%s 预测: 目标窗口=%s ~ %s", window.name, window.target_start_date, window.target_end_date)
        predict_seconds = run_predict(
            model_dir=model_dir,
            stock_data_file=train_data_path,
            result_path=prediction_path,
            scores_path=prediction_scores_path,
            submission_date=window.mock_submission_date,
            as_of_date=window.as_of_date,
            target_start_date=window.target_start_date,
        )

    score = calculate_window_score(full_df, prediction_path, window.target_dates)
    elapsed_seconds = time.perf_counter() - window_start_time
    window_seconds = stage_total_seconds(
        train_seconds,
        predict_seconds,
        fallback=existing_window_seconds if existing_window_seconds is not None else elapsed_seconds,
    )
    score["train_seconds"] = train_seconds
    score["train_duration"] = format_duration(train_seconds)
    score["predict_seconds"] = predict_seconds
    score["predict_duration"] = format_duration(predict_seconds)
    score["window_seconds"] = window_seconds
    score["window_duration"] = format_duration(window_seconds)
    write_json(score_path, score)
    logger.info("%s 验证得分: %.6f | 总耗时=%s", window.name, score["score"], format_duration(window_seconds))

    return {
        "version": version,
        "window": window.name,
        "as_of_date": window.as_of_date,
        "mock_submission_date": window.mock_submission_date,
        "target_start_date": window.target_start_date,
        "target_end_date": window.target_end_date,
        "target_dates": ",".join(window.target_dates),
        "target_window_type": score["target_window_type"],
        "target_trading_day_count": score["target_trading_day_count"],
        "target_calendar_span_days": score["target_calendar_span_days"],
        "is_consecutive_calendar_days": score["is_consecutive_calendar_days"],
        "is_consecutive_trading_days": is_consecutive_trading_dates(trading_dates, window.target_dates),
        "score": score["score"],
        "oracle_equal_weight_top5": score["oracle_equal_weight_top5"],
        "market_equal_weight": score["market_equal_weight"],
        "train_seconds": train_seconds,
        "train_duration": format_duration(train_seconds),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "window_seconds": window_seconds,
        "window_duration": format_duration(window_seconds),
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
        "score_file": str(score_path.relative_to(REPO_ROOT)),
    }


def run_final(
    full_data_file: Path,
    experiment_dir: Path,
    publish_final: bool,
    resume: bool,
    existing_final: dict | None = None,
) -> dict:
    final_dir = experiment_dir / "final"
    model_dir = final_dir / "model"
    result_path = final_dir / "result.csv"
    result_scores_path = final_dir / "result_scores.csv"
    final_start_time = time.perf_counter()
    train_seconds = optional_float(existing_final.get("train_seconds")) if existing_final else None
    predict_seconds = optional_float(existing_final.get("predict_seconds")) if existing_final else None
    existing_total_seconds = optional_float(existing_final.get("total_seconds")) if existing_final else None

    log_section("最终模型训练与预测")

    if resume and (model_dir / "best_model.pth").exists():
        logger.info("最终模型已存在，跳过训练")
    else:
        logger.info("训练最终模型: %s", model_dir)
        train_seconds = run_train(model_dir=model_dir, stock_data_file=full_data_file, log_path=final_dir / "logs" / "train_command.log")

    if resume and result_path.exists() and result_scores_path.exists():
        logger.info("最终预测已存在，跳过预测")
    else:
        logger.info("生成最终预测: %s", result_path)
        predict_seconds = run_predict(
            model_dir=model_dir,
            stock_data_file=full_data_file,
            result_path=result_path,
            scores_path=result_scores_path,
        )

    published_path = None
    if publish_final:
        published = REPO_ROOT / "output" / "result.csv"
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, published)
        published_path = str(published.relative_to(REPO_ROOT))
        logger.info("已发布最终预测到: %s", published_path)

    elapsed_seconds = time.perf_counter() - final_start_time
    final_seconds = stage_total_seconds(
        train_seconds,
        predict_seconds,
        fallback=existing_total_seconds if existing_total_seconds is not None else elapsed_seconds,
    )
    logger.info("最终流程完成: 耗时=%s", format_duration(final_seconds))

    return {
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "prediction": str(result_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(result_scores_path.relative_to(REPO_ROOT)),
        "published_prediction": published_path,
        "train_seconds": train_seconds,
        "train_duration": format_duration(train_seconds),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "total_seconds": final_seconds,
        "total_duration": format_duration(final_seconds),
    }


def write_summary(experiment_dir: Path, rows: list[dict]) -> None:
    summary_path = experiment_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info("walk-forward 汇总已写入: %s", summary_path.relative_to(REPO_ROOT))


def main() -> None:
    args = parse_args()
    validate_args(args)

    experiment_dir = EXPERIMENTS_DIR / args.version
    prepare_experiment_dir(experiment_dir, resume=args.resume, dry_run=args.dry_run)
    log_path = None if args.dry_run else experiment_dir / "walk_forward.log"
    setup_logging("bdc.walk_forward", log_path)
    run_start_time = time.perf_counter()

    full_df, data_file = load_stock_data(
        REPO_ROOT / "data",
        data_file=args.data_file,
        allow_train_fallback=True,
        logger=logger,
    )
    dates = get_trading_dates(full_df)
    windows = build_windows(dates, args.windows, args.target_days, args.step_days)

    logger.info("版本: %s", args.version)
    tune_env = get_tune_env_snapshot()
    if tune_env:
        logger.info("调参参数: %s", ", ".join(f"{key}={value}" for key, value in tune_env.items()))
    logger.info("窗口数量: %s", len(windows))
    for window in windows:
        logger.info(
            "%s: as_of=%s, mock_submission=%s, target=%s ~ %s (%s)",
            window.name,
            window.as_of_date,
            window.mock_submission_date,
            window.target_start_date,
            window.target_end_date,
            ", ".join(window.target_dates),
        )

    if args.dry_run:
        return

    existing_summary = read_existing_summary(experiment_dir) if args.resume else {}
    previous_manifest = read_json(experiment_dir / "manifest.json") if args.resume else {}
    git_info = get_git_info()
    manifest = {
        "version": args.version,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git": git_info,
        "data_file": str(Path(data_file).resolve()),
        "tune_env": tune_env,
        "fast_dev_mode": parse_bool_env("BDC_FAST_DEV", False),
        "windows": [build_window_metadata(window, dates) for window in windows],
        "window_results": [],
        "final": None,
    }
    write_json(experiment_dir / "manifest.json", manifest)

    rows = []
    for window in windows:
        rows.append(
            run_window(
                full_df,
                dates,
                experiment_dir,
                args.version,
                window,
                resume=args.resume,
                existing_row=existing_summary.get(window.name),
            )
        )
        write_summary(experiment_dir, rows)

    if rows:
        scores = [row["score"] for row in rows]
        manifest["window_results"] = rows
        manifest["walk_forward_score_mean"] = float(sum(scores) / len(scores))
        manifest["walk_forward_score_min"] = float(min(scores))
        manifest["walk_forward_score_max"] = float(max(scores))

    if not args.skip_final:
        manifest["final"] = run_final(
            Path(data_file).resolve(),
            experiment_dir,
            args.publish_final,
            resume=args.resume,
            existing_final=previous_manifest.get("final"),
        )

    write_json(experiment_dir / "manifest.json", manifest)
    if args.create_tag:
        create_git_tag(args.version)
    total_seconds = time.perf_counter() - run_start_time
    manifest["total_seconds"] = total_seconds
    manifest["total_duration"] = format_duration(total_seconds)
    write_json(experiment_dir / "manifest.json", manifest)
    logger.info("版本 %s 完成 | 总耗时=%s", args.version, format_duration(total_seconds))


if __name__ == "__main__":
    main()
