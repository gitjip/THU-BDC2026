import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
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


@dataclass
class WalkForwardWindow:
    name: str
    as_of_date: str
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


def build_windows(dates: pd.DatetimeIndex, window_count: int, target_days: int, step_days: int) -> list[WalkForwardWindow]:
    windows: list[WalkForwardWindow] = []
    latest_target_end_idx = len(dates) - 1

    for offset in reversed(range(window_count)):
        target_end_idx = latest_target_end_idx - offset * step_days
        target_start_idx = target_end_idx - target_days + 1
        as_of_idx = target_start_idx - 1
        if as_of_idx < 0:
            raise ValueError("数据交易日不足，无法构建指定数量的 walk-forward 窗口")

        target_dates = pd.DatetimeIndex(dates[target_start_idx : target_end_idx + 1])
        window = WalkForwardWindow(
            name=f"window_{len(windows) + 1:02d}",
            as_of_date=pd.Timestamp(dates[as_of_idx]).strftime("%Y-%m-%d"),
            target_start_date=pd.Timestamp(target_dates[0]).strftime("%Y-%m-%d"),
            target_end_date=pd.Timestamp(target_dates[-1]).strftime("%Y-%m-%d"),
            target_dates=[pd.Timestamp(day).strftime("%Y-%m-%d") for day in target_dates],
        )
        windows.append(window)

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


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("运行命令: %s", " ".join(command))
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
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


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
        "BDC_NUM_PROCESSES",
        "BDC_TORCH_NUM_THREADS",
        "BDC_TENSORBOARD",
    ]
    return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def run_train(model_dir: Path, stock_data_file: Path, log_path: Path) -> None:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file)
    run_logged([sys.executable, "code/src/train.py"], env=env, log_path=log_path)


def run_predict(
    model_dir: Path,
    stock_data_file: Path,
    result_path: Path,
    submission_date: str | None = None,
    target_start_date: str | None = None,
) -> None:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file)
    command = [sys.executable, "code/src/predict.py", "--output", str(result_path)]
    if submission_date:
        command.extend(["--submission-date", submission_date, "--as-of-date", submission_date])
    if target_start_date:
        command.extend(["--target-start-date", target_start_date])
    run_logged(command, env=env, log_path=result_path.with_suffix(".log"))


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
        "target_dates": target_dates,
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


def run_window(full_df: pd.DataFrame, experiment_dir: Path, version: str, window: WalkForwardWindow, resume: bool) -> dict:
    window_dir = experiment_dir / "windows" / window.name
    model_dir = window_dir / "model"
    prediction_path = window_dir / "prediction.csv"
    score_path = window_dir / "score.json"
    metadata_path = window_dir / "metadata.json"

    train_data_path, target_data_path = write_window_data(full_df, window, window_dir)
    write_json(
        metadata_path,
        {
            "version": version,
            "window": asdict(window),
            "train_data": str(train_data_path.relative_to(REPO_ROOT)),
            "target_data": str(target_data_path.relative_to(REPO_ROOT)),
            "model_dir": str(model_dir.relative_to(REPO_ROOT)),
            "prediction": str(prediction_path.relative_to(REPO_ROOT)),
            "score": str(score_path.relative_to(REPO_ROOT)),
        },
    )

    if resume and (model_dir / "best_model.pth").exists():
        logger.info("%s 已有模型，跳过训练", window.name)
    else:
        logger.info("%s 训练: as_of=%s", window.name, window.as_of_date)
        run_train(model_dir=model_dir, stock_data_file=train_data_path, log_path=window_dir / "logs" / "train_command.log")

    if resume and prediction_path.exists():
        logger.info("%s 已有预测结果，跳过预测", window.name)
    else:
        logger.info("%s 预测: 目标窗口=%s ~ %s", window.name, window.target_start_date, window.target_end_date)
        run_predict(
            model_dir=model_dir,
            stock_data_file=train_data_path,
            result_path=prediction_path,
            submission_date=window.as_of_date,
            target_start_date=window.target_start_date,
        )

    score = calculate_window_score(full_df, prediction_path, window.target_dates)
    write_json(score_path, score)
    logger.info("%s 验证得分: %.6f", window.name, score["score"])

    return {
        "version": version,
        "window": window.name,
        "as_of_date": window.as_of_date,
        "target_start_date": window.target_start_date,
        "target_end_date": window.target_end_date,
        "target_dates": ",".join(window.target_dates),
        "score": score["score"],
        "oracle_equal_weight_top5": score["oracle_equal_weight_top5"],
        "market_equal_weight": score["market_equal_weight"],
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "score_file": str(score_path.relative_to(REPO_ROOT)),
    }


def run_final(full_data_file: Path, experiment_dir: Path, publish_final: bool, resume: bool) -> dict:
    final_dir = experiment_dir / "final"
    model_dir = final_dir / "model"
    result_path = final_dir / "result.csv"

    if resume and (model_dir / "best_model.pth").exists():
        logger.info("最终模型已存在，跳过训练")
    else:
        logger.info("训练最终模型: %s", model_dir)
        run_train(model_dir=model_dir, stock_data_file=full_data_file, log_path=final_dir / "logs" / "train_command.log")

    if resume and result_path.exists():
        logger.info("最终预测已存在，跳过预测")
    else:
        logger.info("生成最终预测: %s", result_path)
        run_predict(model_dir=model_dir, stock_data_file=full_data_file, result_path=result_path)

    published_path = None
    if publish_final:
        published = REPO_ROOT / "output" / "result.csv"
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, published)
        published_path = str(published.relative_to(REPO_ROOT))
        logger.info("已发布最终预测到: %s", published_path)

    return {
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "prediction": str(result_path.relative_to(REPO_ROOT)),
        "published_prediction": published_path,
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
            "%s: as_of=%s, target=%s ~ %s (%s)",
            window.name,
            window.as_of_date,
            window.target_start_date,
            window.target_end_date,
            ", ".join(window.target_dates),
        )

    if args.dry_run:
        return

    git_info = get_git_info()
    manifest = {
        "version": args.version,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git": git_info,
        "data_file": str(Path(data_file).resolve()),
        "tune_env": tune_env,
        "fast_dev_mode": parse_bool_env("BDC_FAST_DEV", False),
        "windows": [asdict(window) for window in windows],
        "final": None,
    }
    write_json(experiment_dir / "manifest.json", manifest)

    rows = []
    for window in windows:
        rows.append(run_window(full_df, experiment_dir, args.version, window, resume=args.resume))
        write_summary(experiment_dir, rows)

    if rows:
        scores = [row["score"] for row in rows]
        manifest["walk_forward_score_mean"] = float(sum(scores) / len(scores))
        manifest["walk_forward_score_min"] = float(min(scores))
        manifest["walk_forward_score_max"] = float(max(scores))

    if not args.skip_final:
        manifest["final"] = run_final(Path(data_file).resolve(), experiment_dir, args.publish_final, resume=args.resume)

    write_json(experiment_dir / "manifest.json", manifest)
    if args.create_tag:
        create_git_tag(args.version)
    logger.info("版本 %s 完成", args.version)


if __name__ == "__main__":
    main()
