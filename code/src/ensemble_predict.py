import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import config
from data_utils import load_stock_data, normalize_stock_codes, setup_logging
from ensemble_config import (
    ENSEMBLE_GATE_OVERHEAT_THRESHOLD,
    ENSEMBLE_SELECTION_STRATEGY,
    ENSEMBLE_VOL_WINDOW,
    get_submission_ensemble_sources,
)
from predict import default_scores_output_path, resolve_prediction_task
from stage2_selection import select_ensemble_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="正式提交用两模型集成预测")
    parser.add_argument(
        "--submission-date",
        default=os.environ.get(
            "BDC_SUBMISSION_DATE", config.get("submission_deadline_date", "2026-08-02")
        ),
        help="提交截止日，默认 2026-08-02；预测窗口必须在该日期之后",
    )
    parser.add_argument(
        "--target-start-date",
        default=os.environ.get("BDC_TARGET_START_DATE"),
        help="预测窗口候选起始日，默认 submission-date 后一天；若非交易日会向后跳到合理交易日",
    )
    parser.add_argument(
        "--as-of-date",
        default=os.environ.get("BDC_AS_OF_DATE") or os.environ.get("BDC_PREDICT_DATE"),
        help="仅用于本地调试的数据截止日；默认使用不晚于 submission-date 的最新数据",
    )
    parser.add_argument(
        "--market-holidays",
        default=os.environ.get(
            "BDC_MARKET_HOLIDAYS", config.get("market_holidays", "")
        ),
        help="未来休市日，逗号分隔，例如 2026-10-01,2026-10-02",
    )
    parser.add_argument(
        "--output",
        default=config.get(
            "prediction_output_path", os.path.join("./output/", "result.csv")
        ),
        help="预测结果输出路径，默认 ./output/result.csv",
    )
    parser.add_argument(
        "--scores-output",
        default=os.environ.get("BDC_PREDICTION_SCORES_OUTPUT"),
        help="完整候选股票排名诊断文件；默认写到 output 同目录的 *_scores.csv",
    )
    parser.add_argument(
        "--temp-dir",
        default=os.environ.get("BDC_ENSEMBLE_TEMP_DIR", "./temp/submission_ensemble"),
        help="集成预测中间文件目录",
    )
    parser.add_argument(
        "--ensemble-selection-strategy",
        default=os.environ.get(
            "BDC_ENSEMBLE_SELECTION_STRATEGY", ENSEMBLE_SELECTION_STRATEGY
        ),
        help="集成重排策略，默认 ensemble_low_vol_top5",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("BDC_TOP_K", config.get("top_k", 5))),
        help="提交股票数量，1到5之间，默认5",
    )
    parser.add_argument(
        "--total-exposure",
        type=float,
        default=float(
            os.environ.get("BDC_TOTAL_EXPOSURE", config.get("total_exposure", 1.0))
        ),
        help="总仓位，0到1之间，默认1.0；小于1表示留现金",
    )
    parser.add_argument(
        "--stage2-vol-window",
        type=int,
        default=int(os.environ.get("BDC_STAGE2_VOL_WINDOW", ENSEMBLE_VOL_WINDOW)),
        help="二阶段波动率计算窗口，默认 20 个交易日",
    )
    parser.add_argument(
        "--gate-overheat-threshold",
        type=float,
        default=float(
            os.environ.get(
                "BDC_GATE_OVERHEAT_THRESHOLD", ENSEMBLE_GATE_OVERHEAT_THRESHOLD
            )
        ),
        help="过热门控阈值，仅 ensemble_gate_overheat_top5 使用，默认 0.65",
    )
    parser.add_argument("--debug", action="store_true", help="使用 debug 集成模型目录")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印预测命令，不实际运行"
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def source_env(source_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(source_env)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def source_predict_script(env: dict[str, str]) -> str:
    model_kind = env.get("BDC_MODEL_KIND", "transformer").strip().lower()
    if model_kind == "lgbm":
        return "code/src/predict_lgbm.py"
    if model_kind in {"", "transformer"}:
        return "code/src/predict.py"
    raise ValueError(f"Unsupported BDC_MODEL_KIND for ensemble source: {model_kind}")


def add_optional_date_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(["--submission-date", args.submission_date])
    if args.target_start_date:
        command.extend(["--target-start-date", args.target_start_date])
    if args.as_of_date:
        command.extend(["--as-of-date", args.as_of_date])
    if args.market_holidays:
        command.extend(["--market-holidays", args.market_holidays])


def run_source_predict(
    label: str,
    env: dict[str, str],
    result_path: Path,
    scores_path: Path,
    args: argparse.Namespace,
) -> float:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        source_predict_script(env),
        "--output",
        str(result_path),
        "--scores-output",
        str(scores_path),
        "--selection-strategy",
        "model_top5",
    ]
    add_optional_date_args(command, args)

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "源模型预测: %s | kind=%s -> %s",
        label,
        env.get("BDC_MODEL_KIND", "transformer"),
        scores_path,
    )
    logger.info("=" * 72)
    logger.info("运行命令: %s", " ".join(command))
    if args.dry_run:
        return 0.0

    start = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, env=source_env(env), check=True)
    duration = time.perf_counter() - start
    logger.info("源模型预测完成: %s | 耗时=%s", label, format_duration(duration))
    return duration


def read_history_until_as_of(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str]]:
    raw_df, data_file = load_stock_data(
        config["data_path"],
        data_file=config.get("stock_data_file"),
        allow_train_fallback=True,
        logger=logger,
    )
    _, latest_date, target_window = resolve_prediction_task(raw_df, args)
    history = raw_df[raw_df["日期"] <= latest_date].copy()
    target_dates = [day.strftime("%Y-%m-%d") for day in target_window]
    logger.info(
        "集成重排历史数据: %s | as_of=%s | 目标窗口=%s",
        data_file,
        latest_date.date(),
        ", ".join(target_dates),
    )
    return history, target_dates


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_prediction_scores_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    if "stock_id" not in frame.columns and "股票代码" in frame.columns:
        frame = frame.rename(columns={"股票代码": "stock_id"})
    required = {"rank", "stock_id", "pred_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少 ensemble 所需列: {sorted(missing)}")
    frame = frame.copy()
    frame["stock_id"] = normalize_stock_codes(frame["stock_id"])
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame = (
        frame.dropna(subset=["rank", "stock_id"])
        .sort_values("rank")
        .reset_index(drop=True)
    )
    frame["rank"] = frame["rank"].astype(int)
    return frame


def main() -> None:
    global logger
    args = parse_args()
    output_path = Path(args.output)
    scores_output_path = Path(
        args.scores_output or default_scores_output_path(str(output_path))
    )
    temp_dir = Path(args.temp_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("bdc.ensemble_predict", output_path.parent / "predict.log")

    debug = args.debug or os.environ.get("BDC_FAST_DEV", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    sources = get_submission_ensemble_sources(debug=debug)
    logger.info("提交集成预测模式: %s", "debug" if debug else "official")
    logger.info("集成策略: %s", args.ensemble_selection_strategy)
    logger.info("过热门控阈值: %s", args.gate_overheat_threshold)
    logger.info("最终输出: %s", output_path)
    logger.info("完整候选排名输出: %s", scores_output_path)

    score_paths = []
    source_top5 = {}
    total_predict_seconds = 0.0
    for source in sources:
        source_dir = temp_dir / source.label
        source_result = source_dir / "result.csv"
        source_scores = source_dir / "result_scores.csv"
        total_predict_seconds += run_source_predict(
            source.label, source.env, source_result, source_scores, args
        )
        score_paths.append((source.label, source_scores))
        if not args.dry_run:
            frame = read_prediction_scores_csv(source_scores)
            source_top5[source.label] = (
                frame.sort_values("rank").head(5)["stock_id"].tolist()
            )

    if args.dry_run:
        return

    history, target_dates = read_history_until_as_of(args)
    score_frames = [
        (label, read_prediction_scores_csv(path)) for label, path in score_paths
    ]
    selected, scores = select_ensemble_predictions(
        score_frames,
        history=history,
        strategy=args.ensemble_selection_strategy,
        top_k=args.top_k,
        total_exposure=args.total_exposure,
        volatility_window=args.stage2_vol_window,
        gate_overheat_threshold=args.gate_overheat_threshold,
    )
    selected[["stock_id", "weight"]].to_csv(output_path, index=False)
    scores.to_csv(scores_output_path, index=False)

    metadata = {
        "mode": "submission_ensemble",
        "selection_strategy": args.ensemble_selection_strategy,
        "top_k": args.top_k,
        "total_exposure": args.total_exposure,
        "stage2_vol_window": args.stage2_vol_window,
        "gate_overheat_threshold": args.gate_overheat_threshold,
        "gate_use_defense": bool(
            scores.get("gate_use_defense", pd.Series([False])).iloc[0]
        ),
        "gate_primary_source": (
            str(scores.get("gate_primary_source", pd.Series([""])).iloc[0])
            if len(scores)
            else ""
        ),
        "gate_primary_top5_overheat_mean": (
            None
            if "gate_primary_top5_overheat_mean" not in scores.columns
            or pd.isna(scores["gate_primary_top5_overheat_mean"].iloc[0])
            else float(scores["gate_primary_top5_overheat_mean"].iloc[0])
        ),
        "target_dates": target_dates,
        "sources": [
            {
                "label": source.label,
                "output_dir": source.output_dir,
                "source_top5": source_top5.get(source.label, []),
            }
            for source in sources
        ],
        "selected": selected["stock_id"].tolist(),
        "candidate_count": int((scores["stage2_pool_member"] == True).sum()),
        "source_predict_seconds": total_predict_seconds,
    }
    write_json(output_path.parent / "ensemble_prediction.json", metadata)

    logger.info("集成候选池股票数: %s", metadata["candidate_count"])
    logger.info("源模型 Top5: %s", source_top5)
    logger.info("最终 Selected: %s", ", ".join(metadata["selected"]))
    logger.info("结果已写入: %s", output_path)
    logger.info("完整候选排名已写入: %s", scores_output_path)
    logger.info("集成预测耗时: %s", format_duration(total_predict_seconds))


if __name__ == "__main__":
    main()
