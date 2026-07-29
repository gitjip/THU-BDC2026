import logging
import os
import json

import joblib
import numpy as np
import pandas as pd

from config import config
from data_utils import load_stock_data, setup_logging
from predict import (
    default_scores_output_path,
    parse_args,
    preprocess_predict_data,
    resolve_prediction_task,
)
from stage2_selection import normalize_selection_strategy, select_predictions

logger = logging.getLogger(__name__)


def read_model_metadata(model_dir):
    path = os.path.join(model_dir, "features.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 LightGBM 特征文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_rows_per_stock(processed, latest_date):
    data = processed[processed["日期"] <= latest_date].copy()
    if data.empty:
        raise ValueError(f"没有不晚于 {latest_date.date()} 的可预测特征")
    data = (
        data.sort_values(["股票代码", "日期"])
        .groupby("股票代码", as_index=False)
        .tail(1)
    )
    return data.reset_index(drop=True)


def main():
    global logger
    args = parse_args()
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    setup_logging(
        "bdc.lgbm.predict",
        os.path.join(os.path.dirname(output_path) or ".", "predict.log"),
    )
    logger = logging.getLogger("bdc.lgbm.predict")

    model_dir = config["output_dir"]
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 LightGBM 模型文件: {model_path}")
    metadata = read_model_metadata(model_dir)
    features = list(metadata.get("features") or [])
    if not features:
        raise ValueError(f"{model_dir}/features.json 未记录特征列")

    raw_df, _ = load_stock_data(
        config["data_path"],
        data_file=config.get("stock_data_file"),
        allow_train_fallback=True,
        logger=logger,
    )
    submission_date, latest_date, target_window = resolve_prediction_task(raw_df, args)
    raw_df = raw_df[raw_df["日期"] <= latest_date].copy()

    stock_ids = list(metadata.get("stock_ids") or sorted(raw_df["股票代码"].unique()))
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    logger.info(
        "LightGBM预测任务: 提交截止日=%s | 数据截止日=%s | 目标窗口=%s ~ %s | 目标交易日=%s",
        submission_date.date(),
        latest_date.date(),
        target_window[0].date(),
        target_window[-1].date(),
        ", ".join(day.strftime("%Y-%m-%d") for day in target_window),
    )
    logger.info("股票映射数量: %s", len(stockid2idx))

    processed, _ = preprocess_predict_data(raw_df, stockid2idx)
    inference_rows = latest_rows_per_stock(processed, latest_date)
    for column in features:
        if column not in inference_rows.columns:
            inference_rows[column] = 0.0
    inference_rows[features] = (
        inference_rows[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    model = joblib.load(model_path)
    scores = model.predict(inference_rows[features])
    scores_df = pd.DataFrame(
        {
            "stock_id": inference_rows["股票代码"].astype(str).values,
            "pred_score": scores,
        }
    ).sort_values("pred_score", ascending=False)
    scores_df["rank"] = np.arange(1, len(scores_df) + 1)
    scores_df = scores_df[["rank", "stock_id", "pred_score"]].reset_index(drop=True)

    selection_strategy = normalize_selection_strategy(args.selection_strategy)
    selected_df, scores_df = select_predictions(
        scores_df,
        history=raw_df,
        strategy=selection_strategy,
        top_k=args.top_k,
        total_exposure=args.total_exposure,
        pool_size=args.stage2_pool_size,
        volatility_window=args.stage2_vol_window,
    )

    selected_stock_ids = selected_df["stock_id"].tolist()
    output_df = selected_df[["stock_id", "weight"]].copy()
    output_df.to_csv(output_path, index=False)

    scores_output_path = args.scores_output or default_scores_output_path(output_path)
    os.makedirs(os.path.dirname(scores_output_path) or ".", exist_ok=True)
    scores_df.to_csv(scores_output_path, index=False)

    logger.info("参与排序股票数: %s", len(scores_df))
    logger.info(
        "选股策略: %s | top_k=%s | total_exposure=%s | stage2_pool_size=%s | stage2_vol_window=%s",
        selection_strategy,
        args.top_k,
        args.total_exposure,
        args.stage2_pool_size,
        args.stage2_vol_window,
    )
    logger.info("Selected: %s", ", ".join(selected_stock_ids))
    logger.info("结果已写入: %s", output_path)
    logger.info("完整候选排名已写入: %s", scores_output_path)


if __name__ == "__main__":
    main()
