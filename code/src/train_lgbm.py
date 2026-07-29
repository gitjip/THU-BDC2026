import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd

from config import config
from data_utils import get_trading_dates, load_stock_data, setup_logging
from train import (
    preprocess_data,
    preprocess_val_data,
    split_train_val_by_recent_trading_days,
)

logger = logging.getLogger(__name__)


def format_duration(seconds):
    seconds = float(seconds or 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:02.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "未安装 lightgbm。请先运行 `uv sync`，或安装 pyproject.toml 中的依赖。"
        ) from exc
    return lgb


def clean_features(frame, features):
    result = frame.copy()
    result[features] = result[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    result = result.dropna(subset=["label"]).copy()
    return result


def sample_train_rows(frame, max_stocks_per_day, seed):
    if not max_stocks_per_day or max_stocks_per_day <= 0:
        return frame
    sampled = []
    for _, group in frame.groupby("日期", sort=True):
        if len(group) > max_stocks_per_day:
            group = group.sample(n=max_stocks_per_day, random_state=seed)
        sampled.append(group)
    return pd.concat(sampled, ignore_index=True).sort_values(["日期", "股票代码"])


def target_date_sets(full_df, val_start):
    dates = get_trading_dates(full_df)
    label_horizon = config.get("label_horizon", 5)
    val_days = config.get("val_days", 5)
    train_target_days = config.get("train_target_days", 0)
    labelable_dates = dates[:-label_horizon] if label_horizon > 0 else dates
    val_start = pd.Timestamp(val_start).normalize()
    val_start_matches = np.where(labelable_dates == val_start)[0]
    if len(val_start_matches) == 0:
        raise ValueError(f"验证起始日不在可构建标签日期中: {val_start.date()}")
    val_start_idx = int(val_start_matches[0])
    train_dates = list(labelable_dates[:val_start_idx])
    if train_target_days and train_target_days > 0:
        train_dates = train_dates[-train_target_days:]
    val_dates = list(labelable_dates[val_start_idx : val_start_idx + val_days])
    if not train_dates or not val_dates:
        raise ValueError("LightGBM 训练或验证目标日期为空")
    return set(pd.Timestamp(day).normalize() for day in train_dates), set(
        pd.Timestamp(day).normalize() for day in val_dates
    )


def restrict_target_dates(frame, allowed_dates):
    result = frame.copy()
    result["日期"] = pd.to_datetime(result["日期"], errors="coerce").dt.normalize()
    return result[result["日期"].isin(allowed_dates)].copy()


def rank_metrics(frame, k=5):
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []

    for _, group in frame.groupby("日期", sort=True):
        group = group.dropna(subset=["pred_score", "label"])
        if len(group) < k:
            continue
        pred_top = group.nlargest(k, "pred_score")
        true_top = group.nlargest(k, "label")
        pred_return_sum = float(pred_top["label"].sum())
        max_return_sum = float(true_top["label"].sum())
        random_return_sum = float(k * group["label"].mean())
        ratio_pred = (
            pred_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9
            else 0.0
        )
        ratio_random = (
            random_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9
            else 0.0
        )
        denominator = max_return_sum - random_return_sum
        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-6
            else 0.0
        )
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    return {
        "pred_return_sum": (
            float(np.mean(pred_return_sum_list)) if pred_return_sum_list else 0.0
        ),
        "max_return_sum": (
            float(np.mean(max_return_sum_list)) if max_return_sum_list else 0.0
        ),
        "random_return_sum": (
            float(np.mean(random_return_sum_list)) if random_return_sum_list else 0.0
        ),
        "ratio_pred": float(np.mean(ratio_pred_list)) if ratio_pred_list else 0.0,
        "ratio_random": float(np.mean(ratio_random_list)) if ratio_random_list else 0.0,
        "final_score": float(np.mean(final_score_list)) if final_score_list else 0.0,
    }


def rmse(y_true, y_pred):
    values = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(values * values))) if len(values) else 0.0


def build_model(lgb):
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=config.get("lgbm_n_estimators", 300),
        learning_rate=config.get("lgbm_learning_rate", 0.03),
        num_leaves=config.get("lgbm_num_leaves", 31),
        min_child_samples=config.get("lgbm_min_child_samples", 20),
        subsample=config.get("lgbm_subsample", 0.9),
        colsample_bytree=config.get("lgbm_colsample_bytree", 0.9),
        reg_alpha=config.get("lgbm_reg_alpha", 0.0),
        reg_lambda=config.get("lgbm_reg_lambda", 1.0),
        random_state=config.get("seed", 42),
        n_jobs=config.get("lgbm_num_threads", 8),
        verbosity=-1,
    )


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    global logger
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    setup_logging("bdc.lgbm.train", os.path.join(output_dir, "train.log"))
    logger = logging.getLogger("bdc.lgbm.train")
    run_start_time = time.perf_counter()
    lgb = import_lightgbm()

    write_json(os.path.join(output_dir, "config.json"), config)
    logger.info("运行模式: LightGBM tabular regression")
    logger.info(
        "LightGBM参数: n_estimators=%s, learning_rate=%s, num_leaves=%s, min_child_samples=%s, num_threads=%s",
        config.get("lgbm_n_estimators"),
        config.get("lgbm_learning_rate"),
        config.get("lgbm_num_leaves"),
        config.get("lgbm_min_child_samples"),
        config.get("lgbm_num_threads"),
    )
    logger.info(
        "训练采样: train_target_days=%s, max_stocks_per_day=%s",
        config.get("train_target_days", 0),
        config.get("max_stocks_per_day", 0),
    )

    full_df, data_file = load_stock_data(
        config["data_path"],
        data_file=config.get("stock_data_file"),
        allow_train_fallback=True,
        logger=logger,
    )
    train_df, val_df, val_start, _ = split_train_val_by_recent_trading_days(
        full_df,
        config["sequence_length"],
    )
    train_target_dates, val_target_dates = target_date_sets(full_df, val_start)
    stock_ids = sorted(full_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    logger.info("股票映射数量: %s", len(stockid2idx))

    feature_start_time = time.perf_counter()
    train_data, features = preprocess_data(
        train_df, is_train=True, stockid2idx=stockid2idx
    )
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    train_data = restrict_target_dates(train_data, train_target_dates)
    val_data = restrict_target_dates(val_data, val_target_dates)
    train_data = clean_features(train_data, features)
    val_data = clean_features(val_data, features)
    train_data = sample_train_rows(
        train_data,
        max_stocks_per_day=config.get("max_stocks_per_day", 0),
        seed=config.get("seed", 42),
    )
    feature_seconds = time.perf_counter() - feature_start_time
    if train_data.empty or val_data.empty:
        raise ValueError("LightGBM 特征清洗后训练集或验证集为空")

    logger.info("特征数量: %s", len(features))
    logger.info(
        "训练样本: 行=%s, 目标日=%s | 验证样本: 行=%s, 目标日=%s | 特征耗时=%s",
        len(train_data),
        train_data["日期"].nunique(),
        len(val_data),
        val_data["日期"].nunique(),
        format_duration(feature_seconds),
    )

    model = build_model(lgb)
    train_start_time = time.perf_counter()
    model.fit(train_data[features], train_data["label"])
    train_seconds = time.perf_counter() - train_start_time

    eval_start_time = time.perf_counter()
    train_pred = model.predict(train_data[features])
    val_pred = model.predict(val_data[features])
    eval_seconds = time.perf_counter() - eval_start_time
    train_eval = train_data[["日期", "股票代码", "label"]].copy()
    val_eval = val_data[["日期", "股票代码", "label"]].copy()
    train_eval["pred_score"] = train_pred
    val_eval["pred_score"] = val_pred
    train_metrics = rank_metrics(train_eval)
    val_metrics = rank_metrics(val_eval)
    train_rmse = rmse(train_data["label"], train_pred)
    val_rmse = rmse(val_data["label"], val_pred)

    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    write_json(
        os.path.join(output_dir, "features.json"),
        {
            "model_kind": "lgbm",
            "features": features,
            "stock_ids": stock_ids,
            "feature_num": config.get("feature_num"),
            "data_file": str(data_file),
            "val_start": val_start.strftime("%Y-%m-%d"),
        },
    )
    history = pd.DataFrame(
        [
            {
                "iteration": int(config.get("lgbm_n_estimators", 300)),
                "train_rmse": train_rmse,
                "eval_rmse": val_rmse,
                "train_final_score": train_metrics["final_score"],
                "eval_final_score": val_metrics["final_score"],
                "train_seconds": round(train_seconds, 3),
                "eval_seconds": round(eval_seconds, 3),
                "feature_seconds": round(feature_seconds, 3),
            }
        ]
    )
    history.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    total_seconds = time.perf_counter() - run_start_time
    logger.info(
        "Train RMSE: %.6f | final_score=%.6f", train_rmse, train_metrics["final_score"]
    )
    logger.info(
        "Eval RMSE: %.6f | final_score=%.6f", val_rmse, val_metrics["final_score"]
    )
    logger.info("模型已保存: %s", os.path.join(output_dir, "model.pkl"))
    logger.info("特征说明已保存: %s", os.path.join(output_dir, "features.json"))
    logger.info("训练完成: 总耗时=%s", format_duration(total_seconds))

    with open(os.path.join(output_dir, "final_score.txt"), "w", encoding="utf-8") as f:
        f.write("Model kind: lgbm\n")
        f.write(f"Best epoch: {config.get('lgbm_n_estimators', 300)}\n")
        f.write(f"Best final_score: {val_metrics['final_score']:.6f}\n")
        f.write(f"Train RMSE: {train_rmse:.8f}\n")
        f.write(f"Eval RMSE: {val_rmse:.8f}\n")
        f.write("Stop reason: fixed_estimators\n")
        f.write(f"Total duration: {format_duration(total_seconds)}\n")
        f.write(f"Total seconds: {total_seconds:.3f}\n")
        f.write(
            f"Training history: {os.path.join(output_dir, 'training_history.csv')}\n"
        )

    return val_metrics["final_score"]


if __name__ == "__main__":
    score = main()
    logger.info(
        "########## LightGBM 训练完成！验证 final score: %.4f ##########", score
    )
