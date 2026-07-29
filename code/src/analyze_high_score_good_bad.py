from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import normalize_stock_codes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_K = 20
DEFAULT_GOOD_QUANTILE = 0.8
DEFAULT_BAD_QUANTILE = 0.2


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def experiment_label(path: Path) -> str:
    return path.name


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def recent_return(close: pd.Series, days: int) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna().reset_index(drop=True)
    if len(close) <= days:
        return 0.0
    base = close.iloc[-days - 1]
    if abs(base) <= 1e-12:
        return 0.0
    return float(close.iloc[-1] / base - 1.0)


def rolling_max_drawdown(close: pd.Series, days: int) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna().tail(days)
    if close.empty:
        return 0.0
    running_max = close.cummax()
    drawdown = close / running_max.where(running_max.abs() > 1e-12) - 1.0
    return float(-min(drawdown.min(), 0.0))


def close_position(close: pd.Series, days: int) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna().tail(days)
    if close.empty:
        return 0.5
    low = close.min()
    high = close.max()
    width = high - low
    if abs(width) <= 1e-12:
        return 0.5
    return float((close.iloc[-1] - low) / width)


def slope_ratio(close: pd.Series, days: int) -> float:
    close = (
        pd.to_numeric(close, errors="coerce").dropna().tail(days).reset_index(drop=True)
    )
    if len(close) < max(5, days // 2):
        return 0.0
    y = np.log(close.clip(lower=1e-12).to_numpy())
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    noise = np.std(residual)
    return float(slope / (noise + 1e-12))


def compute_history_features(history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "股票代码",
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "换手率",
        "涨跌幅",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")

    data = history.copy()
    data["stock_id"] = normalize_stock_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.normalize()
    numeric_columns = [
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "换手率",
        "涨跌幅",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["stock_id", "日期", "收盘"]).sort_values(
        ["stock_id", "日期"]
    )

    rows = []
    for stock_id, group in data.groupby("stock_id", sort=False):
        group = group.sort_values("日期")
        close = group["收盘"].reset_index(drop=True)
        open_ = group["开盘"].reset_index(drop=True)
        high = group["最高"].reset_index(drop=True)
        low = group["最低"].reset_index(drop=True)
        volume = group["成交量"].reset_index(drop=True)
        amount = group["成交额"].reset_index(drop=True)
        turnover = group["换手率"].reset_index(drop=True)
        pct_chg = group["涨跌幅"].reset_index(drop=True)
        daily_return = close.pct_change(fill_method=None)

        last_close = close.iloc[-1] if len(close) else np.nan
        last_open = open_.iloc[-1] if len(open_) else np.nan
        rows.append(
            {
                "stock_id": stock_id,
                "hist_return_3": recent_return(close, 3),
                "hist_return_5": recent_return(close, 5),
                "hist_return_10": recent_return(close, 10),
                "hist_return_20": recent_return(close, 20),
                "hist_return_40": recent_return(close, 40),
                "hist_volatility_5": (
                    float(daily_return.tail(5).std())
                    if len(daily_return.dropna()) >= 3
                    else 0.0
                ),
                "hist_volatility_10": (
                    float(daily_return.tail(10).std())
                    if len(daily_return.dropna()) >= 5
                    else 0.0
                ),
                "hist_volatility_20": (
                    float(daily_return.tail(20).std())
                    if len(daily_return.dropna()) >= 10
                    else 0.0
                ),
                "hist_drawdown_10": rolling_max_drawdown(close, 10),
                "hist_drawdown_20": rolling_max_drawdown(close, 20),
                "hist_close_pos_20": close_position(close, 20),
                "hist_close_pos_40": close_position(close, 40),
                "hist_up_ratio_5": (
                    float(pct_chg.tail(5).gt(0).mean()) if len(pct_chg.tail(5)) else 0.0
                ),
                "hist_up_ratio_10": (
                    float(pct_chg.tail(10).gt(0).mean())
                    if len(pct_chg.tail(10))
                    else 0.0
                ),
                "hist_turnover_5": (
                    float(turnover.tail(5).mean()) if len(turnover.tail(5)) else 0.0
                ),
                "hist_turnover_20": (
                    float(turnover.tail(20).mean()) if len(turnover.tail(20)) else 0.0
                ),
                "hist_amount_5": (
                    float(amount.tail(5).mean()) if len(amount.tail(5)) else 0.0
                ),
                "hist_amount_20": (
                    float(amount.tail(20).mean()) if len(amount.tail(20)) else 0.0
                ),
                "hist_volume_5": (
                    float(volume.tail(5).mean()) if len(volume.tail(5)) else 0.0
                ),
                "hist_volume_20": (
                    float(volume.tail(20).mean()) if len(volume.tail(20)) else 0.0
                ),
                "hist_amount_ratio_5_20": float(
                    amount.tail(5).mean() / (amount.tail(20).mean() + 1e-12) - 1.0
                ),
                "hist_turnover_ratio_5_20": float(
                    turnover.tail(5).mean() / (turnover.tail(20).mean() + 1e-12) - 1.0
                ),
                "hist_volume_ratio_5_20": float(
                    volume.tail(5).mean() / (volume.tail(20).mean() + 1e-12) - 1.0
                ),
                "hist_intraday_range": float(
                    (
                        (high.tail(5) - low.tail(5))
                        / close.tail(5).where(close.tail(5).abs() > 1e-12)
                    ).mean()
                ),
                "hist_open_close_gap": float(
                    (
                        (close.tail(5) - open_.tail(5))
                        / open_.tail(5).where(open_.tail(5).abs() > 1e-12)
                    ).mean()
                ),
                "hist_close_gap_5ma": float(
                    last_close / (close.tail(5).mean() + 1e-12) - 1.0
                ),
                "hist_close_gap_20ma": float(
                    last_close / (close.tail(20).mean() + 1e-12) - 1.0
                ),
                "hist_trend_slope_20": slope_ratio(close, 20),
                "hist_no_trade_20": float(
                    (
                        (volume.tail(20).fillna(0) <= 0)
                        & (amount.tail(20).fillna(0) <= 0)
                    ).mean()
                ),
                "hist_last_close": float(last_close),
                "hist_last_open_close": (
                    float((last_close - last_open) / (last_open + 1e-12))
                    if pd.notna(last_open)
                    else 0.0
                ),
            }
        )

    features = pd.DataFrame(rows)
    numeric = [column for column in features.columns if column != "stock_id"]
    for column in numeric:
        features[column] = (
            pd.to_numeric(features[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return features


def add_cross_sectional_versions(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    added = []
    for column in columns:
        if column not in result.columns:
            continue
        rank_column = f"{column}_xrank"
        z_column = f"{column}_xz"
        result[rank_column] = (
            pd.to_numeric(result[column], errors="coerce").rank(pct=True).fillna(0.5)
        )
        result[z_column] = zscore(result[column]).fillna(0.0)
        added.extend([rank_column, z_column])
    return result, added


def window_paths(experiment_dir: Path) -> list[Path]:
    paths = sorted((experiment_dir / "windows").glob("window_*"))
    if not paths:
        raise FileNotFoundError(f"missing window directories: {experiment_dir}")
    return paths


def read_window_frame(
    window_dir: Path, top_k: int, good_quantile: float, bad_quantile: float
) -> pd.DataFrame:
    metadata = read_json(window_dir / "metadata.json")
    train_data = resolve_path(
        metadata.get("train_data", window_dir / "data" / "train_until_as_of.csv")
    )
    scores_path = resolve_path(
        metadata.get("prediction_scores", window_dir / "prediction_scores.csv")
    )
    diagnostics_path = resolve_path(
        metadata.get(
            "prediction_diagnostics", window_dir / "prediction_diagnostics.csv"
        )
    )

    scores = pd.read_csv(scores_path, dtype={"stock_id": str})
    diagnostics = pd.read_csv(diagnostics_path, dtype={"stock_id": str})
    history = pd.read_csv(train_data, dtype={"股票代码": str})

    scores["stock_id"] = normalize_stock_codes(scores["stock_id"])
    diagnostics["stock_id"] = normalize_stock_codes(diagnostics["stock_id"])
    for frame in (scores, diagnostics):
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")

    merged_columns = [
        "rank",
        "stock_id",
        "target_return",
        "target_return_percentile",
        "target_return_rank",
    ]
    merged = scores.merge(
        diagnostics[merged_columns],
        on=["rank", "stock_id"],
        how="left",
        suffixes=("", "_diag"),
    )
    history_features = compute_history_features(history)
    merged = merged.merge(history_features, on="stock_id", how="left")
    numeric_base = [
        column
        for column in merged.columns
        if column.startswith("hist_")
        or column
        in {
            "recent_return_5",
            "recent_return_10",
            "recent_return_20",
            "volatility_20",
            "max_drawdown_20",
            "overheat_return",
            "volatility_20_rank",
            "overheat_rank",
            "drawdown_rank",
            "risk_score",
        }
    ]
    merged, added_columns = add_cross_sectional_versions(merged, numeric_base)
    merged["window"] = window_dir.name
    merged["target_start_date"] = metadata.get("window", {}).get("target_start_date")
    merged["target_end_date"] = metadata.get("window", {}).get("target_end_date")
    merged["as_of_date"] = metadata.get("window", {}).get("as_of_date")
    merged["stock_prefix"] = merged["stock_id"].str[:3]
    merged["in_high_score_pool"] = merged["rank"] <= top_k
    merged["good_threshold"] = good_quantile
    merged["bad_threshold"] = bad_quantile
    merged["quality_label"] = "middle"
    merged.loc[merged["target_return_percentile"] >= good_quantile, "quality_label"] = (
        "good"
    )
    merged.loc[merged["target_return_percentile"] <= bad_quantile, "quality_label"] = (
        "bad"
    )
    merged.loc[~merged["in_high_score_pool"], "quality_label"] = "outside_pool"
    merged["top_pool_feature_columns"] = ",".join(numeric_base + added_columns)
    return merged


def load_candidates(
    experiment_dir: Path, top_k: int, good_quantile: float, bad_quantile: float
) -> pd.DataFrame:
    frames = [
        read_window_frame(
            path, top_k=top_k, good_quantile=good_quantile, bad_quantile=bad_quantile
        )
        for path in window_paths(experiment_dir)
    ]
    combined = pd.concat(frames, ignore_index=True)
    return combined


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "rank",
        "stock_id",
        "stock_prefix",
        "selected",
        "weight",
        "selection_strategy",
        "stage2_pool_member",
        "stage2_selection_rank",
        "window",
        "target_start_date",
        "target_end_date",
        "as_of_date",
        "target_return",
        "target_return_rank",
        "target_return_percentile",
        "quality_label",
        "in_high_score_pool",
        "good_threshold",
        "bad_threshold",
        "top_pool_feature_columns",
    }
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def summarize_features(pool: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    good = pool[pool["quality_label"] == "good"]
    bad = pool[pool["quality_label"] == "bad"]
    rows = []
    for column in columns:
        good_values = pd.to_numeric(good[column], errors="coerce").dropna()
        bad_values = pd.to_numeric(bad[column], errors="coerce").dropna()
        if good_values.empty or bad_values.empty:
            continue
        good_mean = good_values.mean()
        bad_mean = bad_values.mean()
        pooled_std = pd.concat([good_values, bad_values]).std(ddof=0)
        mean_diff = good_mean - bad_mean
        if len(good_values) and len(bad_values):
            pairwise_bad_gt_good = (
                bad_values.to_numpy()[:, None] > good_values.to_numpy()[None, :]
            ).mean()
        else:
            pairwise_bad_gt_good = None
        standardized_diff = float(mean_diff / pooled_std) if pooled_std > 1e-12 else 0.0
        rows.append(
            {
                "feature": column,
                "good_mean": float(good_mean),
                "bad_mean": float(bad_mean),
                "mean_diff_good_minus_bad": float(mean_diff),
                "abs_mean_diff": float(abs(mean_diff)),
                "standardized_diff": standardized_diff,
                "abs_standardized_diff": abs(standardized_diff),
                "good_median": float(good_values.median()),
                "bad_median": float(bad_values.median()),
                "good_count": int(len(good_values)),
                "bad_count": int(len(bad_values)),
                "bad_gt_good_rate": safe_float(pairwise_bad_gt_good),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["abs_standardized_diff", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def summarize_groups(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in pool.groupby("quality_label", sort=True):
        if label not in {"good", "bad", "middle"}:
            continue
        rows.append(
            {
                "quality_label": label,
                "count": int(len(group)),
                "mean_rank": float(group["rank"].mean()),
                "mean_pred_score": float(group["pred_score"].mean()),
                "mean_target_return": float(group["target_return"].mean()),
                "median_target_return": float(group["target_return"].median()),
                "selected_count": (
                    int(
                        group["selected"]
                        .astype(str)
                        .str.lower()
                        .isin({"true", "1"})
                        .sum()
                    )
                    if "selected" in group
                    else 0
                ),
                "unique_stocks": int(group["stock_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_repeated(pool: pd.DataFrame) -> pd.DataFrame:
    top = pool[pool["quality_label"].isin(["good", "bad", "middle"])].copy()
    top["is_good"] = top["quality_label"].eq("good")
    top["is_bad"] = top["quality_label"].eq("bad")
    grouped = top.groupby("stock_id", as_index=False).agg(
        stock_prefix=("stock_prefix", "first"),
        windows=("window", lambda values: ",".join(sorted(set(values)))),
        appearances=("window", "count"),
        good_count=("is_good", "sum"),
        bad_count=("is_bad", "sum"),
        avg_rank=("rank", "mean"),
        best_rank=("rank", "min"),
        avg_target_return=("target_return", "mean"),
        min_target_return=("target_return", "min"),
        max_target_return=("target_return", "max"),
        avg_pred_score=("pred_score", "mean"),
    )
    grouped["bad_minus_good_count"] = grouped["bad_count"] - grouped["good_count"]
    return grouped.sort_values(
        ["bad_minus_good_count", "appearances", "avg_target_return"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def selected_from_pool(
    window_pool: pd.DataFrame,
    keep: pd.Series | None = None,
    score: pd.Series | None = None,
    count: int = 5,
) -> pd.DataFrame:
    pool = window_pool.sort_values("rank").copy()
    if keep is not None:
        keep = keep.reindex(pool.index).fillna(False).astype(bool)
        selected = pool[keep].copy()
    else:
        selected = pool.copy()

    if score is not None:
        selected = selected.assign(_trial_score=score.reindex(selected.index))
        selected = selected.sort_values(["_trial_score", "rank"])
    else:
        selected = selected.sort_values("rank")
    selected = selected.head(count)

    if len(selected) < count:
        filler = (
            pool[~pool["stock_id"].isin(selected["stock_id"])]
            .sort_values("rank")
            .head(count - len(selected))
        )
        selected = pd.concat(
            [selected.drop(columns=["_trial_score"], errors="ignore"), filler],
            ignore_index=True,
        )
    return selected.head(count)


def evaluate_gate_trials(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows = []
    summary_rows = []

    def add_trial(
        window_pool: pd.DataFrame,
        name: str,
        description: str,
        selected: pd.DataFrame,
        baseline_ids: set[str],
    ) -> None:
        selected = selected.copy()
        selected["quality_label"] = selected["quality_label"].fillna("middle")
        returns = pd.to_numeric(selected["target_return"], errors="coerce").fillna(0.0)
        labels = selected["quality_label"].astype(str)
        trial_rows.append(
            {
                "window": str(window_pool["window"].iloc[0]),
                "trial": name,
                "description": description,
                "mean_return": float(returns.mean()),
                "good_count": int(labels.eq("good").sum()),
                "bad_count": int(labels.eq("bad").sum()),
                "middle_count": int(labels.eq("middle").sum()),
                "selected_count": int(len(selected)),
                "replaced_count": int(
                    len(set(selected["stock_id"]) ^ baseline_ids) / 2
                ),
                "selected_stocks": ",".join(selected["stock_id"].astype(str)),
                "avg_rank": float(
                    pd.to_numeric(selected["rank"], errors="coerce").mean()
                ),
            }
        )

    for _, window_pool in pool.groupby("window", sort=True):
        window_pool = window_pool.sort_values("rank").copy()
        baseline = selected_from_pool(window_pool)
        baseline_ids = set(baseline["stock_id"])
        add_trial(
            window_pool,
            "baseline_rank_top5",
            "原始模型 rank 前 5，等权持仓。",
            baseline,
            baseline_ids,
        )

        if "hist_return_3_xrank" in window_pool:
            add_trial(
                window_pool,
                "drop_ret3_xrank_ge_0.85",
                "过滤最近 3 日收益横截面 rank 最高的 15%，不足 5 只时用原始排名补齐。",
                selected_from_pool(
                    window_pool, keep=window_pool["hist_return_3_xrank"] < 0.85
                ),
                baseline_ids,
            )
        if "hist_close_gap_5ma_xrank" in window_pool:
            add_trial(
                window_pool,
                "drop_gap5ma_xrank_ge_0.85",
                "过滤收盘价相对 5 日均线偏离横截面 rank 最高的 15%，不足 5 只时用原始排名补齐。",
                selected_from_pool(
                    window_pool, keep=window_pool["hist_close_gap_5ma_xrank"] < 0.85
                ),
                baseline_ids,
            )
        if "hist_intraday_range_xrank" in window_pool:
            add_trial(
                window_pool,
                "drop_range_xrank_ge_0.85",
                "过滤近 5 日日内振幅横截面 rank 最高的 15%，不足 5 只时用原始排名补齐。",
                selected_from_pool(
                    window_pool, keep=window_pool["hist_intraday_range_xrank"] < 0.85
                ),
                baseline_ids,
            )
        if {"hist_return_3_xrank", "hist_close_gap_5ma_xrank"}.issubset(
            window_pool.columns
        ):
            short_overheat = window_pool[
                ["hist_return_3_xrank", "hist_close_gap_5ma_xrank"]
            ].max(axis=1)
            add_trial(
                window_pool,
                "drop_short_overheat_ge_0.85",
                "过滤 3 日收益 rank 或 5 日均线偏离 rank 达到 0.85 的短期过热股票。",
                selected_from_pool(window_pool, keep=short_overheat < 0.85),
                baseline_ids,
            )
            add_trial(
                window_pool,
                "penalize_short_overheat",
                "不硬过滤，按 rank + 6 * 短期过热分位重排。",
                selected_from_pool(
                    window_pool, score=window_pool["rank"] + 6.0 * short_overheat
                ),
                baseline_ids,
            )
        if {
            "hist_return_3_xrank",
            "hist_close_gap_5ma_xrank",
            "hist_intraday_range_xrank",
        }.issubset(window_pool.columns):
            short_risk = window_pool[
                [
                    "hist_return_3_xrank",
                    "hist_close_gap_5ma_xrank",
                    "hist_intraday_range_xrank",
                ]
            ].mean(axis=1)
            add_trial(
                window_pool,
                "penalize_short_overheat_range",
                "按 rank + 6 * (3 日收益、5MA偏离、日内振幅的平均分位) 重排。",
                selected_from_pool(
                    window_pool, score=window_pool["rank"] + 6.0 * short_risk
                ),
                baseline_ids,
            )

    windows = pd.DataFrame(trial_rows)
    if windows.empty:
        return pd.DataFrame(), windows

    baseline = windows[windows["trial"] == "baseline_rank_top5"][
        ["window", "mean_return"]
    ].rename(columns={"mean_return": "baseline_return"})
    windows = windows.merge(baseline, on="window", how="left")
    windows["return_diff_vs_baseline"] = (
        windows["mean_return"] - windows["baseline_return"]
    )

    for trial, group in windows.groupby("trial", sort=True):
        diffs = pd.to_numeric(group["return_diff_vs_baseline"], errors="coerce")
        returns = pd.to_numeric(group["mean_return"], errors="coerce")
        summary_rows.append(
            {
                "trial": trial,
                "description": group["description"].iloc[0],
                "windows": int(len(group)),
                "mean_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "std_return": float(returns.std(ddof=0)),
                "min_return": float(returns.min()),
                "max_return": float(returns.max()),
                "mean_diff_vs_baseline": float(diffs.mean()),
                "win_windows_vs_baseline": int((diffs > 1e-12).sum()),
                "loss_windows_vs_baseline": int((diffs < -1e-12).sum()),
                "positive_windows": int((returns > 0.0).sum()),
                "avg_bad_count": float(group["bad_count"].mean()),
                "avg_good_count": float(group["good_count"].mean()),
                "avg_replaced_count": float(group["replaced_count"].mean()),
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["mean_diff_vs_baseline", "mean_return"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary, windows.sort_values(["window", "trial"]).reset_index(drop=True)


def write_readme(
    output_dir: Path,
    experiment_dir: Path,
    candidates: pd.DataFrame,
    feature_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    top_k: int,
    good_quantile: float,
    bad_quantile: float,
) -> None:
    pool = candidates[candidates["in_high_score_pool"]].copy()
    group_summary = summarize_groups(pool)
    good_count = int((pool["quality_label"] == "good").sum())
    bad_count = int((pool["quality_label"] == "bad").sum())
    middle_count = int((pool["quality_label"] == "middle").sum())
    strongest = (
        feature_summary.head(12) if not feature_summary.empty else pd.DataFrame()
    )
    lines = [
        "# 高分好股/坏股离线差异分析",
        "",
        "本目录由 `code/src/analyze_high_score_good_bad.py` 生成，只读取已有 walk-forward 预测诊断产物，不重训模型。",
        "",
        "## 口径",
        "",
        f"- 实验目录：`{experiment_dir.relative_to(REPO_ROOT)}`",
        f"- 高分池：每个窗口模型排名前 `{top_k}` 只股票。",
        f"- 高分好股：高分池中未来 5 日收益处于当日全市场前 `{round((1 - good_quantile) * 100)}%`，即 `target_return_percentile >= {good_quantile}`。",
        f"- 高分坏股：高分池中未来 5 日收益处于当日全市场后 `{int(bad_quantile * 100)}%`，即 `target_return_percentile <= {bad_quantile}`。",
        "- 真实未来收益只用于离线打标签和复盘，不可用于正式预测。",
        "",
        "## 文件",
        "",
        "- `candidates.csv`：所有候选股票、历史特征和好坏标签。",
        "- `high_score_pool.csv`：每个窗口 rank 前 N 的高分池。",
        "- `feature_differences.csv`：高分好股与坏股的特征均值差异。",
        "- `group_summary.csv`：good/bad/middle 的数量和收益概览。",
        "- `repeated_stocks.csv`：高分池里反复出现的股票及好坏次数。",
        "- `simple_gate_trials.csv`：只基于预测日前历史特征的简单过滤/重排试算汇总。",
        "- `simple_gate_windows.csv`：简单过滤/重排试算的逐窗口明细。",
        "",
        "## 样本概览",
        "",
        f"- 高分池总样本：`{len(pool)}`",
        f"- good/bad/middle：`{good_count}` / `{bad_count}` / `{middle_count}`",
        f"- 窗口数：`{pool['window'].nunique()}`",
        f"- 高分池平均未来收益：`{pool['target_return'].mean():.6f}`",
        "",
        "## 最明显的差异",
        "",
    ]
    if strongest.empty:
        lines.append("未生成特征差异。")
    else:
        for row in strongest.itertuples(index=False):
            direction = "good 更高" if row.mean_diff_good_minus_bad > 0 else "bad 更高"
            lines.append(
                f"- `{row.feature}`：{direction}，good均值 `{row.good_mean:.6f}`，bad均值 `{row.bad_mean:.6f}`，标准化差 `{row.standardized_diff:.3f}`。"
            )
    lines.extend(["", "## 分组概览", ""])
    if not group_summary.empty:
        for row in group_summary.itertuples(index=False):
            lines.append(
                f"- `{row.quality_label}`：样本 `{row.count}`，均值收益 `{row.mean_target_return:.6f}`，平均rank `{row.mean_rank:.2f}`，唯一股票 `{row.unique_stocks}`。"
            )
    lines.extend(["", "## 简单规则试算", ""])
    if gate_summary.empty:
        lines.append("未生成简单规则试算。")
    else:
        for row in gate_summary.head(8).itertuples(index=False):
            lines.append(
                f"- `{row.trial}`：均值收益 `{row.mean_return:.6f}`，相对原始top5 `{row.mean_diff_vs_baseline:+.6f}`，胜/负窗口 `{row.win_windows_vs_baseline}/{row.loss_windows_vs_baseline}`，平均替换 `{row.avg_replaced_count:.2f}` 只。"
            )
    lines.extend(
        [
            "",
            "## 使用提醒",
            "",
            "这份分析只能说明历史验证窗口里的相关差异。下一步如果要把某个差异做成特征或过滤规则，仍需单独作为小步实验走 6 -> 12 -> 24 窗口验证。",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readme.md").write_text("\n".join(lines), encoding="utf-8")


def default_output_dir(experiment_dir: Path) -> Path:
    return (
        REPO_ROOT
        / "experiments"
        / "analysis"
        / f"high_score_good_bad_{experiment_label(experiment_dir)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze high-score good vs bad stocks from walk-forward diagnostics"
    )
    parser.add_argument(
        "experiment", help="experiment directory, e.g. experiments/v1.4.5"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="high-score pool size per window, default 20",
    )
    parser.add_argument(
        "--good-quantile",
        type=float,
        default=DEFAULT_GOOD_QUANTILE,
        help="target return percentile threshold for good stocks, default 0.8",
    )
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=DEFAULT_BAD_QUANTILE,
        help="target return percentile threshold for bad stocks, default 0.2",
    )
    parser.add_argument(
        "--output-dir", help="output directory, default under experiments/analysis"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not 0.0 < args.bad_quantile < args.good_quantile < 1.0:
        raise ValueError("expected 0 < bad_quantile < good_quantile < 1")

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(experiment_dir)
    )
    candidates = load_candidates(
        experiment_dir,
        top_k=args.top_k,
        good_quantile=args.good_quantile,
        bad_quantile=args.bad_quantile,
    )
    pool = candidates[candidates["in_high_score_pool"]].copy()
    columns = feature_columns(pool)
    feature_summary = summarize_features(pool, columns)
    group_summary = summarize_groups(pool)
    repeated = summarize_repeated(pool)
    gate_summary, gate_windows = evaluate_gate_trials(pool)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    pool.to_csv(output_dir / "high_score_pool.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_differences.csv", index=False)
    group_summary.to_csv(output_dir / "group_summary.csv", index=False)
    repeated.to_csv(output_dir / "repeated_stocks.csv", index=False)
    gate_summary.to_csv(output_dir / "simple_gate_trials.csv", index=False)
    gate_windows.to_csv(output_dir / "simple_gate_windows.csv", index=False)
    write_readme(
        output_dir,
        experiment_dir=experiment_dir,
        candidates=candidates,
        feature_summary=feature_summary,
        gate_summary=gate_summary,
        top_k=args.top_k,
        good_quantile=args.good_quantile,
        bad_quantile=args.bad_quantile,
    )
    print(f"analysis written to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
