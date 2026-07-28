from __future__ import annotations

import pandas as pd

from data_utils import normalize_stock_codes


MODEL_TOP5 = "model_top5"
LOW_VOL_THEN_RANK_TOP5 = "low_vol_then_rank_top5"
ENSEMBLE_AVG_RANK_TOP5 = "ensemble_avg_rank_top5"
ENSEMBLE_LOW_VOL_TOP5 = "ensemble_low_vol_top5"
RISK_METRIC_COLUMNS = [
    "stock_id",
    "stock_prefix",
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
]


def normalize_selection_strategy(value: str | None) -> str:
    strategy = (value or MODEL_TOP5).strip().lower().replace("-", "_")
    aliases = {
        "": MODEL_TOP5,
        "off": MODEL_TOP5,
        "none": MODEL_TOP5,
        "score": MODEL_TOP5,
        "score_top5": MODEL_TOP5,
        "model": MODEL_TOP5,
        "model_top5": MODEL_TOP5,
        "low_vol": LOW_VOL_THEN_RANK_TOP5,
        "low_vol_topn": LOW_VOL_THEN_RANK_TOP5,
        "low_vol_top5": LOW_VOL_THEN_RANK_TOP5,
        "low_vol_then_rank": LOW_VOL_THEN_RANK_TOP5,
        "low_vol_then_rank_top5": LOW_VOL_THEN_RANK_TOP5,
    }
    if strategy not in aliases:
        raise ValueError(
            f"Unsupported BDC_SELECTION_STRATEGY: {value}. "
            f"Supported: {MODEL_TOP5}, {LOW_VOL_THEN_RANK_TOP5}"
        )
    return aliases[strategy]


def normalize_ensemble_selection_strategy(value: str | None) -> str:
    strategy = (value or ENSEMBLE_LOW_VOL_TOP5).strip().lower().replace("-", "_")
    aliases = {
        "ensemble_avg": ENSEMBLE_AVG_RANK_TOP5,
        "ensemble_avg_rank": ENSEMBLE_AVG_RANK_TOP5,
        "ensemble_avg_rank_top5": ENSEMBLE_AVG_RANK_TOP5,
        "ensemble_low_vol": ENSEMBLE_LOW_VOL_TOP5,
        "ensemble_low_vol_top5": ENSEMBLE_LOW_VOL_TOP5,
        "top5_union_low_vol": ENSEMBLE_LOW_VOL_TOP5,
        "union_low_vol": ENSEMBLE_LOW_VOL_TOP5,
    }
    if strategy not in aliases:
        raise ValueError(
            f"Unsupported ensemble selection strategy: {value}. "
            f"Supported: {ENSEMBLE_AVG_RANK_TOP5}, {ENSEMBLE_LOW_VOL_TOP5}"
        )
    return aliases[strategy]


def _recent_return(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return 0.0
    base = close.iloc[-days - 1]
    if base == 0:
        return 0.0
    return float(close.iloc[-1] / base - 1.0)


def compute_recent_risk_metrics(history: pd.DataFrame, volatility_window: int = 20) -> pd.DataFrame:
    required = {"股票代码", "日期", "收盘"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"缺少二阶段风险信号所需列: {sorted(missing)}")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than 1")

    data = history.copy()
    data["stock_id"] = normalize_stock_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.normalize()
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["stock_id", "日期", "收盘"]).sort_values(["stock_id", "日期"])

    rows = []
    for stock_id, group in data.groupby("stock_id", sort=False):
        close = group["收盘"].astype(float).reset_index(drop=True)
        if close.empty:
            continue

        daily_returns = close.pct_change(fill_method=None).tail(volatility_window).dropna()
        volatility = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0
        recent_close = close.tail(volatility_window)
        running_max = recent_close.cummax()
        drawdowns = recent_close / running_max - 1.0
        max_drawdown = float(-min(drawdowns.min(), 0.0)) if len(drawdowns) else 0.0

        rows.append(
            {
                "stock_id": stock_id,
                "stock_prefix": stock_id[:3],
                "recent_return_5": _recent_return(close, 5),
                "recent_return_10": _recent_return(close, 10),
                "recent_return_20": _recent_return(close, 20),
                "volatility_20": volatility,
                "max_drawdown_20": max_drawdown,
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return pd.DataFrame(columns=RISK_METRIC_COLUMNS)

    metrics["overheat_return"] = metrics[["recent_return_5", "recent_return_10"]].max(axis=1)
    metrics["volatility_20_rank"] = metrics["volatility_20"].rank(pct=True)
    metrics["overheat_rank"] = metrics["overheat_return"].rank(pct=True)
    metrics["drawdown_rank"] = metrics["max_drawdown_20"].rank(pct=True)
    metrics["risk_score"] = (
        0.40 * metrics["volatility_20_rank"]
        + 0.35 * metrics["overheat_rank"]
        + 0.25 * metrics["drawdown_rank"]
    )
    return metrics


def annotate_scores_with_risk(
    scores: pd.DataFrame,
    history: pd.DataFrame,
    volatility_window: int = 20,
) -> pd.DataFrame:
    metrics = compute_recent_risk_metrics(history, volatility_window=volatility_window)
    annotated = scores.copy()
    annotated["stock_id"] = normalize_stock_codes(annotated["stock_id"])
    if "stock_prefix" not in annotated.columns:
        annotated["stock_prefix"] = annotated["stock_id"].str[:3]
    annotated = annotated.merge(metrics, on="stock_id", how="left", suffixes=("", "_risk"))
    if "stock_prefix_risk" in annotated.columns:
        annotated = annotated.drop(columns=["stock_prefix_risk"])
    annotated["stock_prefix"] = annotated["stock_prefix"].fillna(annotated["stock_id"].str[:3])

    rank_columns = ["volatility_20_rank", "overheat_rank", "drawdown_rank", "risk_score"]
    for column in rank_columns:
        annotated[column] = pd.to_numeric(annotated[column], errors="coerce").fillna(0.5)

    raw_columns = [
        "recent_return_5",
        "recent_return_10",
        "recent_return_20",
        "volatility_20",
        "max_drawdown_20",
        "overheat_return",
    ]
    for column in raw_columns:
        annotated[column] = pd.to_numeric(annotated[column], errors="coerce").fillna(0.0)
    return annotated


def select_predictions(
    scores: pd.DataFrame,
    history: pd.DataFrame,
    strategy: str = MODEL_TOP5,
    top_k: int = 5,
    pool_size: int = 10,
    volatility_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = normalize_selection_strategy(strategy)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if pool_size < top_k:
        raise ValueError(f"BDC_STAGE2_POOL_SIZE must be >= {top_k}, current: {pool_size}")

    ranked = scores.sort_values("rank").reset_index(drop=True).copy()
    if len(ranked) < top_k:
        raise ValueError(f"可预测股票不足 {top_k} 只，当前仅有 {len(ranked)} 只")

    annotated = annotate_scores_with_risk(
        ranked,
        history=history,
        volatility_window=volatility_window,
    )
    effective_pool_size = min(pool_size, len(annotated))
    annotated["selection_strategy"] = strategy
    annotated["stage2_pool_member"] = False
    annotated["stage2_selection_rank"] = pd.NA

    if strategy == MODEL_TOP5:
        selected = annotated.head(top_k).copy()
    elif strategy == LOW_VOL_THEN_RANK_TOP5:
        pool = annotated.head(effective_pool_size).copy()
        annotated.loc[annotated["rank"].le(effective_pool_size), "stage2_pool_member"] = True
        selected = pool.sort_values(["volatility_20_rank", "rank"]).head(top_k).copy()
    else:
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    selected = selected.reset_index(drop=True)
    selected["stage2_selection_rank"] = range(1, len(selected) + 1)
    selected_ids = set(selected["stock_id"])
    annotated["selected"] = annotated["stock_id"].isin(selected_ids)
    annotated["weight"] = 0.0
    annotated.loc[annotated["selected"], "weight"] = 1.0 / top_k
    rank_map = dict(zip(selected["stock_id"], selected["stage2_selection_rank"]))
    annotated["stage2_selection_rank"] = annotated["stock_id"].map(rank_map)

    selected = annotated[annotated["selected"]].copy()
    selected = selected.sort_values(["stage2_selection_rank", "rank"]).reset_index(drop=True)
    return selected, annotated


def build_ensemble_scores(score_frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if len(score_frames) < 2:
        raise ValueError("ensemble requires at least two score frames")

    combined = None
    rank_columns = []
    for label, frame in score_frames:
        required = {"rank", "stock_id", "pred_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} score frame missing columns: {sorted(missing)}")

        partial = frame[["rank", "stock_id", "pred_score"]].copy()
        partial["stock_id"] = normalize_stock_codes(partial["stock_id"])
        partial["rank"] = pd.to_numeric(partial["rank"], errors="coerce")
        partial["pred_score"] = pd.to_numeric(partial["pred_score"], errors="coerce")
        partial = partial.dropna(subset=["rank", "stock_id"]).copy()
        partial["rank"] = partial["rank"].astype(int)
        partial = partial.rename(
            columns={
                "rank": f"rank_{label}",
                "pred_score": f"pred_score_{label}",
            }
        )
        rank_columns.append(f"rank_{label}")
        combined = partial if combined is None else combined.merge(partial, on="stock_id", how="outer")

    assert combined is not None
    combined["stock_prefix"] = combined["stock_id"].str[:3]
    combined["model_count"] = len(score_frames)
    combined["top5_vote_count"] = sum(combined[column].le(5).fillna(False).astype(int) for column in rank_columns)
    combined["top20_vote_count"] = sum(combined[column].le(20).fillna(False).astype(int) for column in rank_columns)
    combined["top5_borda"] = sum((6 - combined[column]).clip(lower=0).fillna(0) for column in rank_columns)
    combined["top20_borda"] = sum((21 - combined[column]).clip(lower=0).fillna(0) for column in rank_columns)
    combined["avg_rank_fill"] = combined[rank_columns].fillna(9999).mean(axis=1)
    combined["min_rank_fill"] = combined[rank_columns].fillna(9999).min(axis=1)
    combined["ensemble_score"] = (
        combined["top5_borda"] * 10000
        + combined["top20_borda"] * 100
        - combined["avg_rank_fill"]
    )
    combined = combined.sort_values(
        ["top5_borda", "top20_borda", "avg_rank_fill", "min_rank_fill"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    combined["ensemble_rank_before_stage2"] = range(1, len(combined) + 1)
    return combined


def select_ensemble_predictions(
    score_frames: list[tuple[str, pd.DataFrame]],
    history: pd.DataFrame,
    strategy: str = ENSEMBLE_LOW_VOL_TOP5,
    top_k: int = 5,
    volatility_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = normalize_ensemble_selection_strategy(strategy)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    combined = build_ensemble_scores(score_frames)
    candidates = combined[combined["top5_vote_count"] > 0].copy()
    if len(candidates) < top_k:
        raise ValueError(f"ensemble top5 union candidates不足 {top_k} 只，当前仅有 {len(candidates)} 只")

    candidates = annotate_scores_with_risk(
        candidates,
        history=history,
        volatility_window=volatility_window,
    )
    if strategy == ENSEMBLE_AVG_RANK_TOP5:
        selected = candidates.sort_values(["avg_rank_fill", "min_rank_fill"]).head(top_k).copy()
    elif strategy == ENSEMBLE_LOW_VOL_TOP5:
        selected = candidates.sort_values(["volatility_20_rank", "avg_rank_fill", "min_rank_fill"]).head(top_k).copy()
    else:
        raise ValueError(f"Unsupported ensemble selection strategy: {strategy}")

    selected = selected.reset_index(drop=True)
    selected["stage2_selection_rank"] = range(1, len(selected) + 1)
    selected_ids = set(selected["stock_id"])

    annotated = annotate_scores_with_risk(
        combined,
        history=history,
        volatility_window=volatility_window,
    )
    rank_map = dict(zip(selected["stock_id"], selected["stage2_selection_rank"]))
    annotated["selection_strategy"] = strategy
    annotated["stage2_pool_member"] = annotated["top5_vote_count"] > 0
    annotated["stage2_selection_rank"] = annotated["stock_id"].map(rank_map)
    annotated["selected"] = annotated["stock_id"].isin(selected_ids)
    annotated["weight"] = 0.0
    annotated.loc[annotated["selected"], "weight"] = 1.0 / top_k
    annotated = annotated.sort_values(
        ["selected", "stage2_selection_rank", "ensemble_rank_before_stage2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    annotated["rank"] = range(1, len(annotated) + 1)
    annotated["pred_score"] = (
        annotated["selected"].astype(int) * 1000000
        - annotated["stage2_selection_rank"].fillna(9999).astype(float)
        + annotated["ensemble_score"] / 1000000
    )

    selected = annotated[annotated["selected"]].copy()
    selected = selected.sort_values(["stage2_selection_rank", "ensemble_rank_before_stage2"]).reset_index(drop=True)
    return selected, annotated
