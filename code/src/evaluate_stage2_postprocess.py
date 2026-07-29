from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from data_utils import normalize_stock_codes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOLS = (10, 20)
DEFAULT_SHORT_WEIGHTS = (2.0, 4.0, 6.0, 8.0)
DEFAULT_VOL_WEIGHTS = (0.0, 2.0)
DEFAULT_RANGE_WEIGHTS = (0.0, 2.0)
DEFAULT_FILTER_THRESHOLDS = (0.85, 0.90)
DEFAULT_TOP_K = 5


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def experiment_label(experiment_dir: Path) -> str:
    return experiment_dir.name


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("list must not be empty")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("list must not be empty")
    return values


def rank_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(pct=True).fillna(0.5)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def recent_return(close: pd.Series, days: int) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna().reset_index(drop=True)
    if len(close) <= days:
        return 0.0
    base = close.iloc[-days - 1]
    if abs(base) <= 1e-12:
        return 0.0
    return float(close.iloc[-1] / base - 1.0)


def compute_history_signals(history: pd.DataFrame) -> pd.DataFrame:
    required = {"股票代码", "日期", "收盘", "最高", "最低"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")

    data = history.copy()
    data["stock_id"] = normalize_stock_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.normalize()
    for column in ["收盘", "最高", "最低"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["stock_id", "日期", "收盘"]).sort_values(
        ["stock_id", "日期"]
    )

    rows = []
    for stock_id, group in data.groupby("stock_id", sort=False):
        group = group.sort_values("日期")
        close = group["收盘"].reset_index(drop=True)
        high = group["最高"].reset_index(drop=True)
        low = group["最低"].reset_index(drop=True)
        daily_returns = close.pct_change(fill_method=None)
        recent_close = close.tail(5)
        recent_high = high.tail(5)
        recent_low = low.tail(5)

        rows.append(
            {
                "stock_id": stock_id,
                "hist_return_3": recent_return(close, 3),
                "hist_gap_5ma": (
                    float(close.iloc[-1] / (recent_close.mean() + 1e-12) - 1.0)
                    if len(close)
                    else 0.0
                ),
                "hist_intraday_range_5": (
                    float(
                        (
                            (recent_high - recent_low)
                            / recent_close.where(recent_close.abs() > 1e-12)
                        ).mean()
                    )
                    if len(recent_close)
                    else 0.0
                ),
                "hist_volatility_20": (
                    float(daily_returns.tail(20).std())
                    if len(daily_returns.dropna()) >= 10
                    else 0.0
                ),
            }
        )

    signals = pd.DataFrame(rows)
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "ret3_rank",
                "gap5ma_rank",
                "short_overheat_rank",
                "intraday_range_rank",
                "volatility_20_rank",
            ]
        )

    raw_columns = [
        "hist_return_3",
        "hist_gap_5ma",
        "hist_intraday_range_5",
        "hist_volatility_20",
    ]
    for column in raw_columns:
        signals[column] = (
            pd.to_numeric(signals[column], errors="coerce")
            .replace([float("inf"), float("-inf")], pd.NA)
            .fillna(0.0)
        )

    signals["ret3_rank"] = rank_pct(signals["hist_return_3"])
    signals["gap5ma_rank"] = rank_pct(signals["hist_gap_5ma"])
    signals["short_overheat_rank"] = signals[["ret3_rank", "gap5ma_rank"]].max(axis=1)
    signals["intraday_range_rank"] = rank_pct(signals["hist_intraday_range_5"])
    signals["volatility_20_rank"] = rank_pct(signals["hist_volatility_20"])
    return signals


def window_dirs(experiment_dir: Path) -> list[Path]:
    paths = sorted((experiment_dir / "windows").glob("window_*"))
    if not paths:
        raise FileNotFoundError(f"missing window directories: {experiment_dir}")
    return paths


def metadata_path(window_dir: Path, key: str, fallback: Path) -> Path:
    metadata = read_json(window_dir / "metadata.json")
    value = metadata.get(key)
    return resolve_path(value) if value else fallback


def read_window_frame(window_dir: Path) -> tuple[pd.DataFrame, dict]:
    metadata = read_json(window_dir / "metadata.json")
    diagnostics_path = metadata_path(
        window_dir,
        "prediction_diagnostics",
        window_dir / "prediction_diagnostics.csv",
    )
    train_data_path = metadata_path(
        window_dir,
        "train_data",
        window_dir / "data" / "train_until_as_of.csv",
    )
    diagnostics = pd.read_csv(diagnostics_path, dtype={"stock_id": str})
    history = pd.read_csv(train_data_path, dtype={"股票代码": str})

    required = {"rank", "stock_id", "pred_score", "target_return"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise ValueError(f"{diagnostics_path} missing columns: {sorted(missing)}")

    frame = diagnostics.copy()
    frame["stock_id"] = normalize_stock_codes(frame["stock_id"])
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame["target_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
    frame = frame.dropna(subset=["rank", "stock_id", "target_return"]).sort_values(
        "rank"
    )
    frame["rank"] = frame["rank"].astype(int)

    signals = compute_history_signals(history)
    frame = frame.merge(signals, on="stock_id", how="left")
    for column in [
        "ret3_rank",
        "gap5ma_rank",
        "short_overheat_rank",
        "intraday_range_rank",
        "volatility_20_rank",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.5)
    for column in [
        "hist_return_3",
        "hist_gap_5ma",
        "hist_intraday_range_5",
        "hist_volatility_20",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    window_meta = metadata.get("window", {})
    frame["window"] = window_dir.name
    frame["as_of_date"] = window_meta.get("as_of_date")
    frame["target_start_date"] = window_meta.get("target_start_date")
    frame["target_end_date"] = window_meta.get("target_end_date")
    return frame.reset_index(drop=True), metadata


def selected_from_pool(
    frame: pd.DataFrame,
    top_k: int,
    pool_size: int,
    adjusted_rank: pd.Series | None = None,
    keep: pd.Series | None = None,
) -> pd.DataFrame:
    pool = frame.sort_values("rank").head(pool_size).copy()
    if len(pool) < top_k:
        raise ValueError(f"{frame['window'].iloc[0]} has fewer than {top_k} candidates")

    if adjusted_rank is not None:
        pool["_adjusted_rank"] = adjusted_rank.reindex(pool.index)
    else:
        pool["_adjusted_rank"] = pool["rank"]

    if keep is None:
        selected = pool.sort_values(["_adjusted_rank", "rank"]).head(top_k)
    else:
        keep = keep.reindex(pool.index).fillna(False).astype(bool)
        filtered = pool[keep].copy()
        selected = filtered.sort_values(["_adjusted_rank", "rank"]).head(top_k)
        if len(selected) < top_k:
            filler = (
                pool[~pool["stock_id"].isin(selected["stock_id"])]
                .sort_values("rank")
                .head(top_k - len(selected))
            )
            selected = pd.concat([selected, filler], ignore_index=True)
    return selected.head(top_k).copy()


def score_selected(selected: pd.DataFrame, frame: pd.DataFrame) -> dict:
    returns = pd.to_numeric(selected["target_return"], errors="coerce").fillna(0.0)
    actual_top5_ids = set(frame.nlargest(5, "target_return")["stock_id"])
    return {
        "score": float(returns.mean()),
        "selected_count": int(len(selected)),
        "selected_positive_count": int((returns > 0).sum()),
        "selected_negative_count": int((returns < 0).sum()),
        "selected_min_return": float(returns.min()),
        "selected_max_return": float(returns.max()),
        "actual_top5_hits": int(selected["stock_id"].isin(actual_top5_ids).sum()),
        "selected_stock_ids": ",".join(selected["stock_id"].astype(str)),
        "selected_avg_short_overheat_rank": safe_mean(selected["short_overheat_rank"]),
        "selected_avg_volatility_20_rank": safe_mean(selected["volatility_20_rank"]),
        "selected_avg_intraday_range_rank": safe_mean(selected["intraday_range_rank"]),
        "selected_avg_model_rank": safe_mean(selected["rank"]),
    }


def rule_description(rule: str, pool_size: int, a: float, b: float, c: float) -> str:
    if rule == "baseline_model_top5":
        return "原始模型 rank 前 5，等权满仓。"
    if rule.startswith("soft_penalty"):
        return (
            f"在模型 top{pool_size} 内按 rank + {a:g} * short_overheat_rank "
            f"+ {b:g} * volatility_20_rank + {c:g} * intraday_range_rank 重排。"
        )
    if rule.startswith("drop_short_overheat"):
        threshold = rule.rsplit("_ge_", maxsplit=1)[-1].replace("p", ".")
        return f"在模型 top{pool_size} 内过滤 short_overheat_rank >= {threshold}，不足 5 只按原始 rank 补齐。"
    return rule


def build_trials(
    frame: pd.DataFrame,
    pools: list[int],
    short_weights: list[float],
    vol_weights: list[float],
    range_weights: list[float],
    filter_thresholds: list[float],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    window = frame["window"].iloc[0]
    as_of_date = frame["as_of_date"].iloc[0]
    target_start = frame["target_start_date"].iloc[0]
    target_end = frame["target_end_date"].iloc[0]
    baseline = selected_from_pool(frame, top_k=top_k, pool_size=top_k)
    baseline_ids = set(baseline["stock_id"])
    baseline_score = score_selected(baseline, frame)["score"]

    window_rows = []
    candidate_rows = []

    def add_trial(
        rule: str,
        selected: pd.DataFrame,
        pool_size: int,
        adjusted: pd.Series | None = None,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0,
        filter_threshold: float | None = None,
    ) -> None:
        selected_ids = set(selected["stock_id"])
        metrics = score_selected(selected, frame)
        pool = frame.sort_values("rank").head(pool_size).copy()
        pool["adjusted_rank"] = (
            adjusted.reindex(pool.index) if adjusted is not None else pool["rank"]
        )
        pool["selected_by_rule"] = pool["stock_id"].isin(selected_ids)

        row = {
            "window": window,
            "as_of_date": as_of_date,
            "target_start_date": target_start,
            "target_end_date": target_end,
            "rule": rule,
            "description": rule_description(rule, pool_size, a, b, c),
            "pool_size": int(pool_size),
            "top_k": int(top_k),
            "short_overheat_weight": float(a),
            "volatility_weight": float(b),
            "intraday_range_weight": float(c),
            "filter_threshold": filter_threshold,
            "baseline_score": baseline_score,
            "diff_vs_baseline": metrics["score"] - baseline_score,
            "replaced_count": int(len(selected_ids ^ baseline_ids) / 2),
            "pool_mean_return": safe_mean(pool["target_return"]),
            "pool_positive_count": int(
                (pd.to_numeric(pool["target_return"], errors="coerce") > 0).sum()
            ),
            "pool_actual_top5_hits": int(
                pool["stock_id"]
                .isin(set(frame.nlargest(5, "target_return")["stock_id"]))
                .sum()
            ),
        }
        row.update(metrics)
        window_rows.append(row)

        for candidate in pool.itertuples(index=False):
            candidate_rows.append(
                {
                    "window": window,
                    "as_of_date": as_of_date,
                    "target_start_date": target_start,
                    "target_end_date": target_end,
                    "rule": rule,
                    "pool_size": int(pool_size),
                    "stock_id": candidate.stock_id,
                    "rank": int(candidate.rank),
                    "pred_score": float(candidate.pred_score),
                    "adjusted_rank": float(candidate.adjusted_rank),
                    "selected_by_rule": bool(candidate.selected_by_rule),
                    "target_return": float(candidate.target_return),
                    "ret3_rank": float(candidate.ret3_rank),
                    "gap5ma_rank": float(candidate.gap5ma_rank),
                    "short_overheat_rank": float(candidate.short_overheat_rank),
                    "intraday_range_rank": float(candidate.intraday_range_rank),
                    "volatility_20_rank": float(candidate.volatility_20_rank),
                    "hist_return_3": float(candidate.hist_return_3),
                    "hist_gap_5ma": float(candidate.hist_gap_5ma),
                    "hist_intraday_range_5": float(candidate.hist_intraday_range_5),
                    "hist_volatility_20": float(candidate.hist_volatility_20),
                }
            )

    add_trial("baseline_model_top5", baseline, top_k)
    for pool_size in pools:
        if pool_size < top_k:
            continue
        effective_pool = min(pool_size, len(frame))
        for a, b, c in itertools.product(short_weights, vol_weights, range_weights):
            adjusted = (
                frame["rank"]
                + a * frame["short_overheat_rank"]
                + b * frame["volatility_20_rank"]
                + c * frame["intraday_range_rank"]
            )
            rule = f"soft_penalty_pool{effective_pool}_s{a:g}_v{b:g}_r{c:g}".replace(
                ".", "p"
            )
            selected = selected_from_pool(
                frame,
                top_k=top_k,
                pool_size=effective_pool,
                adjusted_rank=adjusted,
            )
            add_trial(
                rule,
                selected,
                effective_pool,
                adjusted=adjusted,
                a=a,
                b=b,
                c=c,
            )

        for threshold in filter_thresholds:
            rule = f"drop_short_overheat_pool{effective_pool}_ge_{threshold:g}".replace(
                ".", "p"
            )
            keep = frame["short_overheat_rank"] < threshold
            selected = selected_from_pool(
                frame,
                top_k=top_k,
                pool_size=effective_pool,
                keep=keep,
            )
            add_trial(
                rule,
                selected,
                effective_pool,
                filter_threshold=threshold,
            )

    return window_rows, candidate_rows


def summarize(window_scores: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        window_scores[window_scores["rule"].eq("baseline_model_top5")]
        .set_index("window")["score"]
        .rename("baseline_score")
    )
    rows = []
    for rule, group in window_scores.groupby("rule", sort=False):
        scores = pd.to_numeric(group["score"], errors="coerce")
        diffs = pd.to_numeric(group["diff_vs_baseline"], errors="coerce")
        positive_diffs = diffs[diffs > 0]
        positive_diff_sum = float(positive_diffs.sum()) if len(positive_diffs) else 0.0
        top2_positive_share = (
            float(
                positive_diffs.nlargest(min(2, len(positive_diffs))).sum()
                / positive_diff_sum
            )
            if positive_diff_sum > 1e-12
            else 0.0
        )
        rows.append(
            {
                "rule": rule,
                "description": group["description"].iloc[0],
                "window_count": int(len(group)),
                "pool_size": int(group["pool_size"].iloc[0]),
                "top_k": int(group["top_k"].iloc[0]),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "std_score": float(scores.std(ddof=0)),
                "score_to_std": (
                    float(scores.mean() / scores.std(ddof=0))
                    if scores.std(ddof=0) > 1e-12
                    else None
                ),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "worst3_mean_score": float(
                    scores.nsmallest(min(3, len(scores))).mean()
                ),
                "positive_windows": int((scores > 0).sum()),
                "negative_windows": int((scores < 0).sum()),
                "mean_diff_vs_baseline": float(diffs.mean()),
                "median_diff_vs_baseline": float(diffs.median()),
                "win_windows_vs_baseline": int((diffs > 1e-12).sum()),
                "loss_windows_vs_baseline": int((diffs < -1e-12).sum()),
                "tie_windows_vs_baseline": int(diffs.abs().le(1e-12).sum()),
                "top2_positive_diff_share": top2_positive_share,
                "mean_replaced_count": float(group["replaced_count"].mean()),
                "mean_pool_return": float(group["pool_mean_return"].mean()),
                "mean_pool_actual_top5_hits": float(
                    group["pool_actual_top5_hits"].mean()
                ),
                "mean_selected_actual_top5_hits": float(
                    group["actual_top5_hits"].mean()
                ),
                "mean_selected_positive_count": float(
                    group["selected_positive_count"].mean()
                ),
                "mean_selected_negative_count": float(
                    group["selected_negative_count"].mean()
                ),
                "mean_selected_short_overheat_rank": float(
                    group["selected_avg_short_overheat_rank"].mean()
                ),
                "mean_selected_volatility_20_rank": float(
                    group["selected_avg_volatility_20_rank"].mean()
                ),
                "mean_selected_intraday_range_rank": float(
                    group["selected_avg_intraday_range_rank"].mean()
                ),
                "baseline_mean_score": float(baseline.mean()),
                "baseline_min_score": float(baseline.min()),
                "baseline_positive_windows": int((baseline > 0).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["mean_diff_vs_baseline", "mean_score", "min_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_paired(window_scores: pd.DataFrame) -> pd.DataFrame:
    baseline = window_scores[window_scores["rule"].eq("baseline_model_top5")][
        ["window", "score", "selected_stock_ids"]
    ].rename(
        columns={
            "score": "baseline_score",
            "selected_stock_ids": "baseline_selected_stock_ids",
        }
    )
    paired = window_scores.merge(
        baseline, on="window", how="left", suffixes=("", "_base")
    )
    paired = paired[~paired["rule"].eq("baseline_model_top5")].copy()
    paired["diff_vs_baseline"] = paired["score"] - paired["baseline_score"]
    return paired.sort_values(["rule", "window"]).reset_index(drop=True)


def build_diagnostic_summary(window_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_rows = window_scores[window_scores["rule"].eq("baseline_model_top5")]
    for row in baseline_rows.itertuples(index=False):
        rows.append(
            {
                "window": row.window,
                "as_of_date": row.as_of_date,
                "target_start_date": row.target_start_date,
                "target_end_date": row.target_end_date,
                "baseline_score": row.score,
                "baseline_selected_stock_ids": row.selected_stock_ids,
                "baseline_selected_positive_count": row.selected_positive_count,
                "baseline_selected_negative_count": row.selected_negative_count,
                "baseline_actual_top5_hits": row.actual_top5_hits,
                "baseline_avg_short_overheat_rank": row.selected_avg_short_overheat_rank,
                "baseline_avg_volatility_20_rank": row.selected_avg_volatility_20_rank,
                "baseline_avg_intraday_range_rank": row.selected_avg_intraday_range_rank,
            }
        )
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def write_readme(
    output_dir: Path,
    experiment_dir: Path,
    summary: pd.DataFrame,
    window_scores: pd.DataFrame,
    pools: list[int],
) -> None:
    baseline = summary[summary["rule"].eq("baseline_model_top5")].iloc[0]
    candidates = summary[~summary["rule"].eq("baseline_model_top5")].copy()
    best = candidates.iloc[0] if not candidates.empty else baseline
    pass_checks = (
        best["mean_score"] > baseline["mean_score"]
        and best["min_score"] >= baseline["min_score"]
        and best["positive_windows"] >= baseline["positive_windows"]
        and best["top2_positive_diff_share"] <= 0.60
    )
    recommendation = (
        "当前离线结果支持把最佳规则作为下一版本可选提交策略继续验证。"
        if pass_checks
        else "当前离线结果不足以支持接入正式预测，建议继续保留原始 top5。"
    )

    lines = [
        "# 二阶段后处理离线评估",
        "",
        "本目录由 `code/src/evaluate_stage2_postprocess.py` 生成，只读取已有 walk-forward 预测诊断和预测日前历史数据，不重训模型，也不修改提交文件。",
        "",
        "## 输入",
        "",
        f"- 实验目录：`{relative_path(experiment_dir)}`",
        f"- 候选池：`{', '.join('top' + str(pool) for pool in pools)}`",
        "- 最终持仓：top5 等权满仓。",
        "",
        "## 文件",
        "",
        "- `summary.csv`：每条规则的跨窗口均值、最差窗口、胜负窗口和风险信号均值。",
        "- `window_scores.csv`：每个窗口、每条规则的得分和入选股票。",
        "- `candidate_scores.csv`：每条规则候选池内每只股票的原始 rank、后处理信号、是否入选和未来收益。",
        "- `paired_vs_baseline.csv`：每条规则相对原始 top5 的逐窗口差值。",
        "- `diagnostic_summary.csv`：原始 top5 的窗口级基础诊断。",
        "",
        "## 结论",
        "",
        (
            f"- 原始 top5：均值 `{baseline['mean_score']:.6f}`，最差 `{baseline['min_score']:.6f}`，"
            f"正分窗口 `{int(baseline['positive_windows'])}/{int(baseline['window_count'])}`。"
        ),
    ]
    if not candidates.empty:
        lines.append(
            f"- 最佳后处理：`{best['rule']}`，均值 `{best['mean_score']:.6f}`，"
            f"相对原始 top5 `{best['mean_diff_vs_baseline']:+.6f}`，最差 `{best['min_score']:.6f}`，"
            f"胜/负窗口 `{int(best['win_windows_vs_baseline'])}/{int(best['loss_windows_vs_baseline'])}`。"
        )
        lines.append(
            f"- 该规则正向改善中，最大两个窗口贡献占比 `{best['top2_positive_diff_share']:.3f}`；"
            "如果占比过高，说明结果可能依赖少数窗口。"
        )
    lines.append(f"- 建议：{recommendation}")
    lines.extend(
        [
            "",
            "## 规则说明",
            "",
            "- `ret3_rank`：预测日前 3 日收益的横截面分位，越高代表越强。",
            "- `gap5ma_rank`：预测日收盘价相对 5 日均线偏离的横截面分位，越高代表越偏离短期均线。",
            "- `short_overheat_rank`：`ret3_rank` 与 `gap5ma_rank` 的较大值，用来近似短期过热。",
            "- `intraday_range_rank`：近 5 日日内振幅横截面分位，越高代表短期波动越大。",
            "- `volatility_20_rank`：近 20 日收益波动横截面分位。",
            "",
            "这些信号只使用预测日前历史数据。`target_return` 只用于离线评估，不可用于正式预测。",
            "",
            "## 最好规则逐窗口",
            "",
        ]
    )
    if not candidates.empty:
        best_windows = window_scores[
            window_scores["rule"].eq(best["rule"])
        ].sort_values("window")
        for row in best_windows.itertuples(index=False):
            lines.append(
                f"- `{row.window}`：得分 `{row.score:.6f}`，相对原始 `{row.diff_vs_baseline:+.6f}`，"
                f"替换 `{row.replaced_count}` 只，入选 `{row.selected_stock_ids}`。"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(experiment_dir: Path) -> Path:
    return (
        REPO_ROOT
        / "experiments"
        / "analysis"
        / f"stage2_postprocess_{experiment_label(experiment_dir)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate stage2 postprocess rules from walk-forward diagnostics"
    )
    parser.add_argument(
        "experiment", help="experiment directory, e.g. experiments/v1.4.5"
    )
    parser.add_argument(
        "--pools",
        default=",".join(str(value) for value in DEFAULT_POOLS),
        help="comma-separated candidate pool sizes, default 10,20",
    )
    parser.add_argument(
        "--short-weights",
        default=",".join(str(value) for value in DEFAULT_SHORT_WEIGHTS),
        help="comma-separated short overheat penalty weights",
    )
    parser.add_argument(
        "--vol-weights",
        default=",".join(str(value) for value in DEFAULT_VOL_WEIGHTS),
        help="comma-separated volatility penalty weights",
    )
    parser.add_argument(
        "--range-weights",
        default=",".join(str(value) for value in DEFAULT_RANGE_WEIGHTS),
        help="comma-separated intraday range penalty weights",
    )
    parser.add_argument(
        "--filter-thresholds",
        default=",".join(str(value) for value in DEFAULT_FILTER_THRESHOLDS),
        help="comma-separated hard filter thresholds for short_overheat_rank",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="final selected stock count"
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

    pools = parse_int_list(args.pools)
    short_weights = parse_float_list(args.short_weights)
    vol_weights = parse_float_list(args.vol_weights)
    range_weights = parse_float_list(args.range_weights)
    filter_thresholds = parse_float_list(args.filter_thresholds)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(experiment_dir)
    )

    window_rows = []
    candidate_rows = []
    for window_dir in window_dirs(experiment_dir):
        frame, _ = read_window_frame(window_dir)
        rows, candidates = build_trials(
            frame,
            pools=pools,
            short_weights=short_weights,
            vol_weights=vol_weights,
            range_weights=range_weights,
            filter_thresholds=filter_thresholds,
            top_k=args.top_k,
        )
        window_rows.extend(rows)
        candidate_rows.extend(candidates)

    window_scores = pd.DataFrame(window_rows)
    candidate_scores = pd.DataFrame(candidate_rows)
    summary = summarize(window_scores)
    paired = build_paired(window_scores)
    diagnostic_summary = build_diagnostic_summary(window_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    window_scores.sort_values(["window", "rule"]).to_csv(
        output_dir / "window_scores.csv", index=False
    )
    candidate_scores.sort_values(["window", "rule", "rank"]).to_csv(
        output_dir / "candidate_scores.csv", index=False
    )
    paired.to_csv(output_dir / "paired_vs_baseline.csv", index=False)
    diagnostic_summary.to_csv(output_dir / "diagnostic_summary.csv", index=False)
    write_readme(output_dir, experiment_dir, summary, window_scores, pools)
    print(f"stage2 postprocess evaluation written to: {relative_path(output_dir)}")


if __name__ == "__main__":
    main()
