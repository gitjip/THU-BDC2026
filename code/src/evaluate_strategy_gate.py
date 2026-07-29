from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from data_utils import normalize_stock_codes
from stage2_selection import compute_recent_risk_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_K = 5
PRIMARY_HIGH_THRESHOLD = 0.03
VERY_BAD_THRESHOLD = -0.03
OVERHEAT_THRESHOLDS = (0.65, 0.70, 0.75, 0.80)
VOLATILITY_THRESHOLDS = (0.65, 0.70, 0.75, 0.80)
OVERLAP_THRESHOLDS = (0, 1)
COMBO_OVERHEAT_THRESHOLDS = (0.65, 0.70, 0.75)
CVAR_TAIL_FRACTION = 0.20
EXPLORATION_TOP2_DIFF_SHARE_LIMIT = 0.70
WORST3_TOLERANCE = 0.005
CVAR_TOLERANCE = 0.005
TOP2_POSITIVE_DIFF_SHARE_LIMIT = 0.60
HIGH_WINDOW_SWITCH_RATE_LIMIT = 0.30


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


def target_key(start: str, end: str) -> str:
    return f"{start}|{end}"


def read_summary(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary.csv: {path}")
    frame = pd.read_csv(path)
    required = {"window", "as_of_date", "target_start_date", "target_end_date", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["target_key"] = [
        target_key(start, end)
        for start, end in zip(frame["target_start_date"], frame["target_end_date"])
    ]
    return frame


def metadata_path(window_dir: Path, key: str, fallback: Path) -> Path:
    metadata = read_json(window_dir / "metadata.json")
    value = metadata.get(key)
    return resolve_path(value) if value else fallback


def read_window_diagnostics(experiment_dir: Path, window: str) -> pd.DataFrame:
    window_dir = experiment_dir / "windows" / window
    path = metadata_path(
        window_dir,
        "prediction_diagnostics",
        window_dir / "prediction_diagnostics.csv",
    )
    if not path.exists():
        raise FileNotFoundError(f"missing prediction diagnostics: {path}")
    frame = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    if "stock_id" not in frame.columns and "股票代码" in frame.columns:
        frame = frame.rename(columns={"股票代码": "stock_id"})
    required = {"rank", "stock_id", "pred_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["stock_id"] = normalize_stock_codes(frame["stock_id"])
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    return (
        frame.dropna(subset=["rank", "stock_id", "pred_score"])
        .sort_values("rank")
        .reset_index(drop=True)
    )


def read_window_history(experiment_dir: Path, window: str) -> pd.DataFrame:
    window_dir = experiment_dir / "windows" / window
    path = metadata_path(
        window_dir,
        "train_data",
        window_dir / "data" / "train_until_as_of.csv",
    )
    if not path.exists():
        raise FileNotFoundError(f"missing train_until_as_of.csv: {path}")
    return pd.read_csv(path, dtype={"股票代码": str})


def top_ids(frame: pd.DataFrame, k: int) -> list[str]:
    return frame.sort_values("rank").head(k)["stock_id"].astype(str).tolist()


def score_gap(frame: pd.DataFrame, k: int) -> float:
    ranked = frame.sort_values("rank").reset_index(drop=True)
    if len(ranked) < k:
        return 0.0
    scores = pd.to_numeric(ranked["pred_score"], errors="coerce")
    return float(scores.iloc[0] - scores.iloc[k - 1])


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def market_state(history: pd.DataFrame, days: int = 5) -> dict:
    required = {"股票代码", "日期", "收盘"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns for market state: {sorted(missing)}")

    data = history.copy()
    data["股票代码"] = normalize_stock_codes(data["股票代码"])
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce").dt.normalize()
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["股票代码", "日期", "收盘"]).sort_values(
        ["股票代码", "日期"]
    )
    data["daily_return"] = data.groupby("股票代码")["收盘"].pct_change(fill_method=None)
    data = data.dropna(subset=["daily_return"]).copy()
    if data.empty:
        return {
            "market_return_5": 0.0,
            "market_up_ratio_5": 0.5,
            "market_volatility_mean_5": 0.0,
        }

    recent_dates = sorted(data["日期"].unique())[-days:]
    recent = data[data["日期"].isin(recent_dates)].copy()
    daily = recent.groupby("日期")["daily_return"].agg(
        market_return_mean="mean",
        market_up_ratio=lambda series: float((series > 0).mean()),
        market_volatility="std",
    )
    return {
        "market_return_5": safe_mean(daily["market_return_mean"]),
        "market_up_ratio_5": safe_mean(daily["market_up_ratio"]),
        "market_volatility_mean_5": safe_mean(daily["market_volatility"]),
    }


def primary_top5_risk_features(
    primary_frame: pd.DataFrame, history: pd.DataFrame
) -> dict:
    metrics = compute_recent_risk_metrics(history, volatility_window=20)
    top5 = primary_frame.sort_values("rank").head(TOP_K).copy()
    top5 = top5.merge(metrics, on="stock_id", how="left", suffixes=("", "_risk"))
    for column in ["overheat_rank", "volatility_20_rank", "drawdown_rank"]:
        top5[column] = pd.to_numeric(top5[column], errors="coerce").fillna(0.5)
    return {
        "tf_top5_overheat_mean": safe_mean(top5["overheat_rank"]),
        "tf_top5_volatility_mean": safe_mean(top5["volatility_20_rank"]),
        "tf_top5_drawdown_mean": safe_mean(top5["drawdown_rank"]),
    }


def indexed_summary(summary: pd.DataFrame) -> dict[str, pd.Series]:
    return {row.target_key: row for row in summary.itertuples(index=False)}


def build_gate_features(
    primary_dir: Path,
    lgbm_dir: Path,
    defense_dir: Path,
) -> pd.DataFrame:
    primary_summary = read_summary(primary_dir)
    lgbm_summary = indexed_summary(read_summary(lgbm_dir))
    defense_summary = indexed_summary(read_summary(defense_dir))
    rows = []

    for primary_row in primary_summary.itertuples(index=False):
        key = primary_row.target_key
        if key not in lgbm_summary or key not in defense_summary:
            continue
        lgbm_row = lgbm_summary[key]
        defense_row = defense_summary[key]

        primary_frame = read_window_diagnostics(primary_dir, primary_row.window)
        lgbm_frame = read_window_diagnostics(lgbm_dir, lgbm_row.window)
        history = read_window_history(primary_dir, primary_row.window)

        primary_top5 = set(top_ids(primary_frame, 5))
        primary_top20 = set(top_ids(primary_frame, 20))
        lgbm_top5 = set(top_ids(lgbm_frame, 5))
        lgbm_top20 = set(top_ids(lgbm_frame, 20))

        row = {
            "target_key": key,
            "as_of_date": primary_row.as_of_date,
            "target_start_date": primary_row.target_start_date,
            "target_end_date": primary_row.target_end_date,
            "primary_window": primary_row.window,
            "lgbm_window": lgbm_row.window,
            "defense_window": defense_row.window,
            "primary_score": float(primary_row.score),
            "lgbm_score": float(lgbm_row.score),
            "defense_score": float(defense_row.score),
            "primary_high_window": float(primary_row.score) > PRIMARY_HIGH_THRESHOLD,
            "tf_lgbm_top5_overlap": len(primary_top5 & lgbm_top5),
            "tf_lgbm_top20_overlap": len(primary_top20 & lgbm_top20),
            "tf_top5_score_gap": score_gap(primary_frame, 5),
            "tf_top20_score_gap": score_gap(primary_frame, 20),
        }
        row.update(primary_top5_risk_features(primary_frame, history))
        row.update(market_state(history))
        rows.append(row)

    if not rows:
        raise ValueError("no common target windows found across experiments")
    return pd.DataFrame(rows).sort_values("target_start_date").reset_index(drop=True)


def rule_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def add_rule_rows(
    rows: list[dict],
    features: pd.DataFrame,
    defense_label: str,
    defense_score_column: str,
    rule_name: str,
    description: str,
    use_defense: pd.Series,
) -> None:
    use_defense = use_defense.reindex(features.index).fillna(False).astype(bool)
    for idx, feature in features.iterrows():
        selected_score = (
            feature[defense_score_column]
            if bool(use_defense.loc[idx])
            else feature["primary_score"]
        )
        selected_strategy = defense_label if bool(use_defense.loc[idx]) else "primary"
        diff = float(selected_score - feature["primary_score"])
        rows.append(
            {
                "target_start_date": feature["target_start_date"],
                "target_end_date": feature["target_end_date"],
                "as_of_date": feature["as_of_date"],
                "rule": rule_name,
                "description": description,
                "defense_target": defense_label,
                "selected_strategy": selected_strategy,
                "use_defense": bool(use_defense.loc[idx]),
                "score": float(selected_score),
                "primary_score": float(feature["primary_score"]),
                "defense_score": float(feature[defense_score_column]),
                "diff_vs_primary": diff,
                "primary_bad_window": bool(feature["primary_score"] < 0),
                "primary_very_bad_window": bool(
                    feature["primary_score"] <= VERY_BAD_THRESHOLD
                ),
                "primary_high_window": bool(feature["primary_high_window"]),
                "bad_window_switched": bool(
                    feature["primary_score"] < 0 and use_defense.loc[idx]
                ),
                "very_bad_window_switched": bool(
                    feature["primary_score"] <= VERY_BAD_THRESHOLD
                    and use_defense.loc[idx]
                ),
                "high_window_switched": bool(
                    feature["primary_high_window"] and use_defense.loc[idx]
                ),
                "high_window_hurt_by_switch": bool(
                    feature["primary_high_window"] and use_defense.loc[idx] and diff < 0
                ),
                "defense_improved_window": bool(use_defense.loc[idx] and diff > 0),
                "defense_hurt_window": bool(use_defense.loc[idx] and diff < 0),
                "tf_top5_overheat_mean": float(feature["tf_top5_overheat_mean"]),
                "tf_top5_volatility_mean": float(feature["tf_top5_volatility_mean"]),
                "tf_lgbm_top5_overlap": int(feature["tf_lgbm_top5_overlap"]),
                "market_return_5": float(feature["market_return_5"]),
                "market_up_ratio_5": float(feature["market_up_ratio_5"]),
            }
        )


def build_rule_scores(
    features: pd.DataFrame,
    lgbm_label: str,
    defense_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    defense_targets = [
        (lgbm_label, "lgbm_score"),
        (defense_label, "defense_score"),
    ]
    false_mask = pd.Series(False, index=features.index)
    true_mask = pd.Series(True, index=features.index)

    for target_label, score_column in defense_targets:
        add_rule_rows(
            rows,
            features,
            target_label,
            score_column,
            "always_primary",
            "始终使用 rank-replace Transformer。",
            false_mask,
        )
        add_rule_rows(
            rows,
            features,
            target_label,
            score_column,
            "always_defense",
            f"始终使用防守目标 {target_label}。",
            true_mask,
        )

        for threshold in OVERHEAT_THRESHOLDS:
            add_rule_rows(
                rows,
                features,
                target_label,
                score_column,
                f"gate_tf_overheat_ge_{rule_label(threshold)}",
                f"Transformer top5 平均过热 rank >= {threshold:g} 时切防守。",
                features["tf_top5_overheat_mean"] >= threshold,
            )
        for threshold in VOLATILITY_THRESHOLDS:
            add_rule_rows(
                rows,
                features,
                target_label,
                score_column,
                f"gate_tf_volatility_ge_{rule_label(threshold)}",
                f"Transformer top5 平均波动 rank >= {threshold:g} 时切防守。",
                features["tf_top5_volatility_mean"] >= threshold,
            )
        for threshold in OVERLAP_THRESHOLDS:
            add_rule_rows(
                rows,
                features,
                target_label,
                score_column,
                f"gate_tf_lgbm_top5_overlap_le_{threshold}",
                f"Transformer 与 LightGBM top5 重合数 <= {threshold} 时切防守。",
                features["tf_lgbm_top5_overlap"] <= threshold,
            )

        add_rule_rows(
            rows,
            features,
            target_label,
            score_column,
            "gate_market_return_5_lt_0",
            "预测日前近 5 个交易日市场平均收益 < 0 时切防守。",
            features["market_return_5"] < 0,
        )
        add_rule_rows(
            rows,
            features,
            target_label,
            score_column,
            "gate_market_up_ratio_5_lt_0p45",
            "预测日前近 5 个交易日平均上涨比例 < 0.45 时切防守。",
            features["market_up_ratio_5"] < 0.45,
        )

        for overheat_threshold in COMBO_OVERHEAT_THRESHOLDS:
            for overlap_threshold in OVERLAP_THRESHOLDS:
                add_rule_rows(
                    rows,
                    features,
                    target_label,
                    score_column,
                    (
                        "gate_tf_overheat_ge_"
                        f"{rule_label(overheat_threshold)}"
                        f"_and_overlap_le_{overlap_threshold}"
                    ),
                    (
                        f"Transformer top5 过热 >= {overheat_threshold:g} "
                        f"且 top5 重合数 <= {overlap_threshold} 时切防守。"
                    ),
                    (features["tf_top5_overheat_mean"] >= overheat_threshold)
                    & (features["tf_lgbm_top5_overlap"] <= overlap_threshold),
                )
    return pd.DataFrame(rows)


def top2_positive_diff_share(diffs: pd.Series) -> float:
    positive = pd.to_numeric(diffs, errors="coerce")
    positive = positive[positive > 0]
    if positive.empty:
        return 0.0
    total = float(positive.sum())
    if total <= 1e-12:
        return 0.0
    return float(positive.nlargest(min(2, len(positive))).sum() / total)


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def bool_count(values: pd.Series) -> int:
    return int(values.fillna(False).astype(bool).sum())


def cvar_score(scores: pd.Series, tail_fraction: float = CVAR_TAIL_FRACTION) -> float:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    tail_size = max(1, math.ceil(len(numeric) * tail_fraction))
    return float(numeric.nsmallest(tail_size).mean())


def worst_n_mean(scores: pd.Series, n: int = 3) -> float:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.nsmallest(min(n, len(numeric))).mean())


def risk_adjusted_score(scores: pd.Series) -> float:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean() - 0.5 * numeric.std(ddof=0))


def mean_or_zero(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def summarize_rules(window_scores: pd.DataFrame) -> pd.DataFrame:
    primary_rows = window_scores[window_scores["rule"].eq("always_primary")]
    baseline_by_target = {
        target: group.copy()
        for target, group in primary_rows.groupby("defense_target", sort=False)
    }
    defense_rows = window_scores[window_scores["rule"].eq("always_defense")]
    defense_mean_by_target = (
        defense_rows.groupby("defense_target")["score"].mean().to_dict()
    )

    rows = []
    for (defense_target, rule), group in window_scores.groupby(
        ["defense_target", "rule"], sort=False
    ):
        baseline = baseline_by_target[defense_target]
        baseline_scores = pd.to_numeric(baseline["score"], errors="coerce")
        baseline_mean = float(baseline_scores.mean())
        baseline_std = float(baseline_scores.std(ddof=0))
        baseline_min = float(baseline_scores.min())
        baseline_worst3 = worst_n_mean(baseline_scores, 3)
        baseline_cvar20 = cvar_score(baseline_scores)
        baseline_risk_adjusted = risk_adjusted_score(baseline_scores)
        baseline_positive = int((baseline_scores > 0).sum())

        scores = pd.to_numeric(group["score"], errors="coerce")
        diffs = pd.to_numeric(group["diff_vs_primary"], errors="coerce")
        switch_mask = group["use_defense"].fillna(False).astype(bool)
        bad_mask = group["primary_bad_window"].fillna(False).astype(bool)
        very_bad_mask = group["primary_very_bad_window"].fillna(False).astype(bool)
        high_windows = group[group["primary_high_window"]]

        window_count = int(len(group))
        win_count = int((diffs > 1e-12).sum())
        loss_count = int((diffs < -1e-12).sum())
        tie_count = int(diffs.abs().le(1e-12).sum())
        active_count = win_count + loss_count
        defense_use_count = bool_count(group["use_defense"])
        bad_window_count = bool_count(group["primary_bad_window"])
        very_bad_window_count = bool_count(group["primary_very_bad_window"])
        bad_switch_count = bool_count(group["bad_window_switched"])
        very_bad_switch_count = bool_count(group["very_bad_window_switched"])
        high_switch_count = int(group["high_window_switched"].sum())
        high_window_count = int(len(high_windows))
        high_switch_rate = (
            high_switch_count / high_window_count if high_window_count else 0.0
        )
        positive_diff_share = top2_positive_diff_share(diffs)
        mean_score = float(scores.mean())
        std_score = float(scores.std(ddof=0))
        worst3 = worst_n_mean(scores, 3)
        cvar20 = cvar_score(scores)
        paired_mean_diff = float(diffs.mean())
        active_win_rate = safe_rate(win_count, active_count)
        is_candidate_rule = rule not in {"always_primary", "always_defense"}

        passes_exploration = (
            is_candidate_rule
            and paired_mean_diff > 1e-12
            and positive_diff_share <= EXPLORATION_TOP2_DIFF_SHARE_LIMIT
        )
        passes_candidate = (
            passes_exploration
            and active_win_rate >= 0.50
            and int((scores > 0).sum()) >= baseline_positive
            and worst3 >= baseline_worst3 - WORST3_TOLERANCE
            and cvar20 >= baseline_cvar20 - CVAR_TOLERANCE
            and positive_diff_share <= TOP2_POSITIVE_DIFF_SHARE_LIMIT
        )
        passes_default = (
            passes_candidate
            and worst3 >= baseline_worst3 - 1e-12
            and cvar20 >= baseline_cvar20 - 1e-12
            and high_switch_rate <= HIGH_WINDOW_SWITCH_RATE_LIMIT
        )

        rows.append(
            {
                "defense_target": defense_target,
                "rule": rule,
                "description": group["description"].iloc[0],
                "window_count": window_count,
                "mean_score": mean_score,
                "median_score": float(scores.median()),
                "std_score": std_score,
                "risk_adjusted_score": risk_adjusted_score(scores),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "worst3_mean_score": worst3,
                "cvar_20_score": cvar20,
                "positive_windows": int((scores > 0).sum()),
                "negative_windows": int((scores < 0).sum()),
                "paired_mean_diff": paired_mean_diff,
                "mean_diff_vs_primary": paired_mean_diff,
                "median_diff_vs_primary": float(diffs.median()),
                "win_windows_vs_primary": win_count,
                "loss_windows_vs_primary": loss_count,
                "tie_windows_vs_primary": tie_count,
                "win_rate_vs_primary": safe_rate(win_count, window_count),
                "active_win_rate_vs_primary": active_win_rate,
                "non_loss_rate_vs_primary": safe_rate(
                    win_count + tie_count, window_count
                ),
                "defense_use_count": defense_use_count,
                "defense_use_rate": safe_rate(defense_use_count, window_count),
                "bad_window_count": bad_window_count,
                "bad_window_switch_count": bad_switch_count,
                "bad_window_recall": safe_rate(bad_switch_count, bad_window_count),
                "bad_window_precision": safe_rate(bad_switch_count, defense_use_count),
                "very_bad_window_count": very_bad_window_count,
                "very_bad_window_switch_count": very_bad_switch_count,
                "very_bad_window_recall": safe_rate(
                    very_bad_switch_count, very_bad_window_count
                ),
                "switched_window_mean_diff": mean_or_zero(diffs[switch_mask]),
                "bad_window_switch_mean_diff": mean_or_zero(
                    diffs[bad_mask & switch_mask]
                ),
                "rescued_window_count": int(
                    ((bad_mask & switch_mask) & (diffs > 1e-12)).sum()
                ),
                "rescued_score_mean": mean_or_zero(
                    diffs[(bad_mask & switch_mask) & (diffs > 1e-12)]
                ),
                "hurt_window_count": int((switch_mask & (diffs < -1e-12)).sum()),
                "hurt_score_mean": mean_or_zero(diffs[switch_mask & (diffs < -1e-12)]),
                "primary_high_window_count": high_window_count,
                "high_window_switch_count": high_switch_count,
                "high_window_switch_rate": high_switch_rate,
                "high_window_false_switch_rate": high_switch_rate,
                "high_window_hurt_by_switch_count": int(
                    group["high_window_hurt_by_switch"].sum()
                ),
                "mean_high_window_diff_vs_primary": (
                    float(high_windows["diff_vs_primary"].mean())
                    if high_window_count
                    else 0.0
                ),
                "top2_positive_diff_share": positive_diff_share,
                "baseline_primary_mean": baseline_mean,
                "baseline_primary_std": baseline_std,
                "baseline_primary_min": baseline_min,
                "baseline_primary_worst3_mean": baseline_worst3,
                "baseline_primary_cvar_20": baseline_cvar20,
                "baseline_primary_risk_adjusted": baseline_risk_adjusted,
                "baseline_primary_positive_windows": baseline_positive,
                "target_always_defense_mean": (
                    float(defense_mean_by_target[defense_target])
                    if defense_target in defense_mean_by_target
                    else None
                ),
                "passes_exploration_checks": passes_exploration,
                "passes_candidate_checks": passes_candidate,
                "passes_default_checks": passes_default,
                "passes_basic_checks": passes_default,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "passes_default_checks",
                "passes_candidate_checks",
                "passes_exploration_checks",
                "mean_score",
                "cvar_20_score",
                "positive_windows",
            ],
            ascending=[False, False, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def write_readme(
    output_dir: Path,
    primary_dir: Path,
    lgbm_dir: Path,
    defense_dir: Path,
    summary: pd.DataFrame,
) -> None:
    baseline = summary[summary["rule"].eq("always_primary")].iloc[0]
    candidates = summary[~summary["rule"].isin(["always_primary", "always_defense"])]
    best = candidates.iloc[0] if not candidates.empty else baseline
    default_passed = summary[summary["passes_default_checks"]]
    candidate_passed = summary[summary["passes_candidate_checks"]]
    exploration_passed = summary[summary["passes_exploration_checks"]]
    recommendation = (
        "已有规则通过指标层默认候选检查，但本脚本仍只是离线诊断；需要另起版本接入 walk-forward 复现后才可考虑改正式预测。"
        if not default_passed.empty
        else (
            "已有规则通过候选检查，可继续围绕该信号做小步复核；暂不直接改正式预测。"
            if not candidate_passed.empty
            else (
                "已有规则通过探索检查，说明方向有信号，但还不足以作为候选策略。"
                if not exploration_passed.empty
                else "没有门控规则通过探索检查，暂不继续接入。"
            )
        )
    )

    lines = [
        "# 策略门控离线诊断",
        "",
        "本目录由 `code/src/evaluate_strategy_gate.py` 生成，只读取已有 walk-forward 结果和预测日前历史数据，不训练、不预测、不修改提交文件。",
        "",
        "## 输入",
        "",
        f"- 进攻主线：`{relative_path(primary_dir)}`",
        f"- 防守候选 1：`{relative_path(lgbm_dir)}`",
        f"- 防守候选 2：`{relative_path(defense_dir)}`",
        "",
        "## 文件",
        "",
        "- `gate_features.csv`：每个窗口一行，包含可用于门控的历史信号和事后得分。",
        "- `gate_rule_summary.csv`：每条门控规则的跨窗口表现。",
        "- `gate_window_scores.csv`：每个窗口、每条规则实际选择哪个策略。",
        "",
        "## 结论",
        "",
        (
            f"- Transformer 基线：均值 `{baseline['mean_score']:.6f}`，"
            f"CVaR20 `{baseline['cvar_20_score']:.6f}`，"
            f"worst3 `{baseline['worst3_mean_score']:.6f}`，"
            f"正分窗口 `{int(baseline['positive_windows'])}/{int(baseline['window_count'])}`。"
        ),
    ]
    if not candidates.empty:
        lines.append(
            f"- 最好门控：`{best['defense_target']} / {best['rule']}`，"
            f"均值 `{best['mean_score']:.6f}`，CVaR20 `{best['cvar_20_score']:.6f}`，"
            f"worst3 `{best['worst3_mean_score']:.6f}`，"
            f"正分 `{int(best['positive_windows'])}/{int(best['window_count'])}`，"
            f"配对均值差 `{best['paired_mean_diff']:+.6f}`。"
        )
        lines.append(
            f"- 负分窗口召回 `{best['bad_window_recall']:.3f}`，"
            f"防守切换精确率 `{best['bad_window_precision']:.3f}`，"
            f"高分窗口误切比例 `{best['high_window_false_switch_rate']:.3f}`，"
            f"最大两个正向改善贡献占比 `{best['top2_positive_diff_share']:.3f}`。"
        )
        lines.append(
            f"- 三层检查：探索 `{bool(best['passes_exploration_checks'])}`，"
            f"候选 `{bool(best['passes_candidate_checks'])}`，"
            f"默认候选 `{bool(best['passes_default_checks'])}`。"
        )
    lines.append(f"- 建议：{recommendation}")
    lines.extend(
        [
            "",
            "## 指标口径",
            "",
            "- `mean_score` 和 `paired_mean_diff` 是主目标，近似衡量期望得分；",
            "- `worst3_mean_score` 和 `cvar_20_score` 比单个 `min_score` 更适合作风险判断；",
            "- `min_score` 只作为警报，不再作为一票否决；",
            "- `bad_window_recall`、`bad_window_precision`、`high_window_false_switch_rate` 是门控专用诊断；",
            "- `passes_exploration_checks`、`passes_candidate_checks`、`passes_default_checks` 分别对应探索、候选和默认候选三层标准。",
            "",
            "门控规则只使用预测日前可见信号。`primary_score`、`lgbm_score`、`defense_score` 只用于事后评分，不可作为线上门控输入。",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(primary_dir: Path, lgbm_dir: Path, defense_dir: Path) -> Path:
    labels = "_vs_".join(
        experiment_label(path) for path in [primary_dir, lgbm_dir, defense_dir]
    )
    return REPO_ROOT / "experiments" / "analysis" / f"strategy_gate_{labels}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple offline gates between primary and defensive strategies"
    )
    parser.add_argument("primary", help="primary Transformer experiment directory")
    parser.add_argument("lgbm", help="LightGBM defensive experiment directory")
    parser.add_argument("defense", help="stage2 defensive experiment directory")
    parser.add_argument("--output-dir", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_dir = resolve_path(args.primary)
    lgbm_dir = resolve_path(args.lgbm)
    defense_dir = resolve_path(args.defense)
    for path in [primary_dir, lgbm_dir, defense_dir]:
        if not path.exists():
            raise FileNotFoundError(f"experiment directory not found: {path}")

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_dir(primary_dir, lgbm_dir, defense_dir)
    )

    features = build_gate_features(primary_dir, lgbm_dir, defense_dir)
    window_scores = build_rule_scores(
        features,
        lgbm_label=experiment_label(lgbm_dir),
        defense_label=experiment_label(defense_dir),
    )
    summary = summarize_rules(window_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "gate_features.csv", index=False)
    summary.to_csv(output_dir / "gate_rule_summary.csv", index=False)
    window_scores.sort_values(["defense_target", "rule", "target_start_date"]).to_csv(
        output_dir / "gate_window_scores.csv",
        index=False,
    )
    write_readme(output_dir, primary_dir, lgbm_dir, defense_dir, summary)
    print(f"strategy gate evaluation written to: {relative_path(output_dir)}")


if __name__ == "__main__":
    main()
