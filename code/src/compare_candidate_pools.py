import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd

from stage2_selection import compute_recent_risk_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_KS = (5, 10, 20, 50, 100)
STAGE2_METHODS = (
    "union_avg_rank_top5",
    "prefix_cap2_avg_rank_top5",
    "risk_w1_top5",
    "risk_w2_top5",
    "risk_w3_top5",
    "risk_gate_moderate_top5",
    "prefix_cap2_risk_w2_top5",
    "low_risk_then_rank_top5",
    "low_vol_then_rank_top5",
    "low_overheat_then_rank_top5",
    "low_drawdown_then_rank_top5",
)


def clean_top_ks(values) -> list[int]:
    top_ks = sorted({int(value) for value in values if int(value) > 0})
    if not top_ks:
        raise ValueError("top_ks must contain at least one positive integer")
    return top_ks


def parse_top_ks(raw: str) -> list[int]:
    return clean_top_ks(part.strip() for part in raw.split(",") if part.strip())


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def experiment_label(experiment_dir: Path) -> str:
    return experiment_dir.name


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "prediction_diagnostics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prediction diagnostics summary: {path}")
    frame = pd.read_csv(path)
    frame["experiment"] = experiment_label(experiment_dir)
    return frame


def read_window_diagnostics(experiment_dir: Path) -> dict[str, pd.DataFrame]:
    windows = {}
    for path in sorted((experiment_dir / "windows").glob("window_*/prediction_diagnostics.csv")):
        frame = pd.read_csv(path, dtype={"stock_id": str, "stock_prefix": str})
        frame["experiment"] = experiment_label(experiment_dir)
        frame["window"] = path.parent.name
        windows[path.parent.name] = frame
    if not windows:
        raise FileNotFoundError(f"missing window prediction diagnostics: {experiment_dir}")
    return windows


def window_train_data_path(experiment_dir: Path, window: str) -> Path:
    window_dir = experiment_dir / "windows" / window
    metadata = read_json(window_dir / "metadata.json")
    train_data = metadata.get("train_data")
    if train_data:
        return resolve_path(train_data)
    return window_dir / "data" / "train_until_as_of.csv"


def scalar_summary(experiment_dir: Path, summary: pd.DataFrame) -> dict:
    aggregate = read_json(experiment_dir / "prediction_diagnostics.json")
    score = summary["score_from_diagnostics"]
    row = {
        "experiment": experiment_label(experiment_dir),
        "window_count": int(len(summary)),
        "mean_score": float(score.mean()),
        "min_score": float(score.min()),
        "max_score": float(score.max()),
        "positive_score_windows": int((score > 0).sum()),
        "mean_spearman_pred_score_target_return": float(summary["spearman_pred_score_target_return"].mean()),
        "mean_pred_top5_return": float(summary["pred_top5_equal_weight_return"].mean()),
        "mean_pred_top20_return": float(summary["pred_top20_equal_weight_return"].mean()),
        "mean_pred_top50_return": float(summary["pred_top50_equal_weight_return"].mean()),
        "mean_actual_top5_hits_in_pred_top20": float(summary["actual_top5_hits_in_pred_top20"].mean()),
        "mean_actual_top5_hits_in_pred_top50": float(summary["actual_top5_hits_in_pred_top50"].mean()),
    }
    for key in ("unique_selected_count", "unique_top20_count", "repeated_top20_stock_count"):
        row[key] = aggregate.get(key)
    return row


def align_windows(window_maps: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    common = None
    for windows in window_maps.values():
        names = set(windows)
        common = names if common is None else common & names
    if not common:
        raise ValueError("experiments do not have common diagnostic windows")
    return sorted(common)


def top_set(frame: pd.DataFrame, top_k: int) -> set[str]:
    return set(frame.sort_values("rank").head(top_k)["stock_id"])


def compare_overlaps(window_maps: dict[str, dict[str, pd.DataFrame]], top_ks: list[int]) -> pd.DataFrame:
    rows = []
    common_windows = align_windows(window_maps)
    for left, right in itertools.combinations(sorted(window_maps), 2):
        for top_k in top_ks:
            counts = []
            ratios = []
            for window in common_windows:
                left_set = top_set(window_maps[left][window], top_k)
                right_set = top_set(window_maps[right][window], top_k)
                overlap = len(left_set & right_set)
                counts.append(overlap)
                ratios.append(overlap / top_k)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "top_k": top_k,
                    "window_count": len(common_windows),
                    "mean_overlap_count": float(pd.Series(counts).mean()),
                    "mean_overlap_ratio": float(pd.Series(ratios).mean()),
                    "min_overlap_ratio": float(pd.Series(ratios).min()),
                    "max_overlap_ratio": float(pd.Series(ratios).max()),
                }
            )
    return pd.DataFrame(rows)


def candidate_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    records = []
    for idx, frame in enumerate(frames):
        for row in frame[["stock_id", "rank", "target_return"]].itertuples(index=False):
            records.append(
                {
                    "stock_id": row.stock_id,
                    f"rank_{idx}": int(row.rank),
                    f"target_return_{idx}": float(row.target_return),
                }
            )
    merged = pd.DataFrame({"stock_id": sorted(set(record["stock_id"] for record in records))})
    for idx in range(len(frames)):
        partial = pd.DataFrame(
            [
                {
                    "stock_id": record["stock_id"],
                    f"rank_{idx}": record.get(f"rank_{idx}"),
                    f"target_return_{idx}": record.get(f"target_return_{idx}"),
                }
                for record in records
                if f"rank_{idx}" in record
            ]
        )
        merged = merged.merge(partial, on="stock_id", how="left")
    rank_cols = [f"rank_{idx}" for idx in range(len(frames))]
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged["min_rank"] = merged[rank_cols].min(axis=1)
    merged["borda_top20"] = sum((21 - merged[column]).clip(lower=0).fillna(0) for column in rank_cols)
    return_cols = [f"target_return_{idx}" for idx in range(len(frames))]
    merged["target_return"] = merged[return_cols].bfill(axis=1).iloc[:, 0]
    return merged


def stock_recent_metrics(train_data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(train_data_path, dtype={"股票代码": str})
    return compute_recent_risk_metrics(data, volatility_window=20)


def union_top5_candidates(frames: list[pd.DataFrame], recent_metrics: pd.DataFrame) -> pd.DataFrame:
    union_ids = sorted(set().union(*(set(frame.sort_values("rank").head(5)["stock_id"]) for frame in frames)))
    candidates = candidate_frame(frames)
    candidates = candidates[candidates["stock_id"].isin(union_ids)].copy()
    rank_cols = [column for column in candidates.columns if re.fullmatch(r"rank_\d+", column)]
    candidates["top5_borda"] = sum((6 - candidates[column]).clip(lower=0).fillna(0) for column in rank_cols)
    candidates["avg_rank_fill"] = candidates[rank_cols].fillna(999).mean(axis=1)
    candidates["min_rank_fill"] = candidates[rank_cols].fillna(999).min(axis=1)

    candidates = candidates.merge(recent_metrics, on="stock_id", how="left", suffixes=("", "_risk"))
    candidates["stock_prefix"] = candidates["stock_prefix"].fillna(candidates["stock_id"].str[:3])
    neutral_columns = ["volatility_20_rank", "overheat_rank", "drawdown_rank", "risk_score"]
    for column in neutral_columns:
        candidates[column] = candidates[column].fillna(0.5)
    raw_columns = ["recent_return_5", "recent_return_10", "recent_return_20", "volatility_20", "max_drawdown_20"]
    for column in raw_columns:
        candidates[column] = candidates[column].fillna(0.0)

    for weight in (1.0, 2.0, 3.0):
        label = str(weight).replace(".", "p")
        candidates[f"risk_adjusted_w{label}"] = candidates["top5_borda"] - weight * candidates["risk_score"]
    candidates["risk_gate_moderate"] = ~(
        (candidates["volatility_20_rank"] >= 0.85)
        | (candidates["overheat_rank"] >= 0.90)
        | (candidates["drawdown_rank"] >= 0.90)
    )
    return candidates


def select_candidates(candidates: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "avg_rank":
        return candidates.sort_values(["avg_rank", "min_rank"]).head(5)
    if method == "min_rank":
        return candidates.sort_values(["min_rank", "avg_rank"]).head(5)
    if method == "borda_top20":
        scored = candidates[candidates["borda_top20"] > 0].copy()
        return scored.sort_values(["borda_top20", "avg_rank"], ascending=[False, True]).head(5)
    raise ValueError(f"unknown candidate selection method: {method}")


def select_with_prefix_cap(sorted_candidates: pd.DataFrame, cap: int = 2) -> pd.DataFrame:
    selected_indices = []
    prefix_counts: dict[str, int] = {}
    for idx, row in sorted_candidates.iterrows():
        prefix = str(row["stock_prefix"])
        if prefix_counts.get(prefix, 0) >= cap:
            continue
        selected_indices.append(idx)
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        if len(selected_indices) == 5:
            break

    if len(selected_indices) < 5:
        for idx in sorted_candidates.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) == 5:
                break
    return sorted_candidates.loc[selected_indices].copy()


def select_stage2_candidates(candidates: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "union_avg_rank_top5":
        return candidates.sort_values(["avg_rank_fill", "min_rank_fill"]).head(5).copy()
    if method == "prefix_cap2_avg_rank_top5":
        sorted_candidates = candidates.sort_values(["avg_rank_fill", "min_rank_fill"])
        return select_with_prefix_cap(sorted_candidates, cap=2)
    if method == "risk_w1_top5":
        return candidates.sort_values(["risk_adjusted_w1p0", "avg_rank_fill"], ascending=[False, True]).head(5).copy()
    if method == "risk_w2_top5":
        return candidates.sort_values(["risk_adjusted_w2p0", "avg_rank_fill"], ascending=[False, True]).head(5).copy()
    if method == "risk_w3_top5":
        return candidates.sort_values(["risk_adjusted_w3p0", "avg_rank_fill"], ascending=[False, True]).head(5).copy()
    if method == "risk_gate_moderate_top5":
        sorted_candidates = candidates.sort_values(["risk_adjusted_w2p0", "avg_rank_fill"], ascending=[False, True])
        filtered = sorted_candidates[sorted_candidates["risk_gate_moderate"]].copy()
        selected = filtered.head(5)
        if len(selected) < 5:
            fill = sorted_candidates[~sorted_candidates["stock_id"].isin(selected["stock_id"])].head(5 - len(selected))
            selected = pd.concat([selected, fill], ignore_index=True)
        return selected.copy()
    if method == "prefix_cap2_risk_w2_top5":
        sorted_candidates = candidates.sort_values(["risk_adjusted_w2p0", "avg_rank_fill"], ascending=[False, True])
        return select_with_prefix_cap(sorted_candidates, cap=2)
    if method == "low_risk_then_rank_top5":
        return candidates.sort_values(["risk_score", "avg_rank_fill", "min_rank_fill"]).head(5).copy()
    if method == "low_vol_then_rank_top5":
        return candidates.sort_values(["volatility_20_rank", "avg_rank_fill", "min_rank_fill"]).head(5).copy()
    if method == "low_overheat_then_rank_top5":
        return candidates.sort_values(["overheat_rank", "avg_rank_fill", "min_rank_fill"]).head(5).copy()
    if method == "low_drawdown_then_rank_top5":
        return candidates.sort_values(["drawdown_rank", "avg_rank_fill", "min_rank_fill"]).head(5).copy()
    raise ValueError(f"unknown stage2 method: {method}")


def compare_ensemble_trials(window_maps: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    labels = sorted(window_maps)
    common_windows = align_windows(window_maps)
    for pair_size in range(2, len(labels) + 1):
        for selected_labels in itertools.combinations(labels, pair_size):
            returns_by_method: dict[str, list[float]] = {
                "top5_union_equal": [],
                "avg_rank_top5": [],
                "min_rank_top5": [],
                "borda_top20_top5": [],
            }
            counts_by_method: dict[str, list[int]] = {key: [] for key in returns_by_method}
            for window in common_windows:
                frames = [window_maps[label][window].sort_values("rank") for label in selected_labels]
                union = sorted(set().union(*(set(frame.head(5)["stock_id"]) for frame in frames)))
                candidates = candidate_frame(frames)
                union_frame = candidates[candidates["stock_id"].isin(union)]
                returns_by_method["top5_union_equal"].append(float(union_frame["target_return"].mean()))
                counts_by_method["top5_union_equal"].append(int(len(union_frame)))

                for method, output_name in [
                    ("avg_rank", "avg_rank_top5"),
                    ("min_rank", "min_rank_top5"),
                    ("borda_top20", "borda_top20_top5"),
                ]:
                    chosen = select_candidates(candidates, method)
                    returns_by_method[output_name].append(float(chosen["target_return"].mean()))
                    counts_by_method[output_name].append(int(len(chosen)))

            for method, values in returns_by_method.items():
                series = pd.Series(values)
                rows.append(
                    {
                        "experiments": "+".join(selected_labels),
                        "method": method,
                        "window_count": len(common_windows),
                        "mean_return": float(series.mean()),
                        "min_return": float(series.min()),
                        "positive_windows": int((series > 0).sum()),
                        "mean_candidate_count": float(pd.Series(counts_by_method[method]).mean()),
                    }
                )
    return pd.DataFrame(rows)


def compare_stage2_rerank_trials(
    experiment_dirs: dict[str, Path],
    window_maps: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    candidate_rows = []
    labels = list(window_maps)
    common_windows = align_windows(window_maps)

    for pair_size in range(2, len(labels) + 1):
        for selected_labels in itertools.combinations(labels, pair_size):
            experiment_key = "+".join(selected_labels)
            base_label = selected_labels[0]
            for window in common_windows:
                frames = [window_maps[label][window].sort_values("rank") for label in selected_labels]
                recent_metrics = stock_recent_metrics(window_train_data_path(experiment_dirs[base_label], window))
                candidates = union_top5_candidates(frames, recent_metrics)
                if candidates.empty:
                    continue

                single_returns = {
                    label: float(frame.sort_values("rank").head(5)["target_return"].mean())
                    for label, frame in zip(selected_labels, frames)
                }
                first_single_return = single_returns[base_label]
                best_single_return = max(single_returns.values())
                union_return = float(candidates["target_return"].mean())
                total_negative_in_union = int((candidates["target_return"] < 0).sum())

                for method in STAGE2_METHODS:
                    selected = select_stage2_candidates(candidates, method)
                    selected_ids = set(selected["stock_id"])
                    removed = candidates[~candidates["stock_id"].isin(selected_ids)]
                    selected_return = float(selected["target_return"].mean()) if len(selected) else 0.0
                    selected_negative_count = int((selected["target_return"] < 0).sum())
                    removed_negative_count = int((removed["target_return"] < 0).sum())
                    removed_negative_share = (
                        removed_negative_count / total_negative_in_union
                        if total_negative_in_union
                        else 0.0
                    )

                    rows.append(
                        {
                            "experiments": experiment_key,
                            "window": window,
                            "method": method,
                            "selected_return": selected_return,
                            "union_return": union_return,
                            "first_single_return": first_single_return,
                            "best_single_return": best_single_return,
                            "delta_vs_union": selected_return - union_return,
                            "delta_vs_first": selected_return - first_single_return,
                            "delta_vs_best_single": selected_return - best_single_return,
                            "selected_count": int(len(selected)),
                            "union_count": int(len(candidates)),
                            "selected_negative_count": selected_negative_count,
                            "removed_negative_count": removed_negative_count,
                            "total_negative_in_union": total_negative_in_union,
                            "removed_negative_share": removed_negative_share,
                            "selected_avg_risk_score": float(selected["risk_score"].mean()) if len(selected) else 0.0,
                            "union_avg_risk_score": float(candidates["risk_score"].mean()),
                            "selected_stocks": ",".join(selected["stock_id"]),
                            "removed_stocks": ",".join(removed["stock_id"]),
                        }
                    )

                    for row in candidates.to_dict(orient="records"):
                        candidate_rows.append(
                            {
                                "experiments": experiment_key,
                                "window": window,
                                "method": method,
                                "stock_id": row["stock_id"],
                                "stock_prefix": row["stock_prefix"],
                                "selected": row["stock_id"] in selected_ids,
                                "target_return": row["target_return"],
                                "top5_borda": row["top5_borda"],
                                "avg_rank_fill": row["avg_rank_fill"],
                                "min_rank_fill": row["min_rank_fill"],
                                "risk_score": row["risk_score"],
                                "volatility_20_rank": row["volatility_20_rank"],
                                "overheat_rank": row["overheat_rank"],
                                "drawdown_rank": row["drawdown_rank"],
                                "recent_return_5": row["recent_return_5"],
                                "recent_return_10": row["recent_return_10"],
                                "recent_return_20": row["recent_return_20"],
                                "volatility_20": row["volatility_20"],
                                "max_drawdown_20": row["max_drawdown_20"],
                            }
                        )

    windows = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    if windows.empty:
        return pd.DataFrame(), windows, candidates

    summary_rows = []
    for (experiment_key, method), group in windows.groupby(["experiments", "method"], sort=False):
        summary_rows.append(
            {
                "experiments": experiment_key,
                "method": method,
                "window_count": int(len(group)),
                "mean_return": float(group["selected_return"].mean()),
                "min_return": float(group["selected_return"].min()),
                "positive_windows": int((group["selected_return"] > 0).sum()),
                "mean_union_return": float(group["union_return"].mean()),
                "mean_first_single_return": float(group["first_single_return"].mean()),
                "mean_best_single_return": float(group["best_single_return"].mean()),
                "mean_delta_vs_union": float(group["delta_vs_union"].mean()),
                "better_than_union_windows": int((group["delta_vs_union"] > 0).sum()),
                "mean_delta_vs_first": float(group["delta_vs_first"].mean()),
                "better_than_first_windows": int((group["delta_vs_first"] > 0).sum()),
                "mean_delta_vs_best_single": float(group["delta_vs_best_single"].mean()),
                "better_than_best_single_windows": int((group["delta_vs_best_single"] > 0).sum()),
                "mean_selected_risk_score": float(group["selected_avg_risk_score"].mean()),
                "mean_union_risk_score": float(group["union_avg_risk_score"].mean()),
                "mean_selected_negative_count": float(group["selected_negative_count"].mean()),
                "mean_removed_negative_count": float(group["removed_negative_count"].mean()),
                "mean_total_negative_in_union": float(group["total_negative_in_union"].mean()),
                "mean_removed_negative_share": float(group["removed_negative_share"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return summary, windows, candidates


def write_readme(output_dir: Path, experiments: list[Path]) -> None:
    lines = [
        "# Candidate Pool Analysis",
        "",
        "This directory is generated from prediction diagnostics. It does not change training or submission files.",
        "",
        "## Experiments",
        "",
    ]
    for experiment_dir in experiments:
        lines.append(f"- `{experiment_label(experiment_dir)}`: `{experiment_dir}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: one row per experiment.",
            "- `window_metrics.csv`: raw window-level diagnostic rows.",
            "- `overlap.csv`: topK overlap between experiment pairs.",
            "- `ensemble_trials.csv`: diagnostic-only candidate pool combination trials.",
            "- `stage2_rerank_trials.csv`: diagnostic-only two-stage risk reranking summary.",
            "- `stage2_rerank_windows.csv`: window-level two-stage reranking results.",
            "- `stage2_rerank_candidates.csv`: candidate-level risk signals and selections.",
            "",
            "`top5_union_equal` can include more than five candidates and is not a valid submission by itself.",
            "It only measures whether the union candidate pool contains useful signal.",
            "Stage2 risk signals are calculated from each window's train_until_as_of data only.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(experiments: list[Path]) -> Path:
    labels = "_vs_".join(experiment_label(path) for path in experiments)
    return REPO_ROOT / "experiments" / "analysis" / f"candidate_pool_{labels}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare walk-forward prediction candidate pools")
    parser.add_argument("experiments", nargs="+", help="experiment directories, e.g. experiments/v1.2.13 experiments/v1.3.2")
    parser.add_argument("--top-ks", default="5,10,20,50,100", help="comma separated topK values")
    parser.add_argument("--output-dir", help="output directory; default under experiments/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [resolve_path(value) for value in args.experiments]
    if len(experiments) < 1:
        raise ValueError("at least one experiment is required")
    top_ks = parse_top_ks(args.top_ks)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(experiments)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {experiment_label(path): read_summary(path) for path in experiments}
    window_maps = {experiment_label(path): read_window_diagnostics(path) for path in experiments}
    experiment_dirs = {experiment_label(path): path for path in experiments}

    summary_rows = [scalar_summary(path, summaries[experiment_label(path)]) for path in experiments]
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(summaries.values(), ignore_index=True).to_csv(output_dir / "window_metrics.csv", index=False)
    if len(experiments) >= 2:
        compare_overlaps(window_maps, top_ks).to_csv(output_dir / "overlap.csv", index=False)
        compare_ensemble_trials(window_maps).to_csv(output_dir / "ensemble_trials.csv", index=False)
        stage2_summary, stage2_windows, stage2_candidates = compare_stage2_rerank_trials(experiment_dirs, window_maps)
        stage2_summary.to_csv(output_dir / "stage2_rerank_trials.csv", index=False)
        stage2_windows.to_csv(output_dir / "stage2_rerank_windows.csv", index=False)
        stage2_candidates.to_csv(output_dir / "stage2_rerank_candidates.csv", index=False)
    write_readme(output_dir, experiments)

    print(f"analysis written to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
