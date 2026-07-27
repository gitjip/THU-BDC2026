import argparse
import itertools
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_KS = (5, 10, 20, 50, 100)


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
    return_col = "target_return_0"
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged["min_rank"] = merged[rank_cols].min(axis=1)
    merged["borda_top20"] = sum((21 - merged[column]).clip(lower=0).fillna(0) for column in rank_cols)
    merged["target_return"] = merged[return_col]
    return merged


def select_candidates(candidates: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "avg_rank":
        return candidates.sort_values(["avg_rank", "min_rank"]).head(5)
    if method == "min_rank":
        return candidates.sort_values(["min_rank", "avg_rank"]).head(5)
    if method == "borda_top20":
        scored = candidates[candidates["borda_top20"] > 0].copy()
        return scored.sort_values(["borda_top20", "avg_rank"], ascending=[False, True]).head(5)
    raise ValueError(f"unknown candidate selection method: {method}")


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
                base = frames[0]
                union = sorted(set().union(*(set(frame.head(5)["stock_id"]) for frame in frames)))
                union_frame = base[base["stock_id"].isin(union)]
                returns_by_method["top5_union_equal"].append(float(union_frame["target_return"].mean()))
                counts_by_method["top5_union_equal"].append(int(len(union_frame)))

                candidates = candidate_frame(frames)
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
            "",
            "`top5_union_equal` can include more than five candidates and is not a valid submission by itself.",
            "It only measures whether the union candidate pool contains useful signal.",
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

    summary_rows = [scalar_summary(path, summaries[experiment_label(path)]) for path in experiments]
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(summaries.values(), ignore_index=True).to_csv(output_dir / "window_metrics.csv", index=False)
    if len(experiments) >= 2:
        compare_overlaps(window_maps, top_ks).to_csv(output_dir / "overlap.csv", index=False)
        compare_ensemble_trials(window_maps).to_csv(output_dir / "ensemble_trials.csv", index=False)
    write_readme(output_dir, experiments)

    print(f"analysis written to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
