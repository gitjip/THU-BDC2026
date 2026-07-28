import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def safe_mean(frame: pd.DataFrame, column: str):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def safe_sum(frame: pd.DataFrame, column: str):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def read_score_summary(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary.csv: {path}")
    frame = pd.read_csv(path)
    if "window" not in frame or "score" not in frame:
        raise ValueError(f"{path} must contain window and score columns")
    frame["experiment"] = experiment_label(experiment_dir)
    return frame


def read_diagnostics_summary(experiment_dir: Path) -> pd.DataFrame:
    path = experiment_dir / "prediction_diagnostics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prediction_diagnostics_summary.csv: {path}")
    frame = pd.read_csv(path)
    frame["experiment"] = experiment_label(experiment_dir)
    return frame


def read_repeated_summary(experiment_dir: Path) -> dict:
    aggregate = read_json(experiment_dir / "prediction_diagnostics.json")
    repeated_path = experiment_dir / "prediction_repeated_stocks.csv"
    repeated = pd.read_csv(repeated_path, dtype={"stock_id": str}) if repeated_path.exists() else pd.DataFrame()
    return {
        "unique_selected_count": aggregate.get("unique_selected_count"),
        "unique_top20_count": aggregate.get("unique_top20_count"),
        "repeated_top20_stock_count": aggregate.get("repeated_top20_stock_count"),
        "selected_stock_count": int((repeated.get("selected_count", pd.Series(dtype=int)) > 0).sum()) if not repeated.empty else None,
    }


def build_summary_rows(experiment_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for experiment_dir in experiment_dirs:
        scores = read_score_summary(experiment_dir)
        diagnostics = read_diagnostics_summary(experiment_dir)
        repeated = read_repeated_summary(experiment_dir)
        score = pd.to_numeric(scores["score"], errors="coerce")
        market = pd.to_numeric(scores.get("market_equal_weight"), errors="coerce") if "market_equal_weight" in scores else None
        rows.append(
            {
                "experiment": experiment_label(experiment_dir),
                "window_count": int(len(scores)),
                "mean_score": float(score.mean()),
                "median_score": float(score.median()),
                "std_score": float(score.std(ddof=0)),
                "min_score": float(score.min()),
                "max_score": float(score.max()),
                "positive_windows": int((score > 0).sum()),
                "mean_market_equal_weight": float(market.mean()) if market is not None else None,
                "mean_score_minus_market": float((score - market).mean()) if market is not None else None,
                "mean_spearman": safe_mean(diagnostics, "spearman_pred_score_target_return"),
                "mean_pred_top5_return": safe_mean(diagnostics, "pred_top5_equal_weight_return"),
                "mean_pred_top20_return": safe_mean(diagnostics, "pred_top20_equal_weight_return"),
                "mean_pred_top50_return": safe_mean(diagnostics, "pred_top50_equal_weight_return"),
                "mean_actual_top5_hits_in_pred_top20": safe_mean(diagnostics, "actual_top5_hits_in_pred_top20"),
                "mean_actual_top5_hits_in_pred_top50": safe_mean(diagnostics, "actual_top5_hits_in_pred_top50"),
                **repeated,
            }
        )
    return pd.DataFrame(rows)


def build_paired_diffs(experiment_dirs: list[Path], ensemble_label: str) -> pd.DataFrame:
    score_frames = {
        experiment_label(path): read_score_summary(path)[["window", "score", "market_equal_weight"]].rename(
            columns={"score": experiment_label(path)}
        )
        for path in experiment_dirs
    }
    if ensemble_label not in score_frames:
        raise ValueError(f"ensemble label not found: {ensemble_label}")

    merged = score_frames[ensemble_label].copy()
    for label, frame in score_frames.items():
        if label == ensemble_label:
            continue
        merged = merged.merge(frame[["window", label]], on="window", how="inner")
        merged[f"{ensemble_label}_minus_{label}"] = merged[ensemble_label] - merged[label]
        merged[f"{ensemble_label}_beats_{label}"] = merged[f"{ensemble_label}_minus_{label}"] > 0
    return merged


def build_diagnostic_summary(experiment_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for experiment_dir in experiment_dirs:
        diagnostics = read_diagnostics_summary(experiment_dir)
        repeated = read_repeated_summary(experiment_dir)
        rows.append(
            {
                "experiment": experiment_label(experiment_dir),
                "window_count": int(len(diagnostics)),
                "mean_spearman": safe_mean(diagnostics, "spearman_pred_score_target_return"),
                "mean_top5_return": safe_mean(diagnostics, "pred_top5_equal_weight_return"),
                "mean_top20_return": safe_mean(diagnostics, "pred_top20_equal_weight_return"),
                "mean_top50_return": safe_mean(diagnostics, "pred_top50_equal_weight_return"),
                "sum_actual_top5_hits_in_pred_top20": safe_sum(diagnostics, "actual_top5_hits_in_pred_top20"),
                "sum_actual_top5_hits_in_pred_top50": safe_sum(diagnostics, "actual_top5_hits_in_pred_top50"),
                **repeated,
            }
        )
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame, paired: pd.DataFrame, ensemble_label: str) -> list[str]:
    lines = []
    ensemble = summary[summary["experiment"].eq(ensemble_label)].iloc[0]
    baselines = summary[~summary["experiment"].eq(ensemble_label)]
    best_baseline_mean = float(baselines["mean_score"].max())
    best_baseline_min = float(baselines["min_score"].max())
    lines.append(
        f"- `{ensemble_label}` 均值 `{ensemble['mean_score']:.6f}`，最差窗口 `{ensemble['min_score']:.6f}`，"
        f"正分窗口 `{int(ensemble['positive_windows'])}/{int(ensemble['window_count'])}`。"
    )
    lines.append(f"- 最好源模型均值 `{best_baseline_mean:.6f}`，最好源模型最差窗口 `{best_baseline_min:.6f}`。")
    for column in paired.columns:
        if column.startswith(f"{ensemble_label}_minus_"):
            target = column.removeprefix(f"{ensemble_label}_minus_")
            mean_diff = float(paired[column].mean())
            win_count = int((paired[column] > 0).sum())
            lines.append(f"- 相对 `{target}`：平均差值 `{mean_diff:.6f}`，胜出窗口 `{win_count}/{len(paired)}`。")

    if float(ensemble["mean_score"]) > best_baseline_mean and float(ensemble["min_score"]) >= best_baseline_min - 0.01:
        lines.append("- 结论：支持继续使用 `ensemble-lowvol`。")
    elif float(ensemble["mean_score"]) > best_baseline_mean:
        lines.append("- 结论：均值支持 `ensemble-lowvol`，但最差窗口需要谨慎复核。")
    else:
        lines.append("- 结论：暂不支持 `ensemble-lowvol` 作为默认提交主线。")
    return lines


def write_readme(output_dir: Path, experiment_dirs: list[Path], ensemble_label: str, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# 验证汇总",
        "",
        "本文比较同一批 walk-forward 窗口上的多个实验。",
        "",
        "## 实验",
        "",
    ]
    for experiment_dir in experiment_dirs:
        lines.append(f"- `{experiment_label(experiment_dir)}`: `{experiment_dir}`")
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `summary.csv`：总体分数指标。",
            "- `paired_diffs.csv`：集成相对源模型的逐窗口差值。",
            "- `diagnostic_summary.csv`：排序诊断聚合指标。",
            "",
            "## 结论",
            "",
        ]
    )
    lines.extend(judge(summary, paired, ensemble_label))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(experiment_dirs: list[Path]) -> Path:
    labels = "_vs_".join(experiment_label(path) for path in experiment_dirs)
    return REPO_ROOT / "experiments" / "analysis" / f"validation_{labels}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize walk-forward validation experiments")
    parser.add_argument("experiments", nargs="+", help="experiment directories")
    parser.add_argument("--ensemble-label", help="experiment label to compare against others; default is the last experiment")
    parser.add_argument("--output-dir", help="output directory; default under experiments/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dirs = [resolve_path(value) for value in args.experiments]
    if len(experiment_dirs) < 2:
        raise ValueError("at least two experiments are required")
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")

    ensemble_label = args.ensemble_label or experiment_label(experiment_dirs[-1])
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(experiment_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_rows(experiment_dirs)
    paired = build_paired_diffs(experiment_dirs, ensemble_label=ensemble_label)
    diagnostic = build_diagnostic_summary(experiment_dirs)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_diffs.csv", index=False)
    diagnostic.to_csv(output_dir / "diagnostic_summary.csv", index=False)
    write_readme(output_dir, experiment_dirs, ensemble_label, summary, paired)

    print(f"validation summary written to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
