from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_KS = (1, 2, 3, 4, 5)
DEFAULT_EXPOSURES = (0.2, 0.4, 0.6, 0.8, 1.0)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("top-k list must not be empty")
    invalid = [value for value in values if value < 1 or value > 5]
    if invalid:
        raise ValueError(f"top-k must be between 1 and 5: {invalid}")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("exposure list must not be empty")
    invalid = [value for value in values if value < 0.0 or value > 1.0]
    if invalid:
        raise ValueError(f"exposure must be between 0 and 1: {invalid}")
    return values


def experiment_label(experiment_dir: Path) -> str:
    return experiment_dir.name


def read_window_diagnostics(experiment_dir: Path) -> list[pd.DataFrame]:
    paths = sorted((experiment_dir / "windows").glob("window_*/prediction_diagnostics.csv"))
    if not paths:
        raise FileNotFoundError(f"missing window prediction diagnostics: {experiment_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"stock_id": str})
        missing = {"rank", "stock_id", "target_return"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        frame = frame.copy()
        frame["experiment"] = experiment_label(experiment_dir)
        frame["window"] = path.parent.name
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["target_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
        frame = frame.dropna(subset=["rank", "target_return"]).sort_values("rank").reset_index(drop=True)
        frame["rank"] = frame["rank"].astype(int)
        if "target_start_open" in frame and "target_end_open" in frame:
            # Keep these columns available for future checks without affecting this evaluation.
            frame["target_start_open"] = pd.to_numeric(frame["target_start_open"], errors="coerce")
            frame["target_end_open"] = pd.to_numeric(frame["target_end_open"], errors="coerce")
        frames.append(frame)
    return frames


def evaluate_window(frame: pd.DataFrame, top_k: int, exposure: float) -> dict:
    selected = frame.head(top_k).copy()
    if len(selected) < top_k:
        raise ValueError(f"{frame['window'].iloc[0]} has only {len(selected)} candidates for top_k={top_k}")

    topk_equal_return = float(selected["target_return"].mean())
    score = topk_equal_return * exposure
    market_equal_return = float(frame["target_return"].mean())
    oracle_top5_return = float(frame["target_return"].nlargest(5).mean()) if len(frame) >= 5 else None
    return {
        "experiment": frame["experiment"].iloc[0],
        "window": frame["window"].iloc[0],
        "top_k": top_k,
        "total_exposure": exposure,
        "weight_per_stock": exposure / top_k if top_k else 0.0,
        "selected_count": top_k,
        "score": score,
        "topk_equal_weight_return": topk_equal_return,
        "market_equal_weight_return": market_equal_return,
        "score_minus_market": score - market_equal_return,
        "oracle_equal_weight_top5_return": oracle_top5_return,
        "selected_stock_ids": ",".join(selected["stock_id"].astype(str).tolist()),
    }


def summarize(window_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["experiment", "top_k", "total_exposure"]
    for (experiment, top_k, exposure), group in window_scores.groupby(group_cols, sort=True):
        score = pd.to_numeric(group["score"], errors="coerce")
        market = pd.to_numeric(group["market_equal_weight_return"], errors="coerce")
        unscaled = pd.to_numeric(group["topk_equal_weight_return"], errors="coerce")
        rows.append(
            {
                "experiment": experiment,
                "top_k": int(top_k),
                "total_exposure": float(exposure),
                "weight_per_stock": float(exposure) / int(top_k),
                "window_count": int(len(group)),
                "mean_score": float(score.mean()),
                "median_score": float(score.median()),
                "std_score": float(score.std(ddof=0)),
                "score_to_std": float(score.mean() / score.std(ddof=0)) if score.std(ddof=0) > 0 else None,
                "min_score": float(score.min()),
                "max_score": float(score.max()),
                "worst3_mean_score": float(score.nsmallest(min(3, len(score))).mean()),
                "positive_windows": int((score > 0).sum()),
                "negative_windows": int((score < 0).sum()),
                "mean_unscaled_topk_return": float(unscaled.mean()),
                "mean_market_equal_weight_return": float(market.mean()),
                "mean_score_minus_market": float((score - market).mean()),
                "cumulative_score": float(score.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment", "mean_score", "score_to_std", "min_score"],
        ascending=[True, False, False, False],
    )


def build_scores(experiment_dirs: list[Path], top_ks: list[int], exposures: list[float]) -> pd.DataFrame:
    rows = []
    for experiment_dir in experiment_dirs:
        for frame in read_window_diagnostics(experiment_dir):
            for top_k in top_ks:
                for exposure in exposures:
                    rows.append(evaluate_window(frame, top_k=top_k, exposure=exposure))
    return pd.DataFrame(rows)


def format_score(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def write_readme(output_dir: Path, summary: pd.DataFrame, window_scores: pd.DataFrame) -> None:
    lines = [
        "# TopK 与现金仓位离线评估",
        "",
        "本目录由 `code/src/evaluate_submission_controls.py` 生成，只复用已有 walk-forward `prediction_diagnostics.csv`，不重跑训练或预测。",
        "",
        "## 文件",
        "",
        "- `summary.csv`：每个 topK/仓位组合的跨窗口汇总。",
        "- `window_scores.csv`：每个窗口、每个组合的明细分数和入选股票。",
        "",
        "## 解释",
        "",
        "- `top_k` 控制最终选几只股票，范围 1 到 5。",
        "- `total_exposure` 控制总仓位，未使用部分等价现金，收益按 0 计算。",
        "- 在固定 `top_k` 下，`total_exposure` 只会线性缩放收益和亏损，不会改变窗口胜负方向。",
        "",
        "## 结论速查",
        "",
    ]
    for experiment, group in summary.groupby("experiment", sort=True):
        best_mean = group.sort_values("mean_score", ascending=False).iloc[0]
        full_exposure = group[group["total_exposure"].eq(1.0)].copy()
        best_full = full_exposure.sort_values("mean_score", ascending=False).iloc[0] if not full_exposure.empty else best_mean
        default = group[group["top_k"].eq(5) & group["total_exposure"].eq(1.0)]
        default_row = default.iloc[0] if not default.empty else None
        lines.extend(
            [
                f"### `{experiment}`",
                "",
                (
                    f"- 最高均值组合：topK=`{int(best_mean['top_k'])}`，仓位=`{best_mean['total_exposure']:.2f}`，"
                    f"均值 `{format_score(best_mean['mean_score'])}`，最差 `{format_score(best_mean['min_score'])}`，"
                    f"正分窗口 `{int(best_mean['positive_windows'])}/{int(best_mean['window_count'])}`。"
                ),
                (
                    f"- 满仓时最高均值：topK=`{int(best_full['top_k'])}`，"
                    f"均值 `{format_score(best_full['mean_score'])}`，最差 `{format_score(best_full['min_score'])}`，"
                    f"std `{format_score(best_full['std_score'])}`。"
                ),
            ]
        )
        if default_row is not None:
            lines.append(
                f"- 当前默认 top5 满仓：均值 `{format_score(default_row['mean_score'])}`，"
                f"最差 `{format_score(default_row['min_score'])}`，std `{format_score(default_row['std_score'])}`。"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readme.md").write_text("\n".join(lines), encoding="utf-8")


def default_output_dir(experiment_dirs: list[Path]) -> Path:
    labels = "_vs_".join(experiment_label(path) for path in experiment_dirs)
    return REPO_ROOT / "experiments" / "analysis" / f"topk_exposure_{labels}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate topK and cash exposure from walk-forward prediction diagnostics")
    parser.add_argument("experiments", nargs="+", help="experiment directories, e.g. experiments/v1.4.5")
    parser.add_argument("--top-ks", default=",".join(str(value) for value in DEFAULT_TOP_KS), help="comma-separated topK values, default 1,2,3,4,5")
    parser.add_argument(
        "--exposures",
        default=",".join(str(value) for value in DEFAULT_EXPOSURES),
        help="comma-separated total exposure values, default 0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--output-dir", help="output directory, default under experiments/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dirs = [resolve_path(value) for value in args.experiments]
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")
    top_ks = parse_int_list(args.top_ks)
    exposures = parse_float_list(args.exposures)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(experiment_dirs)

    window_scores = build_scores(experiment_dirs, top_ks=top_ks, exposures=exposures)
    summary = summarize(window_scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    window_scores.to_csv(output_dir / "window_scores.csv", index=False)
    write_readme(output_dir, summary=summary, window_scores=window_scores)
    print(f"topK/exposure evaluation written to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
