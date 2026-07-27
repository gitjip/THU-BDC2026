import argparse
import json
import math
from pathlib import Path

import pandas as pd

from data_utils import normalize_stock_codes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_KS = (5, 10, 20, 50, 100)


def clean_top_ks(top_ks) -> list[int]:
    values = sorted({int(value) for value in top_ks if int(value) > 0})
    if not values:
        raise ValueError("top_k 至少需要包含一个正整数")
    return values


def parse_top_ks(raw: str) -> list[int]:
    return clean_top_ks(part.strip() for part in raw.split(",") if part.strip())


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def safe_mean(values) -> float | None:
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return float(series.mean())


def to_jsonable(payload):
    if isinstance(payload, dict):
        return {key: to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [to_jsonable(value) for value in payload]
    return json_safe(payload)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def resolve_path(path_value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def parse_selected(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})


def read_prediction_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    if "stock_id" not in scores.columns and "股票代码" in scores.columns:
        scores = scores.rename(columns={"股票代码": "stock_id"})
    required = {"rank", "stock_id", "pred_score", "selected", "weight"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{path} 缺少诊断所需列: {sorted(missing)}")

    scores = scores.copy()
    scores["stock_id"] = normalize_stock_codes(scores["stock_id"])
    scores["rank"] = pd.to_numeric(scores["rank"], errors="coerce").astype("Int64")
    scores["pred_score"] = pd.to_numeric(scores["pred_score"], errors="coerce")
    scores["selected"] = parse_selected(scores["selected"])
    scores["weight"] = pd.to_numeric(scores["weight"], errors="coerce").fillna(0.0)
    scores = scores.dropna(subset=["rank"]).sort_values("rank").reset_index(drop=True)
    scores["rank"] = scores["rank"].astype(int)
    scores["stock_prefix"] = scores["stock_id"].str[:3]
    return scores


def calculate_target_returns(target_data_path: Path) -> tuple[pd.DataFrame, list[str]]:
    target = pd.read_csv(target_data_path, dtype={"股票代码": str})
    required = {"股票代码", "日期", "开盘"}
    missing = required - set(target.columns)
    if missing:
        raise ValueError(f"{target_data_path} 缺少诊断所需列: {sorted(missing)}")

    target = target.copy()
    target["股票代码"] = normalize_stock_codes(target["股票代码"])
    target["日期"] = pd.to_datetime(target["日期"], errors="coerce").dt.normalize()
    target["开盘"] = pd.to_numeric(target["开盘"], errors="coerce")
    if target["日期"].isna().any():
        raise ValueError(f"{target_data_path} 存在无法解析的日期")

    target_dates = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in sorted(target["日期"].dropna().unique())]
    if len(target_dates) < 2:
        raise ValueError(f"{target_data_path} 目标窗口交易日不足，无法计算未来收益")

    first_day = pd.Timestamp(target_dates[0])
    last_day = pd.Timestamp(target_dates[-1])
    first_open = target[target["日期"].eq(first_day)][["股票代码", "开盘"]].rename(columns={"开盘": "target_start_open"})
    last_open = target[target["日期"].eq(last_day)][["股票代码", "开盘"]].rename(columns={"开盘": "target_end_open"})
    returns = first_open.merge(last_open, on="股票代码", how="inner").rename(columns={"股票代码": "stock_id"})
    returns["target_return"] = (returns["target_end_open"] - returns["target_start_open"]) / (returns["target_start_open"] + 1e-12)
    returns["target_return_rank"] = returns["target_return"].rank(ascending=False, method="min").astype("Int64")
    returns["target_return_percentile"] = returns["target_return"].rank(pct=True)
    return returns, target_dates


def records_from_frame(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    records = []
    for row in frame[columns].to_dict(orient="records"):
        records.append({key: json_safe(value) for key, value in row.items()})
    return records


def diagnose_prediction_scores(
    prediction_scores_path: Path,
    target_data_path: Path,
    output_csv_path: Path | None = None,
    output_json_path: Path | None = None,
    top_ks=DEFAULT_TOP_KS,
    window_name: str | None = None,
) -> dict:
    top_ks = clean_top_ks(top_ks)
    scores = read_prediction_scores(prediction_scores_path)
    returns, target_dates = calculate_target_returns(target_data_path)
    diagnostics = scores.merge(returns, on="stock_id", how="left")
    diagnostics["weighted_return"] = diagnostics["target_return"] * diagnostics["weight"]
    diagnostics["target_return_rank"] = diagnostics["target_return_rank"].astype("Int64")

    output_columns = [
        "rank",
        "stock_id",
        "stock_prefix",
        "pred_score",
        "selected",
        "weight",
        "target_start_open",
        "target_end_open",
        "target_return",
        "target_return_rank",
        "target_return_percentile",
        "weighted_return",
    ]
    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics[output_columns].to_csv(output_csv_path, index=False)

    valid = diagnostics.dropna(subset=["target_return"]).copy()
    selected = diagnostics[diagnostics["selected"]].copy()
    actual_top5 = returns.nlargest(5, "target_return").merge(
        diagnostics[["stock_id", "rank", "pred_score"]],
        on="stock_id",
        how="left",
    )

    summary = {
        "window": window_name or prediction_scores_path.parent.name,
        "target_start_date": target_dates[0],
        "target_end_date": target_dates[-1],
        "target_dates": target_dates,
        "candidate_count": int(len(diagnostics)),
        "available_target_return_count": int(len(valid)),
        "missing_target_return_count": int(diagnostics["target_return"].isna().sum()),
        "selected_count": int(selected["selected"].sum()),
        "selected_weight_sum": float(selected["weight"].sum()),
        "score_from_diagnostics": float(selected["weighted_return"].sum(skipna=True)),
        "selected_equal_weight_return": float(selected["target_return"].mean()) if len(selected) else None,
        "market_equal_weight_return": float(returns["target_return"].mean()) if len(returns) else None,
        "oracle_equal_weight_top5_return": float(returns["target_return"].nlargest(5).mean()) if len(returns) >= 5 else None,
        "spearman_pred_score_target_return": float(valid["pred_score"].corr(valid["target_return"], method="spearman")) if len(valid) > 1 else None,
        "top1_pred_score": float(diagnostics["pred_score"].iloc[0]) if len(diagnostics) else None,
        "top5_score_gap": float(diagnostics["pred_score"].iloc[0] - diagnostics["pred_score"].iloc[4]) if len(diagnostics) >= 5 else None,
        "top20_score_gap": float(diagnostics["pred_score"].iloc[0] - diagnostics["pred_score"].iloc[19]) if len(diagnostics) >= 20 else None,
        "actual_top5_mean_model_rank": float(actual_top5["rank"].mean()) if actual_top5["rank"].notna().any() else None,
        "actual_top5_worst_model_rank": int(actual_top5["rank"].max()) if actual_top5["rank"].notna().any() else None,
        "selected": records_from_frame(
            selected.sort_values("rank"),
            ["rank", "stock_id", "pred_score", "weight", "target_return", "target_return_rank"],
        ),
        "actual_top5": records_from_frame(
            actual_top5,
            ["stock_id", "target_return", "target_return_rank", "rank", "pred_score"],
        ),
    }

    for top_k in top_ks:
        top_frame = diagnostics.head(top_k)
        summary[f"pred_top{top_k}_equal_weight_return"] = float(top_frame["target_return"].mean()) if len(top_frame) else None
        summary[f"actual_top5_hits_in_pred_top{top_k}"] = int(actual_top5["rank"].le(top_k).sum())

    if output_json_path is not None:
        write_json(output_json_path, summary)
    return summary


def load_window_diagnostic_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_repeated_stock_rows(experiment_dir: Path, top_ks: list[int]) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((experiment_dir / "windows").glob("window_*/prediction_diagnostics.csv")):
        window = csv_path.parent.name
        frame = pd.read_csv(csv_path, dtype={"stock_id": str, "stock_prefix": str})
        frame["stock_id"] = normalize_stock_codes(frame["stock_id"])
        frame["stock_prefix"] = frame["stock_id"].str[:3]
        frame["selected"] = parse_selected(frame["selected"])
        frame["window"] = window
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    top_ks = clean_top_ks(top_ks)
    max_top_k = max(top_ks)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["rank"] <= max_top_k].copy()
    for top_k in top_ks:
        combined[f"in_top{top_k}"] = combined["rank"] <= top_k

    aggregations = {
        "stock_prefix": ("stock_prefix", "first"),
        "windows": ("window", lambda values: ",".join(sorted(set(values)))),
        "appearances": ("window", "count"),
        "selected_count": ("selected", "sum"),
        "avg_model_rank": ("rank", "mean"),
        "best_model_rank": ("rank", "min"),
        "avg_target_return": ("target_return", "mean"),
        "min_target_return": ("target_return", "min"),
        "max_target_return": ("target_return", "max"),
    }
    for top_k in top_ks:
        aggregations[f"top{top_k}_count"] = (f"in_top{top_k}", "sum")

    repeated = combined.groupby("stock_id", as_index=False).agg(**aggregations)
    sort_columns = [f"top{min(20, max_top_k)}_count"] if f"top{min(20, max_top_k)}_count" in repeated.columns else ["appearances"]
    sort_columns.extend(["selected_count", "avg_target_return"])
    ascending = [False, False, True]
    repeated = repeated.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    return repeated


def write_experiment_prediction_diagnostics(
    experiment_dir: Path,
    top_ks=DEFAULT_TOP_KS,
    repo_root: Path = REPO_ROOT,
) -> dict:
    top_ks = clean_top_ks(top_ks)
    summary_paths = sorted((experiment_dir / "windows").glob("window_*/prediction_diagnostics.json"))
    summaries = [load_window_diagnostic_summary(path) for path in summary_paths]
    if not summaries:
        return {}

    summary_path = experiment_dir / "prediction_diagnostics_summary.csv"
    summary_rows = []
    for item in summaries:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"selected", "actual_top5", "target_dates"}
        }
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    repeated_path = experiment_dir / "prediction_repeated_stocks.csv"
    repeated = build_repeated_stock_rows(experiment_dir, top_ks)
    repeated.to_csv(repeated_path, index=False)

    aggregate = {
        "window_count": len(summaries),
        "summary_csv": relative_path(summary_path, repo_root),
        "repeated_stocks_csv": relative_path(repeated_path, repo_root),
        "mean_score_from_diagnostics": safe_mean(item.get("score_from_diagnostics") for item in summaries),
        "mean_spearman_pred_score_target_return": safe_mean(item.get("spearman_pred_score_target_return") for item in summaries),
        "mean_pred_top5_equal_weight_return": safe_mean(item.get("pred_top5_equal_weight_return") for item in summaries),
        "mean_pred_top20_equal_weight_return": safe_mean(item.get("pred_top20_equal_weight_return") for item in summaries) if "pred_top20_equal_weight_return" in summaries[0] else None,
        "unique_selected_count": int(repeated.loc[repeated["selected_count"] > 0, "stock_id"].nunique()) if not repeated.empty else 0,
        "unique_top20_count": int(repeated.loc[repeated.get("top20_count", pd.Series(dtype=int)) > 0, "stock_id"].nunique()) if "top20_count" in repeated else None,
        "repeated_top20_stock_count": int((repeated.get("top20_count", pd.Series(dtype=int)) >= 2).sum()) if "top20_count" in repeated else None,
    }

    aggregate_path = experiment_dir / "prediction_diagnostics.json"
    write_json(aggregate_path, aggregate)
    aggregate["aggregate_json"] = relative_path(aggregate_path, repo_root)
    return aggregate


def diagnose_window_dir(window_dir: Path, repo_root: Path = REPO_ROOT, top_ks=DEFAULT_TOP_KS) -> dict:
    metadata_path = window_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prediction_scores_path = resolve_path(metadata["prediction_scores"], repo_root)
        target_data_path = resolve_path(metadata["target_data"], repo_root)
    else:
        prediction_scores_path = window_dir / "prediction_scores.csv"
        target_data_path = window_dir / "data" / "target_window.csv"

    return diagnose_prediction_scores(
        prediction_scores_path=prediction_scores_path,
        target_data_path=target_data_path,
        output_csv_path=window_dir / "prediction_diagnostics.csv",
        output_json_path=window_dir / "prediction_diagnostics.json",
        top_ks=top_ks,
        window_name=window_dir.name,
    )


def diagnose_experiment(experiment_dir: Path, repo_root: Path = REPO_ROOT, top_ks=DEFAULT_TOP_KS) -> dict:
    for window_dir in sorted((experiment_dir / "windows").glob("window_*")):
        if (window_dir / "prediction_scores.csv").exists():
            diagnose_window_dir(window_dir, repo_root=repo_root, top_ks=top_ks)
    return write_experiment_prediction_diagnostics(experiment_dir, top_ks=top_ks, repo_root=repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 walk-forward prediction_scores.csv 补充真实收益诊断")
    parser.add_argument("experiment", help="实验目录，例如 experiments/v1.3.4")
    parser.add_argument("--top-ks", default="5,10,20,50,100", help="逗号分隔的 topK 统计口径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment, REPO_ROOT)
    aggregate = diagnose_experiment(experiment_dir, top_ks=parse_top_ks(args.top_ks))
    if not aggregate:
        raise ValueError(f"未找到可诊断的窗口: {experiment_dir}")
    print(f"诊断汇总: {aggregate['summary_csv']}")
    print(f"重复股票: {aggregate['repeated_stocks_csv']}")


if __name__ == "__main__":
    main()
