import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_utils import get_trading_dates, load_stock_data, normalize_stock_codes, setup_logging
from diagnose_predictions import diagnose_prediction_scores, write_experiment_prediction_diagnostics
from stage2_selection import select_ensemble_predictions


SEMVER_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
DEFAULT_VERSION = "v1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
VERSION_FILE = REPO_ROOT / "VERSION"
logger = logging.getLogger("bdc.walk_forward")

PROFILE_DESCRIPTIONS = {
    "quick": "快速调试档，小模型、小样本，用于检查流程是否跑通。",
    "balanced": "常规调参档，保留 instrument 输入，用于和 noid 系列做对照。",
    "noid": "去股票编号对照档，从模型输入中移除 instrument，股票代码只用于分组和输出。",
    "noid-rank": "noid + 横截面 rank 特征，用于测试当日相对强弱特征。",
    "noid-rank-lite": "noid + 小范围横截面 rank 替代，只替换最确定的原始价量尺度列。",
    "noid-rank-replace": "noid + 横截面 rank 替代绝对量价尺度特征，用于测试更少重复噪声的 rank 方案。",
    "noid-rank-sharp": "noid + rank 替代 + 更尖锐的 listwise 目标分布，用于测试训练监督信号是否过弱。",
    "noid-rank-trendq": "noid + 横截面 rank 替代 + 趋势质量特征，用于测试上涨质量信号。",
    "noid-rank-cleanrisk": "noid + 横截面 rank 替代 + 流动性/无成交/回撤风险特征，用于测试清洗启发的风险信号。",
    "noid-rank-multiperiod": "noid + 横截面 rank 替代 + 多周期基础特征，用于测试多窗口量价信号。",
    "noid-rank-breadth": "noid + 横截面 rank 替代 + 市场宽度特征，用于测试市场环境信号。",
    "noid-rank-momdelta": "noid + 横截面 rank 替代 + 短中期动量 rank 差，用于测试横截面动量变化信号。",
    "noid-rank-riskadj": "noid + 横截面 rank 替代 + 风险调整短期动量 rank 差，用于测试高收益但非高波动的相对强度信号。",
    "noid-rank-ret5rank": "noid + 横截面 rank 替代 + return_5 单列横截面 rank，用于测试最小短期相对强弱信号。",
    "noid-marketrel": "noid + 市场相对特征，用于测试个股相对市场强弱。",
    "noid-stable": "noid + 更强正则化，用于测试 dropout/weight_decay 是否缓解高波动偏好。",
    "noid-full": "noid + 完整训练数据，用于测试取消训练目标日和每日股票抽样后的效果。",
    "noid-lowvol": "noid + 低波动二阶段后处理，用于测试候选池低波动优先是否改善 top5。",
    "ensemble-lowvol": "复用 noid 与 rank-replace 模型，测试两模型 top5 并集低波动重排。",
    "smooth": "Lookahead 优化器对照档，用于测试优化器抗震荡效果。",
    "stable": "更强正则化对照档，保留 instrument 输入。",
    "large": "慢速候选复核档，较大前馈层和更多层数。",
    "full": "完整配置复核档，使用配置默认特征和模型设置。",
}

NOTE_CONFIG_KEYS = [
    "BDC_TUNE_PROFILE",
    "BDC_FEATURE_NUM",
    "BDC_SEQUENCE_LENGTH",
    "BDC_TRAIN_TARGET_DAYS",
    "BDC_VAL_DAYS",
    "BDC_MAX_STOCKS_PER_DAY",
    "BDC_NUM_EPOCHS",
    "BDC_LEARNING_RATE",
    "BDC_WEIGHT_DECAY",
    "BDC_DROPOUT",
    "BDC_USE_INSTRUMENT_FEATURE",
    "BDC_USE_MARKET_RELATIVE_FEATURES",
    "BDC_USE_MARKET_BREADTH_FEATURES",
    "BDC_USE_RANK_MOMENTUM_FEATURES",
    "BDC_USE_RANK_RISKADJ_FEATURES",
    "BDC_USE_RET5_RANK_FEATURES",
    "BDC_USE_TREND_QUALITY_FEATURES",
    "BDC_USE_CLEAN_RISK_FEATURES",
    "BDC_USE_MULTI_PERIOD_FEATURES",
    "BDC_SELECTION_STRATEGY",
    "BDC_TOP_K",
    "BDC_TOTAL_EXPOSURE",
    "BDC_LOSS_TEMPERATURE",
    "BDC_LOSS_TARGET_TEMPERATURE",
    "BDC_STAGE2_POOL_SIZE",
    "BDC_STAGE2_VOL_WINDOW",
    "BDC_ENSEMBLE_SOURCES",
    "BDC_USE_CROSS_SECTIONAL_RANKS",
    "BDC_CROSS_SECTIONAL_RANK_MODE",
    "BDC_CROSS_SECTIONAL_RANK_REPLACE_SET",
    "BDC_OPTIMIZER",
    "BDC_LR_SCHEDULER",
    "BDC_EARLY_STOPPING_PATIENCE",
]

RESUME_IGNORED_TUNE_ENV_KEYS = {
    "BDC_WF_WINDOWS",
    "BDC_NUM_PROCESSES",
    "BDC_TORCH_NUM_THREADS",
    "BDC_TENSORBOARD",
}

RESUME_BOOL_FALSE_ENV_KEYS = {
    "BDC_USE_INSTRUMENT_FEATURE",
    "BDC_USE_MARKET_RELATIVE_FEATURES",
    "BDC_USE_MARKET_BREADTH_FEATURES",
    "BDC_USE_RANK_MOMENTUM_FEATURES",
    "BDC_USE_RANK_RISKADJ_FEATURES",
    "BDC_USE_RET5_RANK_FEATURES",
    "BDC_USE_TREND_QUALITY_FEATURES",
    "BDC_USE_CLEAN_RISK_FEATURES",
    "BDC_USE_MULTI_PERIOD_FEATURES",
    "BDC_USE_CROSS_SECTIONAL_RANKS",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def format_optional_score(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def log_section(title: str) -> None:
    line = "=" * 72
    logger.info("\n%s\n%s\n%s", line, title, line)


@dataclass
class WalkForwardWindow:
    name: str
    as_of_date: str
    mock_submission_date: str
    target_start_date: str
    target_end_date: str
    target_dates: list[str]


def read_default_version() -> str:
    if VERSION_FILE.exists():
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        if version:
            return version
    return DEFAULT_VERSION


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def parse_args() -> argparse.Namespace:
    fast_dev = parse_bool_env("BDC_FAST_DEV", False)
    default_windows = parse_int_env("BDC_WF_WINDOWS", 2 if fast_dev else 3)

    parser = argparse.ArgumentParser(description="按语义版本运行多窗口 walk-forward 调参流程")
    parser.add_argument(
        "version",
        nargs="?",
        default=read_default_version(),
        help="实验语义版本号，例如 v1.3.10；默认读取 VERSION",
    )
    parser.add_argument("--windows", type=int, default=default_windows, help="walk-forward 窗口数量")
    parser.add_argument("--target-days", type=int, default=parse_int_env("BDC_WF_TARGET_DAYS", 5), help="每个窗口验证的连续交易日数量")
    parser.add_argument("--step-days", type=int, default=parse_int_env("BDC_WF_STEP_DAYS", 5), help="相邻窗口向前滚动的交易日步长")
    parser.add_argument("--data-file", default=os.environ.get("BDC_STOCK_DATA_FILE"), help="可选数据文件；默认按项目数据规则自动寻找")
    parser.add_argument("--skip-final", action="store_true", help="只跑 walk-forward，不训练最终模型")
    parser.add_argument("--publish-final", action="store_true", help="把最终预测复制到 output/result.csv；默认只保存在 experiments 下")
    parser.add_argument("--create-tag", action="store_true", help="流程成功后创建同名本地 Git tag；要求工作区无未提交改动")
    parser.add_argument("--resume", action="store_true", help="复用已有版本目录中已完成的窗口")
    parser.add_argument("--rerun-predictions", action="store_true", help="配合 --resume 时强制重跑预测和评分，不重训已有模型")
    parser.add_argument("--reuse-models-from", help="复用另一个实验目录的已训练模型，只重跑当前版本预测和评分")
    parser.add_argument("--dry-run", action="store_true", help="只打印窗口计划，不执行训练、预测和打分")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not SEMVER_PATTERN.fullmatch(args.version):
        raise ValueError(f"版本号必须是 vMAJOR.MINOR.PATCH 格式（各段可为多位数字），例如 v1.3.10。当前: {args.version}")
    if args.windows <= 0:
        raise ValueError("--windows 必须大于 0")
    if args.target_days != 5:
        raise ValueError("当前比赛提交窗口固定为连续 5 个交易日，请保持 --target-days 5")
    if args.step_days <= 0:
        raise ValueError("--step-days 必须大于 0")


def get_git_info() -> dict:
    def run_git(command: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *command],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def create_git_tag(version: str) -> None:
    git_info = get_git_info()
    if git_info["status_short"]:
        raise ValueError("创建 Git tag 前请先提交当前代码改动，确保 tag 能准确对应代码版本")

    existing = subprocess.run(
        ["git", "tag", "--list", version],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if existing:
        raise ValueError(f"Git tag 已存在: {version}")

    subprocess.run(["git", "tag", "-a", version, "-m", version], cwd=REPO_ROOT, check=True)
    logger.info("已创建本地 Git tag: %s", version)


def format_date_list(dates: pd.DatetimeIndex | list[str]) -> list[str]:
    return [pd.Timestamp(day).normalize().strftime("%Y-%m-%d") for day in dates]


def is_consecutive_trading_dates(
    trading_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex | list[str],
) -> bool:
    trading_date_list = format_date_list(trading_dates)
    target_date_list = format_date_list(target_dates)
    positions = {date: idx for idx, date in enumerate(trading_date_list)}

    if not target_date_list or any(date not in positions for date in target_date_list):
        return False

    target_positions = [positions[date] for date in target_date_list]
    expected_positions = list(range(target_positions[0], target_positions[0] + len(target_positions)))
    return target_positions == expected_positions


def is_consecutive_calendar_dates(target_dates: pd.DatetimeIndex | list[str]) -> bool:
    normalized = pd.DatetimeIndex(pd.to_datetime(target_dates).normalize())
    if len(normalized) == 0:
        return False
    expected = pd.date_range(normalized[0], periods=len(normalized), freq="D")
    return normalized.equals(expected)


def build_window_metadata(window: WalkForwardWindow, trading_dates: pd.DatetimeIndex) -> dict:
    target_dates = format_date_list(window.target_dates)
    first_day = pd.Timestamp(target_dates[0])
    last_day = pd.Timestamp(target_dates[-1])
    calendar_span_days = int((last_day - first_day).days + 1)
    return {
        **asdict(window),
        "target_window_type": "consecutive_calendar_days_with_stock_data",
        "target_trading_dates": target_dates,
        "target_trading_day_count": len(target_dates),
        "target_calendar_span_days": calendar_span_days,
        "is_consecutive_calendar_days": is_consecutive_calendar_dates(target_dates),
        "is_consecutive_trading_days": is_consecutive_trading_dates(trading_dates, target_dates),
        "date_note": "验证窗口要求连续自然日，并且这些自然日都必须在数据中有交易记录；周末或节假日窗口会被跳过。",
    }


def build_windows(dates: pd.DatetimeIndex, window_count: int, target_days: int, step_days: int) -> list[WalkForwardWindow]:
    candidates: list[tuple[int, int, pd.DatetimeIndex]] = []
    for target_start_idx in range(1, len(dates) - target_days + 1):
        target_end_idx = target_start_idx + target_days - 1
        target_dates = pd.DatetimeIndex(dates[target_start_idx : target_end_idx + 1])
        if len(target_dates) != target_days:
            raise ValueError(f"验证窗口交易日数量不等于 {target_days}")
        if not is_consecutive_trading_dates(dates, target_dates):
            raise ValueError("验证窗口不是连续交易日，请检查数据日期")
        if not is_consecutive_calendar_dates(target_dates):
            continue

        candidates.append((target_start_idx, target_end_idx, target_dates))

    windows: list[WalkForwardWindow] = []
    last_selected_end_idx: int | None = None

    for target_start_idx, target_end_idx, target_dates in reversed(candidates):
        if last_selected_end_idx is not None and target_end_idx > last_selected_end_idx - step_days:
            continue

        as_of_idx = target_start_idx - 1
        window = WalkForwardWindow(
            name=f"window_{len(windows) + 1:02d}",
            as_of_date=pd.Timestamp(dates[as_of_idx]).strftime("%Y-%m-%d"),
            mock_submission_date=(pd.Timestamp(target_dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            target_start_date=pd.Timestamp(target_dates[0]).strftime("%Y-%m-%d"),
            target_end_date=pd.Timestamp(target_dates[-1]).strftime("%Y-%m-%d"),
            target_dates=[pd.Timestamp(day).strftime("%Y-%m-%d") for day in target_dates],
        )
        windows.append(window)
        last_selected_end_idx = target_end_idx
        if len(windows) == window_count:
            break

    if len(windows) < window_count:
        raise ValueError(
            f"可用的连续 {target_days} 个自然日且都有交易数据的验证窗口不足: "
            f"需要 {window_count} 个，实际 {len(windows)} 个"
        )

    windows.reverse()
    for idx, window in enumerate(windows, start=1):
        window.name = f"window_{idx:02d}"

    return windows


def prepare_experiment_dir(experiment_dir: Path, resume: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if experiment_dir.exists() and any(experiment_dir.iterdir()) and not resume:
        raise FileExistsError(f"版本目录已存在: {experiment_dir}。如需继续旧任务，请加 --resume")
    experiment_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_experiment_ref(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        direct = REPO_ROOT / path
        if direct.exists():
            path = direct
        else:
            path = EXPERIMENTS_DIR / value
    if not path.exists():
        raise FileNotFoundError(f"复用模型实验目录不存在: {path}")
    return path


def resolve_experiment_refs(raw_value: str | None) -> list[Path]:
    if raw_value in (None, ""):
        return []
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    refs = [resolve_experiment_ref(value) for value in values]
    return [ref for ref in refs if ref is not None]


def experiment_source_label(experiment_dir: Path) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_dir.name).strip("_")
    return label or "source"


def source_tune_env(source_experiment_dir: Path) -> dict[str, str]:
    manifest = read_json(source_experiment_dir / "manifest.json")
    tune_env = manifest.get("tune_env") or {}
    env = {key: str(value) for key, value in tune_env.items() if value is not None}
    env["BDC_SELECTION_STRATEGY"] = "model_top5"
    env["BDC_STAGE2_POOL_SIZE"] = "5"
    env.pop("BDC_ENSEMBLE_SOURCES", None)
    return env


def source_window_model_dir(source_experiment_dir: Path | None, window: WalkForwardWindow) -> Path | None:
    if source_experiment_dir is None:
        return None
    source_window_dir = source_experiment_dir / "windows" / window.name
    metadata = read_json(source_window_dir / "metadata.json")
    source_window = metadata.get("window") or {}
    if source_window:
        source_target_dates = format_date_list(source_window.get("target_dates") or [])
        if source_window.get("as_of_date") != window.as_of_date or source_target_dates != format_date_list(window.target_dates):
            raise ValueError(
                f"{source_window_dir} 的窗口日期与当前 {window.name} 不一致，不能复用模型"
            )

    model_dir = source_window_dir / "model"
    if not model_artifacts_exist(model_dir):
        raise FileNotFoundError(f"复用模型缺少 best_model.pth 或 scaler.pkl: {model_dir}")
    return model_dir


def model_artifacts_exist(model_dir: Path) -> bool:
    return (model_dir / "best_model.pth").exists() and (model_dir / "scaler.pkl").exists()


def source_final_model_dir(source_experiment_dir: Path | None) -> Path | None:
    if source_experiment_dir is None:
        return None
    model_dir = source_experiment_dir / "final" / "model"
    if not model_artifacts_exist(model_dir):
        raise FileNotFoundError(f"复用最终模型缺少 best_model.pth 或 scaler.pkl: {model_dir}")
    return model_dir


def read_existing_summary(experiment_dir: Path) -> dict[str, dict]:
    summary_path = experiment_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    if "window" not in summary.columns:
        return {}
    rows = {}
    for _, row in summary.iterrows():
        row_dict = row.to_dict()
        rows[str(row_dict["window"])] = row_dict
        key = row_identity_key(row_dict)
        if key:
            rows[f"date:{key}"] = row_dict
    return rows


def normalize_resume_env_value(key: str, value) -> str:
    if value is None or value == "":
        return "0" if key in RESUME_BOOL_FALSE_ENV_KEYS else ""
    normalized = str(value)
    if key in RESUME_BOOL_FALSE_ENV_KEYS:
        lowered = normalized.strip().lower()
        if lowered in {"0", "false", "no", "off"}:
            return "0"
        if lowered in {"1", "true", "yes", "on"}:
            return "1"
    return normalized


def resume_tune_env(tune_env: dict) -> dict[str, str]:
    keys = set(tune_env) - RESUME_IGNORED_TUNE_ENV_KEYS
    return {key: normalize_resume_env_value(key, tune_env.get(key)) for key in sorted(keys)}


def window_identity(window: WalkForwardWindow | dict) -> dict[str, str | list[str]]:
    if isinstance(window, WalkForwardWindow):
        return {
            "name": window.name,
            "as_of_date": window.as_of_date,
            "target_dates": format_date_list(window.target_dates),
        }
    return {
        "name": str(window.get("name") or ""),
        "as_of_date": str(window.get("as_of_date") or ""),
        "target_dates": format_date_list(window.get("target_dates") or window.get("target_trading_dates") or []),
    }


def window_date_identity(window: WalkForwardWindow | dict) -> dict[str, str | list[str]]:
    identity = window_identity(window)
    return {
        "as_of_date": identity["as_of_date"],
        "target_dates": identity["target_dates"],
    }


def window_identity_key(window: WalkForwardWindow | dict) -> str:
    identity = window_date_identity(window)
    return f"{identity['as_of_date']}__{','.join(identity['target_dates'])}"


def row_identity_key(row: dict) -> str:
    as_of_date = str(row.get("as_of_date") or "")
    target_dates = str(row.get("target_dates") or "")
    if not as_of_date or not target_dates:
        return ""
    parsed_target_dates = [item.strip() for item in target_dates.split(",") if item.strip()]
    if not parsed_target_dates:
        return ""
    return f"{as_of_date}__{','.join(parsed_target_dates)}"


def existing_summary_row(existing_summary: dict[str, dict], window: WalkForwardWindow) -> dict | None:
    date_row = existing_summary.get(f"date:{window_identity_key(window)}")
    if date_row:
        return date_row

    name_row = existing_summary.get(window.name)
    if name_row and row_identity_key(name_row) == window_identity_key(window):
        return name_row

    return None


def build_data_signature(full_df: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    return {
        "row_count": int(len(full_df)),
        "stock_count": int(full_df["股票代码"].nunique()) if "股票代码" in full_df.columns else None,
        "trading_day_count": int(len(dates)),
        "start_date": pd.Timestamp(dates[0]).strftime("%Y-%m-%d") if len(dates) else "",
        "end_date": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") if len(dates) else "",
    }


def validate_existing_window_row(window: WalkForwardWindow, existing_row: dict | None) -> None:
    if not existing_row:
        return

    row_as_of = str(existing_row.get("as_of_date") or "")
    row_target_dates = str(existing_row.get("target_dates") or "")
    if row_as_of and row_as_of != window.as_of_date:
        raise ValueError(
            f"{window.name} 的 summary.csv 日期与当前窗口不一致: "
            f"as_of 旧={row_as_of}, 新={window.as_of_date}。请换版本号或清理旧窗口后重跑。"
        )
    if row_target_dates:
        parsed_target_dates = [item.strip() for item in row_target_dates.split(",") if item.strip()]
        if parsed_target_dates != format_date_list(window.target_dates):
            raise ValueError(
                f"{window.name} 的 summary.csv 目标日期与当前窗口不一致: "
                f"旧={parsed_target_dates}, 新={format_date_list(window.target_dates)}。请换版本号或清理旧窗口后重跑。"
            )


def validate_resume_window(
    window_dir: Path,
    window: WalkForwardWindow,
    previous_metadata: dict,
    existing_row: dict | None,
    requires_existing_metadata: bool,
) -> None:
    validate_existing_window_row(window, existing_row)
    if not requires_existing_metadata:
        return
    if not previous_metadata:
        raise ValueError(f"{window_dir} 缺少 metadata.json，不能安全 --resume。请换版本号或清理旧窗口后重跑。")

    previous_window = previous_metadata.get("window") or {}
    if window_date_identity(previous_window) != window_date_identity(window):
        raise ValueError(
            f"{window_dir} 的 metadata.json 窗口日期与当前窗口不一致，不能安全 --resume。"
            f"旧={window_identity(previous_window)}, 新={window_identity(window)}。请换版本号或清理旧窗口后重跑。"
        )


def normalize_resume_window_dirs(experiment_dir: Path, windows: list[WalkForwardWindow]) -> None:
    windows_dir = experiment_dir / "windows"
    if not windows_dir.exists():
        return

    desired_name_by_key = {window_identity_key(window): window.name for window in windows}
    existing_dir_by_key = {}
    for child in sorted(windows_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("window_"):
            continue
        metadata = read_json(child / "metadata.json")
        previous_window = metadata.get("window") or {}
        if not previous_window:
            continue
        key = window_identity_key(previous_window)
        if key not in desired_name_by_key:
            continue
        if key in existing_dir_by_key:
            raise ValueError(
                f"{windows_dir} 中存在重复日期窗口: {existing_dir_by_key[key].name}, {child.name}。请清理后重跑。"
            )
        existing_dir_by_key[key] = child

    moves = [
        (source_dir, windows_dir / desired_name_by_key[key])
        for key, source_dir in existing_dir_by_key.items()
        if source_dir.name != desired_name_by_key[key]
    ]
    if not moves:
        return

    source_dirs = {source_dir.resolve() for source_dir, _ in moves}
    for _, target_dir in moves:
        if target_dir.exists() and target_dir.resolve() not in source_dirs:
            raise ValueError(f"无法重排窗口目录，目标目录已存在且不属于可移动旧窗口: {target_dir}")

    temp_dir = windows_dir / ".resume_reindex_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    staged_moves = []
    for source_dir, target_dir in moves:
        staged_dir = temp_dir / source_dir.name
        shutil.move(str(source_dir), str(staged_dir))
        staged_moves.append((staged_dir, target_dir))

    for staged_dir, target_dir in staged_moves:
        shutil.move(str(staged_dir), str(target_dir))

    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("已按窗口日期重排已有目录，便于增量扩跑: %s 个窗口", len(moves))


def validate_resume_experiment(
    experiment_dir: Path,
    previous_manifest: dict,
    tune_env: dict,
    windows: list[WalkForwardWindow],
    data_signature: dict,
) -> None:
    if not previous_manifest:
        if experiment_dir.exists() and any(experiment_dir.iterdir()):
            raise ValueError(f"{experiment_dir} 已有内容但缺少 manifest.json，不能安全 --resume。请换版本号或清理目录。")
        return

    previous_signature = previous_manifest.get("data_signature")
    if previous_signature and previous_signature != data_signature:
        raise ValueError(
            "当前 stock_data 摘要与旧 manifest 不一致，不能安全 --resume。"
            f"旧={previous_signature}, 新={data_signature}。请换版本号重新跑。"
        )

    previous_windows = [window_date_identity(item) for item in previous_manifest.get("windows") or []]
    previous_window_keys = {window_identity_key(item) for item in previous_manifest.get("windows") or []}
    current_window_keys = {window_identity_key(window) for window in windows}
    if previous_windows and previous_window_keys - current_window_keys:
        missing = [
            item
            for item in previous_windows
            if window_identity_key(item) not in current_window_keys
        ]
        raise ValueError(
            "当前 walk-forward 窗口不包含旧 manifest 中的全部日期，不能安全 --resume。"
            f"缺失旧窗口={missing}。请换版本号重新跑。"
        )

    previous_env = resume_tune_env(previous_manifest.get("tune_env") or {})
    current_env = resume_tune_env(tune_env)
    keys = sorted(set(previous_env) | set(current_env))
    mismatches = [
        f"{key}: 旧={previous_env.get(key, '')}, 新={current_env.get(key, '')}"
        for key in keys
        if normalize_resume_env_value(key, previous_env.get(key)) != normalize_resume_env_value(key, current_env.get(key))
    ]
    if mismatches:
        detail = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            detail = f"{detail}; ..."
        raise ValueError(f"当前调参配置与旧 manifest 不一致，不能安全 --resume。{detail}。请换版本号重新跑。")


def optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    if pd.isna(value):
        return None
    return float(value)


def stage_total_seconds(*values: float | None, fallback: float | None = None) -> float | None:
    if values and all(value is not None for value in values):
        return float(sum(value for value in values if value is not None))
    if fallback is not None:
        return fallback
    known = [value for value in values if value is not None]
    if known:
        return float(sum(known))
    return fallback


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("运行命令: %s", " ".join(command))
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    duration = time.perf_counter() - start_time
    logger.info("命令完成: 耗时=%s", format_duration(duration))
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return duration


def make_child_env(model_dir: Path, stock_data_file: Path, env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env["PYTHONUNBUFFERED"] = "1"
    env["BDC_OUTPUT_DIR"] = str(model_dir)
    env["BDC_STOCK_DATA_FILE"] = str(stock_data_file)
    env.setdefault("BDC_TENSORBOARD", "0")
    return env


def get_tune_env_snapshot() -> dict[str, str | None]:
    keys = [
        "BDC_TUNE_PROFILE",
        "BDC_FAST_DEV",
        "BDC_WF_WINDOWS",
        "BDC_FEATURE_NUM",
        "BDC_SEQUENCE_LENGTH",
        "BDC_TRAIN_TARGET_DAYS",
        "BDC_VAL_DAYS",
        "BDC_MAX_STOCKS_PER_DAY",
        "BDC_D_MODEL",
        "BDC_NHEAD",
        "BDC_NUM_LAYERS",
        "BDC_DIM_FEEDFORWARD",
        "BDC_BATCH_SIZE",
        "BDC_NUM_EPOCHS",
        "BDC_LEARNING_RATE",
        "BDC_WEIGHT_DECAY",
        "BDC_DROPOUT",
        "BDC_USE_INSTRUMENT_FEATURE",
        "BDC_USE_MARKET_RELATIVE_FEATURES",
        "BDC_USE_MARKET_BREADTH_FEATURES",
        "BDC_USE_RANK_MOMENTUM_FEATURES",
        "BDC_USE_RANK_RISKADJ_FEATURES",
        "BDC_USE_RET5_RANK_FEATURES",
        "BDC_USE_TREND_QUALITY_FEATURES",
        "BDC_USE_CLEAN_RISK_FEATURES",
        "BDC_USE_MULTI_PERIOD_FEATURES",
        "BDC_SELECTION_STRATEGY",
        "BDC_TOP_K",
        "BDC_TOTAL_EXPOSURE",
        "BDC_LOSS_TEMPERATURE",
        "BDC_LOSS_TARGET_TEMPERATURE",
        "BDC_STAGE2_POOL_SIZE",
        "BDC_STAGE2_VOL_WINDOW",
        "BDC_ENSEMBLE_SOURCES",
        "BDC_USE_CROSS_SECTIONAL_RANKS",
        "BDC_CROSS_SECTIONAL_RANK_MODE",
        "BDC_CROSS_SECTIONAL_RANK_REPLACE_SET",
        "BDC_OPTIMIZER",
        "BDC_LOOKAHEAD_K",
        "BDC_LOOKAHEAD_ALPHA",
        "BDC_LR_SCHEDULER",
        "BDC_LR_PATIENCE",
        "BDC_LR_FACTOR",
        "BDC_LR_THRESHOLD",
        "BDC_MIN_LR",
        "BDC_EARLY_STOPPING_PATIENCE",
        "BDC_EARLY_STOPPING_MIN_DELTA",
        "BDC_GRAD_CLIP",
        "BDC_NUM_PROCESSES",
        "BDC_TORCH_NUM_THREADS",
        "BDC_TENSORBOARD",
    ]
    return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def get_walk_forward_args_snapshot(args: argparse.Namespace, generated_window_count: int) -> dict:
    return {
        "requested_windows": args.windows,
        "generated_windows": generated_window_count,
        "target_days": args.target_days,
        "step_days": args.step_days,
        "data_file_arg": args.data_file,
        "skip_final": args.skip_final,
        "publish_final": args.publish_final,
        "create_tag": args.create_tag,
        "resume": args.resume,
        "rerun_predictions": args.rerun_predictions,
        "reuse_models_from": args.reuse_models_from,
        "dry_run": args.dry_run,
    }


def run_train(model_dir: Path, stock_data_file: Path, log_path: Path) -> float:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file)
    return run_logged([sys.executable, "code/src/train.py"], env=env, log_path=log_path)


def run_predict(
    model_dir: Path,
    stock_data_file: Path,
    result_path: Path,
    scores_path: Path | None = None,
    submission_date: str | None = None,
    as_of_date: str | None = None,
    target_start_date: str | None = None,
    env_overrides: dict[str, str] | None = None,
    selection_strategy: str | None = None,
) -> float:
    env = make_child_env(model_dir=model_dir, stock_data_file=stock_data_file, env_overrides=env_overrides)
    command = [sys.executable, "code/src/predict.py", "--output", str(result_path)]
    if scores_path:
        command.extend(["--scores-output", str(scores_path)])
    if submission_date:
        command.extend(["--submission-date", submission_date, "--as-of-date", as_of_date or submission_date])
    if target_start_date:
        command.extend(["--target-start-date", target_start_date])
    if selection_strategy:
        command.extend(["--selection-strategy", selection_strategy])
    return run_logged(command, env=env, log_path=result_path.with_suffix(".log"))


def normalize_prediction(prediction_path: Path) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path, dtype={"stock_id": str, "股票代码": str})
    id_col = "stock_id" if "stock_id" in prediction.columns else "股票代码" if "股票代码" in prediction.columns else None
    weight_col = "weight" if "weight" in prediction.columns else "权重" if "权重" in prediction.columns else None
    if id_col is None or weight_col is None:
        raise ValueError(f"{prediction_path} 缺少 stock_id/股票代码 或 weight/权重 字段")

    prediction = prediction.rename(columns={id_col: "stock_id", weight_col: "weight"})[["stock_id", "weight"]].copy()
    prediction["stock_id"] = normalize_stock_codes(prediction["stock_id"])
    prediction["weight"] = pd.to_numeric(prediction["weight"], errors="coerce")

    if prediction["weight"].isna().any():
        raise ValueError(f"{prediction_path} 存在无法解析的权重")
    if len(prediction) > 5:
        raise ValueError(f"{prediction_path} 最多只能包含 5 只股票")
    weight_sum = float(prediction["weight"].sum())
    if weight_sum < 0 or weight_sum > 1.0:
        raise ValueError(f"{prediction_path} 权重和必须在 0 到 1 之间，当前为 {weight_sum}")
    return prediction


def calculate_window_score(full_df: pd.DataFrame, prediction_path: Path, target_dates: list[str]) -> dict:
    prediction = normalize_prediction(prediction_path)
    target_index = pd.DatetimeIndex(pd.to_datetime(target_dates).normalize())
    first_day = target_index[0]
    last_day = target_index[-1]

    window_data = full_df[full_df["日期"].isin(target_index)].copy()
    first_open = window_data[window_data["日期"] == first_day][["股票代码", "开盘"]].rename(columns={"开盘": "start_open"})
    last_open = window_data[window_data["日期"] == last_day][["股票代码", "开盘"]].rename(columns={"开盘": "end_open"})
    returns = first_open.merge(last_open, on="股票代码", how="inner")
    returns["return"] = (returns["end_open"] - returns["start_open"]) / (returns["start_open"] + 1e-12)

    selected = prediction.merge(returns, left_on="stock_id", right_on="股票代码", how="left")
    if selected["return"].isna().any():
        missing = selected.loc[selected["return"].isna(), "stock_id"].tolist()
        raise ValueError(f"目标窗口首尾交易日缺少这些预测股票的数据: {missing}")

    selected["weighted_return"] = selected["return"] * selected["weight"]
    score = float(selected["weighted_return"].sum())
    oracle_equal_weight_top5 = float(returns["return"].nlargest(5).mean()) if len(returns) >= 5 else 0.0
    market_equal_weight = float(returns["return"].mean()) if len(returns) else 0.0

    return {
        "score": score,
        "oracle_equal_weight_top5": oracle_equal_weight_top5,
        "market_equal_weight": market_equal_weight,
        "target_window_type": "consecutive_calendar_days_with_stock_data",
        "target_dates": target_dates,
        "target_trading_dates": target_dates,
        "target_trading_day_count": len(target_dates),
        "target_calendar_span_days": int((last_day - first_day).days + 1),
        "is_consecutive_calendar_days": is_consecutive_calendar_dates(target_dates),
        "selected": [
            {
                "stock_id": row.stock_id,
                "weight": float(row.weight),
                "return": float(row.return_),
                "weighted_return": float(row.weighted_return),
            }
            for row in selected.rename(columns={"return": "return_"}).itertuples(index=False)
        ],
    }


def write_window_data(full_df: pd.DataFrame, window: WalkForwardWindow, window_dir: Path) -> tuple[Path, Path]:
    data_dir = window_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(window.as_of_date)
    target_dates = pd.DatetimeIndex(pd.to_datetime(window.target_dates).normalize())

    train_data_path = data_dir / "train_until_as_of.csv"
    target_data_path = data_dir / "target_window.csv"
    full_df[full_df["日期"] <= as_of].to_csv(train_data_path, index=False)
    full_df[full_df["日期"].isin(target_dates)].to_csv(target_data_path, index=False)
    return train_data_path, target_data_path


def read_prediction_scores_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"stock_id": str, "股票代码": str})
    if "stock_id" not in frame.columns and "股票代码" in frame.columns:
        frame = frame.rename(columns={"股票代码": "stock_id"})
    required = {"rank", "stock_id", "pred_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少 ensemble 所需列: {sorted(missing)}")
    frame = frame.copy()
    frame["stock_id"] = normalize_stock_codes(frame["stock_id"])
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["pred_score"] = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame = frame.dropna(subset=["rank", "stock_id"]).sort_values("rank").reset_index(drop=True)
    frame["rank"] = frame["rank"].astype(int)
    return frame


def write_ensemble_prediction(
    score_paths: list[tuple[str, Path]],
    train_data_path: Path,
    prediction_path: Path,
    prediction_scores_path: Path,
) -> dict:
    history = pd.read_csv(train_data_path, dtype={"股票代码": str})
    score_frames = [
        (label, read_prediction_scores_csv(path))
        for label, path in score_paths
    ]
    strategy = os.environ.get("BDC_SELECTION_STRATEGY", "ensemble_low_vol_top5")
    selected, scores = select_ensemble_predictions(
        score_frames,
        history=history,
        strategy=strategy,
        top_k=parse_int_env("BDC_TOP_K", 5),
        total_exposure=parse_float_env("BDC_TOTAL_EXPOSURE", 1.0),
        volatility_window=parse_int_env("BDC_STAGE2_VOL_WINDOW", 20),
    )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_scores_path.parent.mkdir(parents=True, exist_ok=True)
    selected[["stock_id", "weight"]].to_csv(prediction_path, index=False)
    scores.to_csv(prediction_scores_path, index=False)

    source_top5 = {}
    for label, frame in score_frames:
        source_top5[label] = frame.sort_values("rank").head(5)["stock_id"].tolist()

    return {
        "selection_strategy": strategy,
        "top_k": parse_int_env("BDC_TOP_K", 5),
        "total_exposure": parse_float_env("BDC_TOTAL_EXPOSURE", 1.0),
        "source_top5": source_top5,
        "selected": selected["stock_id"].tolist(),
        "candidate_count": int((scores["stage2_pool_member"] == True).sum()),
    }


def run_window(
    full_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    experiment_dir: Path,
    version: str,
    window: WalkForwardWindow,
    resume: bool,
    rerun_predictions: bool,
    source_experiment_dir: Path | None,
    existing_row: dict | None = None,
) -> dict:
    window_dir = experiment_dir / "windows" / window.name
    model_dir = window_dir / "model"
    prediction_path = window_dir / "prediction.csv"
    prediction_scores_path = window_dir / "prediction_scores.csv"
    prediction_diagnostics_path = window_dir / "prediction_diagnostics.csv"
    prediction_diagnostics_summary_path = window_dir / "prediction_diagnostics.json"
    score_path = window_dir / "score.json"
    metadata_path = window_dir / "metadata.json"
    reused_model_dir = source_window_model_dir(source_experiment_dir, window)
    active_model_dir = reused_model_dir or model_dir
    window_start_time = time.perf_counter()
    train_seconds = optional_float(existing_row.get("train_seconds")) if existing_row else None
    predict_seconds = optional_float(existing_row.get("predict_seconds")) if existing_row else None
    existing_window_seconds = optional_float(existing_row.get("window_seconds")) if existing_row else None

    log_section(
        f"{window.name} | as_of={window.as_of_date} | target={window.target_start_date} ~ {window.target_end_date}"
    )

    previous_metadata = read_json(metadata_path) if resume else {}
    will_skip_local_model = resume and reused_model_dir is None and model_artifacts_exist(model_dir)
    will_skip_prediction = resume and not rerun_predictions and prediction_path.exists() and prediction_scores_path.exists()
    validate_resume_window(
        window_dir=window_dir,
        window=window,
        previous_metadata=previous_metadata,
        existing_row=existing_row,
        requires_existing_metadata=will_skip_local_model or will_skip_prediction,
    )

    train_data_path, target_data_path = write_window_data(full_df, window, window_dir)
    window_metadata = {
        "version": version,
        "window": build_window_metadata(window, trading_dates),
        "train_data": str(train_data_path.relative_to(REPO_ROOT)),
        "target_data": str(target_data_path.relative_to(REPO_ROOT)),
        "model_dir": display_path(active_model_dir),
        "reused_model_dir": display_path(reused_model_dir) if reused_model_dir else None,
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics": str(prediction_diagnostics_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics_summary": str(prediction_diagnostics_summary_path.relative_to(REPO_ROOT)),
        "score": str(score_path.relative_to(REPO_ROOT)),
    }
    write_json(metadata_path, window_metadata)

    if reused_model_dir:
        logger.info("%s 复用模型，跳过训练: %s", window.name, display_path(reused_model_dir))
        train_seconds = 0.0
    elif will_skip_local_model:
        logger.info("%s 已有模型，跳过训练", window.name)
    else:
        logger.info("%s 训练: as_of=%s", window.name, window.as_of_date)
        train_seconds = run_train(model_dir=model_dir, stock_data_file=train_data_path, log_path=window_dir / "logs" / "train_command.log")

    if will_skip_prediction:
        logger.info("%s 已有预测结果，跳过预测", window.name)
    else:
        logger.info("%s 预测: 目标窗口=%s ~ %s", window.name, window.target_start_date, window.target_end_date)
        predict_seconds = run_predict(
            model_dir=active_model_dir,
            stock_data_file=train_data_path,
            result_path=prediction_path,
            scores_path=prediction_scores_path,
            submission_date=window.mock_submission_date,
            as_of_date=window.as_of_date,
            target_start_date=window.target_start_date,
        )

    score = calculate_window_score(full_df, prediction_path, window.target_dates)
    diagnostics = diagnose_prediction_scores(
        prediction_scores_path=prediction_scores_path,
        target_data_path=target_data_path,
        output_csv_path=prediction_diagnostics_path,
        output_json_path=prediction_diagnostics_summary_path,
        window_name=window.name,
    )
    elapsed_seconds = time.perf_counter() - window_start_time
    window_seconds = stage_total_seconds(
        train_seconds,
        predict_seconds,
        fallback=existing_window_seconds if existing_window_seconds is not None else elapsed_seconds,
    )
    score["train_seconds"] = train_seconds
    score["train_duration"] = format_duration(train_seconds)
    score["predict_seconds"] = predict_seconds
    score["predict_duration"] = format_duration(predict_seconds)
    score["window_seconds"] = window_seconds
    score["window_duration"] = format_duration(window_seconds)
    write_json(score_path, score)
    logger.info("%s 验证得分: %.6f | 总耗时=%s", window.name, score["score"], format_duration(window_seconds))

    return {
        "version": version,
        "window": window.name,
        "as_of_date": window.as_of_date,
        "mock_submission_date": window.mock_submission_date,
        "target_start_date": window.target_start_date,
        "target_end_date": window.target_end_date,
        "target_dates": ",".join(window.target_dates),
        "target_window_type": score["target_window_type"],
        "target_trading_day_count": score["target_trading_day_count"],
        "target_calendar_span_days": score["target_calendar_span_days"],
        "is_consecutive_calendar_days": score["is_consecutive_calendar_days"],
        "is_consecutive_trading_days": is_consecutive_trading_dates(trading_dates, window.target_dates),
        "score": score["score"],
        "oracle_equal_weight_top5": score["oracle_equal_weight_top5"],
        "market_equal_weight": score["market_equal_weight"],
        "train_seconds": train_seconds,
        "train_duration": format_duration(train_seconds),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "window_seconds": window_seconds,
        "window_duration": format_duration(window_seconds),
        "model_dir": display_path(active_model_dir),
        "reused_model_dir": display_path(reused_model_dir) if reused_model_dir else "",
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics": str(prediction_diagnostics_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics_summary": str(prediction_diagnostics_summary_path.relative_to(REPO_ROOT)),
        "pred_score_target_return_spearman": diagnostics.get("spearman_pred_score_target_return"),
        "pred_top5_equal_weight_return": diagnostics.get("pred_top5_equal_weight_return"),
        "pred_top20_equal_weight_return": diagnostics.get("pred_top20_equal_weight_return"),
        "actual_top5_hits_in_pred_top20": diagnostics.get("actual_top5_hits_in_pred_top20"),
        "score_file": str(score_path.relative_to(REPO_ROOT)),
    }


def run_ensemble_window(
    full_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    experiment_dir: Path,
    version: str,
    window: WalkForwardWindow,
    source_experiment_dirs: list[Path],
    resume: bool,
    rerun_predictions: bool,
    existing_row: dict | None = None,
) -> dict:
    window_dir = experiment_dir / "windows" / window.name
    prediction_path = window_dir / "prediction.csv"
    prediction_scores_path = window_dir / "prediction_scores.csv"
    prediction_diagnostics_path = window_dir / "prediction_diagnostics.csv"
    prediction_diagnostics_summary_path = window_dir / "prediction_diagnostics.json"
    score_path = window_dir / "score.json"
    metadata_path = window_dir / "metadata.json"
    window_start_time = time.perf_counter()
    predict_seconds = optional_float(existing_row.get("predict_seconds")) if existing_row else None
    existing_window_seconds = optional_float(existing_row.get("window_seconds")) if existing_row else None

    log_section(
        f"{window.name} | ensemble | as_of={window.as_of_date} | target={window.target_start_date} ~ {window.target_end_date}"
    )

    previous_metadata = read_json(metadata_path) if resume else {}
    will_skip_prediction = (
        resume
        and not rerun_predictions
        and prediction_path.exists()
        and prediction_scores_path.exists()
    )
    validate_resume_window(
        window_dir=window_dir,
        window=window,
        previous_metadata=previous_metadata,
        existing_row=existing_row,
        requires_existing_metadata=will_skip_prediction,
    )

    train_data_path, target_data_path = write_window_data(full_df, window, window_dir)
    source_entries = []
    source_run_entries = []
    source_score_paths = []
    ensemble_source_dir = window_dir / "ensemble_sources"
    for source_dir in source_experiment_dirs:
        label = experiment_source_label(source_dir)
        model_dir = source_window_model_dir(source_dir, window)
        assert model_dir is not None
        source_prediction_path = ensemble_source_dir / label / "prediction.csv"
        source_scores_path = ensemble_source_dir / label / "prediction_scores.csv"
        source_entries.append(
            {
                "label": label,
                "experiment": display_path(source_dir),
                "model_dir": display_path(model_dir),
                "prediction": display_path(source_prediction_path),
                "prediction_scores": display_path(source_scores_path),
            }
        )
        source_run_entries.append(
            {
                "label": label,
                "source_dir": source_dir,
                "model_dir": model_dir,
                "prediction": source_prediction_path,
                "prediction_scores": source_scores_path,
            }
        )
        source_score_paths.append((label, source_scores_path))

    window_metadata = {
        "version": version,
        "window": build_window_metadata(window, trading_dates),
        "train_data": str(train_data_path.relative_to(REPO_ROOT)),
        "target_data": str(target_data_path.relative_to(REPO_ROOT)),
        "ensemble_sources": source_entries,
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics": str(prediction_diagnostics_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics_summary": str(prediction_diagnostics_summary_path.relative_to(REPO_ROOT)),
        "score": str(score_path.relative_to(REPO_ROOT)),
    }
    write_json(metadata_path, window_metadata)

    should_predict_sources = not (
        will_skip_prediction
        and all(path.exists() for _, path in source_score_paths)
    )
    if should_predict_sources:
        total_predict_seconds = 0.0
        for entry in source_run_entries:
            logger.info("%s 源模型预测: %s", window.name, entry["label"])
            total_predict_seconds += run_predict(
                model_dir=entry["model_dir"],
                stock_data_file=train_data_path,
                result_path=entry["prediction"],
                scores_path=entry["prediction_scores"],
                submission_date=window.mock_submission_date,
                as_of_date=window.as_of_date,
                target_start_date=window.target_start_date,
                env_overrides=source_tune_env(entry["source_dir"]),
                selection_strategy="model_top5",
            )

        ensemble_info = write_ensemble_prediction(
            source_score_paths,
            train_data_path=train_data_path,
            prediction_path=prediction_path,
            prediction_scores_path=prediction_scores_path,
        )
        predict_seconds = total_predict_seconds
        write_json(window_dir / "ensemble_prediction.json", ensemble_info)
    else:
        logger.info("%s 已有 ensemble 预测结果，跳过预测", window.name)

    score = calculate_window_score(full_df, prediction_path, window.target_dates)
    diagnostics = diagnose_prediction_scores(
        prediction_scores_path=prediction_scores_path,
        target_data_path=target_data_path,
        output_csv_path=prediction_diagnostics_path,
        output_json_path=prediction_diagnostics_summary_path,
        window_name=window.name,
    )
    elapsed_seconds = time.perf_counter() - window_start_time
    window_seconds = stage_total_seconds(
        0.0,
        predict_seconds,
        fallback=existing_window_seconds if existing_window_seconds is not None else elapsed_seconds,
    )
    score["train_seconds"] = 0.0
    score["train_duration"] = format_duration(0.0)
    score["predict_seconds"] = predict_seconds
    score["predict_duration"] = format_duration(predict_seconds)
    score["window_seconds"] = window_seconds
    score["window_duration"] = format_duration(window_seconds)
    write_json(score_path, score)
    logger.info("%s ensemble 验证得分: %.6f | 总耗时=%s", window.name, score["score"], format_duration(window_seconds))

    return {
        "version": version,
        "window": window.name,
        "as_of_date": window.as_of_date,
        "mock_submission_date": window.mock_submission_date,
        "target_start_date": window.target_start_date,
        "target_end_date": window.target_end_date,
        "target_dates": ",".join(window.target_dates),
        "target_window_type": score["target_window_type"],
        "target_trading_day_count": score["target_trading_day_count"],
        "target_calendar_span_days": score["target_calendar_span_days"],
        "is_consecutive_calendar_days": score["is_consecutive_calendar_days"],
        "is_consecutive_trading_days": is_consecutive_trading_dates(trading_dates, window.target_dates),
        "score": score["score"],
        "oracle_equal_weight_top5": score["oracle_equal_weight_top5"],
        "market_equal_weight": score["market_equal_weight"],
        "train_seconds": 0.0,
        "train_duration": format_duration(0.0),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "window_seconds": window_seconds,
        "window_duration": format_duration(window_seconds),
        "model_dir": "ensemble",
        "reused_model_dir": ",".join(display_path(path) for path in source_experiment_dirs),
        "prediction": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(prediction_scores_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics": str(prediction_diagnostics_path.relative_to(REPO_ROOT)),
        "prediction_diagnostics_summary": str(prediction_diagnostics_summary_path.relative_to(REPO_ROOT)),
        "pred_score_target_return_spearman": diagnostics.get("spearman_pred_score_target_return"),
        "pred_top5_equal_weight_return": diagnostics.get("pred_top5_equal_weight_return"),
        "pred_top20_equal_weight_return": diagnostics.get("pred_top20_equal_weight_return"),
        "actual_top5_hits_in_pred_top20": diagnostics.get("actual_top5_hits_in_pred_top20"),
        "score_file": str(score_path.relative_to(REPO_ROOT)),
    }


def run_final(
    full_data_file: Path,
    experiment_dir: Path,
    publish_final: bool,
    resume: bool,
    rerun_predictions: bool,
    source_experiment_dir: Path | None,
    existing_final: dict | None = None,
) -> dict:
    final_dir = experiment_dir / "final"
    model_dir = final_dir / "model"
    result_path = final_dir / "result.csv"
    result_scores_path = final_dir / "result_scores.csv"
    reused_model_dir = source_final_model_dir(source_experiment_dir)
    active_model_dir = reused_model_dir or model_dir
    final_start_time = time.perf_counter()
    train_seconds = optional_float(existing_final.get("train_seconds")) if existing_final else None
    predict_seconds = optional_float(existing_final.get("predict_seconds")) if existing_final else None
    existing_total_seconds = optional_float(existing_final.get("total_seconds")) if existing_final else None

    log_section("最终模型训练与预测")

    if reused_model_dir:
        logger.info("复用最终模型，跳过训练: %s", display_path(reused_model_dir))
        train_seconds = 0.0
    elif resume and model_artifacts_exist(model_dir):
        logger.info("最终模型已存在，跳过训练")
    else:
        logger.info("训练最终模型: %s", model_dir)
        train_seconds = run_train(model_dir=model_dir, stock_data_file=full_data_file, log_path=final_dir / "logs" / "train_command.log")

    if resume and not rerun_predictions and result_path.exists() and result_scores_path.exists():
        logger.info("最终预测已存在，跳过预测")
    else:
        logger.info("生成最终预测: %s", result_path)
        predict_seconds = run_predict(
            model_dir=active_model_dir,
            stock_data_file=full_data_file,
            result_path=result_path,
            scores_path=result_scores_path,
        )

    published_path = None
    if publish_final:
        published = REPO_ROOT / "output" / "result.csv"
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, published)
        published_path = str(published.relative_to(REPO_ROOT))
        logger.info("已发布最终预测到: %s", published_path)

    elapsed_seconds = time.perf_counter() - final_start_time
    final_seconds = stage_total_seconds(
        train_seconds,
        predict_seconds,
        fallback=existing_total_seconds if existing_total_seconds is not None else elapsed_seconds,
    )
    logger.info("最终流程完成: 耗时=%s", format_duration(final_seconds))

    return {
        "model_dir": display_path(active_model_dir),
        "reused_model_dir": display_path(reused_model_dir) if reused_model_dir else "",
        "prediction": str(result_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(result_scores_path.relative_to(REPO_ROOT)),
        "published_prediction": published_path,
        "train_seconds": train_seconds,
        "train_duration": format_duration(train_seconds),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "total_seconds": final_seconds,
        "total_duration": format_duration(final_seconds),
    }


def run_final_ensemble(
    full_data_file: Path,
    experiment_dir: Path,
    source_experiment_dirs: list[Path],
    publish_final: bool,
    resume: bool,
    rerun_predictions: bool,
    existing_final: dict | None = None,
) -> dict:
    final_dir = experiment_dir / "final"
    result_path = final_dir / "result.csv"
    result_scores_path = final_dir / "result_scores.csv"
    final_start_time = time.perf_counter()
    predict_seconds = optional_float(existing_final.get("predict_seconds")) if existing_final else None
    existing_total_seconds = optional_float(existing_final.get("total_seconds")) if existing_final else None

    log_section("最终 ensemble 预测")

    source_score_paths = []
    source_entries = []
    source_run_entries = []
    source_root = final_dir / "ensemble_sources"
    for source_dir in source_experiment_dirs:
        label = experiment_source_label(source_dir)
        model_dir = source_final_model_dir(source_dir)
        assert model_dir is not None
        source_prediction_path = source_root / label / "result.csv"
        source_scores_path = source_root / label / "result_scores.csv"
        source_entries.append(
            {
                "label": label,
                "experiment": display_path(source_dir),
                "model_dir": display_path(model_dir),
                "prediction": display_path(source_prediction_path),
                "prediction_scores": display_path(source_scores_path),
            }
        )
        source_run_entries.append(
            {
                "label": label,
                "source_dir": source_dir,
                "model_dir": model_dir,
                "prediction": source_prediction_path,
                "prediction_scores": source_scores_path,
            }
        )
        source_score_paths.append((label, source_scores_path))

    should_predict = not (
        resume
        and not rerun_predictions
        and result_path.exists()
        and result_scores_path.exists()
        and all(path.exists() for _, path in source_score_paths)
    )
    if should_predict:
        total_predict_seconds = 0.0
        for entry in source_run_entries:
            logger.info("最终源模型预测: %s", entry["label"])
            total_predict_seconds += run_predict(
                model_dir=entry["model_dir"],
                stock_data_file=full_data_file,
                result_path=entry["prediction"],
                scores_path=entry["prediction_scores"],
                env_overrides=source_tune_env(entry["source_dir"]),
                selection_strategy="model_top5",
            )
        ensemble_info = write_ensemble_prediction(
            source_score_paths,
            train_data_path=full_data_file,
            prediction_path=result_path,
            prediction_scores_path=result_scores_path,
        )
        predict_seconds = total_predict_seconds
        write_json(final_dir / "ensemble_prediction.json", ensemble_info)
    else:
        logger.info("最终 ensemble 预测已存在，跳过预测")

    published_path = None
    if publish_final:
        published = REPO_ROOT / "output" / "result.csv"
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, published)
        published_path = str(published.relative_to(REPO_ROOT))
        logger.info("已发布最终预测到: %s", published_path)

    elapsed_seconds = time.perf_counter() - final_start_time
    final_seconds = stage_total_seconds(
        0.0,
        predict_seconds,
        fallback=existing_total_seconds if existing_total_seconds is not None else elapsed_seconds,
    )
    logger.info("最终 ensemble 流程完成: 耗时=%s", format_duration(final_seconds))
    return {
        "model_dir": "ensemble",
        "reused_model_dir": ",".join(display_path(path) for path in source_experiment_dirs),
        "ensemble_sources": source_entries,
        "prediction": str(result_path.relative_to(REPO_ROOT)),
        "prediction_scores": str(result_scores_path.relative_to(REPO_ROOT)),
        "published_prediction": published_path,
        "train_seconds": 0.0,
        "train_duration": format_duration(0.0),
        "predict_seconds": predict_seconds,
        "predict_duration": format_duration(predict_seconds),
        "total_seconds": final_seconds,
        "total_duration": format_duration(final_seconds),
    }


def write_summary(experiment_dir: Path, rows: list[dict]) -> None:
    summary_path = experiment_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    logger.info("walk-forward 汇总已写入: %s", summary_path.relative_to(REPO_ROOT))


def write_experiment_note(experiment_dir: Path, manifest: dict) -> None:
    tune_env = manifest.get("tune_env") or {}
    walk_forward_args = manifest.get("walk_forward_args") or {}
    rows = manifest.get("window_results") or []
    profile = tune_env.get("BDC_TUNE_PROFILE", "")
    profile_desc = PROFILE_DESCRIPTIONS.get(profile, "自定义调参档。")
    git_info = manifest.get("git") or {}

    lines = [
        f"# {manifest.get('version', experiment_dir.name)} 实验说明",
        "",
        "## 目的",
        "",
        f"- Profile: `{profile or 'manual'}`",
        f"- 说明: {profile_desc}",
        f"- Git: `{git_info.get('commit', '')}` on `{git_info.get('branch', '')}`",
        f"- 数据: `{manifest.get('data_file', '')}`",
        "",
        "## 关键配置",
        "",
    ]

    for key in NOTE_CONFIG_KEYS:
        if key in tune_env:
            lines.append(f"- `{key}`: `{tune_env[key]}`")

    if walk_forward_args:
        lines.extend(
            [
                f"- `requested_windows`: `{walk_forward_args.get('requested_windows', '')}`",
                f"- `generated_windows`: `{walk_forward_args.get('generated_windows', '')}`",
                f"- `step_days`: `{walk_forward_args.get('step_days', '')}`",
                f"- `skip_final`: `{walk_forward_args.get('skip_final', '')}`",
                f"- `rerun_predictions`: `{walk_forward_args.get('rerun_predictions', '')}`",
                f"- `reuse_models_from`: `{walk_forward_args.get('reuse_models_from', '')}`",
            ]
        )

    lines.extend(["", "## 结果", ""])
    if rows:
        lines.extend(
            [
                f"- mean score: `{format_optional_score(manifest.get('walk_forward_score_mean'))}`",
                f"- min score: `{format_optional_score(manifest.get('walk_forward_score_min'))}`",
                f"- max score: `{format_optional_score(manifest.get('walk_forward_score_max'))}`",
                f"- total duration: `{manifest.get('total_duration', '')}`",
                "",
                "| window | target | score | train | predict |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            target = f"{row.get('target_start_date', '')} ~ {row.get('target_end_date', '')}"
            lines.append(
                "| {window} | {target} | {score} | {train} | {predict} |".format(
                    window=row.get("window", ""),
                    target=target,
                    score=format_optional_score(row.get("score")),
                    train=row.get("train_duration", ""),
                    predict=row.get("predict_duration", ""),
                )
            )
    else:
        lines.append("- walk-forward 窗口尚未完成。")

    final_result = manifest.get("final")
    prediction_diagnostics = manifest.get("prediction_diagnostics") or {}
    lines.extend(["", "## 预测诊断", ""])
    if prediction_diagnostics:
        lines.append(f"- summary: `{prediction_diagnostics.get('summary_csv', '')}`")
        lines.append(f"- repeated stocks: `{prediction_diagnostics.get('repeated_stocks_csv', '')}`")
        lines.append(
            f"- mean score from diagnostics: `{format_optional_score(prediction_diagnostics.get('mean_score_from_diagnostics'))}`"
        )
        lines.append(
            f"- mean pred-score/return spearman: `{format_optional_score(prediction_diagnostics.get('mean_spearman_pred_score_target_return'))}`"
        )
        lines.append(f"- unique selected stocks: `{prediction_diagnostics.get('unique_selected_count', '')}`")
        lines.append(f"- unique top20 stocks: `{prediction_diagnostics.get('unique_top20_count', '')}`")
    else:
        lines.append("- 未生成预测诊断，通常是因为没有运行 walk-forward 窗口。")

    lines.extend(["", "## 最终模型", ""])
    if final_result:
        lines.append(f"- final result: `{final_result.get('prediction', '')}`")
        lines.append(f"- published: `{final_result.get('published_prediction', '')}`")
    else:
        lines.append("- 未运行最终模型训练或最终预测。")

    note_path = experiment_dir / "experiment_note.md"
    note_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("实验说明已写入: %s", display_path(note_path))


def main() -> None:
    args = parse_args()
    validate_args(args)

    experiment_dir = EXPERIMENTS_DIR / args.version
    prepare_experiment_dir(experiment_dir, resume=args.resume, dry_run=args.dry_run)
    log_path = None if args.dry_run else experiment_dir / "walk_forward.log"
    setup_logging("bdc.walk_forward", log_path)
    run_start_time = time.perf_counter()

    full_df, data_file = load_stock_data(
        REPO_ROOT / "data",
        data_file=args.data_file,
        allow_train_fallback=True,
        logger=logger,
    )
    dates = get_trading_dates(full_df)
    windows = build_windows(dates, args.windows, args.target_days, args.step_days)

    logger.info("版本: %s", args.version)
    tune_env = get_tune_env_snapshot()
    ensemble_sources_raw = os.environ.get("BDC_ENSEMBLE_SOURCES", "")
    if tune_env:
        logger.info("调参参数: %s", ", ".join(f"{key}={value}" for key, value in tune_env.items()))
    logger.info(
        "walk-forward参数: windows=%s, target_days=%s, step_days=%s, skip_final=%s, resume=%s, rerun_predictions=%s, reuse_models_from=%s, dry_run=%s",
        args.windows,
        args.target_days,
        args.step_days,
        args.skip_final,
        args.resume,
        args.rerun_predictions,
        args.reuse_models_from,
        args.dry_run,
    )
    if ensemble_sources_raw:
        logger.info("ensemble源实验: %s", ensemble_sources_raw)
    logger.info("窗口数量: %s", len(windows))
    for window in windows:
        logger.info(
            "%s: as_of=%s, mock_submission=%s, target=%s ~ %s (%s)",
            window.name,
            window.as_of_date,
            window.mock_submission_date,
            window.target_start_date,
            window.target_end_date,
            ", ".join(window.target_dates),
        )

    if args.dry_run:
        return

    ensemble_source_dirs = resolve_experiment_refs(ensemble_sources_raw)
    source_experiment_dir = resolve_experiment_ref(args.reuse_models_from)
    if ensemble_source_dirs and source_experiment_dir:
        raise ValueError("BDC_ENSEMBLE_SOURCES 与 --reuse-models-from 不能同时使用")
    previous_manifest = read_json(experiment_dir / "manifest.json") if args.resume else {}
    data_signature = build_data_signature(full_df, dates)
    if args.resume:
        validate_resume_experiment(
            experiment_dir=experiment_dir,
            previous_manifest=previous_manifest,
            tune_env=tune_env,
            windows=windows,
            data_signature=data_signature,
        )
        normalize_resume_window_dirs(experiment_dir, windows)
    existing_summary = read_existing_summary(experiment_dir) if args.resume else {}
    git_info = get_git_info()
    manifest = {
        "version": args.version,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git": git_info,
        "data_file": str(Path(data_file).resolve()),
        "data_signature": data_signature,
        "reuse_models_from": display_path(source_experiment_dir) if source_experiment_dir else None,
        "ensemble_sources": [display_path(path) for path in ensemble_source_dirs],
        "tune_env": tune_env,
        "walk_forward_args": get_walk_forward_args_snapshot(args, len(windows)),
        "fast_dev_mode": parse_bool_env("BDC_FAST_DEV", False),
        "windows": [build_window_metadata(window, dates) for window in windows],
        "window_results": [],
        "final": None,
    }
    write_json(experiment_dir / "manifest.json", manifest)

    rows = []
    for window in windows:
        if ensemble_source_dirs:
            rows.append(
                run_ensemble_window(
                    full_df,
                    dates,
                    experiment_dir,
                    args.version,
                    window,
                    source_experiment_dirs=ensemble_source_dirs,
                    resume=args.resume,
                    rerun_predictions=args.rerun_predictions,
                    existing_row=existing_summary_row(existing_summary, window),
                )
            )
        else:
            rows.append(
                run_window(
                    full_df,
                    dates,
                    experiment_dir,
                    args.version,
                    window,
                    resume=args.resume,
                    rerun_predictions=args.rerun_predictions,
                    source_experiment_dir=source_experiment_dir,
                    existing_row=existing_summary_row(existing_summary, window),
                )
            )
        write_summary(experiment_dir, rows)

    if rows:
        manifest["prediction_diagnostics"] = write_experiment_prediction_diagnostics(experiment_dir)
        scores = [row["score"] for row in rows]
        manifest["window_results"] = rows
        manifest["walk_forward_score_mean"] = float(sum(scores) / len(scores))
        manifest["walk_forward_score_min"] = float(min(scores))
        manifest["walk_forward_score_max"] = float(max(scores))

    if not args.skip_final:
        if ensemble_source_dirs:
            manifest["final"] = run_final_ensemble(
                Path(data_file).resolve(),
                experiment_dir,
                ensemble_source_dirs,
                args.publish_final,
                resume=args.resume,
                rerun_predictions=args.rerun_predictions,
                existing_final=previous_manifest.get("final"),
            )
        else:
            manifest["final"] = run_final(
                Path(data_file).resolve(),
                experiment_dir,
                args.publish_final,
                resume=args.resume,
                rerun_predictions=args.rerun_predictions,
                source_experiment_dir=source_experiment_dir,
                existing_final=previous_manifest.get("final"),
            )

    write_json(experiment_dir / "manifest.json", manifest)
    if args.create_tag:
        create_git_tag(args.version)
    total_seconds = time.perf_counter() - run_start_time
    manifest["total_seconds"] = total_seconds
    manifest["total_duration"] = format_duration(total_seconds)
    write_json(experiment_dir / "manifest.json", manifest)
    write_experiment_note(experiment_dir, manifest)
    logger.info("版本 %s 完成 | 总耗时=%s", args.version, format_duration(total_seconds))


if __name__ == "__main__":
    main()
