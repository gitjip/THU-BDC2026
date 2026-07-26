import logging
import os
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
}


def setup_logging(name: str, log_file: str | Path | None = None) -> logging.Logger:
    def build_handlers() -> list[logging.Handler]:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter(fmt="%(message)s")
        handlers: list[logging.Handler] = []

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(console_formatter)
        handlers.append(stream_handler)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        return handlers

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    for handler in build_handlers():
        logger.addHandler(handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for handler in build_handlers():
        root_logger.addHandler(handler)

    return logger


def normalize_stock_codes(series: pd.Series) -> pd.Series:
    codes = series.astype("string").fillna("").str.strip()
    codes = codes.str.replace(r"\.0$", "", regex=True)

    extracted = codes.str.extract(r"(\d{6})$", expand=False)
    codes = extracted.where(extracted.notna(), codes)

    numeric_mask = codes.str.fullmatch(r"\d{1,6}", na=False)
    codes.loc[numeric_mask] = codes.loc[numeric_mask].str.zfill(6)
    return codes.astype(str)


def resolve_stock_data_file(
    data_path: str | Path,
    data_file: str | Path | None = None,
    allow_train_fallback: bool = True,
) -> Path:
    base_dir = Path(data_path)
    candidates: list[Path] = []

    env_file = os.environ.get("BDC_STOCK_DATA_FILE") or os.environ.get("BDC_DATA_FILE")
    for value in (env_file, data_file):
        if value:
            candidates.append(Path(value))

    candidates.extend(
        [
            base_dir / "stock_data.csv",
            base_dir / "stock_data",
        ]
    )
    if allow_train_fallback:
        candidates.append(base_dir / "train.csv")

    for candidate in candidates:
        if candidate.is_dir():
            if (candidate / "stock_data.csv").exists():
                return candidate / "stock_data.csv"
            if (candidate / "stock_data").exists():
                return candidate / "stock_data"
            if list(candidate.glob("*.csv")):
                return candidate
        elif candidate.exists() and candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"未找到股票数据文件，已检查: {searched}")


def read_market_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.is_dir():
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"{data_path} 目录下未找到 CSV 文件")
        df = pd.concat(
            [pd.read_csv(csv_file, dtype={"股票代码": str}) for csv_file in csv_files],
            ignore_index=True,
        )
    else:
        df = pd.read_csv(data_path, dtype={"股票代码": str})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} 缺少必要列: {sorted(missing)}")

    df = df.copy()
    df["股票代码"] = normalize_stock_codes(df["股票代码"])
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.normalize()
    if df["日期"].isna().any():
        bad_rows = int(df["日期"].isna().sum())
        raise ValueError(f"{data_path} 中存在无法解析的日期，共 {bad_rows} 行")

    numeric_columns = list(REQUIRED_COLUMNS - {"股票代码", "日期"})
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    return df


def load_stock_data(
    data_path: str | Path,
    data_file: str | Path | None = None,
    allow_train_fallback: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, Path]:
    path = resolve_stock_data_file(
        data_path=data_path,
        data_file=data_file,
        allow_train_fallback=allow_train_fallback,
    )
    df = read_market_data(path)

    if logger:
        dates = sorted(df["日期"].unique())
        logger.info(
            "读取数据: %s | 行数=%s | 股票数=%s | 交易日=%s | 范围=%s ~ %s",
            path,
            len(df),
            df["股票代码"].nunique(),
            len(dates),
            pd.Timestamp(dates[0]).date(),
            pd.Timestamp(dates[-1]).date(),
        )
    return df, path


def get_trading_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(df["日期"], errors="coerce").dropna().dt.normalize().unique()
    return pd.DatetimeIndex(sorted(dates))


def parse_date(date_value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"{name} 日期格式无效: {date_value}")
    return pd.Timestamp(date).normalize()


def parse_holiday_dates(raw_holidays: str | list[str] | tuple[str, ...] | None) -> set[pd.Timestamp]:
    if raw_holidays in (None, ""):
        return set()
    if isinstance(raw_holidays, str):
        values = [item.strip() for item in raw_holidays.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in raw_holidays if str(item).strip()]
    return {parse_date(value, "market_holidays") for value in values}


def _is_future_business_day(date: pd.Timestamp, holidays: set[pd.Timestamp]) -> bool:
    return date.weekday() < 5 and date not in holidays


def is_trading_day(
    date: pd.Timestamp,
    known_trading_dates: pd.DatetimeIndex,
    holidays: set[pd.Timestamp] | None = None,
) -> bool:
    holidays = holidays or set()
    date = pd.Timestamp(date).normalize()
    if len(known_trading_dates) > 0 and date <= pd.Timestamp(known_trading_dates[-1]):
        return date in known_trading_dates
    return _is_future_business_day(date, holidays)


def next_trading_day(
    start_date: pd.Timestamp,
    known_trading_dates: pd.DatetimeIndex,
    holidays: set[pd.Timestamp] | None = None,
) -> pd.Timestamp:
    holidays = holidays or set()
    current = pd.Timestamp(start_date).normalize()
    for _ in range(370):
        if is_trading_day(current, known_trading_dates, holidays):
            return current
        current += pd.Timedelta(days=1)
    raise ValueError(f"无法在 {start_date.date()} 之后找到合理交易日")


def future_trading_window(
    start_date: pd.Timestamp,
    horizon: int,
    known_trading_dates: pd.DatetimeIndex,
    holidays: set[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    if horizon <= 0:
        raise ValueError("prediction_horizon 必须大于 0")

    days = []
    current = pd.Timestamp(start_date).normalize()
    while len(days) < horizon:
        current = next_trading_day(current, known_trading_dates, holidays)
        days.append(current)
        current += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)


def split_by_last_trading_days(
    df: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = get_trading_dates(df)
    if test_days <= 0:
        raise ValueError("--test-days 必须大于 0")
    if len(dates) <= test_days:
        raise ValueError(f"交易日数量不足，无法划分最后 {test_days} 个交易日为测试集")

    train_dates = dates[:-test_days]
    test_dates = dates[-test_days:]
    train_df = df[df["日期"].isin(train_dates)].copy()
    test_df = df[df["日期"].isin(test_dates)].copy()
    return train_df, test_df, train_dates, test_dates


def split_train_val_by_trading_days(
    df: pd.DataFrame,
    sequence_length: int,
    val_days: int,
    label_horizon: int,
    train_target_days: int = 0,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = get_trading_dates(df)
    min_needed = sequence_length + val_days + label_horizon
    if len(dates) < min_needed:
        raise ValueError(
            f"数据交易日不足: 当前 {len(dates)} 天，至少需要 {min_needed} 天 "
            f"(sequence_length={sequence_length}, val_days={val_days}, label_horizon={label_horizon})"
        )

    labelable_dates = dates[:-label_horizon] if label_horizon > 0 else dates
    if len(labelable_dates) <= val_days:
        raise ValueError(f"可构建标签的交易日不足，无法划分 {val_days} 天验证集")

    val_start_idx = len(labelable_dates) - val_days
    val_start = pd.Timestamp(labelable_dates[val_start_idx])
    val_end = pd.Timestamp(labelable_dates[-1])

    first_train_target_idx = sequence_length - 1
    if train_target_days and train_target_days > 0:
        first_train_target_idx = max(first_train_target_idx, val_start_idx - train_target_days)
    train_context_start_idx = max(0, first_train_target_idx - sequence_length + 1)
    train_context_start = pd.Timestamp(dates[train_context_start_idx])
    train_target_start = pd.Timestamp(dates[first_train_target_idx])
    train_target_end = pd.Timestamp(dates[val_start_idx - 1])

    context_start_idx = max(0, val_start_idx - sequence_length + 1)
    val_context_start = pd.Timestamp(dates[context_start_idx])

    train_df = df[(df["日期"] >= train_context_start) & (df["日期"] < val_start)].copy()
    val_df = df[df["日期"] >= val_context_start].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("训练集或验证集为空，请检查数据时间范围")

    if logger:
        logger.info("训练目标日期: %s ~ %s", train_target_start.date(), train_target_end.date())
        logger.info(
            "训练取数范围: %s ~ %s (包含序列上下文)",
            train_df["日期"].min().date(),
            train_df["日期"].max().date(),
        )
        logger.info("验证目标日期: %s ~ %s", val_start.date(), val_end.date())
        logger.info(
            "验证取数范围: %s ~ %s (包含 %s 个交易日上下文)",
            val_df["日期"].min().date(),
            val_df["日期"].max().date(),
            sequence_length - 1,
        )
    return train_df, val_df, val_start, val_end
