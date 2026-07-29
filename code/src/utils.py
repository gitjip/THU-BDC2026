import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)


def show_progress_bar():
    return sys.stderr.isatty()


CROSS_SECTIONAL_RANK_BASE_COLUMNS = [
    '涨跌幅',
    '换手率',
    '成交额',
    '成交量',
    '振幅',
    'return_1',
    'return_5',
    'return_10',
    'volume_ratio',
    'volatility_10',
    'volatility_20',
    'rsi',
    'atr_14',
]

CROSS_SECTIONAL_RANK_REPLACE_COLUMNS = [
    '开盘',
    '收盘',
    '最高',
    '最低',
    '成交量',
    '成交额',
    '涨跌额',
    'sma_5',
    'sma_20',
    'ema_12',
    'ema_26',
    'ema_60',
    'boll_mid',
    'boll_std',
    'atr_14',
    'obv',
    'volume_ma_5',
    'volume_ma_20',
    'high_low_spread',
    'open_close_spread',
    'high_close_spread',
    'low_close_spread',
]

CROSS_SECTIONAL_RANK_REPLACE_LITE_COLUMNS = [
    '开盘',
    '收盘',
    '最高',
    '最低',
    '成交量',
    '成交额',
    '涨跌额',
]

CROSS_SECTIONAL_RANK_REPLACE_COLUMN_SETS = {
    'default': CROSS_SECTIONAL_RANK_REPLACE_COLUMNS,
    'lite': CROSS_SECTIONAL_RANK_REPLACE_LITE_COLUMNS,
}

MARKET_RELATIVE_FEATURE_SPECS = [
    ('return_1', 'mkt_rel_return_1', 'mean_diff'),
    ('return_5', 'mkt_rel_return_5', 'mean_diff'),
    ('return_10', 'mkt_rel_return_10', 'mean_diff'),
    ('涨跌幅', 'mkt_rel_pct_chg', 'mean_diff'),
    ('换手率', 'mkt_rel_turnover', 'median_ratio'),
    ('volume_ratio', 'mkt_rel_volume_ratio', 'median_ratio'),
    ('volatility_10', 'mkt_rel_volatility_10', 'median_ratio'),
    ('volatility_20', 'mkt_rel_volatility_20', 'median_ratio'),
]

MARKET_BREADTH_FEATURE_COLUMNS = [
    'mkt_breadth_5',
]

MARKET_ENV_FEATURE_COLUMNS = [
    'market_return_mean',
    'market_return_std',
    'up_ratio',
    'market_volatility_mean',
    'market_turnover_median',
]

RANK_MOMENTUM_FEATURE_COLUMNS = [
    'ret5_rank_minus_ret20_rank',
]

RANK_RISKADJ_FEATURE_COLUMNS = [
    'ret5_rank_minus_vol20_rank',
]

RET5_RANK_FEATURE_COLUMNS = [
    'return_5_cs_rank',
]

SHORT_OVERHEAT_FEATURE_COLUMNS = [
    'short_overheat_guard',
]

TREND_QUALITY_FEATURE_COLUMNS = [
    'tq_mom5_vol10',
    'tq_mom10_vol20',
    'tq_close_pos20',
    'tq_drawdown20',
    'tq_up_ratio10',
]

CLEAN_RISK_FEATURE_COLUMNS = [
    'cr_no_trade',
    'cr_no_trade_5',
    'cr_no_trade_20',
    'cr_amount_z20',
    'cr_turnover_z20',
    'cr_drawdown_20',
]

MULTI_PERIOD_FEATURE_COLUMNS = [
    'mp_return_3',
    'mp_return_20',
    'mp_return_40',
    'mp_volatility_5',
    'mp_volatility_40',
    'mp_ma_gap_3',
    'mp_ma_gap_10',
    'mp_ma_gap_40',
    'mp_volume_ratio_3_20',
    'mp_volume_ratio_10_40',
]


def cross_sectional_rank_feature_names(columns=None):
    source_columns = columns or CROSS_SECTIONAL_RANK_BASE_COLUMNS
    return [f'{column}_cs_rank' for column in source_columns]


def add_cross_sectional_rank_features(df, columns=None):
    """Add per-date percentile ranks so the model sees relative strength."""
    if '日期' not in df.columns:
        raise ValueError("横截面 rank 特征需要 日期 列")

    result = df.copy()
    source_columns = columns or CROSS_SECTIONAL_RANK_BASE_COLUMNS
    available_columns = [column for column in source_columns if column in result.columns]
    if not available_columns:
        return result, []

    dates = result['日期']
    added_columns = []
    for column in available_columns:
        rank_column = f'{column}_cs_rank'
        values = pd.to_numeric(result[column], errors='coerce')
        ranks = values.groupby(dates).rank(method='average', pct=True)
        result[rank_column] = ranks.fillna(0.5).astype(float)
        added_columns.append(rank_column)

    return result, added_columns


def apply_cross_sectional_rank_features(df, feature_columns, mode='append', columns=None, replace_set='default'):
    """
    Apply cross-sectional rank features and return the updated feature list.

    append: keep raw features and append rank columns.
    replace: keep the same feature count where possible by replacing selected
    raw absolute price/volume columns with their rank columns.
    """
    mode = (mode or 'off').strip().lower()
    if mode in {'0', 'false', 'none'}:
        mode = 'off'
    if mode == 'add':
        mode = 'append'
    if mode == 'substitute':
        mode = 'replace'

    if mode == 'off':
        return df.copy(), list(feature_columns), []
    if mode not in {'append', 'replace'}:
        raise ValueError(f"Unsupported cross-sectional rank mode: {mode}")

    source_columns = columns
    if source_columns is None:
        if mode == 'replace':
            replace_set = (replace_set or 'default').strip().lower()
            if replace_set in {'', 'full'}:
                replace_set = 'default'
            if replace_set not in CROSS_SECTIONAL_RANK_REPLACE_COLUMN_SETS:
                raise ValueError(f"Unsupported cross-sectional rank replace set: {replace_set}")
            source_columns = CROSS_SECTIONAL_RANK_REPLACE_COLUMN_SETS[replace_set]
        else:
            source_columns = CROSS_SECTIONAL_RANK_BASE_COLUMNS

    ranked, added_columns = add_cross_sectional_rank_features(df, source_columns)
    if mode == 'append':
        updated_columns = list(feature_columns)
        updated_columns.extend(column for column in added_columns if column not in updated_columns)
        return ranked, updated_columns, added_columns

    rank_column_by_source = {
        column: f'{column}_cs_rank'
        for column in source_columns
    }
    added_column_set = set(added_columns)
    updated_columns = []
    for column in feature_columns:
        rank_column = rank_column_by_source.get(column)
        if rank_column in added_column_set:
            replacement = rank_column
        else:
            replacement = column
        if replacement not in updated_columns:
            updated_columns.append(replacement)

    return ranked, updated_columns, added_columns


def market_relative_feature_names():
    return [target for _, target, _ in MARKET_RELATIVE_FEATURE_SPECS]


def add_market_relative_features(df, specs=None):
    """Add per-date market-relative features from existing per-stock indicators."""
    if '日期' not in df.columns:
        raise ValueError("市场相对特征需要 日期 列")

    result = df.copy()
    source_specs = specs or MARKET_RELATIVE_FEATURE_SPECS
    dates = result['日期']
    added_columns = []

    for source_column, target_column, method in source_specs:
        if source_column not in result.columns:
            continue

        values = pd.to_numeric(result[source_column], errors='coerce')
        grouped = values.groupby(dates)
        if method == 'mean_diff':
            relative_values = values - grouped.transform('mean')
        elif method == 'median_ratio':
            baseline = grouped.transform('median')
            baseline = baseline.where(baseline.abs() > 1e-12)
            relative_values = values / baseline - 1.0
        else:
            raise ValueError(f"Unsupported market-relative method: {method}")

        result[target_column] = (
            relative_values.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        added_columns.append(target_column)

    return result, added_columns


def apply_market_relative_features(df, feature_columns):
    processed, added_columns = add_market_relative_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def market_breadth_feature_names():
    return list(MARKET_BREADTH_FEATURE_COLUMNS)


def add_market_breadth_features(df):
    """Add a per-date market breadth feature shared by all stocks on that date."""
    required_columns = {'日期', '涨跌幅'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    pct_chg = pd.to_numeric(result['涨跌幅'], errors='coerce')
    valid = dates.notna() & pct_chg.notna()

    if valid.any():
        daily = pd.DataFrame(
            {
                '日期': dates[valid],
                'up': pct_chg[valid].gt(0).astype(float),
                'down': pct_chg[valid].lt(0).astype(float),
            }
        )
        breadth = daily.groupby('日期', sort=True)[['up', 'down']].mean()
        breadth['mkt_breadth'] = breadth['up'] - breadth['down']
        breadth['mkt_breadth_5'] = breadth['mkt_breadth'].rolling(5, min_periods=1).mean()
        result['mkt_breadth_5'] = dates.map(breadth['mkt_breadth_5'])
    else:
        result['mkt_breadth_5'] = 0.0

    result['mkt_breadth_5'] = (
        pd.to_numeric(result['mkt_breadth_5'], errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-1.0, upper=1.0)
        .fillna(0.0)
        .astype(float)
    )
    return result, list(MARKET_BREADTH_FEATURE_COLUMNS)


def apply_market_breadth_features(df, feature_columns):
    processed, added_columns = add_market_breadth_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def market_env_feature_names():
    return list(MARKET_ENV_FEATURE_COLUMNS)


def add_market_env_features(df):
    """Add per-date market state features shared by all stocks on that date."""
    required_columns = {'日期', 'return_1', 'volatility_20', '换手率'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    return_1 = pd.to_numeric(result['return_1'], errors='coerce')
    volatility_20 = pd.to_numeric(result['volatility_20'], errors='coerce')
    turnover = pd.to_numeric(result['换手率'], errors='coerce')
    valid = dates.notna()

    if valid.any():
        daily = pd.DataFrame(
            {
                '日期': dates[valid],
                'return_1': return_1[valid],
                'up': return_1[valid].gt(0).astype(float),
                'volatility_20': volatility_20[valid],
                'turnover': turnover[valid],
            }
        )
        market = daily.groupby('日期', sort=True).agg(
            market_return_mean=('return_1', 'mean'),
            market_return_std=('return_1', 'std'),
            up_ratio=('up', 'mean'),
            market_volatility_mean=('volatility_20', 'mean'),
            market_turnover_median=('turnover', 'median'),
        )
        for column in MARKET_ENV_FEATURE_COLUMNS:
            result[column] = dates.map(market[column])
    else:
        for column in MARKET_ENV_FEATURE_COLUMNS:
            result[column] = 0.0

    for column in MARKET_ENV_FEATURE_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    result['up_ratio'] = result['up_ratio'].clip(lower=0.0, upper=1.0)
    return result, list(MARKET_ENV_FEATURE_COLUMNS)


def apply_market_env_features(df, feature_columns):
    processed, added_columns = add_market_env_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def rank_momentum_feature_names():
    return list(RANK_MOMENTUM_FEATURE_COLUMNS)


def add_rank_momentum_features(df):
    """Add a one-column cross-sectional short-vs-medium momentum rank delta."""
    required_columns = {'股票代码', '日期', '收盘'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    groups = result.groupby('股票代码', sort=False)

    if 'return_5' in result.columns:
        return_5 = pd.to_numeric(result['return_5'], errors='coerce')
    else:
        return_5 = groups['收盘'].transform(lambda values: pd.to_numeric(values, errors='coerce').pct_change(5, fill_method=None))
    return_20 = groups['收盘'].transform(lambda values: pd.to_numeric(values, errors='coerce').pct_change(20, fill_method=None))

    ret5_rank = return_5.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    ret20_rank = return_20.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    result['ret5_rank_minus_ret20_rank'] = (
        (ret5_rank - ret20_rank)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-1.0, upper=1.0)
        .fillna(0.0)
        .astype(float)
    )

    return result, list(RANK_MOMENTUM_FEATURE_COLUMNS)


def apply_rank_momentum_features(df, feature_columns):
    processed, added_columns = add_rank_momentum_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def rank_riskadj_feature_names():
    return list(RANK_RISKADJ_FEATURE_COLUMNS)


def add_rank_riskadj_features(df):
    """Add a one-column cross-sectional risk-adjusted short momentum signal."""
    required_columns = {'股票代码', '日期', '收盘'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    groups = result.groupby('股票代码', sort=False)

    if 'return_5' in result.columns:
        return_5 = pd.to_numeric(result['return_5'], errors='coerce')
    else:
        return_5 = groups['收盘'].transform(lambda values: pd.to_numeric(values, errors='coerce').pct_change(5, fill_method=None))

    if 'volatility_20' in result.columns:
        volatility_20 = pd.to_numeric(result['volatility_20'], errors='coerce')
    else:
        return_1 = groups['收盘'].transform(lambda values: pd.to_numeric(values, errors='coerce').pct_change(1, fill_method=None))
        volatility_20 = return_1.groupby(result['股票代码'], sort=False).transform(lambda values: values.rolling(20, min_periods=10).std())

    ret5_rank = return_5.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    vol20_rank = volatility_20.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    result['ret5_rank_minus_vol20_rank'] = (
        (ret5_rank - vol20_rank)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=-1.0, upper=1.0)
        .fillna(0.0)
        .astype(float)
    )

    return result, list(RANK_RISKADJ_FEATURE_COLUMNS)


def apply_rank_riskadj_features(df, feature_columns):
    processed, added_columns = add_rank_riskadj_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def ret5_rank_feature_names():
    return list(RET5_RANK_FEATURE_COLUMNS)


def add_ret5_rank_features(df):
    """Add the current-date cross-sectional rank of 5-day return."""
    required_columns = {'日期', 'return_5'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    return_5 = pd.to_numeric(result['return_5'], errors='coerce')
    result['return_5_cs_rank'] = (
        return_5.groupby(dates).rank(method='average', pct=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.5)
        .astype(float)
    )

    return result, list(RET5_RANK_FEATURE_COLUMNS)


def apply_ret5_rank_features(df, feature_columns):
    processed, added_columns = add_ret5_rank_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def short_overheat_feature_names():
    return list(SHORT_OVERHEAT_FEATURE_COLUMNS)


def add_short_overheat_features(df):
    """Add one guard feature for short-term overheated high-score stocks."""
    required_columns = {'股票代码', '日期', '收盘'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    dates = pd.to_datetime(result['日期'], errors='coerce').dt.normalize()
    groups = result.groupby('股票代码', sort=False)
    close = pd.to_numeric(result['收盘'], errors='coerce')

    return_3 = groups['收盘'].transform(
        lambda values: pd.to_numeric(values, errors='coerce').pct_change(3, fill_method=None)
    )
    rolling_ma_5 = groups['收盘'].transform(
        lambda values: pd.to_numeric(values, errors='coerce').rolling(5, min_periods=3).mean()
    )
    close_gap_5ma = close / rolling_ma_5.where(rolling_ma_5.abs() > 1e-12) - 1.0

    ret3_rank = return_3.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    gap5ma_rank = close_gap_5ma.groupby(dates).rank(method='average', pct=True).fillna(0.5)
    result['short_overheat_guard'] = (
        pd.concat([ret3_rank, gap5ma_rank], axis=1)
        .max(axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0.0, upper=1.0)
        .fillna(0.5)
        .astype(float)
    )

    return result, list(SHORT_OVERHEAT_FEATURE_COLUMNS)


def apply_short_overheat_features(df, feature_columns):
    processed, added_columns = add_short_overheat_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def add_trend_quality_features(df):
    """Add per-stock past-only trend quality features."""
    required_columns = {'股票代码', '日期', '收盘', 'return_1', 'return_5', 'return_10', 'volatility_10', 'volatility_20'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    close = pd.to_numeric(result['收盘'], errors='coerce')
    ret1 = pd.to_numeric(result['return_1'], errors='coerce')
    ret5 = pd.to_numeric(result['return_5'], errors='coerce')
    ret10 = pd.to_numeric(result['return_10'], errors='coerce')
    vol10 = pd.to_numeric(result['volatility_10'], errors='coerce')
    vol20 = pd.to_numeric(result['volatility_20'], errors='coerce')
    groups = result.groupby('股票代码', sort=False)

    rolling_high20 = groups['收盘'].transform(
        lambda values: pd.to_numeric(values, errors='coerce').rolling(20, min_periods=5).max()
    )
    rolling_low20 = groups['收盘'].transform(
        lambda values: pd.to_numeric(values, errors='coerce').rolling(20, min_periods=5).min()
    )
    range20 = rolling_high20 - rolling_low20
    up_ratio10 = ret1.gt(0).astype(float).groupby(result['股票代码']).transform(
        lambda values: values.rolling(10, min_periods=5).mean()
    )

    result['tq_mom5_vol10'] = ret5 / vol10.where(vol10.abs() > 1e-6)
    result['tq_mom10_vol20'] = ret10 / vol20.where(vol20.abs() > 1e-6)
    result['tq_close_pos20'] = (close - rolling_low20) / range20.where(range20.abs() > 1e-12)
    result['tq_drawdown20'] = close / rolling_high20.where(rolling_high20.abs() > 1e-12) - 1.0
    result['tq_up_ratio10'] = up_ratio10

    for column in TREND_QUALITY_FEATURE_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-10.0, upper=10.0)
            .fillna(0.0)
            .astype(float)
        )

    return result, list(TREND_QUALITY_FEATURE_COLUMNS)


def apply_trend_quality_features(df, feature_columns):
    processed, added_columns = add_trend_quality_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def clean_risk_feature_names():
    return list(CLEAN_RISK_FEATURE_COLUMNS)


def add_clean_risk_features(df):
    """Add liquidity/no-trade/drawdown features inspired by data cleaning."""
    required_columns = {'股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率', '涨跌幅'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    open_ = pd.to_numeric(result['开盘'], errors='coerce')
    high = pd.to_numeric(result['最高'], errors='coerce')
    low = pd.to_numeric(result['最低'], errors='coerce')
    close = pd.to_numeric(result['收盘'], errors='coerce')
    volume = pd.to_numeric(result['成交量'], errors='coerce')
    amount = pd.to_numeric(result['成交额'], errors='coerce')
    turnover = pd.to_numeric(result['换手率'], errors='coerce')
    pct_chg = pd.to_numeric(result['涨跌幅'], errors='coerce')

    flat_price_mask = open_.eq(close) & close.eq(high) & high.eq(low)
    no_trade_mask = flat_price_mask & volume.le(0).fillna(False) & amount.le(0).fillna(False) & turnover.le(0).fillna(False)
    no_trade_mask &= pct_chg.fillna(0.0).abs().le(1e-12)
    result['cr_no_trade'] = no_trade_mask.astype(float)
    groups = result.groupby('股票代码', sort=False)
    result['cr_no_trade_5'] = groups['cr_no_trade'].transform(lambda values: values.rolling(5, min_periods=1).mean())
    result['cr_no_trade_20'] = groups['cr_no_trade'].transform(lambda values: values.rolling(20, min_periods=5).mean())

    result['_cr_log_amount'] = np.log1p(amount.clip(lower=0))
    result['_cr_log_turnover'] = np.log1p(turnover.clip(lower=0))
    groups = result.groupby('股票代码', sort=False)
    log_amount_mean20 = groups['_cr_log_amount'].transform(lambda values: values.rolling(20, min_periods=5).mean())
    log_amount_std20 = groups['_cr_log_amount'].transform(lambda values: values.rolling(20, min_periods=5).std())
    log_turnover_mean20 = groups['_cr_log_turnover'].transform(lambda values: values.rolling(20, min_periods=5).mean())
    log_turnover_std20 = groups['_cr_log_turnover'].transform(lambda values: values.rolling(20, min_periods=5).std())

    result['cr_amount_z20'] = (result['_cr_log_amount'] - log_amount_mean20) / log_amount_std20.where(log_amount_std20.abs() > 1e-12)
    result['cr_turnover_z20'] = (result['_cr_log_turnover'] - log_turnover_mean20) / log_turnover_std20.where(log_turnover_std20.abs() > 1e-12)

    rolling_high20 = groups['收盘'].transform(
        lambda values: pd.to_numeric(values, errors='coerce').rolling(20, min_periods=5).max()
    )
    result['cr_drawdown_20'] = close / rolling_high20.where(rolling_high20.abs() > 1e-12) - 1.0

    for column in CLEAN_RISK_FEATURE_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-10.0, upper=10.0)
            .fillna(0.0)
            .astype(float)
        )

    result.drop(columns=['_cr_log_amount', '_cr_log_turnover'], inplace=True)

    return result, list(CLEAN_RISK_FEATURE_COLUMNS)


def apply_clean_risk_features(df, feature_columns):
    processed, added_columns = add_clean_risk_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


def multi_period_feature_names():
    return list(MULTI_PERIOD_FEATURE_COLUMNS)


def add_multi_period_features(df):
    """Add simple past-only multi-period return, volatility, MA-gap and volume features."""
    required_columns = {'股票代码', '日期', '收盘', '成交量'}
    missing = required_columns - set(df.columns)
    if missing:
        return df.copy(), []

    result = df.copy()
    result = result.sort_values(['股票代码', '日期']).reset_index(drop=True)
    result['_mp_close'] = pd.to_numeric(result['收盘'], errors='coerce')
    result['_mp_volume'] = pd.to_numeric(result['成交量'], errors='coerce')
    groups = result.groupby('股票代码', sort=False)

    result['mp_return_3'] = groups['_mp_close'].transform(lambda values: values.pct_change(3, fill_method=None))
    result['mp_return_20'] = groups['_mp_close'].transform(lambda values: values.pct_change(20, fill_method=None))
    result['mp_return_40'] = groups['_mp_close'].transform(lambda values: values.pct_change(40, fill_method=None))

    result['_mp_return_1'] = groups['_mp_close'].transform(lambda values: values.pct_change(1, fill_method=None))
    result['mp_volatility_5'] = groups['_mp_return_1'].transform(lambda values: values.rolling(5, min_periods=3).std())
    result['mp_volatility_40'] = groups['_mp_return_1'].transform(lambda values: values.rolling(40, min_periods=20).std())

    for window, min_periods in [(3, 2), (10, 5), (40, 20)]:
        ma = groups['_mp_close'].transform(lambda values, w=window, m=min_periods: values.rolling(w, min_periods=m).mean())
        result[f'mp_ma_gap_{window}'] = result['_mp_close'] / ma.where(ma.abs() > 1e-12) - 1.0

    volume_ma_3 = groups['_mp_volume'].transform(lambda values: values.rolling(3, min_periods=2).mean())
    volume_ma_10 = groups['_mp_volume'].transform(lambda values: values.rolling(10, min_periods=5).mean())
    volume_ma_20 = groups['_mp_volume'].transform(lambda values: values.rolling(20, min_periods=10).mean())
    volume_ma_40 = groups['_mp_volume'].transform(lambda values: values.rolling(40, min_periods=20).mean())
    result['mp_volume_ratio_3_20'] = volume_ma_3 / volume_ma_20.where(volume_ma_20.abs() > 1e-12) - 1.0
    result['mp_volume_ratio_10_40'] = volume_ma_10 / volume_ma_40.where(volume_ma_40.abs() > 1e-12) - 1.0

    for column in MULTI_PERIOD_FEATURE_COLUMNS:
        result[column] = (
            pd.to_numeric(result[column], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-10.0, upper=10.0)
            .fillna(0.0)
            .astype(float)
        )

    result.drop(columns=['_mp_close', '_mp_volume', '_mp_return_1'], inplace=True)
    return result, list(MULTI_PERIOD_FEATURE_COLUMNS)


def apply_multi_period_features(df, feature_columns):
    processed, added_columns = add_multi_period_features(df)
    updated_columns = list(feature_columns)
    updated_columns.extend(column for column in added_columns if column not in updated_columns)
    return processed, updated_columns, added_columns


# 特征工程
def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)
def engineer_features_158plus39(df):
    """
    计算39个技术指标特征和158个Alpha特征，并合并它们。
    """
    # 为了避免修改原始DataFrame，创建一个副本
    df_copy = df.copy()

    # 1. 计算158个Alpha特征
    df_158 = engineer_features(df_copy)
    
    # 2. 计算39个技术指标特征
    df_39 = engineer_features_39(df_copy)

    # 3. 合并两个DataFrame
    # 首先，从df_39中选取我们需要的列，避免与df_158中的原始列（如'开盘'）重复
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    # 确保所有列都存在于df_39中
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    
    # 合并，df_158 已经包含了原始列和158个特征
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 4. 处理可能因为合并产生的重复列（如果两个函数生成了同名特征）
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    # 5. 统一处理inf和NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    
    return df_final

def engineer_features_39(df):
    """
    计算39个技术指标特征。
    'stock_idx','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    """
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    # MACD
    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # 标准差 = (上轨 - 中轨) / 2
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2

    # 删除临时列
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change(fill_method=None)
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1, fill_method=None)
    df['return_5'] = close.pct_change(5, fill_method=None)
    df['return_10'] = close.pct_change(10, fill_method=None)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 填充 NaN 值（注意：这可能引入偏差，根据下游任务决定是否保留）
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15 features) - 使用 talib 加速
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')
        
        # R-squared can be calculated as CORREL^2
        time_period_series = pd.Series(range(w), index=close.index[:w])
        rolling_corr = close.rolling(w).corr(time_period_series)
        rsquare = rolling_corr**2
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15 features) - 向量化操作，无需更改
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15 features) - 向量化操作，无需更改
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15 features) - 向量化操作，无需更改
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)
    
    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高
            day_targets = np.array(day_targets)
            # 使用涨跌幅的排序作为相关性得分（值越大排名越高）
            sorted_indices = np.argsort(day_targets)[::-1]  # 降序排列
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank  # 最高涨跌幅得分最高
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates), disable=not show_progress_bar()):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理", disable=not show_progress_bar()):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(
    data,
    features,
    sequence_length,
    ranking_data_path=None,
    min_window_end_date=None,
    max_stocks_per_date=0,
    stock_sample_seed=42,
):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    保持与原函数完全相同的输出格式。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    logger.info("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 确保每只股票都有 'label'（次日涨跌幅），否则无法作为 target
    data = data.dropna(subset=['label'])
    
    # 3. 为每只股票生成所有滑动窗口
    # label 已在预处理阶段按股票 shift(-label_horizon) 生成；
    # 这里按数据中的真实交易日滑窗，不要求自然日连续。
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target)

    logger.info("Step 1: 为每只股票生成滑动窗口")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks", disable=not show_progress_bar()):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # (T,)

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天的次日涨跌幅
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((end_date, stock_code, seq, target))

    # 4. 转为 DataFrame 便于按日期聚合
    logger.info("Step 2: 按日期聚合窗口")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []

    logger.info("Step 3: 构建每日样本并计算 relevance")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date", disable=not show_progress_bar()):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue

        if len(group) < 10:
            continue

        if max_stocks_per_date and len(group) > max_stocks_per_date:
            date_seed = int(pd.Timestamp(date).strftime('%Y%m%d')) + stock_sample_seed
            group = group.sample(n=max_stocks_per_date, random_state=date_seed).sort_values('stock_code')
        
        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)
        day_stocks = group['stock_code'].tolist()         # [str]

        # 计算 relevance（与原逻辑一致）
        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)

    logger.info("成功创建 %s 个训练样本", len(sequences))
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        logger.info("每个样本平均包含 %.1f 只股票", avg_stocks)

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    return sequences, targets, relevance_scores, stock_indices
