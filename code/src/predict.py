import os
import multiprocessing as mp
import logging
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from data_utils import (
	future_trading_window,
	get_trading_dates,
	load_stock_data,
	parse_date,
	parse_holiday_dates,
	setup_logging,
)
from model import StockTransformer
from stage2_selection import normalize_selection_strategy, select_predictions
from utils import (
	apply_cross_sectional_rank_features,
	apply_clean_risk_features,
	apply_market_breadth_features,
	apply_market_relative_features,
	apply_multi_period_features,
	apply_rank_momentum_features,
	apply_rank_riskadj_features,
	apply_trend_quality_features,
	engineer_features_39,
	engineer_features_158plus39,
)

logger = logging.getLogger(__name__)


def show_progress_bar():
	return sys.stderr.isatty()


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def select_feature_columns(feature_columns):
	if config.get('use_instrument_feature', True):
		return list(feature_columns)
	return [column for column in feature_columns if column != 'instrument']


def parse_args():
	parser = argparse.ArgumentParser(description='基于提交截止日后的未来交易窗口生成 result.csv')
	parser.add_argument(
		'--submission-date',
		default=os.environ.get('BDC_SUBMISSION_DATE', config.get('submission_deadline_date', '2026-08-02')),
		help='提交截止日，默认 2026-08-02；预测窗口必须在该日期之后',
	)
	parser.add_argument(
		'--target-start-date',
		default=os.environ.get('BDC_TARGET_START_DATE'),
		help='预测窗口候选起始日，默认 submission-date 后一天；若非交易日会向后跳到合理交易日',
	)
	parser.add_argument(
		'--as-of-date',
		default=os.environ.get('BDC_AS_OF_DATE') or os.environ.get('BDC_PREDICT_DATE'),
		help='仅用于本地调试的数据截止日；默认使用不晚于 submission-date 的最新数据',
	)
	parser.add_argument(
		'--market-holidays',
		default=os.environ.get('BDC_MARKET_HOLIDAYS', config.get('market_holidays', '')),
		help='未来休市日，逗号分隔，例如 2026-10-01,2026-10-02',
	)
	parser.add_argument(
		'--output',
		default=config.get('prediction_output_path', os.path.join('./output/', 'result.csv')),
		help='预测结果输出路径，默认 ./output/result.csv',
	)
	parser.add_argument(
		'--scores-output',
		default=os.environ.get('BDC_PREDICTION_SCORES_OUTPUT'),
		help='完整候选股票排名诊断文件；默认写到 output 同目录的 *_scores.csv',
	)
	parser.add_argument(
		'--selection-strategy',
		default=os.environ.get('BDC_SELECTION_STRATEGY', config.get('selection_strategy', 'model_top5')),
		help='选股后处理策略，默认 model_top5；可选 low_vol_then_rank_top5',
	)
	parser.add_argument(
		'--top-k',
		type=int,
		default=int(os.environ.get('BDC_TOP_K', config.get('top_k', 5))),
		help='提交股票数量，1到5之间，默认5',
	)
	parser.add_argument(
		'--total-exposure',
		type=float,
		default=float(os.environ.get('BDC_TOTAL_EXPOSURE', config.get('total_exposure', 1.0))),
		help='总仓位，0到1之间，默认1.0；小于1表示留现金',
	)
	parser.add_argument(
		'--stage2-pool-size',
		type=int,
		default=int(os.environ.get('BDC_STAGE2_POOL_SIZE', config.get('stage2_pool_size', 10))),
		help='二阶段策略候选池大小，默认 10；必须不小于 5',
	)
	parser.add_argument(
		'--stage2-vol-window',
		type=int,
		default=int(os.environ.get('BDC_STAGE2_VOL_WINDOW', config.get('stage2_vol_window', 20))),
		help='二阶段波动率计算窗口，默认 20 个交易日',
	)
	return parser.parse_args()


def default_scores_output_path(output_path):
	root, ext = os.path.splitext(output_path)
	if not ext:
		ext = '.csv'
	return f'{root}_scores{ext}'


def preprocess_predict_data(df, stockid2idx):
	if config['feature_num'] not in feature_engineer_func_map:
		raise ValueError(f"Unsupported feature_num: {config['feature_num']}")
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = select_feature_columns(feature_cloums_map[config['feature_num']])

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(config.get('num_processes', 10), mp.cpu_count(), len(groups))
	logger.info("开始预测集特征工程: 股票数=%s, 进程数=%s", len(groups), num_processes)
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程', disable=not show_progress_bar()))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])
	if config.get('use_market_relative_features', False):
		processed, feature_columns, market_relative_columns = apply_market_relative_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加市场相对特征 %s 个，输入特征=%s",
			len(market_relative_columns),
			len(feature_columns),
		)
	if config.get('use_market_breadth_features', False):
		processed, feature_columns, market_breadth_columns = apply_market_breadth_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加市场宽度特征 %s 个，输入特征=%s",
			len(market_breadth_columns),
			len(feature_columns),
		)
	if config.get('use_trend_quality_features', False):
		processed, feature_columns, trend_quality_columns = apply_trend_quality_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加趋势质量特征 %s 个，输入特征=%s",
			len(trend_quality_columns),
			len(feature_columns),
		)
	if config.get('use_cross_sectional_rank_features', False):
		rank_mode = config.get('cross_sectional_rank_mode', 'append')
		rank_replace_set = config.get('cross_sectional_rank_replace_set', 'default')
		processed, feature_columns, rank_feature_columns = apply_cross_sectional_rank_features(
			processed,
			feature_columns,
			mode=rank_mode,
			replace_set=rank_replace_set,
		)
		logger.info(
			"预测集: 已应用横截面 rank 特征 mode=%s, replace_set=%s, rank列=%s, 输入特征=%s",
			rank_mode,
			rank_replace_set,
			len(rank_feature_columns),
			len(feature_columns),
		)
	if config.get('use_rank_momentum_features', False):
		processed, feature_columns, rank_momentum_columns = apply_rank_momentum_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加横截面动量变化特征 %s 个，输入特征=%s",
			len(rank_momentum_columns),
			len(feature_columns),
		)
	if config.get('use_rank_riskadj_features', False):
		processed, feature_columns, rank_riskadj_columns = apply_rank_riskadj_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加横截面风险调整动量特征 %s 个，输入特征=%s",
			len(rank_riskadj_columns),
			len(feature_columns),
		)
	if config.get('use_multi_period_features', False):
		processed, feature_columns, multi_period_columns = apply_multi_period_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加多周期基础特征 %s 个，输入特征=%s",
			len(multi_period_columns),
			len(feature_columns),
		)
	if config.get('use_clean_risk_features', False):
		processed, feature_columns, clean_risk_columns = apply_clean_risk_features(
			processed,
			feature_columns,
		)
		logger.info(
			"预测集: 已添加清洗启发的风险特征 %s 个，输入特征=%s",
			len(clean_risk_columns),
			len(feature_columns),
		)
	if not config.get('use_instrument_feature', True):
		logger.info("预测集: 已从输入特征中移除 instrument，股票代码仅用于分组和输出")

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def resolve_data_cutoff_date(raw_df, cutoff_date):
	available_dates = pd.DatetimeIndex(sorted(raw_df['日期'].unique()))
	cutoff = pd.Timestamp(cutoff_date).normalize()

	candidates = available_dates[available_dates <= cutoff]
	if len(candidates) == 0:
		raise ValueError(f'数据中没有不晚于 {cutoff.date()} 的交易日')

	as_of_date = pd.Timestamp(candidates[-1])
	if as_of_date != cutoff:
		logger.info("数据截止候选日 %s 无数据，改用最近已知交易日 %s", cutoff.date(), as_of_date.date())
	return as_of_date


def resolve_prediction_task(raw_df, args):
	submission_date = parse_date(args.submission_date, "--submission-date")
	as_of_cutoff = parse_date(args.as_of_date, "--as-of-date") if args.as_of_date else submission_date
	if as_of_cutoff > submission_date:
		raise ValueError(
			f"--as-of-date 不能晚于提交截止日: {as_of_cutoff.date()} > {submission_date.date()}"
		)

	target_start_candidate = (
		parse_date(args.target_start_date, "--target-start-date")
		if args.target_start_date
		else submission_date + pd.Timedelta(days=1)
	)
	if target_start_candidate <= submission_date:
		raise ValueError(
			f"预测窗口起始日必须晚于提交截止日: {target_start_candidate.date()} <= {submission_date.date()}"
		)

	known_dates = get_trading_dates(raw_df)
	holidays = parse_holiday_dates(args.market_holidays)
	target_window = future_trading_window(
		target_start_candidate,
		config.get('prediction_horizon', 5),
		known_dates,
		holidays,
	)
	if target_window[0] != target_start_candidate:
		logger.info("预测窗口候选起始日 %s 非交易日，改用 %s", target_start_candidate.date(), target_window[0].date())

	as_of_date = resolve_data_cutoff_date(raw_df, as_of_cutoff)
	if as_of_date >= target_window[0]:
		raise ValueError(
			f"数据截止日必须早于预测窗口首日，避免未来信息泄漏: {as_of_date.date()} >= {target_window[0].date()}"
		)

	if as_of_date < submission_date:
		logger.info(
			"当前 stock_data 最新可用日早于提交截止日: as_of=%s, submission=%s；正式提交前应更新数据",
			as_of_date.date(),
			submission_date.date(),
		)

	return submission_date, as_of_date, target_window


def main():
	global logger
	args = parse_args()
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = args.output
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
	setup_logging("bdc.predict", os.path.join(os.path.dirname(output_path) or '.', 'predict.log'))
	logger = logging.getLogger("bdc.predict")

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df, data_file = load_stock_data(
		config['data_path'],
		data_file=config.get('stock_data_file'),
		allow_train_fallback=True,
		logger=logger,
	)
	submission_date, latest_date, target_window = resolve_prediction_task(raw_df, args)
	raw_df = raw_df[raw_df['日期'] <= latest_date].copy()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
	logger.info(
		"预测任务: 提交截止日=%s | 数据截止日=%s | 目标窗口=%s ~ %s | 目标交易日=%s",
		submission_date.date(),
		latest_date.date(),
		target_window[0].date(),
		target_window[-1].date(),
		", ".join(day.strftime('%Y-%m-%d') for day in target_window),
	)
	logger.info("股票映射数量: %s", len(stockid2idx))

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')
	logger.info("预测设备: %s", device)

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	state_dict = torch.load(model_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]
	ranked_scores = scores[order]

	scores_df = pd.DataFrame({
		'rank': np.arange(1, len(ranked_stock_ids) + 1),
		'stock_id': ranked_stock_ids,
		'pred_score': ranked_scores,
	})
	selection_strategy = normalize_selection_strategy(args.selection_strategy)
	selected_df, scores_df = select_predictions(
		scores_df,
		history=raw_df,
		strategy=selection_strategy,
		top_k=args.top_k,
		total_exposure=args.total_exposure,
		pool_size=args.stage2_pool_size,
		volatility_window=args.stage2_vol_window,
	)

	selected_stock_ids = selected_df['stock_id'].tolist()
	output_df = selected_df[['stock_id', 'weight']].copy()
	output_df.to_csv(output_path, index=False)

	scores_output_path = args.scores_output or default_scores_output_path(output_path)
	os.makedirs(os.path.dirname(scores_output_path) or '.', exist_ok=True)
	scores_df.to_csv(scores_output_path, index=False)

	logger.info("参与排序股票数: %s", len(ranked_stock_ids))
	logger.info(
		"选股策略: %s | top_k=%s | total_exposure=%s | stage2_pool_size=%s | stage2_vol_window=%s",
		selection_strategy,
		args.top_k,
		args.total_exposure,
		args.stage2_pool_size,
		args.stage2_vol_window,
	)
	logger.info("Selected: %s", ", ".join(selected_stock_ids))
	logger.info("结果已写入: %s", output_path)
	logger.info("完整候选排名已写入: %s", scores_output_path)


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
