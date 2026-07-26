import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from data_utils import load_stock_data, split_by_last_trading_days


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="按数据中的真实交易日切分 train.csv 和 test.csv"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/stock_data.csv",
		help="原始数据文件或目录，默认 data/stock_data.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data",
		help="输出目录，默认 data",
	)
	parser.add_argument(
		"--test-days",
		type=int,
		default=5,
		help="取最后几个真实交易日作为测试集，默认 5",
	)
	return parser.parse_args()


def _format_dates_for_csv(df: pd.DataFrame) -> pd.DataFrame:
	out = df.sort_values(["股票代码", "日期"]).reset_index(drop=True).copy()
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def main() -> None:
	args = parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	df, input_path = load_stock_data(
		data_path=Path(args.input).parent,
		data_file=args.input,
		allow_train_fallback=False,
	)
	train_df, test_df, train_dates, test_dates = split_by_last_trading_days(df, args.test_days)

	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	_format_dates_for_csv(train_df).to_csv(train_path, index=False)
	_format_dates_for_csv(test_df).to_csv(test_path, index=False)

	print(f"原始数据: {input_path}")
	print(f"训练集: {train_path}，共 {len(train_df)} 行，股票数 {train_df['股票代码'].nunique()}")
	print(f"测试集: {test_path}，共 {len(test_df)} 行，股票数 {test_df['股票代码'].nunique()}")
	print(
		f"训练集交易日范围: {train_dates[0].date()} ~ {train_dates[-1].date()} | "
		f"测试集交易日范围: {test_dates[0].date()} ~ {test_dates[-1].date()}"
	)


if __name__ == "__main__":
	main()
