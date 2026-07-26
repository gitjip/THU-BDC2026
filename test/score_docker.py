"""
本脚本用于将预测出的五支股票与实际的股票数据进行对比，计算加权收益，形成最终得分。
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

parser = argparse.ArgumentParser(description='Calculate stock prediction score.')
parser.add_argument('team_name', type=str, help='The name of the team (used for file naming).')
args = parser.parse_args()
OUTPUT_PATH = Path("./test/results_output") / f"{args.team_name}.csv"
TEST_DATA_PATH = Path("./data/test.csv")
TEMP_SCORE_PATH = Path("./temp/latest_score.csv")


def write_failed_score() -> None:
    TEMP_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        {
            "Team Name": [args.team_name],
            "Final Score": [-999],
        }
    )
    result.to_csv(TEMP_SCORE_PATH, index=False)


def is_valid_prediction(prediction_data):
    """
    验证选手输出的结果是否合法：需要包含最多五支股票，并且权重之和0到1之间.
    """
    id_col = 'stock_id' if 'stock_id' in prediction_data.columns else '股票代码' if '股票代码' in prediction_data.columns else None
    weight_col = 'weight' if 'weight' in prediction_data.columns else '权重' if '权重' in prediction_data.columns else None
    if id_col is None or weight_col is None:
        raise ValueError('预测结果缺少必要字段，必须包含 stock_id/股票代码 和 weight/权重。')

    if len(prediction_data) > 5:
        raise ValueError('预测结果不合法：最多只能包含五支股票。')

    weight_sum = prediction_data[weight_col].sum()
    if not (0 <= float(weight_sum) <= 1.0):
        raise ValueError(f"预测结果不合法：权重之和必须为0到1之间. 当前权重之和为 {weight_sum}.")


def calculate_portfolio_return_score(output_data, test_data):
    # 选择输出指定的5个股票
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    # 只选最后五个记录
    test_data = test_data.sort_values(['股票代码', '日期']).groupby('股票代码', group_keys=False).tail(5)
    # 分别计算收益率
    result = test_data.groupby('股票代码', as_index=False).agg(
        start_open=('开盘', 'first'),
        end_open=('开盘', 'last'),
    )
    result['收益率'] = (result['end_open'] - result['start_open']) / result['start_open']
    result = result.merge(output_data, on='股票代码')
    # 计算加权收益率
    final_score = (result['收益率'] * result['权重']).sum()
    return final_score


try:
    test_data = pd.read_csv(TEST_DATA_PATH, dtype={'股票代码': str})
    raw_output_data = pd.read_csv(OUTPUT_PATH, dtype={'stock_id': str, '股票代码': str})
    is_valid_prediction(raw_output_data)
except Exception as e:
    print(f"读取本地测试数据或验证预测结果失败: {e}")
    write_failed_score()
    sys.exit(0)

test_data = test_data[['股票代码', '日期', '开盘', '收盘']]
test_data['股票代码'] = test_data['股票代码'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
# 读取输出数据
output_data = raw_output_data.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
output_data['股票代码'] = output_data['股票代码'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
output_data['权重'] = pd.to_numeric(output_data['权重'], errors='coerce')

required_columns = {'股票代码', '权重'}
if not required_columns.issubset(output_data.columns):
    print('读取本地测试数据或验证预测结果失败: 输出结果缺少股票代码或权重字段。')
    write_failed_score()
    sys.exit(0)

portfolio_return_score = calculate_portfolio_return_score(output_data, test_data)


TEMP_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
result = pd.DataFrame(
    {
        "Team Name": [args.team_name],
        "Final Score": [portfolio_return_score],
    }
)
result.to_csv(TEMP_SCORE_PATH, index=False)
print(f"预测股票的加权收益率得分: {portfolio_return_score}")
print(f"本地评分结果已写入: {TEMP_SCORE_PATH}")
