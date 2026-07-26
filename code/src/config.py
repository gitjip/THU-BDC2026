import os


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ''):
        return default
    return int(value)


# 配置参数
fast_dev_mode = _env_bool('BDC_FAST_DEV', False)
sequence_length = _env_int('BDC_SEQUENCE_LENGTH', 60)
feature_num = os.environ.get('BDC_FEATURE_NUM', '158+39')
output_dir_prefix = 'debug_' if fast_dev_mode else ''
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'label_horizon': 5,     # 标签为未来第5个交易日相对未来第1个交易日的收益
    'prediction_horizon': 5,
    'submission_deadline_date': os.environ.get('BDC_SUBMISSION_DATE', '2026-08-02'),
    'market_holidays': os.environ.get('BDC_MARKET_HOLIDAYS', ''),
    'val_days': _env_int('BDC_VAL_DAYS', 5 if fast_dev_mode else 20),
    'train_target_days': _env_int('BDC_TRAIN_TARGET_DAYS', 48 if fast_dev_mode else 0),
    'max_stocks_per_day': _env_int('BDC_MAX_STOCKS_PER_DAY', 120 if fast_dev_mode else 0),
    'fast_dev_mode': fast_dev_mode,
    'd_model': _env_int('BDC_D_MODEL', 256),          # Transformer输入维度
    'nhead': _env_int('BDC_NHEAD', 4),             # 注意力头数量
    'num_layers': _env_int('BDC_NUM_LAYERS', 3),        # Transformer层数
    'dim_feedforward': _env_int('BDC_DIM_FEEDFORWARD', 512), # 前馈网络维度
    'batch_size': _env_int('BDC_BATCH_SIZE', 4),        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': _env_int('BDC_NUM_EPOCHS', 1),       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    'seed': 42,
    'num_processes': _env_int('BDC_NUM_PROCESSES', 10),
    'torch_num_threads': _env_int('BDC_TORCH_NUM_THREADS', 0),
    'enable_tensorboard': _env_bool('BDC_TENSORBOARD', not fast_dev_mode),

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': os.environ.get('BDC_OUTPUT_DIR', f'./model/{output_dir_prefix}{sequence_length}_{feature_num}'),
    'data_path': './data',
    'stock_data_file': None,  # 默认自动寻找 data/stock_data.csv 或 data/stock_data
    'prediction_output_path': './output/result.csv',
}
