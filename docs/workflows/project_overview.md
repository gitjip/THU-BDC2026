# THU-BigDataCompetition-2026-baseline 项目说明

本文件从根目录 `README.md` 迁移而来，用于保留项目结构和开发说明。根目录 `README.md` 已改为赛事代码规范要求的审查说明格式。

本项目是一个面向沪深300成分股的排序学习选股方案：

- 输入：每只股票过去一段时间的量价与技术特征序列；
- 模型：`StockTransformer`，同时建模单股票时序模式与股票间交互；
- 输出：对同一天全部候选股票打分并排序，最终输出前5只股票，当前默认等权重 `0.2`。

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：

1. 读取历史行情数据；
2. 做特征工程，支持 39 特征或 `158+39` 特征；
3. 构建标签：未来收益率，代码中为 `open_t1` 到 `open_t5` 的相对收益；
4. 按真实交易日组织排序样本：每个样本是一日内多只股票的序列与目标；
5. 训练排序模型，监控 `final_score` 并保存最优权重；
6. 使用训练好的 `best_model.pth` 和 `scaler.pkl` 对提交截止日之后的未来5个交易日生成 top5 选股结果。

## 2. 代码结构说明

### [config.py](../../code/src/config.py)

统一管理训练与推理参数，包括：

- 序列长度 `sequence_length`；
- 模型超参数：`d_model`、`nhead`、`num_layers` 等；
- 训练超参数：`batch_size`、`num_epochs`、`learning_rate`、学习率调度、早停；
- 排序损失权重参数：`pairwise_weight`、`top5_weight`、`base_weight`；
- 数据路径和输出路径。

### [model.py](../../code/src/model.py)

定义核心模型 `StockTransformer`，主要由以下模块组成：

- `PositionalEncoding`：时序位置编码；
- `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- `CrossStockAttention`：在同一交易日内建模股票间关系；
- `ranking_layers` 和 `score_head`：输出每只股票的排序分数。

输入形状：`[batch, num_stocks, seq_len, feature_dim]`  
输出形状：`[batch, num_stocks]`。

### [utils.py](../../code/src/utils.py)

包含特征工程与数据集构建逻辑：

- `engineer_features_39()`：39 个技术指标特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本。

说明：特征工程使用 `TA-Lib`，若未正确安装会报错。

### [train.py](../../code/src/train.py)

训练主脚本，关键内容：

- `_preprocess_common()`：按股票分组并行特征工程、股票 ID 映射、标签构建；
- `split_train_val_by_recent_trading_days()`：按最近真实交易日切分训练/验证集，并保留序列上下文；
- `RankingDataset` 和 `collate_fn`：处理每日股票数量不一致问题；
- `WeightedRankingLoss`：组合 listwise 和 pairwise 排序损失；
- `calculate_ranking_metrics()`：计算 `pred_return_sum`、`max_return_sum`、`ratio_pred`、`final_score` 等。

训练产物：

- `best_model.pth`：最佳模型参数；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：最佳分数记录；
- `training_history.csv`：逐 epoch 训练记录；
- `train.log`：训练日志。

### [ensemble_predict.py](../../code/src/ensemble_predict.py) 和 [predict.py](../../code/src/predict.py)

正式推理入口 `test.sh` 默认调用 `ensemble_predict.py`，它会再调用 `predict.py` 分别运行两个源模型。流程：

1. 加载历史数据，确定提交截止日和未来5个目标交易日；
2. 使用 `model/ensemble/noid` 和 `model/ensemble/noid_rank_replace` 分别对全部可预测股票打分；
3. 取两个源模型 top5 的并集；
4. 基于提交截止日前已知历史数据计算候选股票最近 20 个交易日收益波动率；
5. 在并集中按低波动优先选回5只，输出到 `output/result.csv`。

默认提交截止日为 `2026-08-02`。如果候选起始日不是交易日，代码会向后跳到合理交易日。`--as-of-date` 仅用于本地调试的数据截断，不能晚于提交截止日。

### [get_stock_data.py](../../get_stock_data.py)

数据抓取脚本：

- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式。

正式复现阶段默认使用赛事方挂载的 `data/stock_data`，不要依赖运行时联网抓取数据。

## 3. 运行入口

完整训练：

```bash
sh train.sh
```

正式预测：

```bash
sh test.sh
```

快速调试：

```bash
sh train.sh debug
sh test.sh debug --output /tmp/bdc_debug_result.csv
```

更详细的运行说明见 [train_predict.md](train_predict.md)。调参用的多窗口 walk-forward 流程见 [walk_forward.md](walk_forward.md)，训练速度与硬件取舍见 [training_strategy.md](../experiments/training_strategy.md)。

## 4. 常见问题

### TA-Lib 安装失败

本项目特征工程依赖 `TA-Lib`。如果 Python 包安装失败，先确认系统层面的 `ta-lib` 库已安装。

### 多进程相关问题

`train.py` 与 `predict.py` 均在入口使用 `spawn` 模式。请通过脚本入口运行，不要在交互式环境里直接调用多进程主逻辑。

### GPU/CPU 自动选择

代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无 GPU 时可直接 CPU 运行。
