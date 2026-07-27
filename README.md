# 代码说明

本项目用于参加 2026 大数据挑战赛“基于历史数据预测未来股价收益”赛题。代码从赛事 baseline 发展而来，主要流程是：读取历史股票行情，训练排序模型，对提交日之后未来连续 5 个交易日的股票收益进行预测，并输出不超过 5 只股票及其权重到 `output/result.csv`。

## 环境配置

推荐使用 Python 3.10 到 3.12。项目依赖由 `pyproject.toml` 和 `uv.lock` 固定，主要依赖包括：

- `torch`：深度学习模型训练与推理；
- `pandas`、`numpy`、`scikit-learn`、`joblib`：数据处理、标准化和模型产物保存；
- `TA-Lib`：技术指标特征工程；
- `tqdm`、`tensorboard`、`tensorboardX`：进度展示和训练日志；
- `baostock`、`akshare`：本地数据获取脚本可能使用，正式复现运行不依赖联网调用。

安装依赖：

```bash
uv sync
```

训练和预测脚本会优先使用 `.venv/bin/python`。如果没有 `.venv`，会依次尝试 `uv run python`、`python`、`python3`。

当前 `init.sh` 不执行额外初始化；依赖安装和系统库准备应在镜像构建或运行环境准备阶段完成。

赛事文档要求复现训练和预测时不得联网；本项目正式训练和预测默认只读取本地 `data/` 下的数据文件。

## 数据

默认数据来源为赛事方提供或本地准备的历史股票行情数据。训练和预测会按以下顺序自动寻找数据：

1. `data/stock_data.csv`
2. `data/stock_data`
3. `data/train.csv`，仅作为本地调试兜底

正式验证阶段赛事方会在 `data/` 下挂载 `stock_data`。代码不要求固定文件后缀；如果 `data/stock_data` 是文件或目录，都会按项目数据加载逻辑处理。

输入数据至少需要包含以下列：

- `股票代码`
- `日期`
- `开盘`
- `收盘`
- `最高`
- `最低`
- `成交量`
- `成交额`
- `振幅`
- `涨跌额`
- `换手率`
- `涨跌幅`

代码会把股票代码标准化为 6 位字符串，并按数据中真实存在的交易日构造训练标签和预测窗口。周末、节假日没有股票数据是正常情况。

## 预训练模型

本项目当前不使用外部预训练模型，也不依赖外部 embedding、词典或额外不可公开数据。

训练产物由本项目 `train.sh` 从输入数据重新训练得到，主要保存到 `model/` 目录。默认完整训练产物目录为：

```text
model/60_158+39/
```

其中包括：

- `best_model.pth`：内部验证集 `final_score` 最优的模型参数；
- `scaler.pkl`：训练集特征标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：最佳 epoch、停止原因和训练摘要；
- `training_history.csv`：逐 epoch 训练记录；
- `train.log`：训练日志。

## 算法

### 整体思路介绍

本项目把股票选择建模为横截面排序任务。对每个目标交易日，模型同时看到多只股票各自过去一段时间的量价和技术指标序列，输出每只股票的排序分数。预测时按分数降序选择前 5 只股票，当前默认等权重 `0.2`。

标签使用未来第 1 个交易日开盘价到未来第 5 个交易日开盘价的收益率：

```text
return = (open_t5 - open_t1) / open_t1
```

这与赛题要求的 T+1 开盘买入、T+5 开盘卖出收益口径保持一致。

### 方法的创新点

相对官方 baseline，本项目主要做了以下工程化和训练流程改进：

- 严格按时间顺序切分训练、验证和预测，避免未来信息泄漏；
- 预测窗口自动寻找提交日之后合理的连续 5 个交易日；
- 训练过程加入学习率平台调度、早停、训练耗时和梯度范数日志；
- 支持 walk-forward 历史窗口验证，平时调参不依赖正式未来标签；
- 生成完整候选排名和预测诊断文件，便于检查固定选股池和排序质量；
- 支持通过环境变量控制特征组、模型规模、训练窗口、股票抽样和调参 profile。

### 网络结构

核心模型为 `StockTransformer`，定义在 `code/src/model.py`。主要模块：

- `PositionalEncoding`：为股票历史序列加入时间位置信息；
- `TransformerEncoder`：编码每只股票的历史序列；
- `FeatureAttention`：对时间维度表示进行加权聚合；
- `CrossStockAttention`：建模同一交易日内不同股票之间的横截面关系；
- `ranking_layers` 和 `score_head`：输出每只股票的排序分数。

模型输入形状为：

```text
[batch, num_stocks, sequence_length, feature_dim]
```

模型输出形状为：

```text
[batch, num_stocks]
```

### 特征工程

代码支持两类主要特征配置：

- `39`：基础量价和技术指标特征；
- `158+39`：Alpha 类特征加基础技术指标，默认完整训练使用该配置。

特征工程位于 `code/src/utils.py`，包括移动均线、指数均线、成交量变化、RSI、MACD、KDJ、布林带、ATR、波动率、收益率、价差等。

部分实验还支持横截面 rank 特征、市场相对特征和移除股票编号特征，这些由环境变量控制，默认完整训练不强制启用。

### 损失函数

训练使用 `WeightedRankingLoss`，定义在 `code/src/train.py`。它组合了：

- listwise 排序损失：让模型学习同一交易日内股票收益的整体排序分布；
- pairwise 排序损失：强化股票两两之间的相对顺序；
- top-k 加权：对真实收益靠前的股票给予更高权重。

训练过程中使用内部验证集 `final_score` 保存最佳模型。`final_score` 只用于选择训练过程中的最优权重，不代表正式预测窗口的真实得分。

### 数据扩增

当前没有使用图像或文本类数据扩增。训练样本扩展主要来自时间滚动：每个可构造未来收益标签的交易日都会形成一个横截面排序样本。

调试模式会限制训练目标日数量和每日股票数量，以缩短本地运行时间；完整训练默认不限制训练目标日和每日股票数量。

### 模型集成

当前正式预测默认使用单模型输出，不默认启用模型集成。仓库中保留了候选池对比和预测诊断脚本，用于后续评估是否值得做模型集成或候选重排。

### 算法的其他细节

- 固定随机种子：`seed=42`；
- 优化器：默认 `AdamW`；
- 学习率调度：默认 `ReduceLROnPlateau` 风格的平台调度；
- 早停：验证分数连续若干 epoch 未明显提升时停止训练；
- 梯度裁剪：默认启用；
- 设备选择：按 `CUDA -> MPS -> CPU` 自动选择；
- 股票代码仍用于分组、构造序列和输出结果，不代表一定作为模型输入特征。

## 训练流程

正式训练入口：

```bash
sh train.sh
```

训练流程：

1. 读取 `data/stock_data.csv` 或赛事方挂载的 `data/stock_data`；
2. 标准化股票代码和日期；
3. 按股票分组并行计算技术指标特征；
4. 构造未来第 1 到第 5 个交易日开盘收益标签；
5. 按时间顺序切分训练集和内部验证集；
6. 把每个目标交易日组织成一个横截面排序样本；
7. 使用 `StockTransformer` 训练排序模型；
8. 根据验证集 `final_score` 保存 `best_model.pth`；
9. 保存 `scaler.pkl`、配置快照、训练历史和日志。

快速调试入口：

```bash
sh train.sh debug
```

debug 模式会使用较小模型、较短序列、较少训练目标日和股票抽样，只用于检查流程和代码错误，不用于最终提交成绩。

常用配置可通过环境变量覆盖，例如：

```bash
BDC_NUM_EPOCHS=8 sh train.sh
BDC_USE_INSTRUMENT_FEATURE=0 sh train.sh
BDC_STOCK_DATA_FILE=data/stock_data.csv sh train.sh
```

更多训练和调参说明见：

- [docs/workflows/train_predict.md](docs/workflows/train_predict.md)
- [docs/workflows/walk_forward.md](docs/workflows/walk_forward.md)
- [docs/experiments/training_strategy.md](docs/experiments/training_strategy.md)

## 推理流程

正式预测入口：

```bash
sh test.sh
```

推理流程：

1. 读取训练阶段保存的 `best_model.pth`、`scaler.pkl` 和 `config.json`；
2. 读取不晚于提交截止日的历史行情数据；
3. 根据提交截止日寻找未来连续 5 个合理交易日；
4. 对全部可预测股票执行与训练一致的特征工程；
5. 使用训练时保存的标准化器处理特征；
6. 用模型对全部候选股票打分；
7. 按分数降序选择前 5 只股票；
8. 生成比赛提交文件 `output/result.csv`。

默认提交截止日为 `2026-08-02`。如果预测窗口候选起始日是周末、节假日或无交易数据日期，代码会向后跳到合理交易日。

输出文件：

```text
output/result.csv
output/result_scores.csv
```

`output/result.csv` 是提交文件，格式为：

```csv
stock_id,weight
600000,0.2
```

最多 5 只股票，权重和不超过 1。当前默认策略是模型分数前 5 名等权重 `0.2`。`output/result_scores.csv` 是完整候选排名诊断文件，不是提交文件。

可选参数示例：

```bash
sh test.sh --submission-date 2026-08-02
sh test.sh --target-start-date 2026-08-03
sh test.sh --output output/result.csv
```

## 其他注意事项

- 最终提交结果必须由训练和预测流程产生，不能硬编码股票代码或固定结果。
- 正式预测窗口没有真实标签，本地无法提前知道正式提交得分。
- 平时模型好坏通过历史 walk-forward 窗口验证判断，入口为 `sh tune.sh ...`。
- `data/`、`output/`、`temp/` 在最终验证中可能被挂载覆盖，不要把不可替代的自有文件只放在这些目录。
- 复现阶段不得联网；如使用自定义外部公开数据，需要按赛事要求提前报备来源和 md5。
- 赛事代码规范要求训练不超过 8 小时、预测不超过 5 分钟；提交前应在接近赛事机器的环境复核耗时。
- 本项目当前默认不使用外部预训练模型，也不依赖运行时联网获取数据。
- 原项目概览已迁移到 [docs/workflows/project_overview.md](docs/workflows/project_overview.md)。
