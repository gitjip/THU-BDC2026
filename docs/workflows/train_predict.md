# 训练与预测流程

本文只说明当前阶段需要的流程：训练模型、生成 `output/result.csv`。暂不包含 Docker 打包。

## 1. 准备环境

先安装依赖：

```bash
uv sync
```

之后可以不手动激活虚拟环境，根目录的 `train.sh` 和 `test.sh` 会优先使用 `.venv/bin/python`。

## 2. 数据约定

训练和预测默认自动寻找数据：

1. `data/stock_data.csv`
2. `data/stock_data`
3. `data/train.csv`（仅作为本地调试兜底）

官方最终验证会挂载 `data/`，并提供 `stock_data`，不会提供 `train.csv` 和 `test.csv`。因此正式提交前按默认配置运行即可，不要依赖自己放在 `data/` 下的其它文件。

代码会按数据中真实存在的交易日切分和构造标签。周末、节假日没有股票数据是正常情况；walk-forward 平时验证会跳过包含周末或节假日的 5 天窗口，以贴近最终预测窗口。

读取 `stock_data` 时，代码会对“开高低收完全相同且成交量/成交额/换手率/涨跌幅缺失”的平盘样本做保守补零，避免把停牌或无成交行直接丢掉；其余仍然缺失必要字段的异常行会在数据入口直接删除并记录日志。

## 3. 训练

完整训练运行：

```bash
sh train.sh
```

训练时会：

1. 读取 `stock_data`；
2. 按 `rank-replace` 配置训练单模型；
3. 统一股票代码为 6 位字符串；
4. 用最近 `val_days` 个可打标签交易日做验证集；
5. 用未来第 1 个到第 5 个交易日开盘价计算标签；
6. 按验证集 `final_score` 保存最佳模型；
7. 根据验证表现自动降低学习率，并在连续无明显提升时早停；
8. 保存模型、标准化器和日志。

主要输出：

- `model/rank_replace/best_model.pth`
- `model/rank_replace/scaler.pkl`
- `model/rank_replace/config.json`

快速调试运行：

```bash
sh train.sh debug
```

debug 模式默认会：

- 使用 39 特征、30 日序列和较小的 Transformer；
- 只使用最近 24 个训练目标交易日；
- 验证集只取最近 5 个目标交易日；
- 每个交易日固定抽样 60 只股票参与训练；
- 最多训练 4 个 epoch，若验证分数连续 2 个 epoch 无明显提升会提前停止；
- 保存到 `model/rank_replace_debug/`，不会覆盖正式模型；
- 关闭 TensorBoard，减少调试时生成的文件。

这样做是为了调试流程和代码错误，不是为了获得最佳成绩。完整训练仍使用 `sh train.sh`。

上次训练变慢的主要原因：旧逻辑错误要求未来 5 天自然日连续，因此大量跨周末的正常样本被过滤，训练样本很少；修正为真实交易日后，训练样本从约每周 1 个变成每个交易日 1 个，CPU 上训练会明显变慢。debug 模式通过主动限量采样提速，而不是恢复错误的数据过滤。

## 4. 预测

训练完成后运行：

```bash
sh test.sh
```

正式预测只负责生成提交用的 `result.csv`，不负责验证模型好坏。因为提交日之后的未来 5 个交易日还没有真实收益，不能打分，只能等线上评测或之后数据更新后再回看。

当前默认提交截止日是 `2026-08-02`，这是周日。代码会从提交截止日之后开始找第一个合理交易日，所以默认预测窗口是：

```text
2026-08-03, 2026-08-04, 2026-08-05, 2026-08-06, 2026-08-07
```

预测会读取不晚于提交截止日的最新可用历史数据，使用 `rank-replace` 模型生成候选排名，输出：

```text
output/result.csv
output/result_scores.csv
```

`result.csv` 是比赛提交文件。`result_scores.csv` 是本地诊断文件，记录完整候选排名、是否入选和模型分数；它不是提交目标。

如果要指定提交截止日或预测窗口候选起始日：

```bash
sh test.sh --submission-date 2026-08-02
sh test.sh --target-start-date 2026-08-08
```

如果预测窗口候选起始日是周末、节假日或历史数据中没有交易记录，代码会一直向后跳到合理交易日。未来休市日可用 `BDC_MARKET_HOLIDAYS` 或 `--market-holidays` 指定。

`--as-of-date` 只用于本地调试，表示最多使用哪一天之前的历史数据；它不能晚于提交截止日。

提交文件格式为：

```csv
stock_id,weight
600000,0.2
```

最多 5 只股票，权重和不能超过 1。当前默认逻辑是 `rank-replace` 模型 top5 等权重 `0.2`。

如果要运行原始单模型配置或可选集成，可显式启用：

```bash
BDC_SUBMISSION_MODE=single sh train.sh
BDC_SUBMISSION_MODE=single sh test.sh
sh train.sh ensemble
sh test.sh ensemble
```

其中 `single` 是原始单模型配置，`ensemble` 是两模型集成候选。

## 5. 本地调试与自测

如果只是确认训练和预测能跑通，优先执行：

```bash
sh train.sh debug
sh test.sh debug --output /tmp/bdc_debug_result.csv
```

本地验证和正式预测要分开理解：

- 验证：只在历史数据中做模拟，例如用较早数据训练，用后面 5 个已发生交易日计算分数。它用于平时判断模型方向是否变好。
- 预测：面对提交日之后还没发生的未来 5 个交易日，只能输出 `result.csv`，没有真实标签可算分。

需要更接近提交逻辑的多窗口验证时，使用独立调参入口，见 [walk_forward.md](walk_forward.md)。

如果想用项目已有 `stock_data.csv` 做一次本地后验评分，可以先按最后 5 个真实交易日生成本地 `train.csv` 和 `test.csv`：

```bash
.venv/bin/python data/split_train_test.py --test-days 5
```

然后用 `train.csv` 作为临时数据源训练和预测，避免预测看到 `test.csv` 的未来数据：

```bash
BDC_STOCK_DATA_FILE=data/train.csv sh train.sh
BDC_STOCK_DATA_FILE=data/train.csv sh test.sh
.venv/bin/python test/score_self.py
```

本地评分结果会写入 `temp/latest_score.csv`。它只是最近一次本地评分缓存，已加入 Git 忽略，不是比赛提交文件；比赛提交文件仍是 `output/result.csv`。

## 6. 常用配置

配置文件：`code/src/config.py`

- `sequence_length`：每只股票输入过去多少个交易日；默认提交模型为 45，原始单模型配置默认 60。
- `label_horizon`：标签跨度，默认未来第 5 个交易日。
- `val_days`：训练过程内部验证集目标交易日数量，默认 5。
- `train_target_days`：训练目标交易日数量限制；`config.py` 默认 0 表示不限制，当前 `train.sh` 提交默认 `rank-replace` 使用 60，debug 使用 24。
- `max_stocks_per_day`：每个交易日抽样股票数；`config.py` 默认 0 表示不抽样，当前 `train.sh` 提交默认 `rank-replace` 使用 120，debug 使用 60。
- `num_epochs`：最大训练轮数；默认提交模型为 30，早停可能提前结束。
- `learning_rate`：默认提交模型为 `3e-5`，debug `3e-5`。
- `lr_scheduler`：默认 `plateau`，验证 `final_score` 停滞时降低学习率。
- `early_stopping_patience`：当前 `train.sh` 提交默认 `rank-replace` 为 5，debug 为 2；设为 0 可关闭早停。
- `loss_temperature`：默认 1.0；控制模型预测分数在 listwise loss 中的 softmax 温度。
- `loss_target_temperature`：默认等于 `loss_temperature`；可用 `BDC_LOSS_TARGET_TEMPERATURE=0.05` 让真实收益目标分布更尖锐，强化 top 股票排序信号。
- `use_instrument_feature`：默认开启；可用 `BDC_USE_INSTRUMENT_FEATURE=0` 从模型输入中移除股票编号特征。
- `use_market_relative_features`：默认关闭；可用 `BDC_USE_MARKET_RELATIVE_FEATURES=1` 追加市场相对特征。
- `use_market_breadth_features`：默认关闭；可用 `BDC_USE_MARKET_BREADTH_FEATURES=1` 追加市场宽度特征 `mkt_breadth_5`。
- `use_rank_momentum_features`：默认关闭；可用 `BDC_USE_RANK_MOMENTUM_FEATURES=1` 追加短中期动量 rank 差信号 `ret5_rank_minus_ret20_rank`。
- `use_rank_riskadj_features`：默认关闭；可用 `BDC_USE_RANK_RISKADJ_FEATURES=1` 追加风险调整短期动量 rank 差信号 `ret5_rank_minus_vol20_rank`。
- `use_ret5_rank_features`：默认关闭；可用 `BDC_USE_RET5_RANK_FEATURES=1` 追加单列 `return_5_cs_rank`。
- `use_trend_quality_features`：默认关闭；可用 `BDC_USE_TREND_QUALITY_FEATURES=1` 追加趋势质量特征。
- `use_clean_risk_features`：默认关闭；可用 `BDC_USE_CLEAN_RISK_FEATURES=1` 追加无成交、流动性和回撤风险特征。
- `use_multi_period_features`：默认关闭；可用 `BDC_USE_MULTI_PERIOD_FEATURES=1` 追加 3/5/10/20/40 日多周期基础特征。
- `BDC_SUBMISSION_MODE`：默认 `rank-replace`；设为 `single` 可回退原始单模型入口，设为 `ensemble` 可运行两模型集成候选。
- `BDC_ENSEMBLE_SELECTION_STRATEGY`：仅集成模式使用，默认 `ensemble_low_vol_top5`，即两个源模型 top5 并集后按低波动重排。
- `selection_strategy`：仅单模型或源模型预测使用，默认 `model_top5`；可用 `BDC_SELECTION_STRATEGY=low_vol_then_rank_top5` 启用单模型低波动二阶段后处理。
- `top_k`：默认 5；可用 `BDC_TOP_K=3` 只输出前 3 只股票，范围 1 到 5。
- `total_exposure`：默认 1.0；可用 `BDC_TOTAL_EXPOSURE=0.8` 控制总仓位，范围 0 到 1，未使用权重相当于现金。
- `stage2_pool_size`：默认 10；低波动后处理从模型前多少名中选回 5 只。
- `stage2_vol_window`：默认 20；低波动后处理计算历史波动率的交易日窗口。
- `use_cross_sectional_rank_features`：默认关闭；可用 `BDC_USE_CROSS_SECTIONAL_RANKS=1` 增加当日横截面百分位排名特征。
- `cross_sectional_rank_mode`：默认 `off`；`append` 表示追加 rank 特征，`replace` 表示用 rank 替换部分原始绝对量价特征。
- `cross_sectional_rank_replace_set`：默认 `default`；`lite` 只替换开高低收、成交量、成交额和涨跌额。
- `stock_data_file`：默认 `None`，自动寻找数据；也可以用环境变量 `BDC_STOCK_DATA_FILE` 临时覆盖。

注意：`BDC_EARLY_STOPPING_PATIENCE=0` 只表示不提前停止，训练仍会根据内部验证 `final_score` 保存 `best_model.pth`。`test.sh` 和 walk-forward 预测默认读取 `best_model.pth`，不是最后一个 epoch 的权重。如果要比较最后一轮模型，需要单独增加保存和加载 final epoch 模型的实验开关。

常用环境变量：

```bash
BDC_FAST_DEV=1 sh train.sh
BDC_TRAIN_TARGET_DAYS=80 BDC_MAX_STOCKS_PER_DAY=180 sh train.sh debug
BDC_NUM_EPOCHS=6 BDC_EARLY_STOPPING_PATIENCE=2 sh train.sh debug
BDC_USE_INSTRUMENT_FEATURE=0 BDC_USE_MARKET_RELATIVE_FEATURES=1 sh train.sh debug
BDC_USE_INSTRUMENT_FEATURE=0 BDC_USE_CROSS_SECTIONAL_RANKS=1 sh train.sh debug
BDC_USE_INSTRUMENT_FEATURE=0 BDC_CROSS_SECTIONAL_RANK_MODE=replace sh train.sh debug
BDC_USE_INSTRUMENT_FEATURE=0 BDC_CROSS_SECTIONAL_RANK_MODE=replace BDC_CROSS_SECTIONAL_RANK_REPLACE_SET=lite sh train.sh debug
BDC_LOSS_TARGET_TEMPERATURE=0.05 sh tune.sh v1.5.0 noid-rank-replace --windows 3 --skip-final
sh tune.sh v1.6.0 noid-rank-cleanrisk --windows 3 --skip-final
sh tune.sh v1.6.1 noid-rank-multiperiod --windows 3 --skip-final
sh tune.sh v1.6.2 noid-rank-breadth --windows 3 --skip-final
sh tune.sh v1.7.0 noid-rank-momdelta --windows 6 --skip-final
sh tune.sh v1.8.0 noid-rank-riskadj --windows 6 --skip-final
sh tune.sh v1.9.0 noid-rank-ret5rank --windows 6 --skip-final
BDC_NUM_EPOCHS=30 BDC_LR_SCHEDULER=off BDC_EARLY_STOPPING_PATIENCE=0 sh tune.sh v1.2.10 noid --skip-final
BDC_SUBMISSION_MODE=single sh train.sh
BDC_SUBMISSION_MODE=single sh test.sh
BDC_SELECTION_STRATEGY=low_vol_then_rank_top5 BDC_STAGE2_POOL_SIZE=10 sh test.sh
BDC_TOP_K=3 BDC_TOTAL_EXPOSURE=1.0 sh test.sh
BDC_TOP_K=5 BDC_TOTAL_EXPOSURE=0.8 sh test.sh
BDC_SUBMISSION_DATE=2026-08-02 sh test.sh
BDC_TARGET_START_DATE=2026-08-08 sh test.sh
BDC_MARKET_HOLIDAYS=2026-08-03 sh test.sh
```

离线评估不同 topK 和现金仓位时，不需要重训：

```bash
.venv/bin/python code/src/evaluate_submission_controls.py experiments/v1.4.5 --output-dir experiments/analysis/topk_exposure_v1.4.5_rank_replace_24
```

固定 `top_k` 时，`total_exposure` 只会线性缩放收益和亏损；它适合控制风险幅度，但不会让排序本身变好。

离线分析“模型高分但未来表现差”的股票时，也不需要重训：

```bash
.venv/bin/python code/src/analyze_high_score_good_bad.py experiments/v1.4.5 --top-k 20
```

输出目录默认在 `experiments/analysis/high_score_good_bad_<version>/`。重点看 `feature_differences.csv` 和 `simple_gate_trials.csv`：前者解释高分好股/坏股的历史特征差异，后者粗测这些差异能否变成只依赖历史数据的简单过滤或重排规则。

## 7. 注意事项

- 正式提交结果必须由模型训练和预测产生，不能硬编码股票代码。
- 验证分数只能来自历史窗口模拟，不能拿正式预测窗口提前校验模型好坏。
- `data/`、`output/`、`temp/` 在最终验证中可能被挂载覆盖，不要把不可替代的自有数据放在这些目录。
- 如果出现 `TA-Lib` 安装问题，先确认系统层面的 `ta-lib` 库已安装，再安装 Python 依赖。
