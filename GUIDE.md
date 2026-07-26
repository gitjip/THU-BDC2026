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

代码会按数据中真实存在的交易日切分和构造标签，不要求自然日期连续。周末、节假日没有股票数据是正常情况。

## 3. 训练

完整训练运行：

```bash
sh train.sh
```

训练时会：

1. 读取 `stock_data`；
2. 统一股票代码为 6 位字符串；
3. 用最近 `val_days` 个可打标签交易日做验证集；
4. 用未来第 1 个到第 5 个交易日开盘价计算标签；
5. 保存模型、标准化器和日志。

主要输出：

- `model/60_158+39/best_model.pth`
- `model/60_158+39/scaler.pkl`
- `model/60_158+39/config.json`
- `model/60_158+39/final_score.txt`
- `model/60_158+39/train.log`

快速调试运行：

```bash
sh train.sh debug
```

debug 模式默认会：

- 只使用最近 48 个训练目标交易日；
- 验证集只取最近 5 个目标交易日；
- 每个交易日固定抽样 120 只股票参与训练；
- 保存到 `model/debug_60_158+39/`，不会覆盖正式模型；
- 关闭 TensorBoard，减少调试时生成的文件。

这样做是为了调试流程和代码错误，不是为了获得最佳成绩。完整训练仍使用 `sh train.sh`。

上次训练变慢的主要原因：旧逻辑错误要求未来 5 天自然日连续，因此大量跨周末的正常样本被过滤，训练样本很少；修正为真实交易日后，训练样本从约每周 1 个变成每个交易日 1 个，CPU 上训练会明显变慢。debug 模式通过主动限量采样提速，而不是恢复错误的数据过滤。

## 4. 预测

训练完成后运行：

```bash
sh test.sh
```

预测会读取同一个数据源，使用最新一个交易日作为预测基准日，输出：

```text
output/result.csv
```

如果要指定预测基准日：

```bash
sh test.sh --as-of-date 2026-07-15
```

如果指定日期是周末或节假日，代码会自动使用不晚于该日期的最近交易日。

提交文件格式为：

```csv
stock_id,weight
600000,0.2
```

最多 5 只股票，权重和不能超过 1。当前逻辑是取模型分数最高的 5 只股票，等权重 `0.2`。

## 5. 本地调试与自测

如果只是确认训练和预测能跑通，优先执行：

```bash
sh train.sh debug
sh test.sh debug --output /tmp/bdc_debug_result.csv
```

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

本地评分只用于调试参考，不等于线上最终成绩。

## 6. 常用配置

配置文件：`code/src/config.py`

- `sequence_length`：每只股票输入过去多少个交易日，默认 60。
- `label_horizon`：标签跨度，默认未来第 5 个交易日。
- `val_days`：验证集目标交易日数量，默认 20。
- `train_target_days`：训练目标交易日数量限制，默认 0 表示不限制；debug 默认 48。
- `max_stocks_per_day`：每个交易日抽样股票数，默认 0 表示不抽样；debug 默认 120。
- `num_epochs`：训练轮数，当前为 1。
- `stock_data_file`：默认 `None`，自动寻找数据；也可以用环境变量 `BDC_STOCK_DATA_FILE` 临时覆盖。

常用环境变量：

```bash
BDC_FAST_DEV=1 sh train.sh
BDC_TRAIN_TARGET_DAYS=80 BDC_MAX_STOCKS_PER_DAY=180 sh train.sh debug
BDC_PREDICT_DATE=2026-07-15 sh test.sh
```

## 7. 注意事项

- 正式提交结果必须由模型训练和预测产生，不能硬编码股票代码。
- `data/`、`output/`、`temp/` 在最终验证中可能被挂载覆盖，不要把不可替代的自有数据放在这些目录。
- 如果出现 `TA-Lib` 安装问题，先确认系统层面的 `ta-lib` 库已安装，再安装 Python 依赖。
