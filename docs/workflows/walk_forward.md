# Walk-Forward 调参与版本化运行

本文说明调参用的历史模拟流程。它不替代正式预测，默认也不会覆盖 `output/result.csv`。

## 1. 适用场景

正式预测面对的是 `2026-08-02` 之后未来 5 个交易日，没有真实标签，无法本地打分。

walk-forward 的做法是在历史数据中模拟：

1. 选一个历史 `as_of_date`，只使用这一天及以前的数据训练；
2. 将目标窗口前一天作为模拟提交日，用训练出的模型生成一次预测；
3. 用模拟提交日之后连续 5 个自然日且每天都有交易数据的窗口计算分数；
4. 向前滚动多个窗口，观察平均表现。

这样更接近比赛提交逻辑，也避免把未来数据泄漏进训练。

## 2. 快速查看计划

先看将要跑哪些窗口，不训练：

```bash
sh tune.sh quick --dry-run
```

输出里的每个窗口都包含：

- `as_of`：该窗口训练和预测最多能看到的日期；
- `mock_submission`：模拟提交日，通常是目标窗口前一天，可能是周末；
- `target`：该窗口用于验证的连续 5 天。

这里的“连续”指连续自然日，并且这些自然日都必须在数据中有交易记录。跨周末或节假日的候选窗口会被跳过，例如 `2026-07-02` 到 `2026-07-08` 不会作为平时验证窗口。

## 3. 调试运行

快速调试用：

```bash
sh tune.sh quick --skip-final
```

`quick` 会启用小模型、39 特征、较短序列和较少股票，主要用于平时检查流程、看方向，分数不适合直接和完整模型比较。`debug`、`fast`、`lite` 都是 `quick` 的别名。

如果中途失败，修复后可继续：

```bash
sh tune.sh quick --skip-final --resume
```

## 4. 调参档位

`tune.sh` 默认使用 `balanced`。各档位只设置未显式指定的环境变量，你仍可用 `BDC_...` 覆盖。

| 档位 | 用途 | 默认窗口 | 特征 | 序列 | 训练目标日 | 每日股票 | 模型 | 最大 epoch |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| `quick` | 平时调试 | 1 | 39 | 30 | 24 | 60 | d_model=64, layers=1 | 4 |
| `balanced` | 常规调参 | 2 | 39 | 45 | 60 | 120 | d_model=96, layers=2 | 5 |
| `full` | 冲分前复核 | 3 | 配置默认 | 配置默认 | 不限制 | 不抽样 | 配置默认 | 6 |

三个档位都默认使用 `plateau` 学习率调度和早停。`quick`、`balanced` 的早停耐心值为 2，`full` 为 3；也就是说表里的 epoch 是上限，不一定都会跑完。`quick` 和 `balanced` 会明显降低模型表现上限，但能更快暴露代码问题和大致方向。后续冲分时再切回 `full` 或手动放大参数。

## 5. 正式调参运行

默认跑 `balanced`，并在最后训练一次最终模型、生成最终预测：

```bash
sh tune.sh v1.1.0
```

常用参数：

```bash
sh tune.sh v1.1.0 --windows 5
sh tune.sh v1.1.0 --windows 5 --step-days 5
sh tune.sh v1.1.0 --data-file data/stock_data.csv
sh tune.sh v1.1.0 full --windows 3
```

默认不覆盖正式提交文件。最终预测会保存在：

```text
experiments/v1.1.0/final/result.csv
```

确认这个版本就是要提交的结果后，再显式发布：

```bash
sh tune.sh v1.1.0 --resume --publish-final
```

这会把 `experiments/v1.1.0/final/result.csv` 复制到 `output/result.csv`。

如果代码已经提交，并且希望流程完成后自动创建本地 Git tag：

```bash
sh tune.sh v1.1.0 --resume --create-tag
```

`--create-tag` 要求工作区没有未提交改动，避免 tag 指向的代码和实际实验代码不一致。

## 6. 产物位置

每个版本都有独立目录：

```text
experiments/v1.1.0/
  manifest.json
  summary.csv
  walk_forward.log
  windows/
    window_01/
      metadata.json
      prediction.csv
      score.json
      model/
        best_model.pth
        scaler.pkl
        config.json
        final_score.txt
  final/
    result.csv
    model/
      best_model.pth
      scaler.pkl
      config.json
      final_score.txt
```

重点看：

- `summary.csv`：所有窗口的分数汇总；
- `manifest.json`：版本、Git commit、数据文件、窗口计划；
- `windows/*/metadata.json`：窗口元数据，`target_trading_dates` 是实际验证日期，`target_calendar_span_days` 应为 5；
- `windows/*/model/final_score.txt`：该窗口训练早停位置、最佳 epoch 和最佳内部验证分数；
- `windows/*/score.json`：每个窗口选中的股票、权重和真实收益；
- `final/result.csv`：该版本最终预测文件。

## 7. 注意事项

- 单个窗口分数波动很大，调参时优先看多个窗口平均值。
- 窗口越多越稳，但训练次数越多，耗时近似按窗口数线性增加。
- `experiments/` 下模型和日志默认不提交到 Git，避免仓库过大。
- 语义版本号必须是 `vMAJOR.MINOR.PATCH`，例如 `v1.0.0`。
