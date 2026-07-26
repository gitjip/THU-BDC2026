# Walk-Forward 调参与版本化运行

本文说明调参用的历史模拟流程。它不替代正式预测，默认也不会覆盖 `output/result.csv`。

## 1. 适用场景

正式预测面对的是 `2026-08-02` 之后未来 5 个交易日，没有真实标签，无法本地打分。

walk-forward 的做法是在历史数据中模拟：

1. 选一个历史 `as_of_date`，只使用这一天及以前的数据训练；
2. 用训练出的模型生成一次预测；
3. 用 `as_of_date` 之后连续 5 个真实交易日计算分数；
4. 向前滚动多个窗口，观察平均表现。

这样更接近比赛提交逻辑，也避免把未来数据泄漏进训练。

## 2. 快速查看计划

先看将要跑哪些窗口，不训练：

```bash
sh tune.sh v1.0.0 debug --dry-run
```

输出里的每个窗口都包含：

- `as_of`：该窗口训练和预测最多能看到的日期；
- `target`：该窗口用于验证的连续 5 个真实交易日。

## 3. 调试运行

快速调试用：

```bash
sh tune.sh v1.0.0 debug --windows 1 --skip-final
```

`debug` 会启用 `BDC_FAST_DEV=1`，训练会少取训练目标日、少抽样股票、关闭 TensorBoard，主要用于检查流程是否能跑通。

如果中途失败，修复后可继续：

```bash
sh tune.sh v1.0.0 debug --windows 1 --skip-final --resume
```

## 4. 正式调参运行

默认跑 3 个历史窗口，并在最后训练一次最终模型、生成最终预测：

```bash
sh tune.sh v1.0.0
```

常用参数：

```bash
sh tune.sh v1.0.1 --windows 5
sh tune.sh v1.0.1 --windows 5 --step-days 5
sh tune.sh v1.0.1 --data-file data/stock_data.csv
```

默认不覆盖正式提交文件。最终预测会保存在：

```text
experiments/v1.0.0/final/result.csv
```

确认这个版本就是要提交的结果后，再显式发布：

```bash
sh tune.sh v1.0.0 --resume --publish-final
```

这会把 `experiments/v1.0.0/final/result.csv` 复制到 `output/result.csv`。

如果代码已经提交，并且希望流程完成后自动创建本地 Git tag：

```bash
sh tune.sh v1.0.0 --resume --create-tag
```

`--create-tag` 要求工作区没有未提交改动，避免 tag 指向的代码和实际实验代码不一致。

## 5. 产物位置

每个版本都有独立目录：

```text
experiments/v1.0.0/
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
- `windows/*/score.json`：每个窗口选中的股票、权重和真实收益；
- `final/result.csv`：该版本最终预测文件。

## 6. 注意事项

- 单个窗口分数波动很大，调参时优先看多个窗口平均值。
- 窗口越多越稳，但训练次数越多，耗时近似按窗口数线性增加。
- `experiments/` 下模型和日志默认不提交到 Git，避免仓库过大。
- 语义版本号必须是 `vMAJOR.MINOR.PATCH`，例如 `v1.0.0`。
