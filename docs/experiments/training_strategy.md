# 训练策略与硬件取舍

## 1. v1.0.1 到 v1.1.5 观察

`experiments/v1.0.1` 使用 CPU 训练，配置为 39 特征、45 日序列、`d_model=96`、2 层 Transformer、每个目标日抽样 120 只股票、5 个 epoch。

观察到的问题：

- 训练设备为 `cpu`。本机 Intel Arc 130T 不能按常规 CUDA 路线被 PyTorch 自动使用，所以训练主要吃 CPU。
- 多个窗口第 1 到第 3 个 epoch 已达到最佳内部验证分数，后续 epoch 继续运行但验证 `final_score` 下降。
- 单次训练时间不算长，但 walk-forward 会按窗口数线性叠加；4 个窗口加最终模型相当于训练 5 次。
- `v1.1.5` 的慢参数在 3 个 walk-forward 窗口里相对最好，但均值仍为负，且窗口间波动很大。它可以作为候选方向继续复核，不能直接当成稳定提升。

因此当前优先改训练过程，而不是盲目加大模型。

## 2. v1.2.1 large 观察

`experiments/v1.2.1` 使用 `large` 档位：3 个 walk-forward 窗口、39 特征、45 日序列、`d_model=96`、3 层 Transformer、`dim_feedforward=512`、20 个 epoch 上限、学习率 `5e-6`、`BDC_TORCH_NUM_THREADS=14`。

结果摘要：

- 3 个窗口分数分别为 `0.027493`、`-0.128689`、`-0.006744`，均值约 `-0.035980`；
- 窗口训练耗时约 `8m35s`、`4m35s`、`3m50s`，最终模型训练约 `10m14s`，总耗时约 `27m47s`；
- 当前 `BDC_VAL_DAYS=5`，不是“内部验证样本 20”。不要把其他对话里的样本数判断直接套到这次实验；
- 梯度范数没有接近 0，现有证据不像梯度消失或学习率完全冻住，更像窗口敏感、验证噪声和轻度过拟合。

DeepSeek 提到的“高原上有峡谷”的说法可以作为调参直觉：金融数据低信噪比、非平稳，验证分数容易平台和震荡。但它只是帮助理解现象，不能直接推出“立刻上 Lookahead、SWA、cosine restart”。下一步应先做同窗口公平对照，再一次只改少数变量。

因此，`large` 继续保留为慢速候选复核档，不进入日常默认。下一轮优先跑同样 3 个窗口的 `balanced`，确认慢模型是否真的有收益。

## 3. v1.2.4 到 v1.2.5 观察

`v1.2.4 balanced` 使用 15 个 epoch 上限和早停耐心值 5，但仍只跑了默认 2 个窗口，不能和 3 窗口的 `stable/large` 直接比较。两个窗口外部分数仍与之前一致：`-0.006730`、`-0.038214`。

`v1.2.5 stable` 使用 3 个窗口、`dropout=0.2`、`weight_decay=1e-4`、15 个 epoch 上限。3 个窗口分数为 `-0.044707`、`-0.043543`、`-0.055497`，均值约 `-0.047915`，比 `large` 更平稳但均值没有更好。

延长 epoch 后，多数训练仍在第 7 到第 10 轮早停，最佳内部验证 epoch 多在第 2 到第 5 轮。这说明单纯增加 epoch 不是主要突破口。下一步要单独测试优化器稳定性，例如 `smooth` 档位的 Lookahead，而不是继续同时改模型容量、正则化和学习率。

## 4. 当前策略

训练脚本现在默认使用：

- `ReduceLROnPlateau`：验证 `final_score` 停滞时把学习率乘以 `lr_factor`；
- `AdamW`：默认优化器；
- `Lookahead`：可通过 `BDC_OPTIMIZER=lookahead` 开启，用于单独测试抗震荡效果；
- 早停：连续多个 epoch 没有超过最佳分数加 `early_stopping_min_delta` 时停止；
- 梯度裁剪：默认按 `max_grad_norm=5.0` 裁剪，降低训练不稳定风险；
- `training_history.csv`：逐 epoch 记录 train/eval loss、final_score、学习率、梯度范数和耗时；
- `final_score.txt`：记录最佳 epoch、最佳分数、实际停止 epoch、停止原因、总耗时和最终学习率。

这些设置不会改变“用过去预测未来”的训练/预测语义，只是减少无效训练，并让调参日志更容易判断。

## 5. 本机参数

本机配置是 32GB 内存、Intel Core Ultra 5 225H、Intel Arc 130T。实际训练通常走 CPU。

建议日常使用：

```bash
sh tune.sh quick --skip-final
sh tune.sh v1.2.0 balanced --skip-final
```

如果一个 epoch 仍然太慢，优先继续降低这些参数：

```bash
BDC_TRAIN_TARGET_DAYS=36 BDC_MAX_STOCKS_PER_DAY=80 sh tune.sh v1.2.0 balanced --skip-final
```

不要把日常调试直接等同于冲分结果。`quick` 和 `balanced` 是为了快速发现代码问题和大致方向，分数只能和相同档位、相近窗口的版本比较。

`BDC_TORCH_NUM_THREADS` 控制 PyTorch 在 CPU 上做矩阵计算时能用多少线程，和你的 14 核 CPU 直接相关。本机 CPU 训练可以试 8 到 14；赛方机器有 RTX 4060，训练通常走 CUDA，所以普通档位不默认绑死到 14，避免 CPU 线程抢占和内存压力。

如果要复核 `v1.1.5` 风格的慢参数，用独立 `large` 档位：

```bash
sh tune.sh v1.2.0 large --skip-final
```

`large` 默认 3 个窗口、20 个 epoch 上限、较大的前馈层和 `BDC_TORCH_NUM_THREADS=14`。它不适合每次日常调试都跑。

如果要测试更强正则化，而不是更大的模型，用 `stable` 档位：

```bash
sh tune.sh v1.2.3 stable --skip-final
```

`stable` 基于 `balanced` 的小模型，默认 3 个窗口、`dropout=0.2`、`weight_decay=1e-4`。它用于单独观察正则化能否缓解震荡，不用于替代正式冲分配置。

如果要测试优化器抗震荡，而不是更大的模型或更强正则化，用 `smooth` 档位：

```bash
sh tune.sh v1.2.7 smooth --skip-final
```

`smooth` 与 `balanced` 保持相同模型、窗口、epoch、dropout 和 weight decay，只把优化器从 AdamW 换成 Lookahead。先用它判断 Lookahead 是否值得保留；不要同时叠加 cosine、SWA 或 EMA。

`balanced`、`smooth` 和 `stable` 的 epoch 上限设为 15、早停耐心值设为 5。这个上限比最初的 5 轮更接近 `large`，能避免分数平滑增长时被硬截断；同时模型仍比 `large` 小，单个 epoch 更快。如果日志显示大多数窗口长期在第 3 到第 5 轮早停，说明上限不是瓶颈；如果最佳 epoch 多次出现在第 12 轮以后，再考虑把上限提高到 20。

## 6. 平台期和震荡判断

优先看每个模型目录里的：

```text
training_history.csv
final_score.txt
train.log
```

简单判断：

- train loss 也不降，且 `avg_grad_norm` 很小：可能欠拟合、学习率过低或梯度太弱。
- train loss 下降，但 eval loss 上升、eval final_score 震荡：更像过拟合或验证窗口噪声。
- walk-forward 外部分数和训练内部 `final_score` 不一致是正常的，调参最终以后者之外的 `summary.csv` 多窗口均值为主。
- 对比不同 profile 时，先保证窗口一致。`v1.2.1 large`、2 窗口 `balanced`、3 窗口 `stable` 不能直接下定论，只能作为参考。

## 7. 赛方机器约束

赛事文档要求代码可在 i7-13650H、16GB 内存、RTX 4060 8GB 显存、50GB 存储上运行；预测不超过 5 分钟，训练不超过 8 小时，运行时不得联网。

当前取舍：

- `full` 默认最多 6 个 epoch，并启用早停，避免完整训练失控；
- 默认特征工程进程数为 6，避免在 16GB 内存机器上开太多进程；
- 设备选择仍是 `CUDA -> MPS -> CPU`，赛方 4060 会优先用 CUDA，本机没有 CUDA 时自动退回 CPU；
- 正式预测只生成 `result.csv`，不能本地提前知道未来 5 个交易日真实收益。

提交前至少跑一次：

```bash
sh train.sh
sh test.sh
```

确认 `output/result.csv` 是当前代码训练出的模型生成的结果，不要复用旧实验目录里的文件直接提交。
