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

## 4. v1.2.6 到 v1.2.7 观察

`v1.2.6 balanced` 和 `v1.2.7 smooth` 使用同一批 3 个窗口，主要差异只有优化器。结果显示 Lookahead 只带来很小的外部分数变化：`balanced` 均值约 `-0.029883`，`smooth` 均值约 `-0.029580`。

Lookahead 让部分内部验证曲线略平滑，但没有明显改变最终 top5。两版反复选到同一批股票，说明当前瓶颈更可能是排序输出过于固定，而不只是优化器震荡。后续优先查看 `prediction_scores.csv` 的完整候选排名，相关经验见 [prediction_diagnostics.md](prediction_diagnostics.md)。

固定选股池的优先嫌疑是 `instrument` 被当作连续数值输入模型。下一步用 `noid` 档位移除该特征做对照，先看 top20/top50 重复度和外部分数是否改善，再决定是否保留股票身份信息。

## 5. v1.2.8 noid 观察

`v1.2.8 noid` 在 `balanced` 基础上移除了模型输入中的 `instrument`，其余训练预算保持一致。3 个窗口分数约为 `0.062592`、`-0.063489`、`0.053129`，均值约 `0.017410`，好于 `v1.2.6 balanced` 和 `v1.2.7 smooth`。

这说明把股票编号当连续特征可能确实带来了固定选股池问题。但 `noid` 也出现了新现象：不同窗口的 top5 变化明显增加，且更偏向 `688/300/002` 等高波动股票，第二个窗口被明显拖累。下一步不应马上扩大模型，而应先测试 `noid-stable`：继续移除 `instrument`，同时加大 `dropout` 和 `weight_decay`，观察最差窗口和 top20/top50 重复度是否改善。

## 6. v1.2.10 到 v1.2.11 关闭调度和早停观察

`v1.2.10` 和 `v1.2.11` 使用 `noid`，并显式关闭学习率调度和早停：`BDC_LR_SCHEDULER=off`、`BDC_EARLY_STOPPING_PATIENCE=0`，训练 30 个 epoch。训练历史确认学习率固定为 `3e-5`，停止原因都是 `max_epochs`。

需要注意一个语义点：关闭早停只是不提前结束训练，不代表预测会使用最后一轮权重。训练过程始终按内部验证 `final_score` 保存 `best_model.pth`，预测也读取这个最佳模型。因此如果关闭早停后的最佳 epoch 和早停版相同，外部窗口分数会完全相同，这是合理现象。

结果摘要：

- `v1.2.10` 3 窗口均值约 `-0.003414`，低于 `v1.2.8 noid` 的 `0.017410`。其中前两个窗口最佳 epoch 与 `v1.2.8` 一致，所以预测结果相同；第三个窗口在第 11 轮出现内部验证最佳，但外部窗口分数反而变差。
- `v1.2.11` 扩到 12 个窗口，均值约 `0.016881`，12 个窗口中 8 个为正。最后 3 个窗口与 `v1.2.10` 日期和配置相同，结果完全一致，说明流程是确定性的。
- 事后按 `patience=5` 模拟早停，`v1.2.11` 中多个早期窗口会错过后期内部验证最佳 epoch。这说明过短早停可能漏掉后期改善；但内部验证改善不一定能转化为外部 walk-forward 分数。

当前结论：不要简单认为“关掉早停、硬跑更多 epoch”更好。更稳妥的下一步是做同样 12 个窗口的公平对照：保持 `num_epochs=30`，只打开 `plateau` 和早停，再看均值、最差窗口和耗时。如果要判断最后一轮模型本身，需要另加“保存/预测 final epoch 模型”的实验开关，不能用当前 `best_model.pth` 结果代替。

## 7. v1.2.13 开启调度和早停对照

`v1.2.13` 与 `v1.2.11` 使用同样 12 个 walk-forward 窗口、同样 `noid` 小模型和同样 30 个 epoch 上限，但重新开启 `ReduceLROnPlateau` 和早停。

结果摘要：

- `v1.2.13` 12 窗口均值约 `0.024783`，高于 `v1.2.11` 的 `0.016881`；
- 最差窗口仍为 `-0.063489`，没有改善；
- 正分窗口数仍为 `8/12`；
- 总训练耗时从约 `29m47s` 降到约 `8m16s`，总流程耗时从约 `31m26s` 降到约 `9m37s`；
- 逐窗口看，4 个窗口改善、4 个窗口持平、4 个窗口变差，均值提升主要来自少数窗口的大幅改善。

当前结论：调度和早停应继续默认保留。它们不是稳定提高每个窗口的“冲分技巧”，但在当前实验中显著节省时间，且多窗口均值没有变差。下一步比继续硬跑 30 epoch 更值得做的是 `noid-full`：保持 `noid` 模型不变，只取消训练目标日和每日股票抽样，验证更多训练数据是否能提高泛化。

## 8. 当前策略

训练脚本现在默认使用：

- `ReduceLROnPlateau`：验证 `final_score` 停滞时把学习率乘以 `lr_factor`；
- `AdamW`：默认优化器；
- `Lookahead`：可通过 `BDC_OPTIMIZER=lookahead` 开启，用于单独测试抗震荡效果；
- 早停：连续多个 epoch 没有超过最佳分数加 `early_stopping_min_delta` 时停止；
- 梯度裁剪：默认按 `max_grad_norm=5.0` 裁剪，降低训练不稳定风险；
- `training_history.csv`：逐 epoch 记录 train/eval loss、final_score、学习率、梯度范数和耗时；
- `final_score.txt`：记录最佳 epoch、最佳分数、实际停止 epoch、停止原因、总耗时和最终学习率。

这些设置不会改变“用过去预测未来”的训练/预测语义，只是减少无效训练，并让调参日志更容易判断。

## 9. 本机参数

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

如果要测试固定选股池是否来自股票编号特征，用 `noid` 档位：

```bash
sh tune.sh v1.2.8 noid --skip-final
```

`noid` 与 `balanced` 保持相同模型和训练预算，只从模型输入特征中移除 `instrument`。股票代码仍用于分组和输出。

如果要测试去编号后是否需要更强正则化，用 `noid-stable` 档位：

```bash
sh tune.sh v1.2.9 noid-stable --skip-final
```

`noid-stable` 基于 `noid`，默认 `dropout=0.2`、`weight_decay=1e-4`。它不是新架构，只用于判断高波动股票偏好能否被正则化压住。

如果要测试完整训练数据，而不是更换模型结构，用 `noid-full` 档位：

```bash
sh tune.sh v1.2.14 noid-full --windows 3 --skip-final
```

`noid-full` 保持 `noid` 的 39 特征、小 Transformer 和去编号设置，只把 `train_target_days` 与 `max_stocks_per_day` 设为 0，表示不限制训练目标日、不抽样每日股票。它会明显变慢，建议先跑 1 到 3 个窗口确认耗时，再决定是否扩到 12 个窗口。

`balanced`、`noid`、`smooth` 和 `stable` 的 epoch 上限设为 15、早停耐心值设为 5。这个上限比最初的 5 轮更接近 `large`，能避免分数平滑增长时被硬截断；同时模型仍比 `large` 小，单个 epoch 更快。如果日志显示大多数窗口长期在第 3 到第 5 轮早停，说明上限不是瓶颈；如果最佳 epoch 多次出现在第 12 轮以后，再考虑把上限提高到 20。`noid-full` 为完整数据实验，epoch 上限单独设为 30，但仍保留早停。

## 10. 平台期和震荡判断

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
- 关闭早停后如果外部分数与早停版相同，先看 `training_history.csv` 里的 `best_epoch` 是否相同。当前预测使用的是最佳内部验证权重，不是最后一轮权重。
- 对比不同 profile 时，先保证窗口一致。`v1.2.1 large`、2 窗口 `balanced`、3 窗口 `stable` 不能直接下定论，只能作为参考。

## 11. 赛方机器约束

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
