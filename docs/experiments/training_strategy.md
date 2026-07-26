# 训练策略与硬件取舍

## 1. v1.0.1 观察

`experiments/v1.0.1` 使用 CPU 训练，配置为 39 特征、45 日序列、`d_model=96`、2 层 Transformer、每个目标日抽样 120 只股票、5 个 epoch。

观察到的问题：

- 训练设备为 `cpu`。本机 Intel Arc 130T 不能按常规 CUDA 路线被 PyTorch 自动使用，所以训练主要吃 CPU。
- 多个窗口第 1 到第 3 个 epoch 已达到最佳内部验证分数，后续 epoch 继续运行但验证 `final_score` 下降。
- 单次训练时间不算长，但 walk-forward 会按窗口数线性叠加；4 个窗口加最终模型相当于训练 5 次。

因此当前优先改训练过程，而不是盲目加大模型。

## 2. 当前策略

训练脚本现在默认使用：

- `ReduceLROnPlateau`：验证 `final_score` 停滞时把学习率乘以 `lr_factor`；
- 早停：连续多个 epoch 没有超过最佳分数加 `early_stopping_min_delta` 时停止；
- 梯度裁剪：默认按 `max_grad_norm=5.0` 裁剪，降低训练不稳定风险；
- `final_score.txt`：记录最佳 epoch、最佳分数、实际停止 epoch 和停止原因。

这些设置不会改变“用过去预测未来”的训练/预测语义，只是减少无效训练，并让调参日志更容易判断。

## 3. 本机参数

本机配置是 32GB 内存、Intel Core Ultra 5 225H、Intel Arc 130T。实际训练通常走 CPU。

建议日常使用：

```bash
sh tune.sh quick --skip-final
sh tune.sh v1.1.0 balanced --skip-final
```

如果一个 epoch 仍然太慢，优先继续降低这些参数：

```bash
BDC_TRAIN_TARGET_DAYS=36 BDC_MAX_STOCKS_PER_DAY=80 sh tune.sh v1.1.0 balanced --skip-final
```

不要把日常调试直接等同于冲分结果。`quick` 和 `balanced` 是为了快速发现代码问题和大致方向，分数只能和相同档位、相近窗口的版本比较。

## 4. 赛方机器约束

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
