# 训练策略与硬件取舍

## 1. v1.0.1 到 v1.1.5 观察

`experiments/v1.0.1` 使用 CPU 训练，配置为 39 特征、45 日序列、`d_model=96`、2 层 Transformer、每个目标日抽样 120 只股票、5 个 epoch。

观察到的问题：

- 训练设备为 `cpu`。本机 Intel Arc 130T 不能按常规 CUDA 路线被 PyTorch 自动使用，所以训练主要吃 CPU。
- 多个窗口第 1 到第 3 个 epoch 已达到最佳内部验证分数，后续 epoch 继续运行但验证 `final_score` 下降。
- 单次训练时间不算长，但 walk-forward 会按窗口数线性叠加；4 个窗口加最终模型相当于训练 5 次。
- `v1.1.5` 的慢参数在 3 个 walk-forward 窗口里相对最好，但均值仍为负，且窗口间波动很大。它可以作为候选方向继续复核，不能直接当成稳定提升。

因此当前优先改训练过程，而不是盲目加大模型。

补充一点数据清洗经验：原始 `stock_data` 里确实存在少量“开高低收完全相同、但成交量/成交额/换手率/涨跌幅缺失”的平盘样本。这类样本现在在数据入口会补零保留，不会被整行删掉；如果以后又出现别的缺失行，才在读取阶段按异常行处理。这样更适合后续继续做 walk-forward 对照。

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

## 8. v1.2.15 balanced 对照

`v1.2.15 balanced` 与 `v1.2.13 noid` 使用同样 12 个窗口，差异主要是是否把 `instrument` 输入模型。两者 epoch 上限不同，但最佳 epoch 都没有超过 15，因此仍能作为强参考，不应称为完全严格对照。结果显示：

- `v1.2.13 noid` 12 窗口均值约 `0.024783`，正分窗口 `8/12`；
- `v1.2.15 balanced` 12 窗口均值约 `-0.002820`，正分窗口 `6/12`；
- 同日期逐窗口比较，`noid` 在 9 个窗口更好，`balanced` 在 3 个窗口更好。

当前结论：移除 `instrument` 应继续作为默认方向。股票代码仍用于分组和输出，但不要把整数编码后的股票编号当连续数值特征输入模型。

## 9. 横截面 rank 特征

下一步优先测试横截面 rank 特征。它把同一交易日内的特征值转成百分位排名，让模型学习“当日相对强弱”，而不是绝对价格、绝对成交额或随市场状态漂移的原始数值。

当前实现的 rank 源特征包括：`涨跌幅`、`换手率`、`成交额`、`成交量`、`振幅`、`return_1`、`return_5`、`return_10`、`volume_ratio`、`volatility_10`、`volatility_20`、`rsi`、`atr_14`。对应新增特征名以 `_cs_rank` 结尾。

`v1.3.0 noid-rank` 已完成最近 3 个窗口测试，结果不理想：

- 3 窗口均值约 `-0.035988`，低于同窗口 `v1.2.8/v1.2.13 noid` 的 `0.017410`；
- 仅 1 个窗口为正；
- 同窗口逐项比较，只在 2026-06-29 窗口略好，其余两个窗口明显变差。

当前结论：不要把 `v1.3.0 noid-rank` 直接扩到 12 窗口。横截面 rank 方向仍可能有价值，但第一版“追加 13 个 rank 特征”的方式可能引入了噪声或重复信号。后续若继续 rank 方向，应改成更小的特征组，或尝试用 rank 替代部分原始量价特征，而不是简单追加。

`v1.3.1/v1.3.2 noid-rank-replace` 已完成 3 窗口和 12 窗口测试。它只把绝对量价尺度强、容易随市场状态漂移的列替换为当日横截面百分位排名，保留收益率、换手率、RSI/KDJ、波动率等原始比例或技术指标。这样输入维度基本不变，能更清楚地判断“rank 化绝对尺度”是否有收益。

当前替换列为：`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`涨跌额`、`sma_5`、`sma_20`、`ema_12`、`ema_26`、`ema_60`、`boll_mid`、`boll_std`、`atr_14`、`obv`、`volume_ma_5`、`volume_ma_20`、`high_low_spread`、`open_close_spread`、`high_close_spread`、`low_close_spread`。

结果摘要：

- `v1.3.1` 最近 3 窗口均值约 `0.021452`，略高于同窗口 noid 的 `0.017410`；
- `v1.3.2` 12 窗口均值约 `0.024755`，几乎等于 `v1.2.13 noid` 的 `0.024783`；
- `v1.3.2` 最差窗口约 `-0.055007`，好于 noid 的 `-0.063489`；
- 同窗口比较，replace 在 `6/12` 个窗口更好、`6/12` 个窗口更差；
- 选股集中度没有明显改善，12 窗口仍只选到 31 只不同股票，`688521` 仍高频出现。

当时结论：替代式 rank 明显优于追加式 rank，但没有稳定超过 noid，因此先保留为候选实验档。后续 `v1.4.5/v1.4.7/v1.4.8` 的 24 窗口公平复核更新了判断：`noid-rank-replace` 是当前 Transformer 单模型保底配置。`v1.14.0` 起正式默认提交改为基于它和 LightGBM 的 `ensemble-gate`。后续特征工程细节见 [feature_engineering.md](feature_engineering.md)。

`v1.3.3 noid-marketrel` 追加 8 个市场相对特征后，最近 3 窗口均值约 `-0.036034`，全弱于同窗口 noid，不建议扩跑 12 窗口。训练 loss 和梯度范数没有明显异常，问题更像特征把模型带向了外部收益更差的选股池。

`v1.3.4 noid-rank-lite` 只替换开高低收、成交量、成交额和涨跌额 7 个原始价量尺度列，最近 3 窗口均值约 `-0.018936`，只在 `1/3` 个窗口好于 noid。它比 marketrel 好，但仍明显弱于 noid 和 22 列 `noid-rank-replace`，不建议扩跑 12 窗口。

当前结论：`v1.3.3` 和 `v1.3.4` 都没有显示出继续细分相对特征的价值。后续 24 窗口复核后，主提交配置已改为 `noid-rank-replace`；不要把 rank-lite 或 marketrel 升为默认。

`v1.6.0 noid-rank-cleanrisk` 已跑完 24 窗口，均值和最差窗口都明显变差，说明这组清洗启发的流动性/回撤风险特征当前实现未通过。`v1.6.1 noid-rank-multiperiod` 也在 24 窗口明显变差，均值约 `-0.012037`，相对 `v1.4.5 rank-replace` 只在 `5/24` 个窗口胜出。`v1.6.2 noid-rank-breadth` 已跑 6 窗口并明显变差，单列市场宽度暂不扩跑。`v1.7.0 noid-rank-momdelta` 和 `v1.8.0 noid-rank-riskadj` 的 6 窗口也偏弱，不扩 12/24。后续新方向按 [实验纪律与分级验证](experiment_protocol.md) 走 `6 -> 12 -> 24`，单模型对照仍优先使用 `noid-rank-replace`；同版本只增大 `--windows` 时可加 `--resume` 增量扩跑。

## 10. 当前策略

训练脚本现在默认使用：

- `ReduceLROnPlateau`：验证 `final_score` 停滞时把学习率乘以 `lr_factor`；
- `AdamW`：默认优化器；
- `Lookahead`：可通过 `BDC_OPTIMIZER=lookahead` 开启，用于单独测试抗震荡效果；
- 早停：连续多个 epoch 没有超过最佳分数加 `early_stopping_min_delta` 时停止；
- 梯度裁剪：默认按 `max_grad_norm=5.0` 裁剪，降低训练不稳定风险；
- `training_history.csv`：逐 epoch 记录 train/eval loss、final_score、学习率、梯度范数和耗时；
- `final_score.txt`：记录最佳 epoch、最佳分数、实际停止 epoch、停止原因、总耗时和最终学习率。

这些设置不会改变“用过去预测未来”的训练/预测语义，只是减少无效训练，并让调参日志更容易判断。

## 11. 本机参数

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

`noid` 与 `balanced` 保持相同小模型结构，只从模型输入特征中移除 `instrument`。当前 noid 主线使用 30 epoch 上限；如果要和 `balanced` 严格比较，需要给 `balanced` 显式设置同样的 epoch 上限。股票代码仍用于分组和输出。

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

如果要测试横截面 rank 特征，用 `noid-rank` 档位：

```bash
sh tune.sh v1.3.0 noid-rank --windows 3 --skip-final
```

`noid-rank` 基于 `noid`，继续移除 `instrument`，默认开启 `BDC_USE_CROSS_SECTIONAL_RANKS=1`，并把 epoch 上限设为 30 以贴近 `v1.2.13` 严格对照。

如果要测试用 rank 替代绝对量价尺度特征，用 `noid-rank-replace` 档位：

```bash
sh tune.sh v1.3.1 noid-rank-replace --windows 3 --skip-final
```

`noid-rank-replace` 仍基于 `noid`，但设置 `BDC_CROSS_SECTIONAL_RANK_MODE=replace`。它不把所有 rank 列简单追加到输入后面，而是用 `_cs_rank` 列替换部分原始绝对量价列，用于降低重复信号和非平稳尺度噪声。

`balanced`、`smooth` 和 `stable` 的 epoch 上限保持 15、早停耐心值设为 5，继续作为含 `instrument` 的轻量对照。`noid` 主线和它的特征/后处理对照统一使用 30 epoch 上限，包括 `noid`、`noid-rank`、`noid-rank-replace`、`noid-marketrel`、`noid-stable`、`noid-lowvol` 和 `noid-full`。这样后续比较 noid、rank、marketrel、lowvol 时，主要差异来自实验变量本身，而不是最大训练轮数。

可比性复核：`v1.4.4 noid` 使用旧 15 epoch 上限，`v1.4.5 noid-rank-replace` 使用 30 epoch 上限；`v1.4.5` 中有窗口最佳 epoch 到 16，说明较大上限可能影响结论。从 `v1.4.7` 起已修正 profile 和正式集成源模型配置。

`v1.4.7 noid` 补跑后与旧 `v1.4.4` 逐窗口完全一致，说明 noid 弱不是 15 epoch 截断造成的。公平预算下，`v1.4.5 rank-replace` 仍高于 noid；但 `v1.4.8 ensemble-lowvol` 仍低于 rank-replace。因此训练过程层面的下一步不是继续加 epoch，而是回到特征和排序信号质量。

`v1.4.9 noid-rank-trendq` 追加趋势质量特征后，最新 3 窗口均值约 `-0.067912`，相对同日期 `v1.4.5 rank-replace` 平均低约 `0.048780`，且 3 个窗口都未胜出。这个结果更像特征噪声或重复信号，不像训练过程欠优化；不建议用更多 epoch 或更大模型去补救这组特征。

`v1.9.0 noid-rank-ret5rank` 已跑 6 窗口，均值约 `-0.049961`，相同日期 `0/6` 胜过 `rank-replace`。它说明前两个 rank 差信号失败不只是组合方式问题，单独的短期收益 rank 也会强化固定坏股池；训练过程暂时不是这条线的主要瓶颈。

## 12. v1.5.0 排序监督信号实验

当前默认 `WeightedRankingLoss` 的 listwise 部分会对真实 5 日收益率做 softmax。由于单个窗口内的收益率通常只是几个百分点，默认温度 `1.0` 会让目标分布接近均匀，真实第一名和普通股票在 listwise loss 中的差异不够明显。pairwise loss 仍会提供排序信号，但它也在尝试学习完整横截面排序，而最终提交只关心前 5 只。

`v1.5.0` 新增 `BDC_LOSS_TARGET_TEMPERATURE`，默认仍等于 `BDC_LOSS_TEMPERATURE=1.0`，因此不设置时与旧逻辑等价。新 profile：

```bash
sh tune.sh v1.5.0 noid-rank-sharp --windows 3 --skip-final
```

该档位基于当前默认候选 `noid-rank-replace`，只设置 `BDC_LOSS_TARGET_TEMPERATURE=0.05`，不改特征、模型、训练窗口、epoch、采样或优化器。目标是让 listwise 目标更尖锐，先观察源模型的 top5、top20、top50 排序信号是否改善。

验收重点：

- 最近 3 窗口均值是否接近或超过同日期 `v1.4.5 noid-rank-replace`；
- 最差窗口是否没有明显恶化；
- `prediction_diagnostics_summary.csv` 中 top20/top50 后验收益和 Spearman 是否改善；
- `training_history.csv` 是否出现更快过拟合，例如最佳 epoch 过早且后续 eval loss 急升。

如果 6 窗口明显弱于 `rank-replace`，不要扩到 12/24 窗口；这说明当前实现可能只是在放大金融噪声。但结论应写成“当前实现未通过”，不是永久否定整个方向。

`v1.5.0 noid-rank-sharp` 已完成最近 3 窗口短测，结果不通过：

- `v1.5.0 noid-rank-sharp` 均值约 `-0.024860`；
- 同日期 `v1.4.5 noid-rank-replace` 均值约 `-0.019132`；
- 3 个窗口里 `0/3` 胜出；
- 平均 Spearman 约 `-0.231256`，也没有比同日期 `rank-replace` 更好。

当前判断：把 listwise 目标调得更尖并没有改善源模型排序信号，反而略微恶化。短期内不要继续沿这条线加大温度对比，优先回到特征或候选池结构。

## 13. 平台期和震荡判断

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

## 14. 赛方机器约束

赛事文档要求代码可在 i7-13650H、16GB 内存、RTX 4060 8GB 显存、50GB 存储上运行；预测不超过 5 分钟，训练不超过 8 小时，运行时不得联网。

当前正式取舍：

- 默认提交入口为 `ensemble-gate`，训练一个 Transformer 主模型和一个 LightGBM 防守模型；
- 默认正式强度为 `BDC_SUBMISSION_STRENGTH=strong`，Transformer 使用全部可打标签目标日和全部股票，最多 40 epoch、早停耐心值 8；LightGBM 使用最近 240 个训练目标日和 600 棵树；
- `validated` 保留 `v1.13.4` 同口径源模型预算，适合最终前快速复核；
- `max` 会进一步加宽 Transformer 并增加 LightGBM 树数，只应在本地完整计时确认安全后使用；
- 默认特征工程进程数为 6，避免在 16GB 内存机器上开太多进程；
- 设备选择仍是 `CUDA -> MPS -> CPU`，赛方 4060 会优先用 CUDA，本机没有 CUDA 时自动退回 CPU；
- 正式预测只生成 `result.csv`，不能本地提前知道未来 5 个交易日真实收益。

耗时估计依据：

- `v1.13.0` 受控采样 Transformer 24 窗口平均训练约 43 秒；
- `v1.13.2` LightGBM 24 窗口平均训练约 9 秒；
- `v1.2.14 noid-full` 全目标日、全股票 Transformer 3 窗口平均训练约 12 分钟，单窗口约 9 到 19 分钟。

因此默认 `strong` 预计远低于 8 小时，但它改变了正式训练覆盖面，不等价于 `v1.13.4` 的严格验证预算。提交前如果发现耗时、输出或本地后验表现异常，应先回退到 `BDC_SUBMISSION_STRENGTH=validated`，而不是继续加大到 `max`。

提交前至少跑一次：

```bash
sh train.sh
sh test.sh
```

确认 `output/result.csv` 是当前代码训练出的模型生成的结果，不要复用旧实验目录里的文件直接提交。
