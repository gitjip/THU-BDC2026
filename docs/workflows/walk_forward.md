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
| `balanced` | 常规调参 | 3 | 39 | 45 | 60 | 120 | d_model=96, layers=2 | 15 |
| `noid` | 去股票编号对照 | 3 | 39(no instrument) | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-marketrel` | 去编号+市场相对特征 | 3 | 39(no instrument)+8 mktrel | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-rank` | 去编号+横截面rank | 3 | 39(no instrument)+rank | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-rank-lite` | 去编号+小范围rank替代 | 3 | 39(no instrument, 7 rank replace) | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-rank-replace` | 去编号+rank替代绝对量价 | 3 | 39(no instrument, rank replace) | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-rank-sharp` | rank替代+更尖锐listwise目标 | 3 | 39(no instrument, rank replace) | 45 | 60 | 120 | d_model=96, layers=2, target_temp=0.05 | 30 |
| `noid-rank-trendq` | rank替代+趋势质量 | 3 | 39(no instrument, rank replace)+5 trendq | 45 | 60 | 120 | d_model=96, layers=2 | 30 |
| `noid-stable` | 去编号+正则化对照 | 3 | 39(no instrument) | 45 | 60 | 120 | d_model=96, layers=2, dropout=0.2 | 30 |
| `noid-full` | 去编号完整数据对照 | 3 | 39(no instrument) | 45 | 不限制 | 不抽样 | d_model=96, layers=2 | 30 |
| `noid-lowvol` | 去编号+低波动后处理 | 3 | 39(no instrument) | 45 | 60 | 120 | d_model=96, layers=2, low-vol selection | 30 |
| `ensemble-lowvol` | 两模型 top5 并集+低波动重排 | 3 | 复用源模型 | 复用源模型 | 不训练 | 不训练 | noid + rank-replace | 不训练 |
| `smooth` | Lookahead 对照 | 3 | 39 | 45 | 60 | 120 | d_model=96, layers=2, optimizer=lookahead | 15 |
| `stable` | 正则化对照 | 3 | 39 | 45 | 60 | 120 | d_model=96, layers=2, dropout=0.2 | 15 |
| `large` | 慢速候选复核 | 3 | 39 | 45 | 60 | 120 | d_model=96, layers=3, ff=512 | 20 |
| `full` | 冲分前复核 | 3 | 配置默认 | 配置默认 | 不限制 | 不抽样 | 配置默认 | 6 |

这些档位都默认使用 `plateau` 学习率调度和早停。`quick` 的早停耐心值为 2，`balanced`、`noid`、`noid-marketrel`、`noid-rank`、`noid-rank-lite`、`noid-rank-replace`、`noid-rank-sharp`、`noid-stable`、`noid-full`、`noid-lowvol`、`smooth`、`stable`、`large` 为 5，`full` 为 3；也就是说表里的 epoch 是上限，不一定都会跑完。`quick` 用于快速暴露代码问题；`noid` 主线和 rank/marketrel/lowvol 等特征或后处理对照统一使用 30 epoch 上限，减少“某个特征只是多训练了几轮”的干扰。`balanced/smooth/stable` 保持 15 epoch，主要作为早期含 `instrument` 的轻量对照。

`noid` 源自 `balanced` 小模型配置，但当前属于主线对照档：默认 `BDC_USE_INSTRUMENT_FEATURE=0`，从模型输入特征里移除 `instrument`，并使用 30 epoch 上限。股票代码仍用于分组、构造序列和输出结果，但模型不能把股票编号当连续数值直接学习。如果要重新严格比较是否保留 `instrument`，应给 `balanced` 显式设置相同 epoch 上限。

`noid-marketrel` 是 `noid` 的市场相对特征对照：默认 `BDC_USE_MARKET_RELATIVE_FEATURES=1`，追加 8 个 `mkt_rel_*` 特征，用于判断个股是否强于同日市场整体。它不启用 rank，也不改模型结构。

`noid-rank` 是 `noid` 的横截面特征对照：默认 `BDC_USE_CROSS_SECTIONAL_RANKS=1`，会把若干收益、量能、波动和技术指标转成当日百分位排名特征，新增列名以 `_cs_rank` 结尾。它用于判断相对强弱特征是否能提升泛化。

`noid-rank-lite` 是 `noid-rank-replace` 的小范围版本：默认 `BDC_CROSS_SECTIONAL_RANK_MODE=replace`、`BDC_CROSS_SECTIONAL_RANK_REPLACE_SET=lite`，只替换开高低收、成交量、成交额和涨跌额 7 个原始价量尺度列。

`noid-rank-replace` 是 `noid-rank` 之后的更严格对照：默认 `BDC_CROSS_SECTIONAL_RANK_MODE=replace`，只用 rank 列替换部分绝对量价尺度特征，不追加额外输入维度。`v1.4.7/8` 公平预算复核后，它是当前最强单模型候选，也是默认提交配置。

`noid-rank-sharp` 基于 `noid-rank-replace`，只设置 `BDC_LOSS_TARGET_TEMPERATURE=0.05`。它不改特征、不改模型、不改训练预算，目的是让 listwise loss 中的真实收益分布不再接近均匀，从而强化“真实高收益股票应排到前面”的监督信号。它应先跑最近 3 窗口；如果均值、最差窗口、top20/top50 后验收益没有接近或超过 `noid-rank-replace`，不要扩到 24 窗口。

`v1.5.0 noid-rank-sharp` 的 3 窗口短测已经跑完，均值约 `-0.024860`，3/3 个窗口都没有胜出，说明这条线暂时不值得扩跑。

`noid-rank-trendq` 基于 `noid-rank-replace`，追加 5 个趋势质量特征：波动调整动量、20 日区间位置、20 日回撤和 10 日上涨占比。它用于测试“上涨质量”能否改善 rank-replace 的 top20/top50 排序，不改变模型结构。`v1.4.9` 最新 3 窗口短测均值约 `-0.067912`，相对同日期 `v1.4.5 noid-rank-replace` 平均低约 `0.048780`，不建议扩跑。

`noid-stable` 是 `noid` 的正则化对照：继续移除 `instrument`，同时设置 `dropout=0.2`、`weight_decay=1e-4`。它用于检查 `noid` 倾向高波动股票的问题是否能通过更强正则化缓解。

`noid-full` 是 `noid` 的完整数据对照：继续移除 `instrument`，保持 39 特征和小模型，只取消 `train_target_days` 与 `max_stocks_per_day` 限制。它用于判断更多训练数据是否能提升泛化，不用于同时比较 158+39 特征或更大模型。

`noid-lowvol` 是 `noid` 的预测后处理对照：训练配置与 `noid` 相同，但预测时设置 `BDC_SELECTION_STRATEGY=low_vol_then_rank_top5`，先取模型分数前 `BDC_STAGE2_POOL_SIZE=10` 只作为候选池，再按最近 20 个交易日历史波动率从低到高选回 5 只。它不会改变模型训练，只用于验证低波动二阶段过滤是否真的有用。

`ensemble-lowvol` 不训练新模型，默认通过 `BDC_ENSEMBLE_SOURCES=v1.2.13,v1.3.2` 复用两个已有实验的窗口模型。每个源模型先各自输出 top5，再取两者 top5 并集，最后在并集内按最近 20 个交易日历史波动率从低到高选 5 只。它验证的是 `v1.3.7` 诊断中的“两模型 top5 并集低波动重排”，不是 `v1.3.8` 的单模型 top10 低波动策略。

`smooth` 不是新模型，只是 `balanced` 的优化器对照：默认 `BDC_OPTIMIZER=lookahead`、`BDC_LOOKAHEAD_K=5`、`BDC_LOOKAHEAD_ALPHA=0.5`。它用于判断 Lookahead 是否能降低分数震荡和改善最差窗口。

`stable` 不是新架构，只是 `balanced` 的正则化对照：默认 3 个窗口、`dropout=0.2`、`weight_decay=1e-4`。它和 `balanced` 使用相同的模型规模、epoch 上限和早停耐心值，方便只比较正则化差异。如果它比同窗口 `balanced` 更稳，说明后续可以继续沿正则化方向调；如果没有改善，就不要继续在 dropout 上耗时间。

## 5. 下一轮推荐对照

`v1.2.15 balanced` 与 `v1.2.13 noid` 的 12 窗口对照显示，移除 `instrument` 后均值和多数窗口表现更好。两者 epoch 上限不同，但最佳 epoch 都没有超过 15，因此结论仍有参考价值；若后续重新研究 `instrument`，应使用当前统一预算重跑。`v1.3.0 noid-rank` 最近 3 窗口均值明显变差，因此不要直接扩到 12 窗口。

`v1.3.2 noid-rank-replace` 的 12 窗口和 noid 基本打平；后续 `v1.4.5/v1.4.7/v1.4.8` 的 24 窗口公平复核后，`noid-rank-replace` 已成为当前默认提交单模型。如果只是复核已有方向，使用新的版本号运行，不要复用已存在的实验目录。

`v1.3.3 noid-marketrel` 已跑完最近 3 窗口，均值约 `-0.036034`，全弱于同窗口 noid，不建议扩跑 12 窗口。

`v1.3.4 noid-rank-lite` 已跑完最近 3 窗口，均值约 `-0.018936`，只在 `1/3` 个窗口好于 noid；它比 marketrel 好，但仍弱于同窗口 noid 和 22 列 `noid-rank-replace`，不建议扩跑 12 窗口。

如果继续 rank 方向，不要再简单拆替换列或扩展追加式 rank；应先明确新的金融假设，否则容易只是改变选股池但降低外部窗口收益。

比较时优先看：

- `summary.csv` 的多窗口均值、最差窗口和耗时；
- `manifest.json` 里的窗口日期是否一致；
- `training_history.csv` 中 train loss、eval loss、eval final_score 是否出现明显过拟合或平台；
- `prediction_diagnostics_summary.csv` 的 topK 后验收益和模型分数/真实收益相关性；
- `prediction_repeated_stocks.csv` 中 top20/top50 是否仍长期重复。

## 6. 正式调参运行

默认跑 `balanced`，并在最后训练一次最终模型、生成最终预测：

```bash
sh tune.sh v1.2.0
```

常用参数：

```bash
sh tune.sh v1.2.0 --windows 5
sh tune.sh v1.2.0 --windows 5 --step-days 5
sh tune.sh v1.2.0 --data-file data/stock_data.csv
sh tune.sh v1.3.3 noid-marketrel --windows 3 --skip-final
sh tune.sh v1.3.0 noid-rank --windows 3 --skip-final
sh tune.sh v1.3.4 noid-rank-lite --windows 3 --skip-final
sh tune.sh v1.3.1 noid-rank-replace --windows 3 --skip-final
sh tune.sh v1.5.0 noid-rank-sharp --windows 3 --skip-final
sh tune.sh v1.4.9 noid-rank-trendq --windows 3 --skip-final
sh tune.sh v1.2.9 noid-stable --skip-final
sh tune.sh v1.2.14 noid-full --windows 3 --skip-final
sh tune.sh v1.3.8 noid-lowvol --windows 12 --skip-final --reuse-models-from v1.2.13
sh tune.sh v1.2.0 large --windows 3
sh tune.sh v1.2.0 full --windows 3
```

已有实验补完预测诊断后，可以比较候选池：

```bash
.venv/bin/python code/src/compare_candidate_pools.py experiments/v1.2.13 experiments/v1.3.2
```

输出默认在 `experiments/analysis/` 下，只用于本地复盘，不影响提交结果。

比较两个及以上实验时，还会生成二阶段重排诊断：

```text
stage2_rerank_trials.csv
stage2_rerank_windows.csv
stage2_rerank_candidates.csv
```

这个诊断会把多个模型的 top5 并集当候选池，再测试“高波动惩罚、近期过热惩罚、回撤过滤、前缀集中度限制”等规则能否从并集里选回更好的 5 只股票。风险信号只来自窗口训练截止日以前的数据，目标窗口真实收益只用于事后评分。

当前 `v1.2.13 noid` 与 `v1.3.2 noid-rank-replace` 的 12 窗口复盘中，`low_vol_then_rank_top5` 在“两模型 top5 并集”里暂时最有希望；综合风险分和前缀限额表现较差。从 `v1.3.8` 起，它已作为可选预测后处理接入，但默认不启用。

如果要比较多个完整实验的总体表现，可以用验证汇总脚本：

```bash
.venv/bin/python code/src/validate_experiments.py experiments/v1.4.1 experiments/v1.4.2 experiments/v1.4.3 --ensemble-label v1.4.3
```

默认输出到：

```text
experiments/analysis/validation_v1.4.1_vs_v1.4.2_vs_v1.4.3/
```

重点看：

- `summary.csv`：均值、中位数、标准差、最差窗口、正分窗口数、相对市场均值；
- `paired_diffs.csv`：集成逐窗口相对源模型的差值和胜负；
- `diagnostic_summary.csv`：Spearman、top5/top20/top50 后验收益、真实 top5 命中数、重复股票数量；
- `readme.md`：自动生成的简短中文结论。

如果只想复用已有模型、单独评估后处理策略，可以让新实验目录复用旧模型：

```bash
sh tune.sh v1.3.8 noid-lowvol --windows 12 --skip-final --reuse-models-from v1.2.13
```

这会校验窗口日期一致，然后跳过训练，只重跑当前版本的预测和评分。

`v1.3.8 noid-lowvol` 复用 `v1.2.13` 模型后的 12 窗口均值约 `0.018015`，低于原 noid 的 `0.024783`，最差窗口也更差。因此单模型 top10 内低波动优先暂不建议作为默认策略。

下一步用 `ensemble-lowvol` 验证 `v1.3.7` 的诊断假设是否能落地到实际 walk-forward 输出：

```bash
sh tune.sh v1.3.9 ensemble-lowvol --windows 12 --skip-final
```

这个命令默认复用 `v1.2.13 noid` 和 `v1.3.2 noid-rank-replace` 的已训练窗口模型，不重新训练。

`v1.3.9 ensemble-lowvol` 已完成 12 窗口验证，均值约 `0.030457`，最差窗口约 `-0.032689`，高于 `v1.2.13 noid` 的均值 `0.024783` 和最差窗口 `-0.063489`。这说明“两模型 top5 并集 + 低波动重排”比单模型低波动后处理更有希望。

注意：该 profile 只是复用已有源模型。如果不加 `--skip-final`，最终预测阶段需要 `v1.2.13` 和 `v1.3.2` 都存在 `final/model/best_model.pth` 与 `scaler.pkl`。当前本机这两个源实验只有窗口模型，没有最终模型，因此 `ensemble-lowvol` 暂时用于历史验证；正式提交前要先补齐源实验最终模型。

`v1.4.1`、`v1.4.2`、`v1.4.3` 已补跑 18 窗口扩展验证：

```bash
sh tune.sh v1.4.1 noid --windows 18 --skip-final
sh tune.sh v1.4.2 noid-rank-replace --windows 18 --skip-final
BDC_ENSEMBLE_SOURCES=v1.4.1,v1.4.2 sh tune.sh v1.4.3 ensemble-lowvol --windows 18 --skip-final
```

18 窗口下 `ensemble-lowvol` 均值约 `0.017967`，高于 noid 的 `0.007921`，但低于 rank-replace 的 `0.020982`；最差窗口约 `-0.045579`，优于两个源模型。结论是：它仍可作为防守型候选，但还不能视为已证明的默认提交主线。

使用更新到 `2026-07-27` 的 `stock_data` 后，`v1.4.4`、`v1.4.5`、`v1.4.6` 补跑了 24 窗口：

```bash
sh tune.sh v1.4.4 noid --windows 24 --skip-final
sh tune.sh v1.4.5 noid-rank-replace --windows 24 --skip-final
BDC_ENSEMBLE_SOURCES=v1.4.4,v1.4.5 sh tune.sh v1.4.6 ensemble-lowvol --windows 24 --skip-final
```

24 窗口下 `rank-replace` 均值约 `0.017557`，`ensemble-lowvol` 约 `0.014062`，`noid` 约 `0.002716`。最新 2 个窗口三者都为负，说明近期验证段更难；当前不建议因为 12 窗口结果就把 `ensemble-lowvol` 当成已证明默认提交主线。

可比性复核：`v1.4.4 noid` 使用旧 profile，epoch 上限为 15；`v1.4.5 noid-rank-replace` 为 30，其中有窗口最佳 epoch 到 16。因此 `v1.4.4` 与 `v1.4.5` 不能视为完全严格对照。从 `v1.4.7` 起，`noid`、`noid-stable`、`noid-lowvol` 统一改为 30 epoch 上限，正式集成源模型里的 noid 也同步改为 30。

已补这个公平对照：

```bash
sh tune.sh v1.4.7 noid --windows 24 --skip-final
BDC_ENSEMBLE_SOURCES=v1.4.7,v1.4.5 sh tune.sh v1.4.8 ensemble-lowvol --windows 24 --skip-final
.venv/bin/python code/src/validate_experiments.py experiments/v1.4.7 experiments/v1.4.5 experiments/v1.4.8 --ensemble-label v1.4.8
```

结果显示 `v1.4.7 noid` 与旧 `v1.4.4` 逐窗口完全一致，`v1.4.8 ensemble-lowvol` 与旧 `v1.4.6` 也完全一致。24 窗口公平预算下，`rank-replace` 均值约 `0.017557`，高于 noid 的 `0.002716` 和 ensemble 的 `0.014062`。当前应把 `rank-replace` 作为最强单模型候选，`ensemble-lowvol` 只作为防守候选。

比较多个实验时，`validate_experiments.py` 会额外输出 `config_comparison.csv`，并在 `readme.md` 里提示关键训练预算是否不一致。若 `BDC_NUM_EPOCHS`、训练目标日、每日股票数、模型规模等不同，应先把结果当作参考，避免直接下结论。

下一步推荐先做小步特征改动，并和 `v1.4.5 noid-rank-replace` 做 24 窗口对照。不要再优先调低波动重排，因为两模型 top5 并集有多样性，但当前重排规则没超过 rank-replace。

新增标准档位时，只需要在 `tune.sh` 的 `case "$profile" in` 配置区增加一个分支，并用非 `--` 形式调用，例如 `sh tune.sh v1.3.0 my-profile --skip-final`。如果要新增 `--my-profile` 这类别名，才需要额外改上方参数解析。

`manifest.json` 中 `tune_env` 记录的是 profile 写入的环境变量；如果命令行传了 `--windows 12` 这类参数，实际生效值看 `walk_forward_args` 和 `windows` 列表。`summary.csv` 中的行数也是实际完成窗口数。

默认不覆盖正式提交文件。最终预测会保存在：

```text
experiments/v1.2.0/final/result.csv
```

确认这个版本就是要提交的结果后，再显式发布：

```bash
sh tune.sh v1.2.0 --resume --publish-final
```

这会把 `experiments/v1.2.0/final/result.csv` 复制到 `output/result.csv`。

如果代码已经提交，并且希望流程完成后自动创建本地 Git tag：

```bash
sh tune.sh v1.2.0 --resume --create-tag
```

`--create-tag` 要求工作区没有未提交改动，避免 tag 指向的代码和实际实验代码不一致。

## 7. 产物位置

每个版本都有独立目录：

```text
experiments/v1.2.0/
  manifest.json
  summary.csv
  experiment_note.md
  prediction_diagnostics_summary.csv
  prediction_repeated_stocks.csv
  prediction_diagnostics.json
  walk_forward.log
  windows/
    window_01/
      metadata.json
      prediction.csv
      prediction_scores.csv
      prediction_diagnostics.csv
      prediction_diagnostics.json
      score.json
      model/
        best_model.pth
        scaler.pkl
        config.json
        final_score.txt
  final/
    result.csv
    result_scores.csv
    model/
      best_model.pth
      scaler.pkl
      config.json
      final_score.txt
```

重点看：

- `summary.csv`：所有窗口的分数汇总；
- `manifest.json`：版本、Git commit、数据文件、窗口计划；
- `experiment_note.md`：本次实验的本地简要说明，包含 profile、关键配置、窗口分数和耗时；该文件在 `experiments/` 下，不提交到 Git；
- `windows/*/metadata.json`：窗口元数据，`target_trading_dates` 是实际验证日期，`target_calendar_span_days` 应为 5；
- `windows/*/prediction_scores.csv`：完整候选股票排名和模型分数，用于排查固定选股池；
- `windows/*/prediction_diagnostics.csv`：完整候选排名加目标窗口真实收益、真实收益排名和得分贡献；
- `windows/*/prediction_diagnostics.json`：窗口级 topK 后验收益、真实 top5 命中情况和排序相关性；
- `prediction_diagnostics_summary.csv`：实验级窗口诊断汇总；
- `prediction_repeated_stocks.csv`：实验级重复高分股票统计；
- `prediction_diagnostics.json`：实验级诊断概要，也会写入 `manifest.json`；
- `windows/*/model/final_score.txt`：该窗口训练早停位置、最佳 epoch 和最佳内部验证分数；
- `windows/*/model/training_history.csv`：逐 epoch 训练/验证 loss、final_score、学习率、梯度范数和耗时；
- `windows/*/score.json`：每个窗口选中的股票、权重和真实收益；
- `final/result.csv`：该版本最终预测文件。
- `final/result_scores.csv`：最终预测的完整候选排名诊断文件，不是提交文件。

## 8. 注意事项

- 单个窗口分数波动很大，调参时优先看多个窗口平均值。
- 窗口越多越稳，但训练次数越多，耗时近似按窗口数线性增加。
- `experiments/` 下模型和日志默认不提交到 Git，避免仓库过大。
- 语义版本号必须是 `vMAJOR.MINOR.PATCH`，例如 `v1.0.0`、`v1.3.10`；每一段都可以是多位数字。
