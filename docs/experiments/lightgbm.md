# LightGBM 实验线

## v1.12.0 第一版

`v1.12.0 noid-rank-lgbm` 是第一版 LightGBM 表格模型实验。

目的：

- 暂停继续往 Transformer 输入里追加弱小特征；
- 用更适合表格特征的传统模型测试当前 `rank-replace` 特征是否还有潜力；
- 先做独立 walk-forward 验证，不改正式 `train.sh/test.sh` 默认提交路径。

第一版配置：

- `BDC_MODEL_KIND=lgbm`
- `BDC_FEATURE_NUM=39`
- `BDC_USE_INSTRUMENT_FEATURE=0`
- `BDC_USE_CROSS_SECTIONAL_RANKS=1`
- `BDC_CROSS_SECTIONAL_RANK_MODE=replace`
- `BDC_TRAIN_TARGET_DAYS=120`
- `BDC_MAX_STOCKS_PER_DAY=0`
- `BDC_LGBM_N_ESTIMATORS=300`
- `BDC_LGBM_LEARNING_RATE=0.03`
- `BDC_LGBM_NUM_LEAVES=31`

命令：

```bash
sh tune.sh v1.12.0 noid-rank-lgbm --windows 6 --skip-final
```

6 窗口结果：

- `v1.12.0 noid-rank-lgbm`：均值约 `0.000968`，最差窗口约 `-0.065830`，正分窗口 `3/6`；
- 同日期 `v1.4.5 noid-rank-replace`：均值约 `0.018557`，最差窗口约 `-0.066987`，正分窗口 `3/6`；
- 逐窗口胜负为 `3/6`，但 LGBM 输掉两个高分窗口，所以 top5 均值落后约 `0.017589`；
- LGBM 平均 Spearman 约 `-0.123531`，同日期 rank-replace 约 `-0.138490`；
- LGBM top20 后验收益约 `0.008618`，同日期 rank-replace 约 `-0.010901`；
- LGBM top50 后验收益约 `-0.001600`，同日期 rank-replace 约 `-0.011262`；
- LGBM 的真实 top5 进入预测 top20 的平均命中数约 `1.00`，同日期 rank-replace 约 `0.83`。

24 窗口补跑结果：

- `v1.12.0 noid-rank-lgbm`：均值约 `0.009367`，最差窗口约 `-0.075276`，正分窗口 `16/24`；
- `v1.4.5 noid-rank-replace`：均值约 `0.017557`，最差窗口约 `-0.066987`，正分窗口 `14/24`；
- LGBM 逐窗口 `14/24` 胜出，但在 Transformer 高分窗口明显落后，导致均值仍低；
- 按 Transformer 得分分组看，Transformer 负分窗口里 LGBM 基本胜出，Transformer 高分窗口里 LGBM 大多落后；
- LGBM 平均 top20/top50 后验收益和真实 top5 进入 top20 的命中数仍略好，说明它更像“防守型候选池补充”，不是直接 top5 主模型。

基于 `v1.4.5 + v1.12.0` 的候选池诊断：

- 简单 top5 并集等权均值约 `0.013572`，不如 `v1.4.5`；
- 平均排名、最小排名和 Borda 简单融合也不通过；
- `low_overheat_then_rank_top5` 均值约 `0.025834`，最差窗口约 `-0.060607`，正分窗口 `19/24`；
- 该规则只用预测日前历史价格计算短期过热 rank，没有使用目标窗口未来收益。它已接入 `v1.12.1 ensemble-lowoverheat` 作为可选 profile。

注意：上面是读取旧预测诊断文件的离线候选池试算。`v1.12.1` 真正通过 walk-forward 重新调用两个源模型预测后，结果低于这个离线数值。

结论：

- 第一版 LGBM 不能替换默认提交模型，top5 直接收益还没有过关；
- 训练速度明显更好，6 窗口约 1 分钟完成，适合做快速离线对照；
- broader ranking 诊断不差，说明它适合做候选池补充和二阶段重排输入；
- 下一步优先验证 `ensemble-lowoverheat`，再考虑 `LGBMRanker`。

## v1.12.1 低过热集成 profile

命令：

```bash
sh tune.sh v1.12.1 ensemble-lowoverheat --windows 24 --skip-final
```

默认源模型：

- `v1.4.5`：rank-replace Transformer；
- `v1.12.0`：noid-rank-lgbm。

规则：

1. 两个源模型分别输出 top5；
2. 取 top5 并集；
3. 用预测日前历史数据计算 `overheat_rank`，即最近 5/10 日收益较高者的横截面 rank；
4. 按 `overheat_rank` 从低到高排序，分数接近时用源模型平均排名兜底；
5. 选回 5 只，仍等权满仓。

注意：这仍是候选策略，不自动改正式提交默认路径。若要用于正式提交，需要先保证 `v1.4.5` 与 `v1.12.0` 风格的最终模型都能由 `train.sh/test.sh` 完整复现。

24 窗口 walk-forward 结果：

- `v1.12.1 ensemble-lowoverheat`：均值约 `0.014789`，最差窗口约 `-0.052577`，正分窗口 `16/24`；
- 同期 `v1.4.5 rank-replace`：均值约 `0.017557`，最差窗口约 `-0.066987`，正分窗口 `14/24`；
- 同期 `v1.12.0 LightGBM`：均值约 `0.009367`，最差窗口约 `-0.075276`，正分窗口 `16/24`；
- 相比 `rank-replace`，`v1.12.1` 均值略低，但最差窗口、top20/top50 后验收益和真实 top5 命中数更好；
- 它没有解决“保留 Transformer 高分窗口”的问题，尤其 `2026-06-08` 这类高分窗口会被明显削弱。

结论：`ensemble-lowoverheat` 是防守型候选，不应替换默认提交主线。后续若继续这个问题，重点应从“固定规则重排”转向“何时保留 Transformer、何时启用防守模型”的门控诊断。

注意：它不是和 Transformer 的严格架构对照，因为 LightGBM 第一版使用 120 个训练目标日且不做每日股票抽样。这样做是因为 LightGBM 训练很快，第一步更关心“树模型方向是否值得继续”，而不是只比较同预算下的架构差异。

## v1.12.2 门控诊断

`v1.12.2` 新增 `evaluate_strategy_gate.py`，不训练、不预测，只复用已有 24 窗口结果，评估什么时候从 `v1.4.5 rank-replace Transformer` 切到防守候选：

```bash
.venv/bin/python code/src/evaluate_strategy_gate.py \
  experiments/v1.4.5 \
  experiments/v1.12.0 \
  experiments/v1.12.1 \
  --output-dir experiments/analysis/strategy_gate_v1.4.5_vs_v1.12.0_vs_v1.12.1
```

结果：

- 最好规则是当 Transformer top5 平均过热 rank `>= 0.70` 时切到 `v1.12.1 ensemble-lowoverheat`；
- 24 窗口均值约从 `0.017557` 提升到 `0.029874`；
- 正分窗口从 `14/24` 提升到 `17/24`；
- 但最差窗口仍是 `-0.066987`，没有通过“最差窗口优于主线”的验收标准。

结论：LightGBM 和低过热集成仍有防守价值，但当前更适合作为门控诊断对象。`v1.12.3` 统一指标口径后，`v1.12.1 / gate_tf_overheat_ge_0p7` 在均值、CVaR20、worst3 和高分误切约束下通过默认候选检查；但这仍只是离线诊断，需要另起版本接入 walk-forward 复现，不能直接改正式提交默认。

`v1.12.4` 已新增 `ensemble-gate-overheat` 作为复现入口：

```bash
sh tune.sh v1.12.4 ensemble-gate-overheat --windows 24 --skip-final
```

它仍复用 `v1.4.5 + v1.12.0`，只是把“主模型 top5 过热时切防守”的门控逻辑接入实际 walk-forward 预测流程。

验收：

- 先与同日期 `v1.4.5 noid-rank-replace` 配对比较；
- 6 窗口若均值接近或超过主对照，再扩 12；
- 如果 top20/top50 后验收益、Spearman 或真实 top5 命中数明显更好，即使 top5 均值暂时没赢，也值得继续调 LightGBM；
- 如果 6 窗口明显变差，再考虑减小树模型复杂度或改成 `LGBMRanker`，不要直接接入正式提交。

提交影响：

- 新增依赖 `lightgbm`；
- 新增 `train_lgbm.py`、`predict_lgbm.py`；
- walk-forward 通过 `BDC_MODEL_KIND=lgbm` 调用 LightGBM 入口；
- 当前正式提交默认仍是 `rank-replace` Transformer 单模型。
