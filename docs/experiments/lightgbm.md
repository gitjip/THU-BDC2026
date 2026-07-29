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

结论：

- 第一版 LGBM 不能替换默认提交模型，top5 直接收益还没有过关；
- 训练速度明显更好，6 窗口约 1 分钟完成，适合做快速离线对照；
- broader ranking 诊断不差，说明它可能适合做候选池补充、二阶段重排输入，或继续尝试 `LGBMRanker`；
- 下一步不要直接扩大到正式提交，应先用 `LGBMRanker` 或 LightGBM/Transformer 候选池并集做小窗口验证。

注意：它不是和 Transformer 的严格架构对照，因为 LightGBM 第一版使用 120 个训练目标日且不做每日股票抽样。这样做是因为 LightGBM 训练很快，第一步更关心“树模型方向是否值得继续”，而不是只比较同预算下的架构差异。

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
