# 策略门控诊断

本文记录“什么时候保留 Transformer，什么时候切防守策略”的离线诊断方法。门控只应使用预测日前已经可见的信息，不能把目标窗口真实收益作为线上输入。

## 目标

当前主线 `v1.4.5 rank-replace Transformer` 的特点是：

- 高分窗口收益很高；
- 低分窗口回撤明显；
- `v1.12.0 LightGBM` 和 `v1.12.1 ensemble-lowoverheat` 在部分低分窗口更防守，但会削弱 Transformer 的高分窗口。

因此不能简单把防守模型替换为默认模型。更合理的问题是：能否用预测日前历史信号判断某个窗口更适合进攻还是防守。

## v1.12.2 第一版脚本

脚本：

```bash
.venv/bin/python code/src/evaluate_strategy_gate.py \
  experiments/v1.4.5 \
  experiments/v1.12.0 \
  experiments/v1.12.1 \
  --output-dir experiments/analysis/strategy_gate_v1.4.5_vs_v1.12.0_vs_v1.12.1
```

输入：

- 进攻主线：`experiments/v1.4.5`；
- 防守候选 1：`experiments/v1.12.0`；
- 防守候选 2：`experiments/v1.12.1`。

输出：

- `gate_features.csv`：每个窗口的门控信号和事后得分；
- `gate_rule_summary.csv`：每条规则的跨窗口均值、最差窗口、正分窗口、误切高分窗口等指标；
- `gate_window_scores.csv`：每条规则在每个窗口选择了哪个策略；
- `readme.md`：中文摘要。

## 门控信号

第一版只用简单规则，不训练门控模型：

- `tf_top5_overheat_mean`：Transformer top5 的短期过热 rank 均值；
- `tf_top5_volatility_mean`：Transformer top5 的近 20 日波动 rank 均值；
- `tf_top5_drawdown_mean`：Transformer top5 的近 20 日回撤 rank 均值；
- `tf_lgbm_top5_overlap` / `tf_lgbm_top20_overlap`：Transformer 与 LightGBM 的候选重合数；
- `tf_top5_score_gap` / `tf_top20_score_gap`：Transformer 分数断层，只作诊断；
- `market_return_5`、`market_up_ratio_5`、`market_volatility_mean_5`：预测日前近 5 个交易日的历史市场状态。

注意：`primary_score`、`lgbm_score`、`defense_score` 只用于事后评分，不能进入线上门控规则。

## v1.12.2 结果

24 窗口离线诊断结果：

- Transformer 基线：均值约 `0.017557`，最差约 `-0.066987`，正分窗口 `14/24`；
- 最好门控：`v1.12.1 / gate_tf_overheat_ge_0p7`，均值约 `0.029874`，最差仍为 `-0.066987`，正分窗口 `17/24`；
- 该规则在高分窗口切防守比例约 `0.300`，最大两个正向改善贡献占比约 `0.520`；
- 没有规则同时满足“均值不低于 Transformer、最差窗口优于 Transformer、正分窗口不低于 Transformer、高分窗口误切可控、不能只靠少数窗口拉高”的完整验收标准。

结论：门控方向有信号，尤其“Transformer top5 短期过热时切 `ensemble-lowoverheat`”明显提高均值和正分窗口数；但第一版没有覆盖最差窗口，暂不接入正式提交。下一步若继续，应围绕“识别最差窗口”做更小的规则，例如把市场弱势信号与过热信号组合，而不是直接把当前最佳均值规则上线。

