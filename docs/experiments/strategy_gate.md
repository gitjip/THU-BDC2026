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
- `gate_rule_summary.csv`：每条规则的主目标、风险和门控专用指标；
- `gate_window_scores.csv`：每条规则在每个窗口选择了哪个策略；
- `readme.md`：中文摘要。

指标口径统一参考 [evaluation_metrics.md](evaluation_metrics.md)。门控诊断会输出 `passes_exploration_checks`、`passes_candidate_checks` 和 `passes_default_checks` 三层检查结果；通过默认候选检查也只表示“值得另起版本接入复现”，不表示可以直接改提交默认。

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
- 按旧的严格最差窗口口径，没有规则同时满足“均值、最差窗口、正分窗口、高分窗口误切和改善集中度”全部要求。

结论：门控方向有信号，尤其“Transformer top5 短期过热时切 `ensemble-lowoverheat`”明显提高均值和正分窗口数。后续不应继续用单个 `min_score` 一票否决，而应按 `mean_score + paired_mean_diff + worst3/CVaR + 高分误切` 综合判断。若继续门控方向，应围绕“识别低分窗口和保护高分窗口”做更小的规则，例如把市场弱势信号与过热信号组合，而不是直接把当前最佳均值规则上线。

## v1.12.3 指标规范化

`v1.12.3` 不改门控规则，只补充统一指标：

- 主目标：`mean_score`、`paired_mean_diff`、`active_win_rate_vs_primary`；
- 风险：`risk_adjusted_score`、`worst3_mean_score`、`cvar_20_score`、`top2_positive_diff_share`；
- 门控：`bad_window_recall`、`bad_window_precision`、`very_bad_window_recall`、`rescued_score_mean`、`hurt_score_mean`、`high_window_false_switch_rate`；
- 三层检查：探索、候选、默认候选。

按新指标重新生成 `experiments/analysis/strategy_gate_v1.4.5_vs_v1.12.0_vs_v1.12.1/` 后：

- Transformer 基线：均值约 `0.017557`，CVaR20 约 `-0.046114`，worst3 约 `-0.053284`；
- `v1.12.1 / gate_tf_overheat_ge_0p7`：均值约 `0.029874`，CVaR20 约 `-0.038526`，worst3 约 `-0.049893`；
- 负分窗口召回约 `0.400`，切防守精确率约 `0.500`，高分窗口误切比例约 `0.300`；
- 最大两个正向改善贡献占比约 `0.520`；
- 该规则通过探索、候选和默认候选三层指标检查。

这一步的目的不是证明某个规则可提交，而是先固定评估口径。`gate_tf_overheat_ge_0p7` 现在可以视为“值得另起版本接入 walk-forward 复现”的默认候选，但不能直接改正式提交默认。之后无论做市场弱势门控、低过热集成还是 LightGBM 防守，都应使用同一套汇总表判断。

## v1.12.4 walk-forward 复现入口

`v1.12.4` 将 `gate_tf_overheat_ge_0p7` 接入为可选 walk-forward profile：

```bash
sh tune.sh v1.12.4 ensemble-gate-overheat --windows 24 --skip-final
```

默认配置：

- `BDC_ENSEMBLE_SOURCES=v1.4.5,v1.12.0`；
- `BDC_SELECTION_STRATEGY=ensemble_gate_overheat_top5`；
- `BDC_GATE_OVERHEAT_THRESHOLD=0.70`；
- 第一个源实验 `v1.4.5` 是进攻主模型；
- 防守分支等价于 `ensemble_low_overheat_top5`，即两个源模型 top5 并集内按低过热选回 5 只。

规则：

1. 对第一个源模型 top5 计算预测日前历史 `overheat_rank` 均值；
2. 若均值 `>= 0.70`，切到低过热集成防守；
3. 否则保留第一个源模型原始 top5；
4. 最终仍 top5 等权满仓。

复现验收：

- 与 `v1.4.5 rank-replace` 比 `mean_score`、`paired_mean_diff`、`worst3_mean_score`、`cvar_20_score`；
- 与 `v1.12.1 ensemble-lowoverheat` 比高分窗口是否少被削弱；
- 检查 `prediction_scores.csv` 和 `windows/*/ensemble_prediction.json` 中的 `gate_use_defense`、`gate_primary_top5_overheat_mean` 是否符合规则；
- 如果 24 窗口复现仍通过默认候选检查，再讨论是否接入正式提交可选模式。

3 窗口烟测已经跑通：

- 命令：`sh tune.sh v1.12.4 ensemble-gate-overheat --windows 3 --skip-final --resume`；
- 最近 3 个窗口的 `gate_use_defense` 均为 `false`，说明这 3 个窗口没有触发过热门控；
- 3 窗口均值约 `-0.019112`，只用于确认流程可运行，不能判断策略效果；
- 完整复现应继续运行 `sh tune.sh v1.12.4 ensemble-gate-overheat --windows 24 --skip-final --resume`。

24 窗口已完成：

- `v1.12.4 ensemble-gate-overheat`：均值约 `0.018227`，中位数约 `0.021540`，std 约 `0.043023`，worst3 约 `-0.048249`，CVaR20 约 `-0.041475`，最差约 `-0.063170`，正分窗口 `15/24`；
- 旧 `v1.4.5 rank-replace`：均值约 `0.017557`，worst3 约 `-0.053284`，CVaR20 约 `-0.046114`，最差约 `-0.066987`，正分窗口 `14/24`；
- 旧 `v1.12.1 ensemble-lowoverheat`：均值约 `0.014789`，worst3 约 `-0.043938`，CVaR20 约 `-0.037485`，最差约 `-0.052577`，正分窗口 `16/24`。

与旧 `v1.4.5` 配对比较：

- 配对均值差约 `+0.000670`；
- 胜/负/平窗口为 `10/9/5`，active win rate 约 `0.526`；
- `top2_positive_diff_share` 约 `0.488`；
- 旧主线负分窗口召回约 `0.400`，切防守精确率约 `0.571`；
- 旧主线高分窗口误切比例约 `0.200`。

与本次 `v1.12.4` 运行时重跑出来的 primary 源模型比较：

- primary 重跑均值约 `0.009541`，gate 均值约 `0.018227`，配对均值差约 `+0.008686`；
- 胜/负/平窗口为 `6/1/17`；
- 7 个触发窗口中 6 个改善，1 个轻微变差；
- 但 `top2_positive_diff_share` 约 `0.628`，改善集中度偏高。

验收结论：

- `ensemble-gate-overheat` 方向通过 walk-forward 复现的候选检查：均值、worst3、CVaR20 和正分窗口相对旧 `v1.4.5` 都略有改善；
- 但提升幅度很小，且当前代码重跑源模型与旧 `v1.4.5` 保存的预测文件不完全一致，所以不能直接切正式默认；
- 本轮应把它定位为“可保留候选”，而不是最终提交主线；
- 门控仍会漏掉低过热坏窗口，例如最近窗口 `2026-07-20 ~ 2026-07-24` 未触发防守且得分约 `-0.063170`。

同一批 24 窗口上的阈值复盘显示，`BDC_GATE_OVERHEAT_THRESHOLD=0.65` 的事后均值约 `0.020205`，高于 `0.70` 的 `0.018227`。这只说明阈值 `0.65` 值得另起版本做小步复核，不能直接把本次结论改写成“0.65 已证明更好”。

注意：这仍不改变 `train.sh/test.sh` 默认提交路径。正式默认仍是 rank-replace 单模型。
