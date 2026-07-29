# 验证指标规范

本文规定后续实验复盘的统一指标口径。目标是避免每轮实验临时挑指标，也避免把单个最差窗口当成唯一判断标准。

## 基本原则

最终提交只预测一个未来 5 交易日窗口，真实目标应接近期望得分最大化。历史 walk-forward 只是对未来分布的近似抽样，因此：

- `mean_score` 是主指标，近似衡量期望得分；
- 风险指标用于避免收益只靠少数窗口撑起来；
- 单个 `min_score` 重要但噪声很大，只作为警报，不再一票否决；
- 所有比较优先用同日期、同窗口、同数据范围的 paired diff。

## 主目标指标

- `mean_score`：跨窗口平均得分，主目标。
- `paired_mean_diff`：候选策略逐窗口减去主对照后的平均差值，优先于直接比较两个均值。
- `median_diff_vs_primary`：配对差值中位数，用于观察改善是否普遍。
- `win_windows_vs_primary` / `loss_windows_vs_primary` / `tie_windows_vs_primary`：配对胜负窗口数。
- `win_rate_vs_primary`：胜出窗口数除以总窗口数。
- `active_win_rate_vs_primary`：胜出窗口数除以有变化窗口数，适合稀疏门控规则。
- `non_loss_rate_vs_primary`：胜出或打平窗口数除以总窗口数。

## 风险指标

- `std_score`：窗口得分标准差。
- `risk_adjusted_score`：`mean_score - 0.5 * std_score`，用于快速看收益和波动的折中。
- `worst3_mean_score`：最差 3 个窗口均值，比单个最差值更稳。
- `cvar_20_score`：最差 20% 窗口均值，窗口数变化时比固定 worst3 更自然。
- `min_score`：单个最差窗口，只作为警报。
- `top2_positive_diff_share`：最大两个正向改善占全部正向改善的比例。数值太高说明均值可能被少数窗口拉高。

## 门控专用指标

门控策略的核心不是单纯提高平均分，而是判断何时切防守。

- `bad_window_count`：主对照负分窗口数，默认 `primary_score < 0`。
- `very_bad_window_count`：主对照严重负分窗口数，默认 `primary_score <= -0.03`。
- `bad_window_recall`：主对照负分窗口里，被门控切防守的比例。
- `bad_window_precision`：切防守的窗口里，主对照确实负分的比例。
- `very_bad_window_recall`：严重负分窗口里，被门控切防守的比例。
- `switched_window_mean_diff`：所有切防守窗口的平均配对改善。
- `bad_window_switch_mean_diff`：主对照负分且切防守窗口的平均配对改善。
- `rescued_score_mean`：主对照负分、切防守且实际改善窗口的平均改善。
- `hurt_score_mean`：切防守但实际变差窗口的平均损失。
- `high_window_false_switch_rate`：主对照高分窗口中被切防守的比例，默认高分为 `primary_score > 0.03`。

这些指标中的 `primary_score`、`lgbm_score`、`defense_score` 只能用于离线评分，不能作为线上预测输入。

## 三层验收

### 探索通过

用于判断方向是否值得继续，不代表可提交：

- `paired_mean_diff > 0`；
- `top2_positive_diff_share <= 0.70`。

### 候选通过

用于判断是否值得接入为可选 profile 或继续 12/24 窗口复核：

- 满足探索通过；
- `active_win_rate_vs_primary >= 0.50`；
- 正分窗口数不低于主对照；
- `worst3_mean_score` 不比主对照低超过 `0.005`；
- `cvar_20_score` 不比主对照低超过 `0.005`；
- `top2_positive_diff_share <= 0.60`。

### 默认候选通过

用于判断是否有资格另起版本接入正式预测路径验证：

- 满足候选通过；
- `worst3_mean_score` 不低于主对照；
- `cvar_20_score` 不低于主对照；
- `high_window_false_switch_rate <= 0.30`。

即使通过默认候选，也不能直接改最终提交默认。还必须另起版本接入 `train.sh/test.sh` 或对应可复现入口，并用 walk-forward 复现。

## 使用建议

- 6 窗口短测只看方向，不要因为单次短测失败否定整个假设。
- 12 窗口用于排除近期窗口偶然性。
- 24 窗口用于判断是否保留候选或接入默认候选。
- 当 `mean_score` 提升但 `min_score` 不变时，不应直接判失败，应继续看 `worst3_mean_score`、`cvar_20_score`、配对胜率和改善集中度。

