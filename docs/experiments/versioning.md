# 语义化版本与模型速查

本项目用 `vMAJOR.MINOR.PATCH` 记录一次完整实验，例如 `v1.0.0`。

## 1. 版本含义

- `MAJOR`：训练/预测语义发生明显变化，例如标签定义、提交窗口逻辑、模型输入输出改变。
- `MINOR`：增加新能力或较大的调参方案，例如新增特征、模型结构改进、walk-forward 方案变化。
- `PATCH`：小修复或参数微调，例如日志、默认参数、文档修正。

当前默认版本写在根目录：

```text
VERSION
```

## 2. 每个版本应完成的流程

一个可比较的版本至少应完成：

1. 训练：每个 walk-forward 窗口训练一个模型；
2. 验证：每个窗口用之后连续 5 个真实交易日打分；
3. 预测：用最终模型生成该版本的 `final/result.csv`；
4. 记录：保留 `manifest.json` 和 `summary.csv`。

## 3. 快速查看历史

查看代码历史：

```bash
git log --oneline --decorate -10
git show --stat v1.0.0
git diff v1.0.0..HEAD
```

查看本地实验产物：

```bash
ls experiments/v1.0.0
cat experiments/v1.0.0/summary.csv
cat experiments/v1.0.0/manifest.json
```

## 4. Git Tag 规则

当某个版本的代码和实验结果都确认后，可以创建本地 tag：

```bash
git tag -a v1.0.0 -m "v1.0.0"
```

也可以让调参流程在成功后自动创建 tag：

```bash
sh tune.sh v1.0.0 --resume --create-tag
```

如果之后发现版本还没准备好，不要复用旧版本号，建议升到下一个版本，例如 `v1.0.1`。

本项目按你的要求只本地提交，不自动 push。
