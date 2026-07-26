# 项目文档索引

常用文档已按用途分类：

- 训练与正式预测流程：[docs/workflows/train_predict.md](docs/workflows/train_predict.md)
- walk-forward 调参与版本化运行：[docs/workflows/walk_forward.md](docs/workflows/walk_forward.md)
- 语义化版本和实验产物说明：[docs/experiments/versioning.md](docs/experiments/versioning.md)
- 赛事原始文档：[docs/contest/](docs/contest/)

当前阶段只需要提交 `output/result.csv`。默认训练和预测入口仍是：

```bash
sh train.sh
sh test.sh
```

调参实验使用独立入口，不会默认覆盖正式提交文件：

```bash
sh tune.sh v1.0.0 debug --dry-run
```
