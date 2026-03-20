# Evaluator Overview

## 定位

`evaluator` 负责在训练闭环之外加载策略、重置环境、按多个 seed 执行评测，并输出可归档的结果。

## 边界

- 输入：`CheckpointManifest`、`EnvSpec`、seed suite
- 输出：`EvalReport`
- 不负责：训练参数更新
- 评测对象是 checkpoint artifact，不是裸 `policy_version`
