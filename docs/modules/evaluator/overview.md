# Evaluator Overview

## 定位

`evaluator` 负责在训练闭环之外加载策略、重置环境、按多个 seed 执行评测，并输出可归档的结果。

## 边界

- 输入：checkpoint / policy snapshot、`EnvSpec`、seed plan
- 输出：`EvalReport`
- 不负责：训练参数更新
