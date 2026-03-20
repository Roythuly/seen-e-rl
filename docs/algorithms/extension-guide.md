# Algorithm Extension Guide

## 新增算法时必须补齐

1. 新增算法说明文档
2. 指明使用 `TrajectoryBatch` 还是 `ReplayBatch`
3. 列出 sampler 原生字段与 trainer/learner 派生字段
4. 列出所需 model outputs 与 `forward_train(...)` 返回子域
5. 说明 `RuntimeLoop` 调度依赖
6. 说明 `PublishSchedule` 与 checkpoint/eval 触发策略
7. 说明 `info` 指标与 evaluator 触发策略

## 何时需要新 contract

- 仅当现有 `TrajectoryBatch` / `ReplayBatch` 无法表达算法必需数据时
- 优先 add-only 扩展现有 schema
- 破坏性变更先写 RFC
- 只有当 `RuntimeSpec` 无法表达调度需求时，才考虑新增 runtime contract
