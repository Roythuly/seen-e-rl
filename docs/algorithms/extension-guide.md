# Algorithm Extension Guide

## 新增算法时必须补齐

1. 新增算法说明文档
2. 指明使用 `TrajectoryBatch` 还是 `ReplayBatch`
3. 列出所需 model outputs
4. 说明 `RuntimeLoop` 调度依赖
5. 说明 `info` 指标与 evaluator 触发策略

## 何时需要新 contract

- 仅当现有 `TrajectoryBatch` / `ReplayBatch` 无法表达算法必需数据时
- 优先 add-only 扩展现有 schema
- 破坏性变更先写 RFC
