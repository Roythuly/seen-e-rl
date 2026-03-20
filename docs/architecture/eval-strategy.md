# Eval Strategy

## 分层策略

- `smoke`：最小关键链路，确保 docs/contracts/templates 没有断裂
- `regression`：后续用于沉淀历史问题与边界条件
- `stability`：后续用于 NaN、seed、buffer starvation、stale policy 等稳定性问题

## 当前阶段目标

`v0.1` 只交付 eval 结构和占位用例，不运行真实训练评测。

## 后续关注点

- `PPO` rollout 批次完整性
- `SAC/TD3` buffer 与策略刷新节奏
- 双 backend 接口一致性
- evaluator 与训练环解耦
