# PPO

## 依赖字段

- `TrajectoryBatch.observations`
- `TrajectoryBatch.actions`
- `TrajectoryBatch.rewards`
- `TrajectoryBatch.returns` 或可推导回报
- `TrajectoryBatch.advantages`
- `TrajectoryBatch.log_probs`

## 依赖 model outputs

- policy logits / distribution params
- value estimates

## 调度方式

- 冻结当前策略版本
- 采样 N 步或多条轨迹
- 对同一批 rollout 做多轮 epoch/minibatch 更新
- 完成后发布新策略版本

## 接 `info` / `evaluator`

- 记录 rollout reward、KL、entropy、value loss
- 更新后评测 latest 或 milestone checkpoint
