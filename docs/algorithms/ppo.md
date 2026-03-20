# PPO

## 依赖字段

### sampler 原生字段

- `TrajectoryBatch.observations`
- `TrajectoryBatch.actions`
- `TrajectoryBatch.rewards`
- `TrajectoryBatch.terminated`
- `TrajectoryBatch.truncated`
- `TrajectoryBatch.log_probs`
- `TrajectoryBatch.value_estimates`
- `TrajectoryBatch.policy_version`

### trainer / learner 派生字段

- `TrajectoryBatch.advantages`
- `TrajectoryBatch.returns`

## 依赖 model outputs

- `TrainOutputs.policy.distribution_params`
- `TrainOutputs.value.state_values`
- 可选 `TrainOutputs.aux.diagnostics`

## 调度方式

- `CollectionSchedule` 必须在 rollout 期间冻结 `PolicySnapshot`
- `CollectionSchedule` 可以定义采样 N 步或采满若干条轨迹
- `UpdateSchedule` 必须支持同一批 rollout 的多轮 epoch/minibatch 更新
- `PublishSchedule` 默认在一轮 PPO 更新完成后发布一次新 actor
- checkpoint 默认跟随更新轮次或 milestone，而不是跟随每个 minibatch

## 接 `info` / `evaluator`

- 记录 rollout reward、KL、entropy、value loss
- 记录 `env_steps`、`grad_steps` 与 publish/checkpoint 事件
- evaluator 默认评测 `latest` 或 `milestone` checkpoint，而不是裸 `policy_version`
