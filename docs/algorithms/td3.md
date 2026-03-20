# TD3

## 依赖字段

### sampler 原生字段

- 与 `SAC` 相同的 `ReplayBatch` 基本字段：
  `observations`、`actions`、`rewards`、`next_observations`、`terminated`、`truncated`、`policy_version`

### trainer / learner 维护状态

- deterministic actor 参数
- twin critic 参数
- target networks
- optimizer states

## 依赖 model outputs

- `TrainOutputs.policy.actions`
- `TrainOutputs.q.online`
- `TrainOutputs.q.target`
- `TrainOutputs.aux.target_policy_smoothing`

## 调度方式

- 通过 replay buffer 驱动
- `UpdateSchedule` 必须显式写出 `policy_delay`
- learner 需要 target policy smoothing 与 noise clipping 语义
- `PublishSchedule` 应与 actor update 节奏一致，而不是与 critic update 节奏一致
- checkpoint 必须覆盖 actor、critic、target networks 与 optimizer states

## 接 `info` / `evaluator`

- 重点记录 actor/critic 更新比、target update 统计、buffer 状态
- 记录 publish cadence 与 actor update cadence 是否一致
- evaluator 默认评测 checkpoint artifact，并显式记录 selector
