# SAC

## 依赖字段

### sampler 原生字段

- `ReplayBatch.observations`
- `ReplayBatch.actions`
- `ReplayBatch.rewards`
- `ReplayBatch.next_observations`
- `ReplayBatch.terminated`
- `ReplayBatch.truncated`
- `ReplayBatch.policy_version`

### trainer / learner 维护状态

- twin critic 参数
- target critic 参数
- actor 参数
- entropy temperature / `alpha`
- optimizer states

## 依赖 model outputs

- `TrainOutputs.policy.distribution_params`
- `TrainOutputs.q.online`
- `TrainOutputs.q.target`
- `TrainOutputs.aux.alpha`

## 调度方式

- `sampler` 与环境交互并写入 `ReplayBuffer`
- `UpdateSchedule` 需支持 warmup、`min_ready_size` 与“每步交互 + 若干次训练”的同步节奏
- `PublishSchedule` 可配置为每次更新后立刻发布，或每 K 次更新后发布
- `CheckpointManifest` 必须覆盖 actor、critic、target critic、optimizer state 与 entropy state
- `best` checkpoint 选择策略应基于稳定评测，而不是单次最新 publish

## 接 `info` / `evaluator`

- 重点记录 buffer fill ratio、sample age、critic loss、alpha 相关指标
- 记录 actor publish 与 checkpoint save 的解耦关系
- evaluator 默认按最新稳定 checkpoint 评测
