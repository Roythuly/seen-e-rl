# Trainer Contracts

## Owned

- `RuntimeSpec`
- `CollectionSchedule`
- `UpdateSchedule`
- `PublishSchedule`
- `ReplayBatch`
- `ReplayBufferSpec`
- `UpdateResult`
- `PolicySnapshot`

## Consumed

- `TrajectoryBatch`
- `ModelSpec`
- `CheckpointManifest`

## 关键约束

- `UpdateResult` 必须可被 `info` 和 `evaluator` 消费
- `UpdateResult` 必须至少带 `run_id`、`policy_version`、`env_steps`、`grad_steps`
- `PolicySnapshot` 必须表示可发布的 actor snapshot，而不是完整 learner state
- `CheckpointManifest` 必须表示可恢复 learner state
- `ReplayBatch` 需要支持 nested observations
