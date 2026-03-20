# Trainer Contracts

## Owned

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
- `PolicySnapshot` 必须带版本信息
- `ReplayBatch` 需要支持 nested observations
