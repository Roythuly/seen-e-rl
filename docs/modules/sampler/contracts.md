# Sampler Contracts

## Owned

- `TrajectoryBatch`
- `ReplayRecord`

## Consumed

- `EnvSpec`
- `ObservationSpec`
- `ObservationBatch`
- `PolicySnapshot`

## 关键约束

- observation 必须支持 nested dict
- replay record 必须带上 `policy_version`
- trajectory 必须保留回报、终止标记和时间顺序
