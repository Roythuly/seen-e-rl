# Model Contracts

## Owned

- `ModelSpec`
- `BackendSpec`

## Consumed

- `ObservationSpec`
- `ObservationBatch`
- `CheckpointManifest`
- `PolicySnapshot`

## 关键约束

- `ModelSpec` 需要显式描述 encoder、actor head、critic head
- `ModelSpec` 需要显式描述 feature sharing 策略与训练接口模式
- `BackendSpec` 只描述后端能力与名称，不直接携带业务逻辑
- `PolicySnapshot` 与 `CheckpointManifest` 的保存/恢复边界必须一致
