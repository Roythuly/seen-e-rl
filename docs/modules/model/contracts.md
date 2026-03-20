# Model Contracts

## Owned

- `ModelSpec`
- `BackendSpec`

## Consumed

- `ObservationSpec`
- `ObservationBatch`
- `CheckpointManifest`

## 关键约束

- `ModelSpec` 需要显式描述 encoder、actor head、critic head
- `BackendSpec` 只描述后端能力与名称，不直接携带业务逻辑
