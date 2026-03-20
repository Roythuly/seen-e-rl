# Info Contracts

## Owned

- `MetricEvent`

## Consumed

- `UpdateResult`
- `PolicySnapshot`
- `EvalReport`
- `CheckpointManifest`

## 关键约束

- 指标事件必须能表达 algorithm、backend、env、policy_version
- checkpoint 和 evaluation 结果应有统一关联键
