# Info Contracts

## Owned

- `MetricEvent`

## Consumed

- `UpdateResult`
- `PolicySnapshot`
- `EvalReport`
- `CheckpointManifest`

## 关键约束

- `MetricEvent` 最少要能表达 `run_id`、`event_type`、`algorithm`、`backend`、`env_id`
- `MetricEvent` 最少要能表达 `policy_version`、`checkpoint_id`、`env_steps`、`grad_steps`
- checkpoint 和 evaluation 结果应有统一关联键
