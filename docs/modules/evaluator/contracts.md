# Evaluator Contracts

## Owned

- `EvalReport`

## Consumed

- `EnvSpec`
- `PolicySnapshot`
- `CheckpointManifest`

## 关键约束

- `EvalReport` 必须能关联到算法、backend、env、policy_version
- `EvalReport` 必须能关联到 `checkpoint_id`
- checkpoint 选择策略需显式记录
- `PolicySnapshot` 只用于说明 checkpoint 对应的 actor 版本，不直接作为评测主输入
