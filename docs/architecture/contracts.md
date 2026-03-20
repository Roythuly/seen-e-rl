# Contracts

## 作用

`contracts/v0/` 用来存放首版跨模块公共契约的 minimum schema。这些 schema 的目的不是一次定义完所有字段，而是先固定：

- 类型名称
- 最小可实现字段
- 关键对象的身份关系
- 所属版本与演进规则

## 首版 contracts 列表

- `backend_spec.schema.json`
- `observation_spec.schema.json`
- `observation_batch.schema.json`
- `model_spec.schema.json`
- `env_spec.schema.json`
- `runtime_spec.schema.json`
- `collection_schedule.schema.json`
- `update_schedule.schema.json`
- `publish_schedule.schema.json`
- `trajectory_batch.schema.json`
- `replay_record.schema.json`
- `replay_batch.schema.json`
- `replay_buffer_spec.schema.json`
- `policy_snapshot.schema.json`
- `update_result.schema.json`
- `metric_event.schema.json`
- `eval_report.schema.json`
- `checkpoint_manifest.schema.json`
- `error_code.schema.json`

## 演进规则

- 默认 add-only
- 破坏性变更必须先写 RFC
- 文档中的接口语义与 schema 命名保持一致
- 模块内部私有结构不进入 `contracts/`
- `minimum` contract 至少要能支撑本仓库已声明的算法文档与配置示例

## 文档映射

- `model` 依赖 `BackendSpec`、`ModelSpec`、`ObservationSpec`
- `sampler` 依赖 `EnvSpec`、`ObservationBatch`、`TrajectoryBatch`、`ReplayRecord`
- `runtime` 依赖 `RuntimeSpec`、`CollectionSchedule`、`UpdateSchedule`、`PublishSchedule`
- `trainer` 依赖 `ReplayBatch`、`ReplayBufferSpec`、`UpdateResult`、`PolicySnapshot`、`CheckpointManifest`
- `info` 依赖 `MetricEvent`
- `evaluator` 依赖 `EvalReport`、`CheckpointManifest`

## 当前收敛重点

- `TrajectoryBatch` 必须真实支撑 `PPO`
- `ReplayRecord` / `ReplayBatch` 必须真实支撑 `SAC` / `TD3`
- `PolicySnapshot` 必须表示推理态 actor artifact
- `CheckpointManifest` 必须表示训练态 learner checkpoint
- `MetricEvent`、`EvalReport` 必须共享可关联的主键字段
