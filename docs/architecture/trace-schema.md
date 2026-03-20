# Trace Schema

## 目的

虽然首版不实现 tracing，但需要先固定事件维度，避免未来 `info` 模块和各个运行时组件各写一套日志语义。

## 核心事件类型

- `rollout_started`
- `rollout_finished`
- `buffer_write`
- `buffer_sample`
- `update_started`
- `update_finished`
- `policy_published`
- `checkpoint_saved`
- `evaluation_finished`
- `system_health`

## 事件建议字段

- `timestamp`
- `run_id`
- `module`
- `event_type`
- `event_category`
- `policy_version`
- `checkpoint_id`
- `algorithm`
- `backend`
- `env_id`
- `env_steps`
- `grad_steps`
- `metrics`
- `status`
- `error_code`

## 脱敏与约束

- 不记录原始敏感配置
- 不把大体积 observation 直接写入 trace
- 评测失败样本应通过引用或摘要记录，而不是整包落日志
- 事件 envelope 字段应优先与 `MetricEvent` schema 对齐，而不是各模块各自扩张
