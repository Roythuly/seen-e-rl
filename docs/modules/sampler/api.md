# Sampler API

## Public APIs

- `EnvFactory.create(env_spec, seed=None)`
- `EnvAdapter.reset()`
- `EnvAdapter.step(action)`
- `ActorHandle.get_action(observation_batch, policy_version=None)`
- `RolloutWorker.collect(...)`
- `TrajectoryAssembler.build(records)`

## 接口语义

- `ActorHandle` 是 sampler 读取当前策略的唯一入口
- sampler 支持两种输出模式：trajectory 和 replay record
- sampler 通过 `RuntimeLoop` 被调度，不自行决定何时训练
- sampler 未来即使拆成异步推理/采样端，也仍然只依赖 `ActorHandle`、`ReplayBuffer` 和 runtime schedule 语义
