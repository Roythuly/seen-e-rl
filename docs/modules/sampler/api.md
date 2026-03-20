# Sampler API

## Public APIs

- `EnvFactory.create(env_spec, seed=None)`
- `EnvAdapter.reset()`
- `EnvAdapter.step(action)`
- `ActorHandle.act(observation_batch, policy_version=None)`
- `RolloutWorker.collect(...)`
- `TrajectoryAssembler.build(records)`

## 接口语义

- `ActorHandle` 是 sampler 读取当前策略的唯一入口
- `ActorHandle.act(...)` 至少返回 `action` 与 `policy_version`
- `ActorHandle.act(...)` 的返回类型记为 `ActionOutput`
- sampler 支持两种输出模式：trajectory 和 replay record
- sampler 通过 `RuntimeLoop` 被调度，不自行决定何时训练
- sampler 未来即使拆成异步推理/采样端，也仍然只依赖 `ActorHandle`、`ReplayBuffer` 和 runtime schedule 语义
- PPO 路径必须保留 rollout 所需的 `log_prob` 与 `value_estimate`
- SAC/TD3 路径必须保留 transition core 与 `policy_version`
