# Trainer API

## Public APIs

- `RuntimeLoop.run(schedule, actor_handle, sampler, trainer, info)`
- `Algorithm.update_from_rollout(batch)`
- `Algorithm.update_from_replay(batch)`
- `ReplayBuffer.write(record)`
- `ReplayBuffer.sample(spec)`
- `Learner.publish_policy()`

## 接口语义

- `RuntimeLoop` 决定采样边界、更新时机和策略发布时间
- `RuntimeLoop` 是统一 on-policy / off-policy 的核心抽象，而不是某个算法的私有 helper
- `Algorithm` 只关注 batch -> update 的转换
- `ReplayBuffer` 是 off-policy 和未来异步 actor-learner 的共同桥梁
