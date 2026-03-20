# Trainer API

## Public APIs

- `RuntimeLoop.run(runtime_spec, actor_handle, sampler, learner, info)`
- `Learner.update(batch, objective=None)`
- `ReplayBuffer.write(record)`
- `ReplayBuffer.sample(spec)`
- `Learner.publish_policy()`
- `Learner.save_checkpoint()`

## 接口语义

- `RuntimeLoop` 决定采样边界、更新时机和策略发布时间
- `RuntimeLoop` 是统一 on-policy / off-policy 的核心抽象，而不是某个算法的私有 helper
- `Learner.update(...)` 是 trainer 侧统一训练入口
- `Model.forward_train(...)` 是 model 侧统一训练前向入口
- `ReplayBuffer` 是 off-policy 和未来异步 actor-learner 的共同桥梁
- `publish_policy()` 发布的是推理态 `PolicySnapshot`
- `save_checkpoint()` 保存的是训练态 `CheckpointManifest`
