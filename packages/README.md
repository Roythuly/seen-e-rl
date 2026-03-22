# Packages

`packages/rl_training_infra` 是当前运行时代码的共享基础层，按“template / base / concrete”组织：

- `common/`：registry、YAML/JSON I/O、后续可扩展的共享工具
- `contracts/`：runtime artifact builders 和 schema validation helpers
- `model/`：模板、Torch model factory、encoder/actor/critic 组合与 checkpoint 接口
- `sampler/`：env adapter、Gym env factory、actor handle、trajectory/replay collectors
- `trainer/`：learner base、runtime loop、replay buffer
- `info/`：metric event builder 与 sinks
- `evaluator/`：checkpoint selection、eval runtime、report writer

设计边界：

- 这里放模块共享能力，不放算法私有损失实现
- 具体算法 learner 位于 `algorithms/ppo`、`algorithms/sac`、`algorithms/td3`
- `trainer` 负责 runtime 协调，不拥有 PPO/SAC/TD3 的私有更新公式
