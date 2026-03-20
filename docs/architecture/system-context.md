# System Context

## 目标

本系统是一个面向 RL 研究迭代的训练 infra。`v0.1` 目标不是直接提供可运行训练系统，而是先把后续实现所需的架构边界、公共接口和交付事实源稳定下来。

## 范围

- 单层策略训练闭环
- 算法中心框架
- `model`、`sampler`、`trainer`、`info`、`evaluator` 五模块
- `PyTorch` / `JAX` 双 backend 设计
- on-policy 与 off-policy 双训练路径
- `ReplayBuffer` 作为未来异步 actor-learner 的桥梁

## 系统主流程

### On-policy

1. `sampler` 使用固定策略版本采集 N 步或多条轨迹
2. `trainer` 基于 `TrajectoryBatch` 做集中更新
3. `trainer` 发布新的 `PolicySnapshot`
4. `info` 记录采样、更新和 checkpoint 事件
5. `evaluator` 在独立环境中评测 checkpoint

### Off-policy

1. `sampler` 从 `ActorHandle` 读取当前策略动作
2. 交互结果写入 `ReplayBuffer`
3. `trainer` 从 buffer 采样 `ReplayBatch`
4. 更新完成后发布新的 `PolicySnapshot`
5. 同样由 `info` 和 `evaluator` 记录与验证

## 模块边界

- `model`：定义 actor/critic/encoder 的后端无关接口
- `sampler`：负责环境交互与 batch 装配，不直接调用 trainer
- `trainer`：负责算法更新、运行时调度和 buffer 桥接
- `info`：统一 observability、日志和系统健康事件
- `evaluator`：独立运行评测闭环

## Runtime Coordination

`RuntimeLoop` 是本系统的薄运行时协调层。它在概念上位于 `sampler` 和 `trainer` 之间，负责统一 `CollectionSchedule`、`UpdateSchedule`、`PolicySnapshot` 发布和 `ReplayBuffer` 桥接。详细约束见 [execution-model.md](/home/seene/workspace/p2_rl_training/docs/architecture/execution-model.md)。

## 设计原则

- 先稳定 contracts，再扩展实现
- 算法实现不应反向污染采样接口
- observation 必须原生支持 vector、image、multimodal
- 文档必须足以让实现者脱离聊天记录开始编码
