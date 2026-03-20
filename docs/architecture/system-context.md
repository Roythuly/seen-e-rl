# System Context

## 目标

本系统是一个面向 RL 研究迭代的训练 infra。`v0.1` 目标不是直接提供可运行训练系统，而是先把后续实现所需的架构边界、公共接口和交付事实源稳定下来。

## 范围

- 单层策略训练闭环
- 算法中心框架
- `model`、`sampler`、`trainer`、`info`、`evaluator` 五模块
- `PyTorch` 优先、`JAX` 兼容的双 backend 设计
- on-policy 与 off-policy 双训练路径
- `RuntimeLoop` 统一两条训练路径
- `ReplayBuffer` 与 `PublishSchedule` 作为未来训练/推理节点分离的桥梁
- 推理态 `PolicySnapshot` 与训练态 `CheckpointManifest` 显式区分

## 系统主流程

### On-policy

1. `RuntimeLoop` 固定当前 `PolicySnapshot`
2. `sampler` 使用 `ActorHandle` 采集一批 rollout，产出 `TrajectoryBatch`
3. `trainer` 基于 rollout 派生 `advantages` / `returns` 并做集中更新
4. `learner` 视 `PublishSchedule` 发布新的 `PolicySnapshot`
5. `learner` 视 checkpoint 策略落盘 `CheckpointManifest`
6. `info` 记录采样、更新、发布和 checkpoint 事件
7. `evaluator` 在独立环境中评测 checkpoint artifact

### Off-policy

1. `sampler` 从 `ActorHandle` 读取当前策略动作
2. 交互结果以 `ReplayRecord` 写入 `ReplayBuffer`
3. `RuntimeLoop` 依据 `UpdateSchedule` 从 buffer 采样 `ReplayBatch`
4. `trainer` 执行 `Learner.update(...)`
5. `RuntimeLoop` 按 `PublishSchedule` 决定是否发布新的 `PolicySnapshot`
6. `learner` 视 checkpoint 策略落盘 `CheckpointManifest`
7. 同样由 `info` 和 `evaluator` 记录与验证

## 模块边界

- `model`：定义 actor/critic/encoder 的后端无关接口，并提供统一 act/train 入口
- `sampler`：负责环境交互、observation 标准化和 batch 装配，不直接调用 trainer
- `trainer`：负责 `Learner.update(...)`、runtime bridge 和 publish/checkpoint 决策
- `info`：统一 observability、日志和系统健康事件
- `evaluator`：独立运行 checkpoint 评测闭环

## Runtime Coordination

`RuntimeLoop` 是本系统的公共运行时协调层。它在概念上位于 `sampler` 和 `trainer` 之间，负责统一 `CollectionSchedule`、`UpdateSchedule`、`PublishSchedule`、`PolicySnapshot` 发布和 `ReplayBuffer` 桥接。详细约束见 [execution-model.md](/home/seene/workspace/p2_rl_training/docs/architecture/execution-model.md)。

## 设计原则

- 先稳定 contracts，再扩展实现
- 算法实现不应反向污染采样接口
- 推理态与训练态 artifact 必须显式区分
- 未来训练/推理节点分离不应改变上层 batch 与 runtime contract
- observation 必须原生支持 vector、image、multimodal
- 文档必须足以让实现者脱离聊天记录开始编码
