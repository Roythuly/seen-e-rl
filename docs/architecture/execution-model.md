# Execution Model

## 目的

`RuntimeLoop` 是 on-policy 与 off-policy 共享的运行时协调层。它不定义算法损失，也不拥有环境内部逻辑；它只负责把“何时采样、采多少、何时训练、训练多少次、何时发布 actor、何时落 checkpoint”统一收敛成稳定 contract。

## 为什么必须单独定义

如果没有这层，`sampler` 很容易开始感知算法类型，`trainer` 也会开始反向控制环境交互，最后 PPO 与 SAC/TD3 会演化成两套割裂的运行模式。把 `RuntimeLoop` 单独定义出来，有三个直接收益：

1. 让 `on-policy` 和 `off-policy` 在同一套调度框架下表达
2. 让发布 actor 与更新 learner 成为两个可独立配置的动作
3. 让未来“推理端/采样端”和“训练端”异步解耦时，仍保留同一组上层接口

## RuntimeLoop 的职责边界

- 持有并执行 `RuntimeSpec`
- 持有并执行 `CollectionSchedule`
- 持有并执行 `UpdateSchedule`
- 持有并执行 `PublishSchedule`
- 协调 `sampler`、`Learner`、`ActorHandle`、`ReplayBuffer`
- 决定何时发布新的 `PolicySnapshot`
- 决定何时生成新的 `CheckpointManifest`
- 把关键运行时事件发送给 `info`

`RuntimeLoop` 不负责：

- 定义算法损失函数
- 定义 model 结构
- 执行环境内部逻辑
- 决定 metric sink 如何落盘

## RuntimeSpec

`RuntimeSpec` 是 runtime 的顶层配置对象，用来把采样、更新、发布和 checkpoint 规则组合成一条训练链路。

- `collection`：一次 collection 的边界
- `update`：一次 update window 的边界
- `publish`：actor publish 的节奏
- `checkpoint`：checkpoint 触发和命名策略
- `evaluation`：独立 evaluator 的触发策略

## 统一 on-policy / off-policy 的方式

### On-policy

- `CollectionSchedule` 定义一次采样 N 步、N 条轨迹或采到某个 rollout 条件
- `RuntimeLoop` 在采样期间固定 `PolicySnapshot`
- `sampler` 输出 `TrajectoryBatch`
- `Learner.update(batch)` 在 trainer 域内派生 `advantages` / `returns` 并完成多轮 epoch/minibatch 更新
- `PublishSchedule` 通常配置为“本轮更新完成后发布一次”

### Off-policy

- `CollectionSchedule` 可以定义每步、每 K 步或按 episode 持续交互
- `sampler` 每次交互后通过 `ReplayBuffer.write(record)` 写入 transition
- `RuntimeLoop` 根据 `UpdateSchedule` 触发 `ReplayBuffer.sample(spec)` 和 `Learner.update(batch)`
- `PublishSchedule` 决定是每次更新后发布、每 K 次更新后发布，还是仅在 actor update 后发布
- `CheckpointManifest` 由 learner 按 checkpoint 策略落盘

两条路径的差异只体现在 schedule 和 batch 类型上，不体现在 `sampler` 是否直接依赖 `trainer`。

## 关键协作对象

### ActorHandle

- `sampler` 读取动作的唯一入口
- 输入 `ObservationBatch`
- 输出结构化 `ActionOutput`
- `ActionOutput` 至少包含 `action` 与 `policy_version`
- 可选包含 `log_prob`、`value_estimate`、`actor_state`、`diagnostics`
- 屏蔽当前策略来自 `PyTorch` 还是 `JAX`
- 未来异步化时，可演进为远程 actor handle 或版本化策略服务

### ReplayBuffer

- 现在是 off-policy 数据桥梁
- 未来异步 actor-learner 架构里，是推理端和训练端解耦的核心桥梁
- 运行时只依赖 `write/sample` 语义，不依赖具体实现位置

### CollectionSchedule

- 定义采样边界
- 最少字段应能表达：`mode`、`unit`、`amount`
- on-policy 路径需要 `freeze_policy_during_collection`
- off-policy 路径需要 `warmup_env_steps` 或同类 warmup 语义
- 例如“采 1 步再判断”、“采 N 步再更新”、“采满若干条轨迹”

### UpdateSchedule

- 定义更新边界
- 最少字段应能表达：`trigger_unit`、`updates_per_trigger`
- PPO 需要 `epochs` 与 `minibatch_size`
- SAC/TD3 需要 `updates_per_step`、`min_ready_size`

### PublishSchedule

- 定义 actor publish 边界
- 最少字段应能表达：`strategy`
- 可选字段包括 `every_n_updates`、`on_actor_update_only`、`require_checkpoint`
- 用来避免“更新 learner”与“发布 actor”被默认绑定成同一动作

## 面向未来异步训练的兼容性

`RuntimeLoop` 的文档语义必须从第一天就兼容异步模式：

- 同步单机模式下：loop 在一个进程里协调 `sampler`、`ReplayBuffer`、`Learner`
- 异步模式下：采样端和训练端可以拆开，但仍共享 `ActorHandle`、`ReplayBuffer`、`PolicySnapshot`、`CheckpointManifest`、schedule 语义
- 变化的是部署形态，不是 batch contract，也不是 publish/checkpoint contract

换句话说，未来异步化改变的是部署形态，不应该改变上层 contracts。

## 落地要求

- `sampler` 文档必须声明自己只被 `RuntimeLoop` 调度
- `trainer` 文档必须声明自己不直接拥有环境交互逻辑
- 算法文档必须写清依赖哪种 schedule，以及哪些字段是 trainer 派生字段
- `PolicySnapshot` 与 `CheckpointManifest` 必须在各模块文档中语义一致
- 后续若新增独立 `runtime` 目录或模块，应以本文件为事实源
