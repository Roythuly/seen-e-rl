# Execution Model

## 目的

`RuntimeLoop` 是 `sampler` 和 `trainer` 之间那层必须显式存在的运行时协调层。它的职责不是实现算法细节，而是把“何时采样、采多少、何时训练、训练多少次、何时发布新策略”统一收敛到一个稳定抽象里。

## 为什么必须单独定义

如果没有这层，`sampler` 很容易开始感知算法类型，`trainer` 也会开始反向控制环境交互，最后 on-policy 和 off-policy 会演化成两套割裂的运行模式。把 `RuntimeLoop` 单独定义出来，有两个直接收益：

1. 让 `on-policy` 和 `off-policy` 在同一套调度框架下表达
2. 让未来“推理端/采样端”和“训练端”异步解耦时，仍保留同一组上层接口

## RuntimeLoop 的职责边界

- 持有并执行 `CollectionSchedule`
- 持有并执行 `UpdateSchedule`
- 协调 `sampler`、`trainer`、`ActorHandle`、`ReplayBuffer`
- 在训练完成后决定何时发布新的 `PolicySnapshot`
- 把关键运行时事件发送给 `info`

`RuntimeLoop` 不负责：

- 定义算法损失函数
- 定义 model 结构
- 执行环境内部逻辑
- 决定 metric sink 如何落盘

## 统一 on-policy / off-policy 的方式

### On-policy

- `CollectionSchedule` 定义一次采样 N 步、N 条轨迹或采到某个 rollout 条件
- `RuntimeLoop` 在采样期间固定 `PolicySnapshot`
- 采样完成后，把 `TrajectoryBatch` 交给 `Algorithm.update_from_rollout(batch)`
- 完成一轮更新后发布新策略，进入下一轮采样

### Off-policy

- `CollectionSchedule` 可以定义每步、每 K 步或按 episode 持续交互
- `sampler` 每次交互后通过 `ReplayBuffer.write(record)` 写入 transition
- `RuntimeLoop` 根据 `UpdateSchedule` 触发 `ReplayBuffer.sample(spec)` 和 `Algorithm.update_from_replay(batch)`
- 更新完成后可立即或按策略版本门槛发布新策略

两条路径的差异只体现在 schedule 和 batch 类型上，不体现在 `sampler` 是否直接依赖 `trainer`。

## 关键协作对象

### ActorHandle

- `sampler` 读取动作的唯一入口
- 屏蔽当前策略来自 `PyTorch` 还是 `JAX`
- 未来异步化时，可演进为远程 actor handle 或版本化策略服务

### ReplayBuffer

- 现在是 off-policy 数据桥梁
- 未来异步 actor-learner 架构里，是推理端和训练端解耦的核心桥梁
- 运行时只依赖 `write/sample` 语义，不依赖具体实现位置

### CollectionSchedule

- 定义采样边界
- 例如“采 1 步再判断”、“采 N 步再更新”、“采满若干条轨迹”

### UpdateSchedule

- 定义更新边界
- 例如“每次 collection 后更新 1 次”、“每步交互后更新 M 次”、“每隔 K 步发布一次新策略”

## 面向未来异步训练的兼容性

`RuntimeLoop` 的文档语义必须从第一天就兼容异步模式：

- 同步单机模式下：loop 在一个进程里协调 `sampler`、`ReplayBuffer`、`trainer`
- 异步模式下：采样端和训练端可以拆开，但仍共享 `ActorHandle`、`ReplayBuffer`、`PolicySnapshot`、schedule 语义

换句话说，未来异步化改变的是部署形态，不应该改变上层 contracts。

## 落地要求

- `sampler` 文档必须声明自己只被 `RuntimeLoop` 调度
- `trainer` 文档必须声明自己不直接拥有环境交互逻辑
- 算法文档必须写清依赖哪种 schedule
- 后续若新增独立 `runtime` 目录或模块，应以本文件为事实源
