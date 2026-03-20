# Trainer Overview

## 定位

`trainer` 负责算法更新、buffer 桥接与 runtime 执行，是系统的训练控制中心。`RuntimeLoop` 是公共 runtime 抽象，可由 trainer 域承载实现，但不属于 trainer 私有 contract。

## 边界

- 输入：`TrajectoryBatch`、`ReplayBatch`、`PolicySnapshot`、`RuntimeSpec`
- 输出：`UpdateResult`、新的 `PolicySnapshot`、新的 `CheckpointManifest`
- 不负责：环境交互细节、模型底层实现

## 主流程

- on-policy：采样一批 rollout 后统一更新
- off-policy：采样写 buffer，再从 buffer 采样更新
- 两条路径都由 `RuntimeLoop` 驱动策略刷新
- publish actor 与 save checkpoint 允许是两个独立动作

## RuntimeLoop 在本模块中的位置

- 它统一表达 on-policy 和 off-policy 的调度差异
- 它协调 `sampler`、`ActorHandle`、`ReplayBuffer`、`Learner`
- 它不是具体算法的一部分，而是算法运行时的公共外壳
