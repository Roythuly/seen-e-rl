# Sampler Overview

## 定位

`sampler` 负责环境交互、observation 标准化、trajectory/replay record 装配，是训练系统的“数据生产端”。

## 边界

- 输入：`EnvSpec`、`ActorHandle`
- 输出：`TrajectoryBatch` 或 `ReplayRecord`
- 不负责：参数更新、不直接调用 trainer

## 主流程

1. `EnvFactory` 创建环境
2. `EnvAdapter` 统一 observation / action 语义
3. `ActorHandle` 给出动作
4. `TrajectoryAssembler` 产出轨迹或 replay records
