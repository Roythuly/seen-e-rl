# Algorithms Overview

首版算法文档只覆盖单层策略，目标是说明算法如何装配 `model`、`sampler`、`trainer`、`info`、`evaluator` 五模块。

## 首批算法

- `PPO`
- `SAC`
- `TD3`

## 共性约束

- 算法不直接控制 env，统一由 `RuntimeLoop` 调度
- 算法只依赖标准 batch 契约，不依赖具体 observation 模态实现
- 所有算法都必须能写出统一的 `UpdateResult`
- 算法差异通过 `CollectionSchedule` 和 `UpdateSchedule` 表达，而不是通过改写 sampler/trainer 边界表达

## 差异主线

- `PPO`：rollout 驱动，固定策略采样后再更新
- `SAC`：replay 驱动，可配置交互后立即更新
- `TD3`：replay 驱动，与 `SAC` 同属 off-policy，但 actor 更新节奏不同
