# Model Overview

## 定位

`model` 模块负责定义后端无关的 actor/critic/encoder 构建与推理接口，是算法与具体深度学习框架之间的稳定边界。

## 边界

- 输入：`ModelSpec`、`BackendSpec`、`ObservationSpec`
- 输出：可执行的 policy/value 模型实例
- 不负责：采样调度、训练循环、日志写入

## 主流程

1. 根据 `ModelSpec` 解析网络结构
2. 根据 `BackendSpec` 选择 `PyTorch` 或 `JAX`
3. 返回统一接口的 model object

## 设计重点

- 同一份 `ModelSpec` 尽量表达双 backend 共识
- observation encoder 支持 vector、image、multimodal
- actor 与 critic 允许共享 encoder，但共享策略必须显式声明
