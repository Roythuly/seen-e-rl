# Info Overview

## 定位

`info` 是统一的 observability 子系统，负责把训练、采样、buffer、checkpoint、评测等事件汇总成统一日志与指标出口。

## 边界

- 输入：各模块发出的 metric / health / checkpoint / eval 事件
- 输出：`wandb`、`tensorboard`、结构化日志
- 不负责：算法逻辑或环境交互
- 统一事件 envelope 是 `info` 的主 contract，而不是 sink 的私有实现细节
