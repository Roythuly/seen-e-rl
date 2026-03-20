# RL Training Infra

面向 RL 研究迭代的 docs-first 基础设施仓库。

## 当前里程碑

本仓库当前交付 `v0.1` 文档先行版本，重点是把后续实现需要依赖的事实源固定下来：

- 系统与模块架构文档
- 跨模块 minimum contracts
- evals/tests/configs/scripts/infra 模板
- 版本冻结入口与协作模板

本轮**不包含**训练代码、算法实现或可运行服务。

## 系统目标

- 单机研究训练闭环优先
- 算法中心框架
- `RuntimeLoop` 统一 on-policy 与 off-policy 的调度语义
- `sampler` 与 `trainer` 通过 `ReplayBuffer` 和 `PolicySnapshot/ActorHandle` 解耦
- 采样/推理态与训练/恢复态显式区分：`PolicySnapshot` != `CheckpointManifest`
- `PyTorch` 优先，`JAX` 保留兼容约束
- `ReplayBuffer` 与 publish schedule 从第一天兼容未来训练/推理节点分离
- 首批算法文档覆盖 `PPO`、`SAC`、`TD3`

## 仓库导航

- [docs/index.md](/home/seene/workspace/p2_rl_training/docs/index.md)：文档总入口
- [docs/architecture/system-context.md](/home/seene/workspace/p2_rl_training/docs/architecture/system-context.md)：系统全景
- [docs/architecture/contracts.md](/home/seene/workspace/p2_rl_training/docs/architecture/contracts.md)：契约总览
- [docs/prd/v0.1/system-prd.md](/home/seene/workspace/p2_rl_training/docs/prd/v0.1/system-prd.md)：首版 PRD
- [contracts/README.md](/home/seene/workspace/p2_rl_training/contracts/README.md)：contracts 演进规则

## 文档优先规则

- 设计先写进 `docs/`，外部讨论只作为输入，不作为事实源
- 跨模块边界先写进 `contracts/`
- 顶层实验装配先写进 `configs/`
- 评测策略先写进 `evals/`
- 工程门禁尽量以脚本形式固化到 `scripts/`

## 后续实现入口

后续实现将以 `packages/`、`algorithms/`、`examples/` 为主要落点。当前这些目录只保留职责说明，避免误解为代码已开工。
