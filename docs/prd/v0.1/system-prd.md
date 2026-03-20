# v0.1 System PRD

## Why

当前团队需要一个面向 RL 研究迭代的 infra，但如果直接开始写训练代码，容易在模块边界、batch 语义、双 backend 策略和 off-policy 桥接层上不断返工。`v0.1` 的目标是先固定这些事实源。

## What

交付一个 docs-first 的仓库骨架，覆盖：

- 架构设计
- 模块规范
- 算法装配文档
- contracts placeholder schema
- evals/tests/configs/scripts/infra 模板
- `v0.1` 冻结入口

## Who

- 直接读者：后续实现 infra 的工程师和算法研究者
- 间接读者：评审者、后续维护者、需要扩展新算法或新 backend 的开发者

## Acceptance Criteria

- AC-001：五个模块的职责边界和 API 语义在文档中明确
- AC-002：`PPO`、`SAC`、`TD3` 的装配要求有独立文档
- AC-003：`contracts/v0` 全部 schema 占位齐全且可通过校验脚本
- AC-004：`configs/tests/evals/scripts/infra` 的落点全部存在并有说明
- AC-005：实现者仅阅读仓库文档即可开始下一轮编码
