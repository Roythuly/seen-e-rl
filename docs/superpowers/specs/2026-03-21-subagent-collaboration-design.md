# Subagent Collaboration Design

## Summary

本文定义本仓库后续采用的 subagent 协作开发流程，覆盖当前 docs-first 阶段以及未来 `packages/`、`algorithms/` 落地后的实现阶段。

推荐方案为“模块优先、滚动集成”：

- 当前主会话充当唯一 `dispatcher`
- `module-dev` 可重复 spawn，多次实例化，每次只负责一个固定模块
- `integrator` 在每个模块完成后立即持续收口，负责整体阅读、入口装配、wiring、CI 与集成测试
- `reviewer` 在阶段里程碑和最终收尾时做 correctness / regression / coverage review

该流程以固定模块边界、显式 handoff、独立 worktree 和持续集成为核心，避免多 agent 并发时职责漂移、上下文污染和大爆炸式合并。

## Motivation

当前仓库仍处于 docs-first 阶段，但已经稳定了未来实现要依赖的五模块边界：

- `model`
- `sampler`
- `trainer`
- `info`
- `evaluator`

与此同时，仓库也已经明确了 contracts、configs、tests、scripts 和后续实现目录的职责分层。因此，协作流程需要同时满足两类需求：

1. 当前阶段能推进文档、contracts、configs、tests 模板
2. 后续阶段能平滑延续到真实代码实现、main 组装和回归验证

如果直接让多个 agent 在共享工作区中自由发挥，风险主要有三类：

- 模块边界被破坏，agent 顺手修改不属于自己的 consumer
- 对外接口修改没有被显式交接，导致集成期断点集中爆发
- 评审 agent 兼做修复，最终无法分清问题归属和完成定义

因此，本设计选择单一调度中心、固定模块拆分、独立 worktree 隔离和阶段化回流。

## Proposed Design

### 1. Role Model

#### `dispatcher`

`dispatcher` 由最外层当前会话承担，是系统中的唯一调度者。它负责：

- 读取用户目标和当前 integration 基线
- 将需求拆成模块任务卡
- 为每个任务创建 worktree / branch
- spawn 对应角色 agent
- 等待结果、分诊问题、决定下一跳

`dispatcher` 不直接承担持续性接线工作，也不把“修掉所有问题”作为默认职责。它是项目调度与技术分诊中心，而不是第二个 `integrator`。

#### `module-dev`

`module-dev` 是可重复 spawn 的模块开发 agent，默认按固定模块实例化：

- `model`
- `sampler`
- `trainer`
- `info`
- `evaluator`

每个实例只负责：

- 本模块私有或模块级文档、configs、tests、实现代码
- 本模块 module-scoped public interface 定义
- 本模块内部验证

如果任务卡显式授权，`module-dev` 也可以修改少量 shared surface，但只能在以下前提下进行：

- `dispatcher` 已为该任务预留独占路径
- 变更直接服务于该模块任务
- 共享 contract 仍遵守 add-only 与 RFC-first 规则
- 最终合并仍由 `dispatcher` 和 `integrator` 收口

每个实例不负责：

- 下游 consumer 适配
- `main` 或顶层入口 wiring
- 跨模块回归收口
- 系统级 CI 设计

默认规则：`module-dev` 可以修改“本模块实现 + 本模块 module-scoped interface 定义”，也可以在明确授权时提出 shared contract 变更，但不能把下游适配与装配收口也揽到自己身上，更不能绕过现有 contract 治理规则单独落地破坏性变更。

#### `integrator`

`integrator` 是持续集成 agent，在每个模块任务交付后立即介入。它负责：

- 通读最新 integration 基线和新交付模块
- 组装入口、补 `main`、补 wiring
- 修复因接口升级造成的下游断点
- 更新 shared config
- 补最小集成测试
- 补 CI 门禁和跨模块冒烟验证

`integrator` 可以跨模块修改接线层，但不应该回头重写某个模块的内部设计。若发现问题本质上属于模块内部缺陷，应回流给 `dispatcher` 重新派发。

#### `reviewer`

`reviewer` 是最终审查 agent，默认只读，不直接做修复。它负责：

- correctness review
- regression risk review
- test coverage review
- 接口漂移和未验证假设检查

`reviewer` 的输出是 findings，而不是补丁。所有修复都需要由 `dispatcher` 回流到 `module-dev` 或 `integrator`。

### 2. Task Decomposition Model

本设计以模块任务为主，而不是按临时能力切片全面改写拆分逻辑。原因如下：

- 当前仓库的事实源已经按五模块组织
- docs-first 阶段与实现阶段都可以沿用相同边界
- 模块 owner 最容易对本模块 contract 负责
- `integrator` 可以专门消化跨模块 wiring，而不污染模块职责

但当前仓库也存在一类天然不是模块私有的 shared surface，例如：

- `contracts/v0/*`
- `docs/architecture/*`
- `docs/index.md` 与 release / navigation 入口
- `configs/experiment/*`
- `scripts/validate_*.py`
- `tests/contract/*`
- `tests/integration/*`
- `.github/workflows/*`

因此，`dispatcher` 实际上要管理两类任务：

1. 模块任务：默认单位，绑定到单个模块
2. shared-surface task：仍以某个 lead module 或 `integrator` 为责任主体，但必须串行处理共享路径

跨模块能力如 `RuntimeLoop`、`ReplayBuffer`、`PolicySnapshot publish`、`CheckpointManifest` 虽然天然横跨多个模块，但默认处理方式不是新建“能力型 module-dev”，而是：

1. 由 `dispatcher` 将相关工作拆到对应模块任务卡中
2. 若任务触及 shared surface，则在任务卡中显式列出“borrowed shared ownership”
3. `dispatcher` 保证共享路径同一时刻只被一个任务持有
4. 由 `integrator` 负责最终跨模块接线与收口

这里需要额外区分两类 shared surface：

- shared contract / architecture surface
  - 可由 lead `module-dev` 在明确授权下修改
  - 仍受 repo 级 contract 治理约束
- assembly / validation surface
  - 默认归 `integrator` 所有
  - 包括 `configs/experiment/*`、`.github/workflows/*`、`tests/contract/*`、`tests/integration/*`、顶层入口 wiring

这样既保留“模块优先”的主路径，也给 docs-first 阶段大量共享文件留出明确 owner。

### 3. Branching And Worktree Strategy

默认并发策略采用独立 worktree，而不是共享工作区。

本设计将 `integration/mainline` 定义为真实长期分支，并建议配套一个长期 integration worktree。建议保留两个长期语义分支或工作区视角：

- `main`：人类可读、相对稳定
- `integration/mainline`：持续汇总各模块结果，允许短期演进，但必须保持当前阶段对应的门禁可通过

每个模块任务都从当前 `integration/mainline` 派生独立 worktree，例如：

- `module/model-encoder-spec`
- `module/sampler-rollout-assembler`
- `module/trainer-runtime-loop`

这样做的收益是：

- 多个 `module-dev` 可以并发，不互相覆盖工作区
- `integrator` 每次都围绕最新 integration 基线收口
- 合并失败和回归问题更容易定位到具体模块任务

但 worktree 只隔离工作区，不自动消除 shared surface 冲突。因此并发必须附带路径级串行规则：

- 只有当任务卡的 writable paths 与 reserved shared paths 互不重叠时，多个 `module-dev` 才允许并发
- 任何触及 `contracts/v0/*`、`docs/architecture/*`、`docs/index.md`、`configs/experiment/*`、`scripts/validate_*.py`、`tests/contract/*`、`tests/integration/*`、`.github/workflows/*` 的任务，都必须由 `dispatcher` 显式加锁
- 默认由 `integrator` 持有 assembly / validation surface；模块任务只有在被授权时才能临时借用

### 4. Task Lifecycle

每个任务都遵循固定状态机：

`Planned -> Owner In Progress -> Awaiting Integration -> Integrated -> Review Pending -> Done`

具体流转如下：

1. `dispatcher` 根据用户目标创建任务卡，并指定 `Execution Owner`
2. 从 `integration/mainline` 创建对应 worktree 和分支
3. 若 `Execution Owner = module-dev`，则 spawn 一个 `module-dev` 实例执行该任务
4. 若 `Execution Owner = integrator`，则直接 spawn `integrator` 执行 shared-surface task
5. `module-dev` 完成后，提交显式 handoff，状态进入 `Awaiting Integration`
6. `dispatcher` 将模块结果回收到 integration 基线，并立即唤起 `integrator`
7. `integrator` 完成接线、shared surface 收口、冒烟验证和 CI 更新后，状态进入 `Integrated`
8. 对于直接由 `integrator` 执行的 shared-surface task，可跳过单独的 `Awaiting Integration`，但仍需输出 completion note 和验证结果
9. 在阶段里程碑或最终收尾时，`dispatcher` 唤起 `reviewer`
10. 若审查通过，进入 `Done`
11. 若审查或集成失败，则按归属回流为新的模块任务或集成任务

### 5. Dispatcher Playbook

`dispatcher` 的标准运行循环如下：

1. 读取当前目标、已集成模块和风险点
2. 优先派发依赖较少、边界较清晰的模块任务
3. 每完成一个模块，立即触发一次 `integrator`
4. 集成稳定后再决定是否并发派发下一个模块
5. 在里程碑节点触发 `reviewer`
6. 将 findings 按归属重新拆成模块修复单或集成修复单
7. 只有在 integration 基线通过门禁后，才考虑推进到最终完成

为控制复杂度，建议默认限制 subagent 只有单层深度：

- `dispatcher` 可以 spawn `module-dev` / `integrator` / `reviewer`
- 子 agent 默认不再继续 spawn 新 agent

### 6. Required Task Card

每个模块任务卡至少包含以下字段：

```text
Task: <short title>
Task Type: <module|shared-surface>
Execution Owner: <module-dev|integrator>
Module: <model|sampler|trainer|info|evaluator>
Phase: <docs-first|implementation>
Goal: <what changes by the end>
Writable Paths:
- <owned paths>
Reserved Shared Paths:
- <exclusive shared paths or "none">
Allowed Interface Surface:
- <owned contracts / public interfaces>
Out Of Scope:
- <consumer adaptation / wiring / CI etc.>
Definition of Done:
- <observable completion criteria>
Minimum Verification:
- <commands or checks>
Expected Integrator Follow-up:
- <wiring / consumer / CI touchpoints>
```

任务卡是 `module-dev` 的唯一事实源之一；没有任务卡就不应 spawn 模块 agent。

### 7. Required Handoff

每个 `module-dev` 结束时必须输出 handoff，最少包含：

```text
Completed:
- <what was changed>

Files Changed:
- <path list>

Shared Surfaces Touched:
- <shared paths or "none">

Interface Changes:
- <contracts / public APIs changed or "none">

Integrator Follow-up:
- <downstream adaptation points>

Verification Run:
- <commands and outcomes>

Open Risks:
- <known limitations or assumptions>
```

没有 handoff 的模块交付，不能进入 `Awaiting Integration` 之后的流程。

### 8. Verification Gates

所有 gate 都必须 phase-aware。当前仓库仍是 docs-first，所以流程不能强迫 agent 为了“满足 gate”去伪造运行态 wiring、假烟雾测试或无意义 CI 复杂度。

#### 当前 docs-first gate

当前阶段的“integration”应解释为文档、contracts、configs、validation scripts、tests 模板与 CI 模板之间的一致性，而不是可运行训练主程序。

`module-dev` gate：

- 若任务只触及模块文档 / configs / templates，至少运行 `python scripts/validate_docs.py`
- 若任务触及 `contracts/v0/*`，额外运行 `python scripts/validate_contracts.py`
- 若任务触及 `tests/*`、`scripts/validate_*.py` 或会影响仓库校验行为，额外运行 `bash scripts/run_tests.sh`
- 当前阶段的完成表述应为“文档 / contract / config / template 一致”；只有在实现文件真实存在时，才要求“实现一致”

`integrator` gate：

- 在 docs-first 阶段，`integrator` 的首要职责是收口 shared config、shared docs、validation scripts、CI 模板和跨模块引用
- 必跑命令为 `python scripts/validate_docs.py`
- 若本轮触及 contracts，则必跑 `python scripts/validate_contracts.py`
- 必跑 `bash scripts/run_tests.sh`
- CI 更新必须与当前门禁对齐，重点是 docs / contracts / tests validation，而不是假设已经有运行态训练链路
- 若本轮触及 `.github/workflows/*`、`tests/contract/*` 或 `tests/integration/*` 这类尚未被现有脚本完整覆盖的 surface，则该任务必须同时新增或更新可执行校验，使其进入 `bash scripts/run_tests.sh`、`python scripts/validate_docs.py` 或其他写入任务卡的自动化命令；在此之前不应把这类改动视为 fully gated

#### 后续实现阶段 gate

当 `packages/`、`algorithms/` 或真实 runtime 入口开始落地后，再追加实现期 gate。

`module-dev` 只能证明“本模块成立”，其实现期完成门禁包括：

- 本模块文档、contracts、实现一致
- 本模块最小测试或校验通过
- 对外接口变更被显式列出
- 已知风险没有被隐藏

`integrator` 的实现期完成门禁包括：

- 入口装配和 wiring 完成
- shared config 已同步
- 最小集成场景可冒烟通过
- CI 所需校验已补齐或更新

#### `reviewer` gate

`reviewer` 需要输出结构化审查结果，而不是笼统结论。其输出最少包含：

- findings 列表，按严重度排序
- correctness 风险
- regression 风险
- coverage 缺口
- residual risks

### 9. Failure Routing

失败不是异常，而是正常的状态迁移。

回流规则固定为三类：

- 模块内部问题：回流给对应 `module-dev`
- 跨模块接线问题：回流给 `integrator`
- 审查发现问题：由 `dispatcher` 判断归属后重新派单

任何角色都不应通过“顺手多做一点”来吞掉不属于自己的问题，否则长期会导致边界坍塌。

### 10. Configuration Layering

建议采用如下分层：

```text
your-repo/
├─ AGENTS.md
├─ .codex/
│  ├─ config.toml
│  └─ agents/
│     ├─ module-dev.toml
│     ├─ integrator.toml
│     └─ reviewer.toml
```

各层职责如下：

- `AGENTS.md`
  - 存放仓库级共享规则
  - 定义模块边界、事实源优先级、worktree 约定、handoff 模板、验证最小要求、回流原则
- `.codex/config.toml`
  - 存放运行时默认值
  - 定义模型、reasoning effort、sandbox / approval 基线、`agents.max_threads`、`agents.max_depth = 1`
- `.codex/agents/module-dev.toml`
  - 定义单模块开发角色
  - 强调 ownership、可修改范围、禁止项和 handoff 输出
- `.codex/agents/integrator.toml`
  - 定义持续集成角色
  - 强调 wiring、入口装配、CI / integration tests 和问题回流
- `.codex/agents/reviewer.toml`
  - 定义只读审查角色
  - 强调 findings-first、correctness / regression / coverage review

### 11. Recommended Agent Defaults

建议默认配置以下行为：

- `agents.max_depth = 1`
- `module-dev` 默认不再 spawn 子 agent
- `reviewer` 默认只读
- `integrator` 默认以最新 integration 基线为工作事实源
- `dispatcher` 只有在模块边界清晰时才并发多个 `module-dev`

对于当前仓库，推荐优先推进顺序如下：

1. `model`
2. `sampler`
3. `trainer`
4. `info`
5. `evaluator`

该顺序的理由是：

- `model`、`sampler`、`trainer` 共同决定训练主链路
- `info`、`evaluator` 更适合作为收口期补强
- 这与当前 contracts 和 architecture 文档的重心一致

## Contracts Impact

本设计不直接引入新的训练 runtime contract，但会引入以下协作协议层面的稳定约束：

- “模块 owner 拥有本模块 module-scoped interface 定义权”
- “shared contracts 仍是 repo 级治理对象，遵守 add-only / RFC-first”
- “consumer 适配权属于 `integrator`”
- “审查结论与修复职责分离”
- “任务卡与 handoff 为硬性交付物”

这些约束会影响未来 `AGENTS.md`、`.codex/config.toml` 和 `.codex/agents/*.toml` 的内容设计，但不会改变现有 `contracts/v0/` 中的领域 schema 命名。

## Compatibility

该流程同时兼容：

- 当前 docs-first 阶段
- 后续 `packages/` 共享包实现
- 后续 `algorithms/` 算法实现
- `tests/` 中从 contract 校验扩展到 unit / integration 的演进路径

在 docs-first 阶段，`module-dev` 的产物主要表现为文档、contracts、configs、tests 模板；此时 `integrator` 收口的是 shared docs、shared configs、validation scripts、CI 模板和跨模块引用。在实现阶段，则自然扩展为模块代码、模块级测试和真实入口 wiring。角色模型不需要切换，但 gate 会升级。

## Rollout / Migration

建议按以下顺序落地：

1. 写入仓库级 `AGENTS.md`
2. 写入 `.codex/config.toml`
3. 写入 `.codex/agents/module-dev.toml`
4. 写入 `.codex/agents/integrator.toml`
5. 写入 `.codex/agents/reviewer.toml`
6. 在一次真实小任务上试跑完整闭环
7. 根据试跑结果微调 task card、handoff 和 gate

第一次试跑建议选一个边界清晰、跨模块影响可控的任务，不要一开始就用跨 `model / sampler / trainer` 的大任务验证整套机制。

## Alternatives Considered

### 方案 A：模块优先、滚动集成

优点：

- 最贴合当前仓库模块边界
- 接口责任归属清晰
- 适合 docs-first 向代码实现平滑迁移
- 每个模块完成后立刻集成，风险暴露更早

缺点：

- `integrator` 的工作频率较高
- 对 handoff 质量要求更高

### 方案 B：阶段优先、批次推进

优点：

- 阶段划分整齐
- 调度简单

缺点：

- 接口变化会延迟到后期集中爆发
- 与“每个模块完成后立即集成”的目标冲突

### 方案 C：文档期按模块、实现期按能力切片

优点：

- 实现后期在某些场景下更灵活

缺点：

- 角色和边界模型会在阶段切换时变化
- `AGENTS.md` 和 agent 指令会更复杂
- 不利于长期复用同一套调度协议

最终选择方案 A。

## Open Questions

- `module-dev` 是否需要为不同模块设定不同的默认验证命令
