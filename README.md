# RL Training Infra

面向 RL 研究迭代的模板化训练基础设施仓库。

## 当前状态

当前仓库已经包含一套 Torch-first 的可运行 RL 训练栈：

- 根入口 `main.py`
- 稳定五模块：`model`、`sampler`、`trainer`、`info`、`evaluator`
- `algorithms/template` 与独立的 `algorithms/ppo`、`algorithms/sac`、`algorithms/td3`
- 公共 contracts、experiment configs、contract/integration tests、CI/eval scripts
- `Humanoid-v5` 的 `PPO`、`SAC`、`TD3` 短程 smoke config

实现边界：

- 运行时后端以 `PyTorch` 为主
- `JAX` 仍保留为文档与 contract 约束，不在当前运行路径中实现
- 当前目标是可运行、可扩展、可验证的研究基线，不是最终 benchmark 结果

## 代码布局

- `main.py`：顶层 CLI，负责 `train` / `evaluate`
- `packages/rl_training_infra/common`：配置加载、registry、JSON/YAML I/O
- `packages/rl_training_infra/contracts`：schema builders 与 validation helpers
- `packages/rl_training_infra/model`：模板、Torch model factory、checkpoint round-trip
- `packages/rl_training_infra/sampler`：env factory、actor handle、trajectory/replay collectors
- `packages/rl_training_infra/trainer`：runtime loop、replay buffer、learner base
- `packages/rl_training_infra/info`：metric event builder、console/JSONL sinks
- `packages/rl_training_infra/evaluator`：checkpoint selector、eval report runtime
- `algorithms/template`：assembly template、shared algorithm wiring helpers
- `algorithms/ppo` / `algorithms/sac` / `algorithms/td3`：算法私有 learner、默认配置与 assembly
- `configs/experiment`：可直接运行的 experiment entrypoints
- `tests/unit` / `tests/contract` / `tests/integration`：分层验证
- `scripts`：repo gate、runtime preflight、Humanoid smoke commands

## 使用方式

训练：

```bash
python main.py train --config configs/experiment/ppo_humanoid_v5.yaml
python main.py train --config configs/experiment/sac_humanoid_v5.yaml
python main.py train --config configs/experiment/td3_humanoid_v5.yaml
```

评测：

```bash
python main.py evaluate --config configs/experiment/ppo_humanoid_v5.yaml --selector latest
python main.py evaluate --config configs/experiment/sac_humanoid_v5.yaml --selector latest
python main.py evaluate --config configs/experiment/td3_humanoid_v5.yaml --selector latest
```

常用验证：

```bash
python scripts/validate_docs.py
python scripts/validate_contracts.py
python scripts/validate_runtime_env.py
bash scripts/run_tests.sh
bash scripts/run_evals.sh
```

## 模板化边界

- 算法私有逻辑放在 `algorithms/*/learner.py`，包括 PPO 的 rollout update、SAC 的 alpha/update、TD3 的 `policy_delay`
- 模块共享逻辑放在 `packages/rl_training_infra/*`，包括 model factory、env adapter、runtime loop、checkpoint/report plumbing
- `PolicySnapshot` 表示推理态 actor artifact
- `CheckpointManifest` 表示训练态恢复点

## 参考与事实源

- `docs/` 仍然是架构语义与模块边界的事实源
- `contracts/` 定义 batch、artifact 和 runtime minimum contracts
- `configs/experiment/*` 是当前可运行入口
