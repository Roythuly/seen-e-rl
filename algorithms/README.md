# Algorithms

`algorithms/` 负责算法私有部分，而不是公共 runtime：

- `template/`：shared assembly helpers、默认 spec builders、train/evaluate persistence
- `ppo/`：PPO 默认配置、assembly、私有 learner
- `sac/`：SAC 默认配置、assembly、私有 learner
- `td3/`：TD3 默认配置、assembly、私有 learner

约束：

- 每个算法目录独立声明自己的 learner 与装配逻辑
- `algorithms/*/assembly.py` 只通过模块 public API 组合系统
- 公共 runtime、replay buffer、checkpoint plumbing 仍然留在 `packages/rl_training_infra`
