# Scripts README

## 当前脚本

- `validate_docs.py`：检查 docs/configs/README 是否齐全，并验证核心 runtime、算法和配置语义没有漂移
- `validate_contracts.py`：检查 minimum schema 是否存在，且关键字段能支撑 `PPO`、`SAC`、`TD3`
- `validate_runtime_env.py`：导入 `torch/gymnasium/mujoco`，并创建 `Humanoid-v5` 做 runtime preflight
- `run_tests.sh`：运行 docs/contracts/runtime env gate 与全部测试
- `run_evals.sh`：依次执行 `PPO`、`SAC`、`TD3` 的 Humanoid-v5 train/evaluate smoke flow
