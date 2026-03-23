# seen-e-rl

A highly modular, widely scalable, and robust reinforcement learning framework built on PyTorch for continuous control. 

Designed for both lightweight prototyping and large-scale robotics simulation, seen-e-rl seamlessly unifies Gymnasium and Isaac Lab backends under a common set of algorithms and interfaces.

## ✨ Features

- **Algorithm Registry:** Built-in implementations for `SAC`, `TD3`, `PPO`, and `OBAC`, abstracting policy details away from trainers.
- **Environment Factory:** A unified batched API for both Gymnasium and NVIDIA Isaac Lab. Run 1 or 10,000 environments using the exact same generic trainer.
- **Model Factory:** Flexible actor and critic backbone configurations via the YAML `model.*` section. Specify hidden dimensions, standard deviations, and architecture choices without touching python code.
- **Nested YAML Configuration:** Streamlined configuration management with hierarchical `_base_` inheritance and robust command-line parameter overrides.
- **Seamless Resume & Evaluation:** Automatically infers full configuration contexts from saved checkpoints. Evaluate or run inference with just a `--checkpoint` argument.
- **Improved PPO Semantics:** Aligned with state-of-the-art implementations, including per-mini-batch advantage normalization, value clipping, and dual-clip objective support.
- **Unified Standardized Logging:** Clean `[TRAIN]`, `[EVAL]`, and `[INFO]` console formats with out-of-the-box TensorBoard and Weights & Biases (wandb) integration.

## 🚀 Installation

For the lightweight Gymnasium workflow, we recommend using `uv` to create an isolated virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

For extras like development dependencies or wandb:
```bash
uv pip install -e ".[dev]"
uv pip install -e ".[wandb]"
```

### Isaac Lab Runtime Setup
If you plan to use NVIDIA's Isaac Lab, the code must be run inside an Isaac Sim/Isaac Lab Python environment (e.g., `isaaclab` conda environment). 
The env factory dynamically imports custom task packages (such as `isaaclab_tasks`) defined in the `env.isaaclab.task_imports` configuration list. Specific compatibility patches (such as URDF caching reuse) are automatically injected.

## 🎮 Quick Start

### 1. Gymnasium Training
You can start training directly using configurations defined in `configs/`:

```bash
# Single environment (Legacy style)
uv run train.py --config configs/sac.yaml --env_name HalfCheetah-v4

# Parallel environments
uv run train.py --config configs/ppo.yaml --env_name Pendulum-v1 --env.num_envs 16
```

### 2. Isaac Lab Training
For Isaac Lab, pass the corresponding multi-environment configs. Below is the example for the `PickPlace` task:

```bash
python train.py --config configs/isaaclab_pickplace_sac.yaml --env.num_envs 32
python train.py --config configs/isaaclab_pickplace_ppo.yaml --env.num_envs 32
python train.py --config configs/isaaclab_pickplace_obac.yaml --env.num_envs 32
```

### 3. Evaluate & Render
Configurations are automatically retrieved from the checkpoint directory. You only have to pass the checkpoint file!

```bash
# Evaluation (without rendering)
uv run evaluate.py --checkpoint results/xxx/checkpoints/best.pt --num_episodes 10

# Rendering
uv run render/renderer.py --checkpoint results/xxx/checkpoints/best.pt --episodes 5
```

### 4. Resume Training
To seamlessly resume an interrupted run with all previous configurations:
```bash
uv run train.py --resume results/xxx/checkpoints/latest.pt
```

## 🧠 Architecture Overview

`seen-e-rl` is decoupled into neatly separated layers:

1. **Configurations (`seenerl/config.py`)**: Uses a dictionary-like Config object holding flattened hierarchies resolving overrides and backwards compatibilities.
2. **Algorithms (`seenerl/algorithms/registry.py`)**: Maps an algorithm string (`"ppo"`, `"sac"`) to its corresponding class and defines its required `trainer_kind` (`on_policy` or `off_policy`).
3. **Models (`seenerl/models/`)**: Generates model architectures corresponding specifically to the algorithm parameters (e.g. `GaussianActor` for SAC, `GaussianFixedStdActor` for PPO).
4. **Environments (`seenerl/envs/`)**: Standardizes outputs. `obs` and `actions` are always handled as bounded `float32` tensors in a batch dimension.
5. **Trainers (`seenerl/trainers/`)**: The core interaction loop. Only distinguishes between off-policy (replay buffer transitions) and on-policy (rollouts for GAE).

## ⚙️ Configuration Schema Details

Configurations emphasize the separation of RL algorithms, networks, and environments. Configuration values nested inside dictionaries can be overridden using dot notation, e.g. `--env.num_envs 8`.

### Environment Configuration
```yaml
env:
  backend: "gymnasium"      # Supported backends: "gymnasium" | "isaaclab"
  id: "Pendulum-v1"
  num_envs: 8               # Parallelized environments count
  kwargs: {}                # Passed during gym.make
  isaaclab:                 # Specific overrides when using Isaac Lab backend
    headless: true
    use_fabric: true
    task_imports: 
      - "isaaclab_tasks.manager_based.manipulation.pick_place"
```

### Model Architecture Selection
```yaml
model:
  actor:
    name: "gaussian"
    hidden_dim: 256
    kwargs:
      squash: false
  critic:
    name: "q_network"
    hidden_dim: 256
```
*(Passing legacy parameters such as `--hidden_size 256` or `--env_name HalfCheetah-v4` via CLI is fully backward compatible.)*

## 📁 Project Structure 

```text
seen-e-rl/
├── configs/                  # Ready-to-use YAML configurations
├── seenerl/
│   ├── algorithms/           # SAC, TD3, PPO, OBAC and registries
│   ├── buffers/              # Replay buffers & Rollout buffers
│   ├── envs/                 # Batched Gymnasium and Isaac Lab adaptors
│   ├── models/               # Factory to link algorithms and architectures
│   ├── networks/             # Base and concrete Torch networks
│   ├── trainers/             # OnPolicy / OffPolicy trainer loops
│   ├── checkpoint.py         # Checkpointing routines
│   ├── config.py             # Config parser & CLI overrides
│   ├── evaluator.py          # Unified evaluation routines
│   └── logger.py             # wandb & tensorboard unified logging
├── render/                   # Visualization script logic
├── tests/                    # PyTest tests checking unit and structural integrity
├── train.py                  # Single robust training entry point
└── evaluate.py               # Evaluation entry point
```

## 🧪 Testing

The repository relies on `pytest` for functional validation. Run the test suite via:

```bash
pytest -q tests/test_config_normalization.py tests/test_buffers_batched.py tests/test_algorithm_registry.py
```

*Note: The Isaac Lab tests (`pytest tests/test_isaaclab_pickplace.py`) are strictly opt-in via the `SEENERL_RUN_ISAACLAB=1` environment variable, as they require the complete NVIDIA simulator stack.*

## 📄 License
MIT
