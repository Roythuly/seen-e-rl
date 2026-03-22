# seen-e-rl

A modular reinforcement learning framework built on PyTorch, supporting state-of-the-art continuous control algorithms:

- **SAC** (Soft Actor-Critic) — Off-policy, entropy-maximized
- **TD3** (Twin Delayed DDPG) — Off-policy, deterministically exploring
- **PPO** (Proximal Policy Optimization) — On-policy, clipped surrogate objective
- **OBAC** (Offline-Boosted Actor-Critic) — Off-policy, adaptively blends historical behaviors

## Features

- 📝 **YAML Configuration** — Hierarchical configs with CLI override support for hyper-parameter tuning
- 🧩 **Modular Architecture** — Pluggable algorithms, networks, buffers, and trainers
- 📊 **Unified Logging** — TensorBoard + wandb with standardized `[TRAIN]`/`[EVAL]` output metrics
- 💾 **Checkpoint Management** — Multiple strategies (latest, best, interval)
- 🔄 **Training Resumption** — Save/load buffer state for seamless script resume
- 🔌 **Extensible Networks** — Registry pattern for custom architectures (VLA, Transformer)
- 🛡️ **Robust Action Bounding** — Robust scaling and testing evaluators natively capable of handling custom asymmetrical Environment bounds seamlessly for algorithms like PPO/TD3.

## Quick Start
It is highly recommended to manage the virtual environment using `uv` for speed and dependency resolution.

### Installation

```bash
# Create and activate environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Or with wandb support:
uv pip install -e ".[wandb]"
```

### Training

```bash
# SAC on HalfCheetah
uv run train.py --config configs/sac.yaml

# PPO on Humanoid-v5
uv run train.py --config configs/ppo.yaml --env_name Humanoid-v5

# TD3 with custom parameters
uv run train.py --config configs/td3.yaml --env_name Ant-v5 --seed 123

# Resume training from checkpoint
uv run train.py --config configs/sac.yaml --resume results/xxx/checkpoints/latest.pt
```

### Evaluation
Evaluate without generating transitions:

```bash
uv run evaluate.py --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --num_episodes 10
```

### Rendering
Render the trained policies iteratively using Gymnasium rendering wrappers:

```bash
uv run render/renderer.py --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --episodes 5
```

## Project Structure

```text
seen-e-rl/
├── configs/                  # YAML configurations (hyperparameters)
│   ├── default.yaml          # Shared base defaults for all algorithms
│   ├── sac.yaml              # Config overrides specific to SAC
│   ├── td3.yaml              # Config overrides specific to TD3
│   ├── ppo.yaml              # Config overrides specific to PPO
│   └── obac.yaml             # Config overrides specific to OBAC
├── seenerl/                  # Core library and logic package
│   ├── algorithms/           # Implementations of RL algorithms
│   │   ├── base.py           # Base algorithm interface class
│   │   ├── obac.py           # Offline-Boosted Actor-Critic implementation    
│   │   ├── ppo.py            # Proximal Policy Optimization implementation
│   │   ├── sac.py            # Soft Actor-Critic implementation
│   │   └── td3.py            # Twin Delayed DDPG implementation
│   ├── networks/             # Neural network definitions and modules
│   │   ├── base.py           # Abstract BaseActor and BaseCritic
│   │   ├── mlp.py            # GaussianActor, DeterministicActor, and MLP Critics
│   │   └── registry.py       # Centralized registry for dynamic network loads
│   ├── buffers/              # Memory storage for transitions and rollouts
│   │   ├── replay_buffer.py  # Standard O(1) buffer for off-policy algorithms
│   │   └── rollout_buffer.py # Standard batching buffer for on-policy algorithms
│   ├── trainers/             # Training loop logic
│   │   ├── off_policy.py     # Single-env training loop for SAC/TD3/OBAC
│   │   └── on_policy.py      # Batch-gathering training loop for PPO
│   ├── evaluator.py          # Standalone robust action bounds evaluator
│   ├── checkpoint.py         # Multi-strategy checkpoint saving and restoring manager
│   ├── logger.py             # Handles writing to TensorBoard and Weights & Biases
│   ├── config.py             # Safe YAML config loading and CLI argument overriding
│   └── utils.py              # Assorted math, seeds, and device utilities
├── render/                   # Scripts and utilities to easily visualize models
│   └── renderer.py           # Evaluates algorithm with gymnasium's render_mode="human"
├── tests/                    # Minimal environment testing and code smoke tests
├── train.py                  # Main training entry point
├── evaluate.py               # Main evaluating entry point without rendering
└── pyproject.toml            # Project packaging metadata and pip dependencies
```

## Configuration

Configs use YAML with inheritance (`_base_` key) and runtime CLI overrides:

```yaml
# configs/sac.yaml
_base_: "default.yaml"    # Inherits all defaults
algo: "SAC"
alpha: 0.2
automatic_entropy_tuning: false
```

Override any parameter via CLI dynamically: `--key value` or `--nested.key value`.


## License

MIT