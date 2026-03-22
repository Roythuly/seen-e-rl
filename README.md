# seen-e-rl

A modular reinforcement learning framework built on PyTorch, supporting state-of-the-art continuous control algorithms:

- **SAC** (Soft Actor-Critic) — Off-policy, entropy-maximized
- **TD3** (Twin Delayed DDPG) — Off-policy, deterministically exploring
- **PPO** (Proximal Policy Optimization) — On-policy, clipped surrogate objective

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

```
seen-e-rl/
├── configs/                  # YAML configurations (default, sac, td3, ppo)
├── seenerl/                  # Main package
│   ├── algorithms/           # SAC, TD3, PPO implementations
│   ├── networks/             # MLP, registry (extensible)
│   ├── buffers/              # ReplayBuffer, RolloutBuffer
│   ├── trainers/             # Off-policy, On-policy trainers
│   ├── evaluator.py          # Standalone robust evaluator
│   ├── checkpoint.py         # Multi-strategy checkpoint manager
│   ├── logger.py             # TensorBoard + wandb logging
│   ├── config.py             # YAML loader
│   └── utils.py              # Common utilities
├── render/                   # Rendering submodule
├── tests/                    # Smoke tests
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
└── pyproject.toml            # Project metadata
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