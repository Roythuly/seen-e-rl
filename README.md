# seen-e-rl

A modular reinforcement learning framework built on PyTorch, supporting:

- **SAC** (Soft Actor-Critic) — off-policy
- **TD3** (Twin Delayed DDPG) — off-policy
- **PPO** (Proximal Policy Optimization) — on-policy

## Features

- 📝 **YAML Configuration** — Hierarchical configs with CLI override support
- 🧩 **Modular Architecture** — Pluggable algorithms, networks, buffers, and trainers
- 📊 **Unified Logging** — TensorBoard + wandb with standardized `[TRAIN]`/`[EVAL]` output
- 💾 **Checkpoint Management** — Multiple strategies (latest, best, interval)
- 🔄 **Training Resumption** — Save/load buffer state for seamless resume
- 🔌 **Extensible Networks** — Registry pattern for custom architectures (VLA, Transformer)

## Quick Start

### Installation

```bash
pip install -e .
# Or with wandb support:
pip install -e ".[wandb]"
```

### Training

```bash
# SAC on HalfCheetah
python train.py --config configs/sac.yaml

# PPO on Humanoid-v5
python train.py --config configs/ppo.yaml --env_name Humanoid-v5

# TD3 with custom parameters
python train.py --config configs/td3.yaml --env_name Ant-v5 --seed 123

# Resume training from checkpoint
python train.py --config configs/sac.yaml --resume results/xxx/checkpoints/latest.pt
```

### Evaluation

```bash
python evaluate.py --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --num_episodes 10
```

### Rendering

```bash
python -m render.renderer --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --episodes 3
```

## Project Structure

```
seen-e-rl/
├── configs/                  # YAML configurations
│   ├── default.yaml          # Shared defaults
│   ├── sac.yaml              # SAC-specific
│   ├── td3.yaml              # TD3-specific
│   └── ppo.yaml              # PPO-specific
├── seenerl/                  # Main package
│   ├── algorithms/           # SAC, TD3, PPO + base class
│   ├── networks/             # MLP, registry (extensible)
│   ├── buffers/              # ReplayBuffer, RolloutBuffer
│   ├── trainers/             # Off-policy, On-policy trainers
│   ├── evaluator.py          # Standalone evaluator
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

Configs use YAML with inheritance (`_base_` key) and CLI overrides:

```yaml
# configs/sac.yaml
_base_: "default.yaml"    # Inherits all defaults
algo: "SAC"
alpha: 0.2
automatic_entropy_tuning: false
```

Override any parameter via CLI: `--key value` or `--nested.key value`.

## Adding a New Algorithm

1. Inherit from `BaseAlgorithm` in `seenerl/algorithms/`
2. Implement `select_action`, `update_parameters`, `get_state_dict`, `load_state_dict`
3. Add a YAML config in `configs/`
4. Register in the trainer dispatch logic in `train.py`

## Adding a Custom Network

```python
from seenerl.networks.registry import register_actor

@register_actor("my_vla_actor")
class VLAActor(BaseActor):
    def __init__(self, ...): ...
    def forward(self, state): ...
    def sample(self, state): ...
```

## License

MIT