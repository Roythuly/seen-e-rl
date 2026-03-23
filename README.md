# seen-e-rl

A modular reinforcement learning framework built on PyTorch for continuous control, now with:

- batched Gymnasium training
- Isaac Lab backend support
- algorithm and model factories
- shared batched trainer semantics across `SAC`, `TD3`, `PPO`, and `OBAC`

## Features

- YAML config loading with inheritance and CLI overrides
- Algorithm registry: scripts and trainers no longer hardcode algorithm classes
- Model factory: configurable actor / critic / value backbones through `model.*`
- Environment factory: a single batched interface for Gymnasium and Isaac Lab
- Parallel input support: Gym vector envs and Isaac Lab multi-env tasks share the same trainers
- Unified logging, evaluation, and checkpointing
- Legacy compatibility for existing `env_name` / `hidden_size` configs

## Installation

Using `uv` is recommended for the lightweight Gymnasium path:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e ".[dev]"
uv pip install -e ".[wandb]"
```

### Isaac Lab Runtime

Isaac Lab support is designed to run inside an Isaac Sim / Isaac Lab Python environment.

Validated local assumptions for the new smoke setup:

- `isaacsim 4.5.0.0`
- Isaac Lab source at `/home/seene/workspace/IsaacLab`
- GPU device available as `cuda:0`

The code launches Isaac Lab through `AppLauncher` and expects task packages to be imported explicitly for blacklisted tasks such as pick-place.
For the GR1T2 pick-place task, the runtime also applies two local compatibility patches:

- reuse cached USD -> URDF outputs when Isaac Lab asks for forced reconversion
- bypass Pink's `model.hasConfigurationLimit()` binding and derive the limit mask from the joint bounds instead

## Quick Start

### Gymnasium Training

Single-env legacy style still works:

```bash
uv run train.py --config configs/sac.yaml --env_name HalfCheetah-v4
uv run train.py --config configs/ppo.yaml --env_name Humanoid-v5
```

Parallel Gymnasium training uses the new nested env config:

```bash
uv run train.py --config configs/sac.yaml --env_name Pendulum-v1 --env.num_envs 8
uv run train.py --config configs/ppo.yaml --env_name Pendulum-v1 --env.num_envs 16
```

### Isaac Lab Training

Example configs are provided for `Isaac-PickPlace-GR1T2-Abs-v0`:

```bash
python train.py --config configs/isaaclab_pickplace_sac.yaml --env.num_envs 32
python train.py --config configs/isaaclab_pickplace_td3.yaml --env.num_envs 32
python train.py --config configs/isaaclab_pickplace_obac.yaml --env.num_envs 32
python train.py --config configs/isaaclab_pickplace_ppo.yaml --env.num_envs 32
```

The default Isaac Lab validation target is:

- `Isaac-PickPlace-GR1T2-Abs-v0`

`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` is not the default target anymore because it currently depends on a missing remote locomanipulation URDF asset in this environment.

### Evaluation

```bash
uv run evaluate.py --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --num_episodes 10
uv run evaluate.py --config configs/sac.yaml --env_name Ant-v5 --checkpoint results/Ant-v5/SAC/default/.../checkpoints/best.pt --num_episodes 10
python evaluate.py --config configs/isaaclab_pickplace_ppo.yaml --checkpoint results/xxx/checkpoints/latest.pt --num_episodes 3
```

### Rendering

```bash
uv run render/renderer.py --config configs/sac.yaml --checkpoint results/xxx/checkpoints/best.pt --episodes 5
uv run render/renderer.py --config configs/sac.yaml --env_name Ant-v5 --checkpoint results/Ant-v5/SAC/default/.../checkpoints/best.pt --episodes 5
```

## Architecture

The runtime is split into three layers:

1. Algorithms
   `seenerl.algorithms.registry` maps `algo -> class + trainer kind`
2. Models
   `seenerl.models` resolves `model.actor`, `model.critic`, and `model.value`
3. Environments
   `seenerl.envs` creates a batched adapter for Gymnasium or Isaac Lab

Trainer code now consumes one common env API:

```python
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(actions)
```

Where:

- `obs` is always batched
- `actions` are always batched
- `num_envs=1` is treated as a special case of batched execution

## Config Schema

### Environment

New normalized environment block:

```yaml
env:
  backend: "gymnasium"      # or "isaaclab"
  id: "Pendulum-v1"
  num_envs: 8
  kwargs: {}
  isaaclab:
    headless: true
    use_fabric: true
    task_imports: []
```

Backward compatibility is preserved:

- `env_name` still works
- CLI `--env_name Foo-v0` still maps to `env.id`
- `env.backend` defaults to `gymnasium`

### Models

New optional model block:

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
    kwargs: {}
  value:
    name: "value_network"
    hidden_dim: 256
    kwargs: {}
```

Legacy compatibility is preserved:

- `hidden_size` is still accepted
- SAC still honors `policy_type`
- if `model.*` is omitted, algorithm defaults are used

## Project Structure

```text
seen-e-rl/
├── configs/
│   ├── default.yaml
│   ├── sac.yaml
│   ├── td3.yaml
│   ├── ppo.yaml
│   ├── obac.yaml
│   ├── isaaclab_pickplace_base.yaml
│   ├── isaaclab_pickplace_sac.yaml
│   ├── isaaclab_pickplace_td3.yaml
│   ├── isaaclab_pickplace_ppo.yaml
│   └── isaaclab_pickplace_obac.yaml
├── seenerl/
│   ├── algorithms/
│   ├── buffers/
│   ├── envs/
│   ├── models/
│   ├── networks/
│   ├── trainers/
│   ├── checkpoint.py
│   ├── config.py
│   ├── evaluator.py
│   ├── logger.py
│   └── utils.py
├── render/
├── tests/
├── train.py
├── evaluate.py
└── pyproject.toml
```

## Testing

Default lightweight tests:

```bash
pytest -q tests/test_config_normalization.py tests/test_buffers_batched.py tests/test_algorithm_registry.py tests/test_parallel_gym_smoke.py
```

Opt-in Isaac Lab smoke tests:

```bash
SEENERL_RUN_ISAACLAB=1 pytest tests/test_isaaclab_pickplace.py -q
```

These tests expect a working Isaac Sim runtime and intentionally do not run by default.
On the current local `isaacsim 4.5.0.0` setup, `Isaac-PickPlace-GR1T2-Abs-v0` now gets through `parse_env_cfg`, but `gym.make(...)` still fails later during physics backend initialization with `Failed to create simulation view backend`.
The opt-in tests skip that known GR1T2 runtime failure instead of hanging indefinitely.

## Notes for Isaac Lab

- The code imports `isaacsim` before creating `AppLauncher`
- Pick-place tasks are imported explicitly through `env.isaaclab.task_imports`
- Pick-place startup reuses cached USD -> URDF artifacts to keep repeated GR1T2 launches practical
- Pick-place runtime patches Pink's configuration-limit helper to avoid the local `std::vector<bool>` binding failure
- The current default validation task is `Isaac-PickPlace-GR1T2-Abs-v0`
- `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` currently fails in this environment because the remote asset `g1_29dof_with_hand_only_kinematics.urdf` is unavailable
- On this specific `isaacsim 4.5.0.0` machine, `Isaac-PickPlace-GR1T2-Abs-v0` still stops inside Isaac Sim's physics backend after `parse_env_cfg`; upgrading Isaac Sim is the most likely fix for full end-to-end smoke training

## License

MIT
