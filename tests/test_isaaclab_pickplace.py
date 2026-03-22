"""Isaac Lab pick-place smoke tests.

These tests are intentionally opt-in because they require a working Isaac Sim runtime.
Run them with:

    SEENERL_RUN_ISAACLAB=1 pytest tests/test_isaaclab_pickplace.py -q
"""

import importlib.util
import os
import tempfile

import pytest

from seenerl.config import load_config
from seenerl.envs import create_env
from seenerl.envs.runtime import release_isaaclab_app
from seenerl.trainers.off_policy import OffPolicyTrainer
from seenerl.trainers.on_policy import OnPolicyTrainer


RUN_ISAACLAB = os.environ.get("SEENERL_RUN_ISAACLAB") == "1"
HAS_ISAACSIM = importlib.util.find_spec("isaacsim") is not None


def _skip_known_gr1t2_runtime_issue(exc: BaseException) -> None:
    """Skip Isaac Sim 4.5-specific GR1T2 failures that happen after config parsing."""
    message = str(exc)
    known_tokens = (
        "Failed to create simulation view backend",
        "Failed to get DOF velocities from backend",
        "create_articulation_view",
        "Simulation context already exists. Cannot create a new one.",
    )
    if any(token in message for token in known_tokens):
        try:
            from isaaclab.sim import SimulationContext

            if SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
        except Exception:
            pass
        release_isaaclab_app()
        pytest.skip(
            "Isaac-PickPlace-GR1T2-Abs-v0 reaches parse_env_cfg on this machine, "
            "but Isaac Sim 4.5 fails during physics backend initialization.",
        )


@pytest.mark.skipif(not HAS_ISAACSIM or not RUN_ISAACLAB, reason="Isaac Lab smoke disabled")
def test_isaaclab_pickplace_reset():
    config = load_config(
        "configs/isaaclab_pickplace_ppo.yaml",
        ["--env.num_envs", "1", "--device", "cuda:0"],
    )
    env = None
    try:
        env = create_env(config, num_envs=1)
        obs, _ = env.reset(seed=0)
    except Exception as exc:  # pragma: no cover - depends on local Isaac Sim runtime
        _skip_known_gr1t2_runtime_issue(exc)
        raise
    finally:
        if env is not None:
            env.close()
    assert obs.shape[0] == 1


@pytest.mark.skipif(not HAS_ISAACSIM or not RUN_ISAACLAB, reason="Isaac Lab smoke disabled")
@pytest.mark.parametrize(
    ("config_path", "trainer_cls", "extra_overrides"),
    [
        (
            "configs/isaaclab_pickplace_sac.yaml",
            OffPolicyTrainer,
            [
                "--env.num_envs", "1",
                "--num_steps", "4",
                "--start_steps", "0",
                "--batch_size", "2",
                "--replay_size", "32",
                "--eval", "false",
                "--device", "cuda:0",
            ],
        ),
        (
            "configs/isaaclab_pickplace_td3.yaml",
            OffPolicyTrainer,
            [
                "--env.num_envs", "1",
                "--num_steps", "4",
                "--start_steps", "0",
                "--batch_size", "2",
                "--replay_size", "32",
                "--eval", "false",
                "--device", "cuda:0",
            ],
        ),
        (
            "configs/isaaclab_pickplace_obac.yaml",
            OffPolicyTrainer,
            [
                "--env.num_envs", "1",
                "--num_steps", "4",
                "--start_steps", "0",
                "--batch_size", "2",
                "--replay_size", "32",
                "--eval", "false",
                "--device", "cuda:0",
            ],
        ),
        (
            "configs/isaaclab_pickplace_ppo.yaml",
            OnPolicyTrainer,
            [
                "--env.num_envs", "1",
                "--num_steps", "4",
                "--rollout_steps", "4",
                "--ppo_epoch", "1",
                "--num_mini_batch", "1",
                "--eval", "false",
                "--device", "cuda:0",
            ],
        ),
    ],
)
def test_isaaclab_pickplace_short_training(config_path, trainer_cls, extra_overrides):
    config = load_config(config_path, extra_overrides)
    with tempfile.TemporaryDirectory() as tmpdir:
        config["checkpoint"]["save_dir"] = tmpdir
        try:
            trainer = trainer_cls(config)
            trainer.train()
        except Exception as exc:  # pragma: no cover - depends on local Isaac Sim runtime
            _skip_known_gr1t2_runtime_issue(exc)
            raise
        assert os.path.exists(os.path.join(trainer.result_dir, "training.log"))
