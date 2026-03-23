"""Parallel Gym smoke tests across all supported algorithms."""

import os
import tempfile

from seenerl.config import load_config
from seenerl.trainers.off_policy import OffPolicyTrainer
from seenerl.trainers.on_policy import OnPolicyTrainer


def test_parallel_sac_smoke():
    _run_off_policy_smoke("configs/sac.yaml")


def test_parallel_td3_smoke():
    _run_off_policy_smoke("configs/td3.yaml")


def test_parallel_obac_smoke():
    _run_off_policy_smoke("configs/obac.yaml")


def test_parallel_ppo_smoke():
    config = load_config(
        "configs/ppo.yaml",
        [
            "--env_name", "Pendulum-v1",
            "--env.num_envs", "2",
            "--num_steps", "64",
            "--rollout_steps", "8",
            "--ppo_epoch", "2",
            "--num_mini_batch", "2",
            "--eval", "false",
            "--device", "cpu",
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config["checkpoint"]["save_dir"] = tmpdir
        trainer = OnPolicyTrainer(config)
        trainer.train()

        assert trainer.total_steps == 64
        assert os.path.exists(os.path.join(trainer.result_dir, "training.log"))


def _run_off_policy_smoke(config_path: str) -> None:
    config = load_config(
        config_path,
        [
            "--env_name", "Pendulum-v1",
            "--env.num_envs", "2",
            "--num_steps", "24",
            "--start_steps", "0",
            "--batch_size", "4",
            "--replay_size", "128",
            "--eval", "false",
            "--device", "cpu",
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config["checkpoint"]["save_dir"] = tmpdir
        trainer = OffPolicyTrainer(config)
        trainer.train()

        assert trainer.total_steps == 24
        assert os.path.exists(os.path.join(trainer.result_dir, "training.log"))
