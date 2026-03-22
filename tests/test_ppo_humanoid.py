"""
PPO Humanoid-v5 smoke test.

Runs a short PPO training on Humanoid-v5 to verify:
  - Config loading works
  - PPO algorithm initializes correctly
  - On-policy rollout collection works
  - GAE computation runs correctly
  - Multi-epoch mini-batch training works
  - Checkpoints and logs are created
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seenerl.config import load_config
from seenerl.trainers.on_policy import OnPolicyTrainer


def test_ppo_humanoid():
    """Short PPO training test on Humanoid-v5."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "ppo.yaml"
    )

    config = load_config(config_path, [
        "--env_name", "Humanoid-v5",
        "--num_steps", "5000",
        "--rollout_steps", "512",
        "--ppo_epoch", "4",
        "--num_mini_batch", "4",
        "--seed", "42",
        "--eval_interval", "2",
        "--eval_episodes", "2",
        "--device", "cpu",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        config["checkpoint"]["save_dir"] = tmpdir

        trainer = OnPolicyTrainer(config)
        trainer.train()

        full_result_dir = trainer.result_dir
        assert os.path.exists(full_result_dir), f"Result dir {full_result_dir} not found"

        assert os.path.exists(os.path.join(full_result_dir, "training.log")), \
            "training.log not created"

        print("=" * 60)
        print("PPO Humanoid-v5 test PASSED!")
        print(f"Total steps: {trainer.total_steps}")
        print(f"Rollout count: {trainer.rollout_count}")
        print(f"Result dir: {full_result_dir}")
        print("=" * 60)


if __name__ == "__main__":
    test_ppo_humanoid()
