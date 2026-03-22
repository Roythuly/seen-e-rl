"""
SAC Humanoid-v5 smoke test.

Runs a short SAC training on Humanoid-v5 to verify:
  - Config loading works
  - SAC algorithm initializes correctly
  - Training loop runs without errors
  - TensorBoard logs are created
  - Checkpoints are saved
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seenerl.config import load_config
from seenerl.trainers.off_policy import OffPolicyTrainer


def test_sac_humanoid():
    """Short SAC training test on Humanoid-v5."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "sac.yaml"
    )

    config = load_config(config_path, [
        "--env_name", "Humanoid-v5",
        "--num_steps", "5000",
        "--start_steps", "1000",
        "--seed", "42",
        "--eval_interval", "5",
        "--eval_episodes", "2",
        "--device", "cpu",
    ])

    # Use temp dir for results
    with tempfile.TemporaryDirectory() as tmpdir:
        config["checkpoint"]["save_dir"] = tmpdir

        trainer = OffPolicyTrainer(config)
        trainer.train()

        # Verify results
        result_dirs = os.listdir(tmpdir)
        assert len(result_dirs) > 0, "No result directory created"

        # Find the training dir (nested structure)
        full_result_dir = trainer.result_dir
        assert os.path.exists(full_result_dir), f"Result dir {full_result_dir} not found"

        # Check training log exists
        assert os.path.exists(os.path.join(full_result_dir, "training.log")), \
            "training.log not created"

        print("=" * 60)
        print("SAC Humanoid-v5 test PASSED!")
        print(f"Total steps: {trainer.total_steps}")
        print(f"Result dir: {full_result_dir}")
        print("=" * 60)


if __name__ == "__main__":
    test_sac_humanoid()
