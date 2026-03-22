"""
seenerl unified training entry point.

Usage:
    python train.py --config configs/sac.yaml
    python train.py --config configs/ppo.yaml --env_name Humanoid-v5 --seed 42
    python train.py --config configs/sac.yaml --resume results/xxx/checkpoints/latest.pt
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from seenerl.algorithms import get_algorithm_spec
from seenerl.config import parse_args_and_load_config


def main():
    config = parse_args_and_load_config()
    trainer_kind = get_algorithm_spec(config.algo).trainer_kind

    if trainer_kind == "off_policy":
        from seenerl.trainers.off_policy import OffPolicyTrainer

        trainer = OffPolicyTrainer(config)
    elif trainer_kind == "on_policy":
        from seenerl.trainers.on_policy import OnPolicyTrainer

        trainer = OnPolicyTrainer(config)
    else:
        raise ValueError(f"Unknown trainer kind: {trainer_kind}")

    trainer.train()


if __name__ == "__main__":
    main()
