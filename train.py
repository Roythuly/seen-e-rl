"""
seenerl unified training entry point.

Usage:
    python train.py --config configs/sac.yaml
    python train.py --config configs/ppo.yaml --env_name Humanoid-v5 --seed 42
    python train.py --config configs/sac.yaml --resume results/xxx/checkpoints/latest.pt
"""

from seenerl.config import parse_args_and_load_config


def main():
    config = parse_args_and_load_config()

    algo = config.algo.upper()

    if algo in ("SAC", "TD3"):
        from seenerl.trainers.off_policy import OffPolicyTrainer
        trainer = OffPolicyTrainer(config)
    elif algo == "PPO":
        from seenerl.trainers.on_policy import OnPolicyTrainer
        trainer = OnPolicyTrainer(config)
    else:
        raise ValueError(f"Unknown algorithm: {config.algo}")

    trainer.train()


if __name__ == "__main__":
    main()
