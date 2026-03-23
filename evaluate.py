"""
Standalone evaluation entry point.

Usage:
    python evaluate.py --checkpoint results/xxx/checkpoints/best.pt \\
                       --config configs/sac.yaml --env_name Ant-v5 \\
                       --num_episodes 10
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Import sympy first to prevent gymnasium/torch import order conflicts
import sympy

import argparse

from seenerl.algorithms import build_algorithm
from seenerl.checkpoint import CheckpointManager
from seenerl.config import load_config
from seenerl.envs import create_env
from seenerl.evaluator import Evaluator


def parse_eval_args_and_load_config():
    """Parse known CLI args and forward remaining overrides into the config loader."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args, remaining = parser.parse_known_args()

    if not args.config:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        inferred_config_path = os.path.join(os.path.dirname(checkpoint_dir), "config.yaml")
        if not os.path.exists(inferred_config_path):
            raise ValueError(f"Could not find auto-inferred config at {inferred_config_path}.")
        args.config = inferred_config_path

    config = load_config(args.config, remaining)
    return args, config


def main():
    args, config = parse_eval_args_and_load_config()
    render_mode = "human" if args.render else None
    env = create_env(config, num_envs=1, render_mode=render_mode)
    agent = build_algorithm(config, env.observation_space, env.action_space)

    # Load checkpoint
    CheckpointManager.load(args.checkpoint, agent, evaluate=True)

    # Evaluate
    evaluator = Evaluator(env, agent)
    result = evaluator.evaluate(num_episodes=args.num_episodes)

    print("=" * 60)
    print(f"Environment: {config.env.id}")
    print(f"Algorithm:   {config.algo}")
    print(f"Episodes:    {args.num_episodes}")
    print(f"Avg Reward:  {result['avg_reward']:.2f}")
    print(f"Std Reward:  {result['std_reward']:.2f}")
    print(f"Min Reward:  {result['min_reward']:.2f}")
    print(f"Max Reward:  {result['max_reward']:.2f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
