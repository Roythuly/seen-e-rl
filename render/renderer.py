"""
Renderer module for model visualization.

Loads a checkpoint, runs episodes with rendering, and optionally records video.
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from typing import Optional

import gymnasium as gym
import numpy as np

from seenerl.algorithms.sac import SAC
from seenerl.algorithms.td3 import TD3
from seenerl.algorithms.ppo import PPO
from seenerl.algorithms.obac import OBAC
from seenerl.checkpoint import CheckpointManager
from seenerl.config import load_config, Config


class Renderer:
    """
    Renders a trained agent in its environment.

    Supports human-mode rendering and video recording via gymnasium wrappers.
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Args:
            config_path: Path to YAML config file.
            checkpoint_path: Path to checkpoint .pt file.
        """
        self.config = load_config(config_path)
        self.config["device"] = "cpu"

        render_mode = "human"
        self.env = gym.make(self.config.env_name, render_mode=render_mode)

        obs_dim = self.env.observation_space.shape[0]
        algo = self.config.algo.upper()

        if algo == "SAC":
            self.agent = SAC(obs_dim, self.env.action_space, self.config)
        elif algo == "TD3":
            self.agent = TD3(obs_dim, self.env.action_space, self.config)
        elif algo == "PPO":
            self.agent = PPO(obs_dim, self.env.action_space, self.config)
        elif algo == "OBAC":
            self.agent = OBAC(obs_dim, self.env.action_space, self.config)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algo}")

        CheckpointManager.load(checkpoint_path, self.agent, evaluate=True)

    def run(self, num_episodes: int = 5, record_dir: Optional[str] = None) -> None:
        """
        Run and render episodes.

        Args:
            num_episodes: Number of episodes to render.
            record_dir: If provided, wrap with RecordVideo.
        """
        env = self.env
        if record_dir:
            env = gym.wrappers.RecordVideo(env, record_dir)

        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            truncated = False

            while not (done or truncated):
                action = self.agent.select_action(state, evaluate=True)

                if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                    clipped_action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    )
                else:
                    clipped_action = action

                state, reward, done, truncated, info = env.step(clipped_action)
                episode_reward += reward

            print(f"Episode {ep + 1}: reward = {episode_reward:.2f}")

        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a trained agent")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--record_dir", type=str, default=None)
    args = parser.parse_args()

    renderer = Renderer(args.config, args.checkpoint)
    renderer.run(num_episodes=args.episodes, record_dir=args.record_dir)
