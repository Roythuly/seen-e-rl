"""
Renderer module for model visualization.

Loads a checkpoint, runs episodes with rendering, and optionally records video.
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from typing import Optional

import numpy as np

from seenerl.algorithms import build_algorithm
from seenerl.checkpoint import CheckpointManager
from seenerl.config import load_config
from seenerl.envs import create_env


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
        self.env = create_env(self.config, num_envs=1, render_mode=render_mode)
        self.agent = build_algorithm(
            self.config,
            self.env.observation_space,
            self.env.action_space,
        )

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
            raise ValueError("record_dir is not supported by the batched renderer adapter.")

        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = np.array([False], dtype=np.bool_)
            truncated = np.array([False], dtype=np.bool_)

            while not bool(done[0] or truncated[0]):
                action = self.agent.select_action(state, evaluate=True)
                state, reward, done, truncated, info = env.step(action)
                episode_reward += float(np.asarray(reward).reshape(-1)[0])

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
