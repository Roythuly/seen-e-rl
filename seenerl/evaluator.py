"""
Independent evaluator module.

Decoupled from trainers — can be called during training or standalone
via evaluate.py entry point.
"""

from typing import Dict, List

import numpy as np

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.logger import TrainingLogger


class Evaluator:
    """
    Evaluates an agent in an environment.

    Returns statistics: avg_reward, std_reward, min_reward, max_reward.
    """

    def __init__(self, env, agent: BaseAlgorithm,
                 logger: TrainingLogger = None):
        """
        Args:
            env: Gymnasium environment.
            agent: RL algorithm instance.
            logger: Optional logger for recording metrics.
        """
        self.env = env
        self.agent = agent
        self.logger = logger

    def evaluate(self, num_episodes: int, step: int = 0,
                 render: bool = False) -> Dict[str, float]:
        """
        Run evaluation episodes.

        Args:
            num_episodes: Number of episodes to run.
            step: Current training step (for logging).
            render: Whether to render the environment.

        Returns:
            Dict with avg_reward, std_reward, min_reward, max_reward.
        """
        episode_rewards: List[float] = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            done = np.array([False], dtype=np.bool_)
            truncated = np.array([False], dtype=np.bool_)

            while not bool(done[0] or truncated[0]):
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += float(np.asarray(reward).reshape(-1)[0])
                state = next_state

            episode_rewards.append(episode_reward)

        rewards = np.array(episode_rewards)
        result = {
            "avg_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
        }

        # Log metrics
        if self.logger is not None:
            self.logger.log_eval(
                step=step,
                avg_reward=result["avg_reward"],
                std_reward=result["std_reward"],
                num_episodes=num_episodes,
            )
            self.logger.log_scalar("eval/avg_reward", result["avg_reward"], step)
            self.logger.log_scalar("eval/std_reward", result["std_reward"], step)

        return result
