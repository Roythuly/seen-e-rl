"""
On-policy trainer for PPO.

Training loop:
  1. Collect rollout_steps of data using the current policy
  2. Compute GAE advantages and returns
  3. Train for ppo_epoch epochs on collected data (mini-batched)
  4. Discard old data
  5. Repeat
"""

import datetime
import os

import gymnasium as gym
import numpy as np

from seenerl.algorithms.ppo import PPO
from seenerl.buffers.rollout_buffer import RolloutBuffer
from seenerl.checkpoint import CheckpointManager
from seenerl.config import Config
from seenerl.evaluator import Evaluator
from seenerl.logger import TrainingLogger
from seenerl.utils import set_seed


class OnPolicyTrainer:
    """
    On-policy training loop for PPO.

    Key difference from off-policy:
      - Collects a fixed number of steps (rollout)
      - Trains multiple epochs on collected data
      - Discards data after training
    """

    def __init__(self, config: Config):
        self.config = config

        set_seed(config.seed)

        # Environment
        self.env = gym.make(config.env_name)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Agent
        self.agent = PPO(obs_dim, self.env.action_space, config)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # Result directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(
            config.checkpoint.get("save_dir", "results"),
            config.env_name,
            config.algo,
            config.tag,
            f"{timestamp}_seed{config.seed}",
        )

        # Logger
        logger_cfg = config.get("logger", {})
        self.logger = TrainingLogger(
            log_dir=self.result_dir,
            config=dict(config),
            use_tensorboard=logger_cfg.get("use_tensorboard", True),
            use_wandb=logger_cfg.get("use_wandb", False),
            wandb_project=logger_cfg.get("wandb_project", "seen-e-rl"),
            wandb_entity=logger_cfg.get("wandb_entity", None),
            wandb_name=f"PPO_{config.env_name}_{timestamp}",
        )

        # Evaluator
        eval_env = gym.make(config.env_name)
        self.evaluator = Evaluator(eval_env, self.agent, self.logger)

        # Checkpoint manager
        ckpt_cfg = config.get("checkpoint", {})
        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(self.result_dir, "checkpoints"),
            strategies=ckpt_cfg.get("strategies", ["latest", "best"]),
            interval_steps=ckpt_cfg.get("interval_steps", None),
            interval_epochs=ckpt_cfg.get("interval_epochs", None),
            save_buffer=False,  # On-policy doesn't save buffer
        )

        # Training state
        self.total_steps = 0
        self.rollout_count = 0

        if config.get("resume"):
            self._resume(config.resume)

    def _resume(self, checkpoint_path: str) -> None:
        state = CheckpointManager.load(checkpoint_path, self.agent)
        self.total_steps = state["step"]
        self.rollout_count = state["epoch"]
        self.ckpt_manager.best_reward = state["best_reward"]
        self.logger.log_info(
            f"Resumed from {checkpoint_path}, step={self.total_steps}"
        )

    def _collect_rollout(self) -> float:
        """
        Collect rollout_steps of data using the current policy.

        Returns:
            Average episode reward during collection (for logging).
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_rewards = []
        episode_steps = 0

        for step in range(self.config.rollout_steps):
            action, log_prob, value = self.agent.select_action(state)

            next_state, reward, done, truncated, info = self.env.step(action)
            self.total_steps += 1
            episode_reward += reward
            episode_steps += 1

            self.rollout_buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=done or truncated,
                log_prob=log_prob,
                value=value,
            )

            state = next_state

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episode_steps = 0
                state, _ = self.env.reset()

        # Compute GAE with bootstrap value from last state
        last_value = self.agent.get_value(state)
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return float(np.mean(episode_rewards)) if episode_rewards else episode_reward

    def train(self) -> None:
        """Main on-policy training loop."""
        self.logger.log_info(
            f"Starting PPO training on {self.config.env_name} "
            f"(device: {self.agent.device})"
        )

        while self.total_steps < self.config.num_steps:
            self.rollout_count += 1

            # 1. Collect rollout
            avg_rollout_reward = self._collect_rollout()

            # 2. Train on collected data
            losses = self.agent.update_parameters(self.rollout_buffer)

            # 3. Reset buffer (discard old data)
            self.rollout_buffer.reset()

            # Log
            self.logger.log_train(
                step=self.total_steps,
                episode=self.rollout_count,
                episode_steps=self.config.rollout_steps,
                reward=avg_rollout_reward,
                **losses,
            )
            self.logger.log_scalar("train/avg_reward", avg_rollout_reward, self.total_steps)
            self.logger.log_dict(losses, self.total_steps, prefix="loss")

            # 4. Evaluate periodically
            if self.config.eval and self.rollout_count % self.config.eval_interval == 0:
                result = self.evaluator.evaluate(
                    num_episodes=self.config.eval_episodes,
                    step=self.total_steps,
                )
                self.ckpt_manager.save_if_needed(
                    agent=self.agent,
                    step=self.total_steps,
                    epoch=self.rollout_count,
                    eval_reward=result["avg_reward"],
                )

        self.env.close()
        self.logger.log_info("Training completed.")
        self.logger.close()
