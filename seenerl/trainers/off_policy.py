"""
Off-policy trainer for SAC and TD3.

Training loop:
  1. Interact with environment step-by-step
  2. Store transitions in ReplayBuffer
  3. When buffer is large enough, sample and update networks
  4. Periodically evaluate and save checkpoints
"""

import datetime
import itertools
import os

import gymnasium as gym
import numpy as np

from seenerl.algorithms.base import BaseAlgorithm
from seenerl.buffers.replay_buffer import ReplayBuffer
from seenerl.checkpoint import CheckpointManager
from seenerl.config import Config
from seenerl.evaluator import Evaluator
from seenerl.logger import TrainingLogger
from seenerl.utils import set_seed, resolve_device


class OffPolicyTrainer:
    """
    Off-policy training loop for SAC / TD3.

    Handles: environment interaction, buffer management, network updates,
    evaluation, checkpointing, and logging.
    """

    def __init__(self, config: Config):
        self.config = config

        # Seed
        set_seed(config.seed)

        # Environment
        self.env = gym.make(config.env_name)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Agent
        algo_name = config.algo.upper()
        if algo_name == "SAC":
            from seenerl.algorithms.sac import SAC
            self.agent = SAC(obs_dim, self.env.action_space, config)
        elif algo_name == "TD3":
            from seenerl.algorithms.td3 import TD3
            self.agent = TD3(obs_dim, self.env.action_space, config)
        else:
            raise ValueError(f"Unknown off-policy algorithm: {config.algo}")

        # Buffer
        self.memory = ReplayBuffer(
            capacity=config.replay_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=config.seed,
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
            wandb_name=f"{config.algo}_{config.env_name}_{timestamp}",
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
            save_buffer=ckpt_cfg.get("save_buffer", True),
        )

        # Training state
        self.total_steps = 0
        self.updates = 0
        self.best_reward = -float("inf")

        # Resume if specified
        if config.get("resume"):
            self._resume(config.resume)

    def _resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = CheckpointManager.load(
            checkpoint_path, self.agent, self.memory
        )
        self.total_steps = state["step"]
        self.best_reward = state["best_reward"]
        self.ckpt_manager.best_reward = self.best_reward
        self.logger.log_info(
            f"Resumed from {checkpoint_path}, step={self.total_steps}, "
            f"best_reward={self.best_reward:.2f}"
        )

    def train(self) -> None:
        """Main training loop."""
        self.logger.log_info(
            f"Starting {self.config.algo} training on {self.config.env_name} "
            f"(device: {self.agent.device})"
        )

        for i_episode in itertools.count(1):
            episode_reward = 0.0
            episode_steps = 0
            done = False
            truncated = False
            state, _ = self.env.reset(seed=self.config.seed + i_episode)

            while not (done or truncated):
                # Action selection
                if self.total_steps < self.config.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)

                # Environment step
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_steps += 1
                self.total_steps += 1
                episode_reward += reward

                # Done masking (ignore timeout signal)
                mask = 0.0 if done and not truncated else 1.0

                self.memory.push(state, action, reward, next_state, mask)
                state = next_state

                # Update networks
                if len(self.memory) > self.config.batch_size:
                    for _ in range(self.config.updates_per_step):
                        losses = self.agent.update_parameters(
                            self.memory, self.config.batch_size, self.updates
                        )
                        self.logger.log_dict(losses, self.updates, prefix="loss")
                        self.updates += 1

            if self.total_steps > self.config.num_steps:
                break

            # Log training episode
            self.logger.log_train(
                step=self.total_steps,
                episode=i_episode,
                episode_steps=episode_steps,
                reward=episode_reward,
            )
            self.logger.log_scalar("train/reward", episode_reward, self.total_steps)

            # Evaluation
            if self.config.eval and i_episode % self.config.eval_interval == 0:
                result = self.evaluator.evaluate(
                    num_episodes=self.config.eval_episodes,
                    step=self.total_steps,
                )
                # Checkpoint
                self.ckpt_manager.save_if_needed(
                    agent=self.agent,
                    step=self.total_steps,
                    epoch=i_episode,
                    eval_reward=result["avg_reward"],
                    buffer=self.memory,
                )

        self.env.close()
        self.logger.log_info("Training completed.")
        self.logger.close()
