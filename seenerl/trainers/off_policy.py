"""Batched off-policy trainer for SAC, TD3, and OBAC."""

import datetime
import os

import numpy as np

from seenerl.algorithms import build_algorithm
from seenerl.buffers.replay_buffer import ReplayBuffer
from seenerl.checkpoint import CheckpointManager
from seenerl.config import Config, save_config
from seenerl.envs import create_env
from seenerl.evaluator import Evaluator
from seenerl.logger import TrainingLogger
from seenerl.utils import (
    resolve_buffer_next_states,
    sample_batched_actions,
    set_seed,
)


class OffPolicyTrainer:
    """Off-policy training loop for batched Gymnasium / Isaac Lab envs."""

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

        self.env = create_env(config)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.agent = build_algorithm(config, self.env.observation_space, self.env.action_space)
        self.memory = ReplayBuffer(
            capacity=config.replay_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=config.seed,
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env_id = config.env.id
        self.result_dir = os.path.join(
            config.checkpoint.get("save_dir", "results"),
            env_id,
            config.algo,
            config.tag,
            f"{timestamp}_seed{config.seed}",
        )
        save_config(config, self.result_dir)

        logger_cfg = config.get("logger", {})
        self.logger = TrainingLogger(
            log_dir=self.result_dir,
            config=dict(config),
            use_tensorboard=logger_cfg.get("use_tensorboard", True),
            use_wandb=logger_cfg.get("use_wandb", False),
            wandb_project=logger_cfg.get("wandb_project", "seen-e-rl"),
            wandb_entity=logger_cfg.get("wandb_entity", None),
            wandb_name=f"{config.algo}_{env_id}_{timestamp}",
        )

        eval_env = create_env(config, num_envs=1)
        self.evaluator = Evaluator(eval_env, self.agent, self.logger)

        ckpt_cfg = config.get("checkpoint", {})
        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(self.result_dir, "checkpoints"),
            strategies=ckpt_cfg.get("strategies", ["latest", "best"]),
            interval_steps=ckpt_cfg.get("interval_steps", None),
            interval_epochs=ckpt_cfg.get("interval_epochs", None),
            save_buffer=ckpt_cfg.get("save_buffer", True),
        )

        self.total_steps = 0
        self.updates = 0
        self.best_reward = -float("inf")
        self.completed_episodes = 0

        if config.get("resume"):
            self._resume(config.resume)

    def _resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = CheckpointManager.load(checkpoint_path, self.agent, self.memory)
        self.total_steps = state["step"]
        self.completed_episodes = state["epoch"]
        self.best_reward = state["best_reward"]
        self.ckpt_manager.best_reward = self.best_reward
        self.logger.log_info(
            f"Resumed from {checkpoint_path}, step={self.total_steps}, "
            f"best_reward={self.best_reward:.2f}"
        )

    def _maybe_evaluate(self) -> None:
        if not self.config.eval:
            return
        if self.completed_episodes == 0:
            return
        if self.completed_episodes % self.config.eval_interval != 0:
            return

        result = self.evaluator.evaluate(
            num_episodes=self.config.eval_episodes,
            step=self.total_steps,
        )
        self.ckpt_manager.save_if_needed(
            agent=self.agent,
            step=self.total_steps,
            epoch=self.completed_episodes,
            eval_reward=result["avg_reward"],
            buffer=self.memory,
        )

    def train(self) -> None:
        """Main off-policy training loop."""
        num_envs = self.env.num_envs
        episode_rewards = np.zeros(num_envs, dtype=np.float32)
        episode_steps = np.zeros(num_envs, dtype=np.int64)
        state, _ = self.env.reset(seed=self.config.seed)

        self.logger.log_info(
            f"Starting {self.config.algo} training on {self.config.env.id} "
            f"(num_envs={num_envs}, device: {self.agent.device})"
        )

        while self.total_steps < self.config.num_steps:
            if self.total_steps < self.config.start_steps:
                action = sample_batched_actions(self.env.action_space, num_envs)
            else:
                action = self.agent.select_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            reward = np.asarray(reward, dtype=np.float32).reshape(num_envs)
            terminated = np.asarray(terminated, dtype=np.bool_).reshape(num_envs)
            truncated = np.asarray(truncated, dtype=np.bool_).reshape(num_envs)
            done = terminated | truncated

            buffer_next_state = resolve_buffer_next_states(next_state, info)
            mask = (~done).astype(np.float32)
            self.memory.add_batch(state, action, reward, buffer_next_state, mask)

            self.total_steps += num_envs
            episode_rewards += reward
            episode_steps += 1

            if len(self.memory) >= self.config.batch_size:
                num_updates = max(int(self.config.updates_per_step * num_envs), 1)
                for _ in range(num_updates):
                    losses = self.agent.update_parameters(
                        self.memory, self.config.batch_size, self.updates
                    )
                    self.logger.log_dict(losses, self.updates, prefix="loss")
                    self.updates += 1

            finished_envs = np.flatnonzero(done)
            for env_index in finished_envs:
                self.completed_episodes += 1
                self.logger.log_train(
                    step=self.total_steps,
                    episode=self.completed_episodes,
                    episode_steps=int(episode_steps[env_index]),
                    reward=float(episode_rewards[env_index]),
                )
                self.logger.log_scalar(
                    "train/reward",
                    float(episode_rewards[env_index]),
                    self.total_steps,
                )
                episode_rewards[env_index] = 0.0
                episode_steps[env_index] = 0
                self._maybe_evaluate()

            state = next_state

        self.env.close()
        self.logger.log_info("Training completed.")
        self.logger.close()
