"""Batched on-policy trainer for PPO."""

import datetime
import os

import numpy as np

from seenerl.algorithms import build_algorithm
from seenerl.buffers.rollout_buffer import RolloutBuffer
from seenerl.checkpoint import CheckpointManager
from seenerl.config import Config
from seenerl.envs import create_env
from seenerl.evaluator import Evaluator
from seenerl.logger import TrainingLogger
from seenerl.utils import resolve_buffer_next_states, set_seed


class OnPolicyTrainer:
    """On-policy training loop for batched PPO environments."""

    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

        self.env = create_env(config)
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.agent = build_algorithm(config, self.env.observation_space, self.env.action_space)
        self.rollout_buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=self.env.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
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

        logger_cfg = config.get("logger", {})
        self.logger = TrainingLogger(
            log_dir=self.result_dir,
            config=dict(config),
            use_tensorboard=logger_cfg.get("use_tensorboard", True),
            use_wandb=logger_cfg.get("use_wandb", False),
            wandb_project=logger_cfg.get("wandb_project", "seen-e-rl"),
            wandb_entity=logger_cfg.get("wandb_entity", None),
            wandb_name=f"PPO_{env_id}_{timestamp}",
        )

        eval_env = create_env(config, num_envs=1)
        self.evaluator = Evaluator(eval_env, self.agent, self.logger)

        ckpt_cfg = config.get("checkpoint", {})
        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(self.result_dir, "checkpoints"),
            strategies=ckpt_cfg.get("strategies", ["latest", "best"]),
            interval_steps=ckpt_cfg.get("interval_steps", None),
            interval_epochs=ckpt_cfg.get("interval_epochs", None),
            save_buffer=False,
        )

        self.total_steps = 0
        self.rollout_count = 0
        self.completed_episodes = 0
        self._state = None
        self._episode_returns = np.zeros(self.env.num_envs, dtype=np.float32)

        if config.get("resume"):
            self._resume(config.resume)

    def _resume(self, checkpoint_path: str) -> None:
        state = CheckpointManager.load(checkpoint_path, self.agent)
        self.total_steps = state["step"]
        self.rollout_count = state["epoch"]
        self.ckpt_manager.best_reward = state["best_reward"]
        self.logger.log_info(f"Resumed from {checkpoint_path}, step={self.total_steps}")

    def _collect_rollout(self):
        """Collect a batched rollout."""
        if self._state is None:
            self._state, _ = self.env.reset(seed=self.config.seed)

        num_envs = self.env.num_envs
        episode_return_log = []
        rollout_reward_sum = 0.0

        for _ in range(self.config.rollout_steps):
            action, log_prob, value, raw_action = self.agent.select_action(self._state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            reward = np.asarray(reward, dtype=np.float32).reshape(num_envs)
            terminated = np.asarray(terminated, dtype=np.bool_).reshape(num_envs)
            truncated = np.asarray(truncated, dtype=np.bool_).reshape(num_envs)
            done = terminated | truncated
            log_prob = np.asarray(log_prob, dtype=np.float32).reshape(num_envs)
            value = np.asarray(value, dtype=np.float32).reshape(num_envs)
            buffer_next_state = resolve_buffer_next_states(next_state, info)
            next_value = np.asarray(
                self.agent.get_value(buffer_next_state),
                dtype=np.float32,
            ).reshape(num_envs)

            self.rollout_buffer.add(
                state=self._state,
                action=np.asarray(raw_action, dtype=np.float32),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                value=value,
                next_value=next_value,
            )

            self.total_steps += num_envs
            self._episode_returns += reward
            rollout_reward_sum += float(reward.sum())

            for env_index in np.flatnonzero(done):
                self.completed_episodes += 1
                episode_return_log.append(float(self._episode_returns[env_index]))
                self._episode_returns[env_index] = 0.0

            self._state = next_state

        self.rollout_buffer.compute_returns_and_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        avg_episode_reward = None
        if episode_return_log:
            avg_episode_reward = float(np.mean(episode_return_log))

        return {
            "avg_episode_reward": avg_episode_reward,
            "avg_step_reward": rollout_reward_sum / (self.config.rollout_steps * num_envs),
            "completed_episodes": len(episode_return_log),
        }

    def train(self) -> None:
        """Main on-policy training loop."""
        self.logger.log_info(
            f"Starting PPO training on {self.config.env.id} "
            f"(num_envs={self.env.num_envs}, device: {self.agent.device})"
        )

        while self.total_steps < self.config.num_steps:
            self.rollout_count += 1

            rollout_metrics = self._collect_rollout()
            losses = self.agent.update_parameters(self.rollout_buffer)
            self.rollout_buffer.reset()

            avg_episode_reward = rollout_metrics["avg_episode_reward"]
            avg_step_reward = float(rollout_metrics["avg_step_reward"])
            completed_episodes = int(rollout_metrics["completed_episodes"])
            reward_metric = "episode_return"
            logged_reward = avg_episode_reward
            if logged_reward is None:
                reward_metric = "step_reward"
                logged_reward = avg_step_reward

            self.logger.log_train(
                step=self.total_steps,
                episode=self.rollout_count,
                episode_steps=self.config.rollout_steps * self.env.num_envs,
                reward=float(logged_reward),
                reward_metric=reward_metric,
                completed_episodes=completed_episodes,
                **losses,
            )
            self.logger.log_scalar("train/avg_reward", float(logged_reward), self.total_steps)
            self.logger.log_scalar("train/avg_step_reward", avg_step_reward, self.total_steps)
            if avg_episode_reward is not None:
                self.logger.log_scalar(
                    "train/avg_episode_reward",
                    float(avg_episode_reward),
                    self.total_steps,
                )
            self.logger.log_dict(losses, self.total_steps, prefix="loss")

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
