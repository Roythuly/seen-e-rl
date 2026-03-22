from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rl_training_infra.trainer.base import TorchLearnerBase, as_bool_tensor, as_float_tensor, distribution_log_prob


@dataclass(slots=True)
class PPOLearner(TorchLearnerBase):
    optimizer: torch.optim.Optimizer | None = None

    def __post_init__(self) -> None:
        TorchLearnerBase.__post_init__(self)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get("learning_rate", 3e-4)))

    def optimizer_state(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict()}

    def checkpoint_components(self) -> list[str]:
        return ["actor", "critic", "optimizers", "runtime_state"]

    def update(self, batch: dict[str, Any], objective: dict[str, Any] | None = None) -> dict[str, Any]:
        observations = as_float_tensor(batch["observations"])
        actions = as_float_tensor(batch["actions"])
        rewards = as_float_tensor(batch["rewards"]).view(-1)
        terminated = as_bool_tensor(batch["terminated"]).view(-1)
        truncated = as_bool_tensor(batch["truncated"]).view(-1)
        old_log_probs = as_float_tensor(batch["log_probs"]).view(-1)
        old_values = as_float_tensor(batch["value_estimates"]).view(-1)

        gamma = float(self.config.get("gamma", 0.99))
        gae_lambda = float(self.config.get("gae_lambda", 0.95))
        clip_ratio = float(self.config.get("clip_ratio", 0.2))
        value_coef = float(self.config.get("value_coef", 0.5))
        entropy_coef = float(self.config.get("entropy_coef", 0.0))
        epochs = int(self.config.get("epochs", 1))
        minibatch_size = int(self.config.get("minibatch_size", observations.shape[0]))

        advantages = torch.zeros_like(rewards)
        last_advantage = torch.tensor(0.0)
        next_value = torch.tensor(0.0)
        done = torch.logical_or(terminated, truncated)
        for index in range(rewards.shape[0] - 1, -1, -1):
            mask = 1.0 - done[index].float()
            delta = rewards[index] + gamma * next_value * mask - old_values[index]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[index] = last_advantage
            next_value = old_values[index]

        returns = advantages + old_values
        advantages_std = advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = (advantages - advantages.mean()) / advantages_std

        last_policy_loss = torch.tensor(0.0)
        last_value_loss = torch.tensor(0.0)
        last_entropy = torch.tensor(0.0)
        last_kl = torch.tensor(0.0)
        for _ in range(epochs):
            permutation = torch.randperm(observations.shape[0])
            for start in range(0, observations.shape[0], minibatch_size):
                indices = permutation[start : start + minibatch_size]
                train_outputs = self.model.forward_train({"observations": observations[indices]})
                distribution_params = train_outputs["policy"]["distribution_params"]
                new_log_probs, entropy = distribution_log_prob(distribution_params, actions[indices])
                values = train_outputs["value"]["state_values"].view(-1)
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                surrogate = torch.min(ratio * advantages[indices], clipped_ratio * advantages[indices])
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(values, returns[indices])
                entropy_value = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_value

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.grad_steps += 1
                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()
                last_entropy = entropy_value.detach()
                last_kl = (old_log_probs[indices] - new_log_probs).mean().detach()

        self.env_steps = self._resolved_env_steps(batch, objective)
        self._advance_policy_version()
        return self._build_update_result(
            status="ok",
            published_policy=False,
            metrics={
                "policy_loss": float(last_policy_loss),
                "value_loss": float(last_value_loss),
                "entropy": float(last_entropy),
                "approx_kl": float(last_kl),
                "actor_updated": True,
            },
        )


__all__ = ["PPOLearner"]
