from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rl_training_infra.trainer.base import TorchLearnerBase, as_bool_tensor, as_float_tensor, _soft_update


@dataclass(slots=True)
class SACLearner(TorchLearnerBase):
    critic_optimizer: torch.optim.Optimizer | None = None
    actor_optimizer: torch.optim.Optimizer | None = None
    alpha_optimizer: torch.optim.Optimizer | None = None
    _log_alpha: torch.nn.Parameter | None = None

    def __post_init__(self) -> None:
        TorchLearnerBase.__post_init__(self)
        learning_rate = float(self.config.get("learning_rate", 3e-4))
        alpha_learning_rate = float(self.config.get("alpha_learning_rate", learning_rate))
        self.critic_optimizer = self.critic_optimizer or torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.critic.parameters()),
            lr=learning_rate,
        )
        self.actor_optimizer = self.actor_optimizer or torch.optim.Adam(self.model.actor.parameters(), lr=learning_rate)
        initial_alpha = float(self.model._alpha.item())
        self._log_alpha = self._log_alpha or torch.nn.Parameter(torch.log(torch.tensor(initial_alpha)))
        self.alpha_optimizer = self.alpha_optimizer or torch.optim.Adam([self._log_alpha], lr=alpha_learning_rate)
        action_dim = int(self.model.actor.mean_head.out_features)
        self.target_entropy = float(self.config.get("target_entropy", -action_dim))

    def optimizer_state(self) -> dict[str, Any]:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self._log_alpha.detach().cpu(),
        }

    def checkpoint_components(self) -> list[str]:
        return ["actor", "critic", "target_networks", "optimizers", "entropy_state", "runtime_state"]

    def update(self, batch: dict[str, Any], objective: dict[str, Any] | None = None) -> dict[str, Any]:
        rewards = as_float_tensor(batch["rewards"]).view(-1)
        terminated = as_bool_tensor(batch["terminated"]).view(-1)
        truncated = as_bool_tensor(batch["truncated"]).view(-1)

        gamma = float(self.config.get("gamma", 0.99))
        tau = float(self.config.get("tau", 0.005))

        train_outputs = self.model.forward_train(batch)
        online_q1 = train_outputs["q"]["online"]["q1"]
        online_q2 = train_outputs["q"]["online"]["q2"]
        target_q1 = train_outputs["q"]["target"]["q1"].detach()
        target_q2 = train_outputs["q"]["target"]["q2"].detach()
        next_log_prob = train_outputs["target"]["next_action_log_prob"].detach()
        alpha = self._log_alpha.exp()
        not_done = 1.0 - torch.logical_or(terminated, truncated).float()
        target_value = torch.minimum(target_q1, target_q2) - alpha.detach() * next_log_prob
        target = rewards + gamma * not_done * target_value
        critic_loss = F.mse_loss(online_q1, target) + F.mse_loss(online_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        features = self.model.encoder(as_float_tensor(batch["observations"])).detach()
        actor_sample = self.model.actor.sample(features)
        for parameter in self.model.critic.parameters():
            parameter.requires_grad_(False)
        q_pi = self.model.critic(features, actor_sample["action"])
        actor_loss = (alpha.detach() * actor_sample["log_prob"] - torch.minimum(q_pi["q1"], q_pi["q2"])).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.model.critic.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(self._log_alpha * (actor_sample["log_prob"].detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.model._alpha.copy_(self._log_alpha.exp().detach())

        _soft_update(self.model.encoder, self.model.target_encoder, tau)
        _soft_update(self.model.critic, self.model.target_critic, tau)

        self.env_steps = self._resolved_env_steps(batch, objective)
        self.grad_steps += 3
        self._advance_policy_version()
        return self._build_update_result(
            status="ok",
            published_policy=False,
            metrics={
                "critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "alpha_loss": float(alpha_loss.detach()),
                "alpha": float(self.model._alpha.item()),
                "actor_updated": True,
            },
        )


__all__ = ["SACLearner"]
