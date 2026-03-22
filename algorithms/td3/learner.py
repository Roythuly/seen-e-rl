from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rl_training_infra.trainer.base import TorchLearnerBase, as_bool_tensor, as_float_tensor, _soft_update


@dataclass(slots=True)
class TD3Learner(TorchLearnerBase):
    critic_optimizer: torch.optim.Optimizer | None = None
    actor_optimizer: torch.optim.Optimizer | None = None
    _update_index: int = 0

    def __post_init__(self) -> None:
        TorchLearnerBase.__post_init__(self)
        learning_rate = float(self.config.get("learning_rate", 3e-4))
        self.critic_optimizer = self.critic_optimizer or torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.critic.parameters()),
            lr=learning_rate,
        )
        self.actor_optimizer = self.actor_optimizer or torch.optim.Adam(self.model.actor.parameters(), lr=learning_rate)

    def optimizer_state(self) -> dict[str, Any]:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def checkpoint_components(self) -> list[str]:
        return ["actor", "critic", "target_networks", "optimizers", "runtime_state"]

    def update(self, batch: dict[str, Any], objective: dict[str, Any] | None = None) -> dict[str, Any]:
        rewards = as_float_tensor(batch["rewards"]).view(-1)
        terminated = as_bool_tensor(batch["terminated"]).view(-1)
        truncated = as_bool_tensor(batch["truncated"]).view(-1)
        gamma = float(self.config.get("gamma", 0.99))
        tau = float(self.config.get("tau", 0.005))
        policy_delay = int(self.config.get("policy_delay", 2))

        train_outputs = self.model.forward_train(batch)
        online_q1 = train_outputs["q"]["online"]["q1"]
        online_q2 = train_outputs["q"]["online"]["q2"]
        target_q1 = train_outputs["q"]["target"]["q1"].detach()
        target_q2 = train_outputs["q"]["target"]["q2"].detach()
        not_done = 1.0 - torch.logical_or(terminated, truncated).float()
        target = rewards + gamma * not_done * torch.minimum(target_q1, target_q2)
        critic_loss = F.mse_loss(online_q1, target) + F.mse_loss(online_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._update_index += 1
        actor_updated = False
        actor_loss = torch.tensor(0.0)
        if self._update_index % policy_delay == 0:
            features = self.model.encoder(as_float_tensor(batch["observations"])).detach()
            for parameter in self.model.critic.parameters():
                parameter.requires_grad_(False)
            actor_actions = self.model.actor(features)
            actor_q = self.model.critic(features, actor_actions)["q1"]
            actor_loss = -actor_q.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.model.critic.parameters():
                parameter.requires_grad_(True)

            _soft_update(self.model.encoder, self.model.target_encoder, tau)
            _soft_update(self.model.actor, self.model.target_actor, tau)
            _soft_update(self.model.critic, self.model.target_critic, tau)
            self._advance_policy_version()
            actor_updated = True
            self.grad_steps += 2
        else:
            self.grad_steps += 1

        self.env_steps = self._resolved_env_steps(batch, objective)
        return self._build_update_result(
            status="ok",
            published_policy=False,
            metrics={
                "critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "actor_updated": actor_updated,
            },
        )


__all__ = ["TD3Learner"]
