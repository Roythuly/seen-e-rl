from __future__ import annotations

from copy import deepcopy
from typing import Any

from .torch_impl import DeterministicActor, GaussianActor, MLPEncoder, TorchPPOModel, TorchSACModel, TorchTD3Model, TwinQCritic, ValueHead


class ModelFactory:
    @staticmethod
    def build(spec: dict[str, Any], backend: dict[str, Any]) -> Any:
        backend_name = backend.get("name")
        if backend_name != "torch":
            raise ValueError(f"unsupported backend: {backend_name!r}")

        algorithm = spec.get("algorithm") or spec.get("name")
        encoder_spec = spec.get("encoder", {})
        actor_spec = spec.get("actor_head", {})
        critic_spec = spec.get("critic_head", {})

        encoder = MLPEncoder(
            input_dim=int(encoder_spec["input_dim"]),
            hidden_sizes=tuple(encoder_spec.get("hidden_sizes", ())),
        )
        feature_dim = encoder.output_dim

        if algorithm == "ppo":
            actor = GaussianActor(
                input_dim=feature_dim,
                action_dim=int(actor_spec["action_dim"]),
                hidden_sizes=tuple(actor_spec.get("hidden_sizes", ())),
            )
            value_head = ValueHead(
                input_dim=feature_dim,
                hidden_sizes=tuple(critic_spec.get("hidden_sizes", ())),
            )
            return TorchPPOModel(
                encoder=encoder,
                actor=actor,
                value_head=value_head,
                policy_version=int(spec.get("policy_version", 0)),
            )

        if algorithm == "sac":
            actor = GaussianActor(
                input_dim=feature_dim,
                action_dim=int(actor_spec["action_dim"]),
                hidden_sizes=tuple(actor_spec.get("hidden_sizes", ())),
            )
            critic = TwinQCritic(
                input_dim=feature_dim,
                action_dim=int(critic_spec["action_dim"]),
                hidden_sizes=tuple(critic_spec.get("hidden_sizes", ())),
            )
            return TorchSACModel(
                encoder=encoder,
                actor=actor,
                critic=critic,
                target_encoder=deepcopy(encoder),
                target_critic=deepcopy(critic),
                alpha=float(spec.get("alpha", 0.2)),
                policy_version=int(spec.get("policy_version", 0)),
            )

        if algorithm == "td3":
            actor = DeterministicActor(
                input_dim=feature_dim,
                action_dim=int(actor_spec["action_dim"]),
                hidden_sizes=tuple(actor_spec.get("hidden_sizes", ())),
            )
            critic = TwinQCritic(
                input_dim=feature_dim,
                action_dim=int(critic_spec["action_dim"]),
                hidden_sizes=tuple(critic_spec.get("hidden_sizes", ())),
            )
            return TorchTD3Model(
                encoder=encoder,
                actor=actor,
                critic=critic,
                target_encoder=deepcopy(encoder),
                target_actor=deepcopy(actor),
                target_critic=deepcopy(critic),
                target_policy_noise=float(spec.get("target_policy_noise", 0.2)),
                target_noise_clip=float(spec.get("target_noise_clip", 0.5)),
                policy_version=int(spec.get("policy_version", 0)),
            )

        raise ValueError(f"unsupported algorithm for torch backend: {algorithm!r}")
