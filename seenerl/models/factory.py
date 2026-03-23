"""High-level model factory built on top of the network registry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import gymnasium as gym

from seenerl.networks import build_actor, build_critic


def _resolve_model_spec(
    config,
    key: str,
    default_name: str,
    default_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model_cfg = deepcopy(config.get("model", {}))
    spec = deepcopy(model_cfg.get(key, {}))
    if not isinstance(spec, dict):
        spec = {}

    kwargs = deepcopy(default_kwargs or {})
    kwargs.update(deepcopy(spec.get("kwargs", {})))

    return {
        "name": spec.get("name", default_name),
        "hidden_dim": spec.get("hidden_dim", config.get("hidden_size", 256)),
        "kwargs": kwargs,
    }


def build_actor_model(config, obs_dim: int, action_space, default_name: str,
                      default_kwargs: Dict[str, Any] | None = None):
    """Build an actor model from config with compatibility defaults."""
    spec = _resolve_model_spec(config, "actor", default_name, default_kwargs)
    action_dim = gym.spaces.flatdim(action_space)
    return build_actor(
        spec["name"],
        num_inputs=obs_dim,
        num_actions=action_dim,
        hidden_dim=spec["hidden_dim"],
        action_space=action_space,
        **spec["kwargs"],
    )


def build_q_critic_model(config, obs_dim: int, action_space, default_name: str,
                         default_kwargs: Dict[str, Any] | None = None):
    """Build a state-action critic model from config."""
    spec = _resolve_model_spec(config, "critic", default_name, default_kwargs)
    action_dim = gym.spaces.flatdim(action_space)
    return build_critic(
        spec["name"],
        num_inputs=obs_dim,
        num_actions=action_dim,
        hidden_dim=spec["hidden_dim"],
        **spec["kwargs"],
    )


def build_value_model(config, obs_dim: int, default_name: str,
                      default_kwargs: Dict[str, Any] | None = None):
    """Build a state-value model from config."""
    spec = _resolve_model_spec(config, "value", default_name, default_kwargs)
    return build_critic(
        spec["name"],
        num_inputs=obs_dim,
        hidden_dim=spec["hidden_dim"],
        **spec["kwargs"],
    )
