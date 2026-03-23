"""Algorithm registry and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

import gymnasium as gym

from seenerl.algorithms.base import BaseAlgorithm


@dataclass(frozen=True)
class AlgorithmSpec:
    cls: Type[BaseAlgorithm]
    trainer_kind: str


ALGORITHM_REGISTRY: Dict[str, AlgorithmSpec] = {}


def register_algorithm(name: str, trainer_kind: str):
    """Register an algorithm class and its trainer family."""

    def decorator(cls: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        ALGORITHM_REGISTRY[name.upper()] = AlgorithmSpec(cls=cls, trainer_kind=trainer_kind)
        return cls

    return decorator


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    algo_name = name.upper()
    if algo_name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[algo_name]


def build_algorithm(config, observation_space: gym.Space, action_space: gym.Space) -> BaseAlgorithm:
    """Instantiate an algorithm from config."""
    spec = get_algorithm_spec(config.algo)
    obs_dim = gym.spaces.flatdim(observation_space)
    return spec.cls(obs_dim, action_space, config)
