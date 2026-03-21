from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AlgorithmAssemblyTemplate:
    algorithm_name: str
    config: dict[str, Any]
    module_registry: dict[str, Any]


AlgorithmTemplate = AlgorithmAssemblyTemplate
AlgorithmConfigTemplate = dict[str, Any]


def build_default_model_spec() -> dict[str, Any]:
    return {
        "encoder": {"kind": "mlp", "hidden_sizes": [256, 256]},
        "actor_head": {"kind": "gaussian_policy"},
        "critic_head": {"kind": "twin_q"},
        "feature_sharing": {"actor_critic_encoder": False},
        "training_interface": {"forward_train": "structured"},
    }


def build_default_runtime_spec() -> dict[str, Any]:
    return {
        "collection_schedule": {
            "mode": "step",
            "unit": "env_step",
            "amount": 1,
            "freeze_policy_during_collection": False,
        },
        "update_schedule": {"trigger_unit": "env_step", "updates_per_trigger": 1},
        "publish_schedule": {"strategy": "every_n_updates", "every_n_updates": 1},
    }


def build_default_eval_spec() -> dict[str, Any]:
    return {"selector": "latest", "seeds": [1], "episodes_per_seed": 1}


def build_algorithm(config: dict[str, Any], module_registry: dict[str, Any]) -> AlgorithmAssemblyTemplate:
    return AlgorithmAssemblyTemplate(
        algorithm_name=config.get("name", "template"),
        config=config,
        module_registry=module_registry,
    )


__all__ = [
    "AlgorithmAssemblyTemplate",
    "AlgorithmConfigTemplate",
    "AlgorithmTemplate",
    "build_algorithm",
    "build_default_eval_spec",
    "build_default_model_spec",
    "build_default_runtime_spec",
]
