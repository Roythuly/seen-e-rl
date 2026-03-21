from __future__ import annotations

from typing import Any


def build_default_td3_experiment() -> dict[str, Any]:
    return {
        "run_name": "td3-humanoid-v5",
        "seed": 1,
        "backend": {"name": "torch", "runtime_mode": "train", "device": "cpu"},
        "env": {"id": "Humanoid-v5"},
        "model": {
            "encoder": {"kind": "mlp", "input_dim": 348, "hidden_sizes": [256, 256]},
            "actor_head": {"kind": "deterministic_policy", "action_dim": 17},
            "critic_head": {"kind": "twin_q", "action_dim": 17},
        },
        "algo": {"name": "td3"},
        "sampler": {"mode": "replay", "collection": {"unit": "env_step", "amount": 1}},
        "trainer": {
            "runtime": {
                "collection_schedule": {
                    "mode": "step",
                    "unit": "env_step",
                    "amount": 1,
                    "freeze_policy_during_collection": False,
                    "warmup_env_steps": 1000,
                },
                "update_schedule": {
                    "trigger_unit": "env_step",
                    "updates_per_trigger": 1,
                    "min_ready_size": 1000,
                    "policy_delay": 2,
                },
                "publish_schedule": {"strategy": "on_actor_update", "on_actor_update_only": True},
            }
        },
        "buffer": {"capacity": 200000, "batch_size": 256, "min_ready_size": 1000, "sampling_mode": "random"},
        "eval": {"selector": "latest", "seeds": [1, 2, 3], "episodes_per_seed": 2},
    }
