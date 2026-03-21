from __future__ import annotations

from typing import Any


def build_default_sac_experiment() -> dict[str, Any]:
    return {
        "run_name": "sac-humanoid-v5",
        "seed": 1,
        "backend": {"name": "torch", "runtime_mode": "train", "device": "cpu"},
        "env": {"id": "Humanoid-v5"},
        "model": {
            "encoder": {"kind": "mlp", "input_dim": 348, "hidden_sizes": [256, 256]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 17},
            "critic_head": {"kind": "twin_q", "action_dim": 17},
        },
        "algo": {"name": "sac"},
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
                "update_schedule": {"trigger_unit": "env_step", "updates_per_trigger": 1, "min_ready_size": 1000},
                "publish_schedule": {"strategy": "every_n_updates", "every_n_updates": 1},
            }
        },
        "buffer": {"capacity": 200000, "batch_size": 256, "min_ready_size": 1000, "sampling_mode": "random"},
        "eval": {"selector": "latest", "seeds": [1, 2, 3], "episodes_per_seed": 2},
    }
