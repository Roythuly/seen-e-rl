from __future__ import annotations

from pathlib import Path

from rl_training_infra.contracts import validate_contract_payload

from algorithms.ppo.assembly import build_ppo_algorithm
from algorithms.sac.assembly import build_sac_algorithm
from algorithms.td3.assembly import build_td3_algorithm


def _experiment_config(algorithm: str, tmp_path: Path) -> dict:
    actor_kind = "gaussian_policy"
    critic_head = {"kind": "value_head"}
    runtime = {
        "collection_schedule": {
            "mode": "rollout",
            "unit": "env_step",
            "amount": 8,
            "freeze_policy_during_collection": True,
        },
        "update_schedule": {
            "trigger_unit": "collection",
            "updates_per_trigger": 1,
            "epochs": 2,
            "minibatch_size": 4,
        },
        "publish_schedule": {"strategy": "after_update"},
        "checkpoint": {"enabled": True},
        "max_env_steps": 8,
    }
    buffer = {"capacity": 32, "batch_size": 4, "sampling_mode": "fifo", "min_ready_size": 4}
    sampler_mode = "trajectory"
    if algorithm in {"sac", "td3"}:
        actor_kind = "deterministic_policy" if algorithm == "td3" else "gaussian_policy"
        critic_head = {"kind": "twin_q", "action_dim": 1}
        runtime = {
            "collection_schedule": {
                "mode": "step",
                "unit": "env_step",
                "amount": 1,
                "freeze_policy_during_collection": False,
                "warmup_env_steps": 2,
            },
            "update_schedule": {
                "trigger_unit": "env_step",
                "updates_per_trigger": 1,
                "min_ready_size": 4,
                "batch_size": 4,
                "policy_delay": 2 if algorithm == "td3" else 1,
            },
            "publish_schedule": {"strategy": "on_actor_update" if algorithm == "td3" else "every_n_updates", "every_n_updates": 1},
            "checkpoint": {"enabled": True},
            "max_env_steps": 8,
        }
        buffer = {"capacity": 64, "batch_size": 4, "sampling_mode": "random", "min_ready_size": 4}
        sampler_mode = "replay"

    return {
        "run_name": f"{algorithm}-pendulum-test",
        "seed": 7,
        "backend": {"name": "torch", "runtime_mode": "train", "device": "cpu"},
        "env": {"id": "Pendulum-v1"},
        "model": {
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16, 16]},
            "actor_head": {"kind": actor_kind, "action_dim": 1},
            "critic_head": critic_head,
        },
        "algo": {"name": algorithm},
        "sampler": {"mode": sampler_mode},
        "trainer": {"runtime": runtime},
        "buffer": buffer,
        "eval": {"selector": "latest", "seeds": [11], "episodes_per_seed": 1},
        "artifacts": {"root": str(tmp_path / algorithm)},
    }


def test_algorithm_assemblies_train_and_evaluate_end_to_end(tmp_path: Path) -> None:
    for algorithm, builder in (
        ("ppo", build_ppo_algorithm),
        ("sac", build_sac_algorithm),
        ("td3", build_td3_algorithm),
    ):
        assembly = builder(_experiment_config(algorithm, tmp_path))

        train_result = assembly.train()
        report = assembly.evaluate(selector="latest")

        assert train_result["env_steps"] > 0
        assert train_result["checkpoints"]
        validate_contract_payload("eval_report.schema.json", report)
        assert report["algorithm"] == algorithm
        assert report["env_id"] == "Pendulum-v1"
