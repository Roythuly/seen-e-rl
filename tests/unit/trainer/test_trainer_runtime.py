from __future__ import annotations

import math
from pathlib import Path

import torch

from algorithms.ppo.learner import PPOLearner
from algorithms.sac.learner import SACLearner
from algorithms.td3.learner import TD3Learner
from rl_training_infra.contracts import validate_contract_payload
from rl_training_infra.model import ModelFactory
from rl_training_infra.trainer import (
    OffPolicyRuntimeLoop,
    ReplayBuffer,
)


def _build_ppo_trajectory_batch(model, batch_size: int = 8) -> dict[str, torch.Tensor]:
    observations = torch.randn(batch_size, 3)
    action_output = model.forward_act({"observations": observations})
    rewards = torch.randn(batch_size)
    terminated = torch.zeros(batch_size, dtype=torch.bool)
    truncated = torch.zeros(batch_size, dtype=torch.bool)
    terminated[-1] = True
    return {
        "observations": observations,
        "actions": action_output["action"].detach(),
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "log_probs": action_output["log_prob"].detach(),
        "value_estimates": action_output["value_estimate"].detach(),
        "policy_version": action_output["policy_version"],
    }


def _build_replay_batch(batch_size: int = 8, obs_dim: int = 3, action_dim: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.randn(batch_size, obs_dim),
        "actions": torch.randn(batch_size, action_dim).clamp(-1.0, 1.0),
        "rewards": torch.randn(batch_size),
        "next_observations": torch.randn(batch_size, obs_dim),
        "terminated": torch.zeros(batch_size, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, dtype=torch.bool),
        "policy_version": torch.zeros(batch_size, dtype=torch.long),
    }


def test_replay_buffer_batches_nested_records() -> None:
    buffer = ReplayBuffer(capacity=5, batch_size=2, sampling_mode="fifo")
    buffer.write(
        {
            "observations": {"vector": [1.0, 2.0], "aux": {"value": [3.0]}},
            "actions": [0.1, 0.2],
            "rewards": 1.5,
            "next_observations": {"vector": [1.1, 2.1], "aux": {"value": [3.1]}},
            "terminated": False,
            "truncated": False,
            "policy_version": 0,
            "env_step": 1,
        }
    )
    buffer.write(
        {
            "observations": {"vector": [4.0, 5.0], "aux": {"value": [6.0]}},
            "actions": [0.3, 0.4],
            "rewards": 2.5,
            "next_observations": {"vector": [4.1, 5.1], "aux": {"value": [6.1]}},
            "terminated": True,
            "truncated": False,
            "policy_version": 1,
            "env_step": 2,
        }
    )

    batch = buffer.sample({"batch_size": 2})

    assert batch["observations"]["vector"].shape == (2, 2)
    assert batch["observations"]["aux"]["value"].shape == (2, 1)
    assert batch["actions"].shape == (2, 2)
    assert batch["sample_info"]["size"] == 2
    assert batch["sample_info"]["capacity"] == 5


def test_ppo_learner_update_publish_and_checkpoint_are_contract_aligned(tmp_path: Path) -> None:
    model = ModelFactory.build(
        {
            "algorithm": "ppo",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "value_head"},
        },
        {"name": "torch"},
    )
    learner = PPOLearner(
        model=model,
        run_id="run-ppo",
        backend="torch",
        algorithm="ppo",
        artifacts_dir=tmp_path,
        config={"gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2, "epochs": 2, "minibatch_size": 4},
    )

    update_result = learner.update(_build_ppo_trajectory_batch(model))
    snapshot = learner.publish_policy()
    checkpoint = learner.save_checkpoint()

    validate_contract_payload("update_result.schema.json", update_result)
    validate_contract_payload("policy_snapshot.schema.json", snapshot)
    validate_contract_payload("checkpoint_manifest.schema.json", checkpoint)
    assert update_result["policy_version"] == 1
    assert snapshot["policy_version"] == 1
    assert checkpoint["policy_version"] == 1


def test_ppo_learner_handles_single_step_batches_without_nan_metrics(tmp_path: Path) -> None:
    model = ModelFactory.build(
        {
            "algorithm": "ppo",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "value_head"},
        },
        {"name": "torch"},
    )
    learner = PPOLearner(
        model=model,
        run_id="run-ppo-single-step",
        backend="torch",
        algorithm="ppo",
        artifacts_dir=tmp_path,
        config={"gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2, "epochs": 1, "minibatch_size": 1},
    )

    update_result = learner.update(_build_ppo_trajectory_batch(model, batch_size=1))

    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert math.isfinite(update_result["metrics"][key]), key


def test_sac_learner_update_reports_alpha_and_actor_metrics(tmp_path: Path) -> None:
    model = ModelFactory.build(
        {
            "algorithm": "sac",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "twin_q", "action_dim": 2},
        },
        {"name": "torch"},
    )
    learner = SACLearner(
        model=model,
        run_id="run-sac",
        backend="torch",
        algorithm="sac",
        artifacts_dir=tmp_path,
        config={"gamma": 0.99, "tau": 0.005, "batch_size": 8},
    )

    update_result = learner.update(_build_replay_batch())

    validate_contract_payload("update_result.schema.json", update_result)
    assert update_result["policy_version"] == 1
    assert "alpha" in update_result["metrics"]
    assert update_result["metrics"]["actor_updated"] is True


def test_td3_learner_respects_policy_delay_before_publishing_new_actor(tmp_path: Path) -> None:
    model = ModelFactory.build(
        {
            "algorithm": "td3",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16]},
            "actor_head": {"kind": "deterministic_policy", "action_dim": 2},
            "critic_head": {"kind": "twin_q", "action_dim": 2},
        },
        {"name": "torch"},
    )
    learner = TD3Learner(
        model=model,
        run_id="run-td3",
        backend="torch",
        algorithm="td3",
        artifacts_dir=tmp_path,
        config={"gamma": 0.99, "tau": 0.005, "batch_size": 8, "policy_delay": 2},
    )

    first_result = learner.update(_build_replay_batch())
    second_result = learner.update(_build_replay_batch())

    validate_contract_payload("update_result.schema.json", first_result)
    validate_contract_payload("update_result.schema.json", second_result)
    assert first_result["policy_version"] == 0
    assert first_result["metrics"]["actor_updated"] is False
    assert second_result["policy_version"] == 1
    assert second_result["metrics"]["actor_updated"] is True


class _StubInfoHub:
    def __init__(self) -> None:
        self.training_events: list[dict[str, object]] = []
        self.checkpoint_events: list[dict[str, object]] = []

    def record_training(self, **event):
        self.training_events.append(dict(event))
        return event

    def record_checkpoint(self, **event):
        self.checkpoint_events.append(dict(event))
        return event


class _StubCollector:
    def __init__(self, records):
        self.records = list(records)
        self.calls: list[int] = []

    def collect(self, amount: int):
        self.calls.append(amount)
        return [dict(record) for record in self.records]


def test_off_policy_runtime_loop_respects_warmup_and_actor_only_publish(tmp_path: Path) -> None:
    model = ModelFactory.build(
        {
            "algorithm": "td3",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [16]},
            "actor_head": {"kind": "deterministic_policy", "action_dim": 2},
            "critic_head": {"kind": "twin_q", "action_dim": 2},
        },
        {"name": "torch"},
    )
    learner = TD3Learner(
        model=model,
        run_id="run-runtime",
        backend="torch",
        algorithm="td3",
        artifacts_dir=tmp_path,
        config={"gamma": 0.99, "tau": 0.005, "batch_size": 2, "policy_delay": 2},
    )
    collector = _StubCollector(
        [
            {
                "observations": [1.0, 2.0, 3.0],
                "actions": [0.1, 0.2],
                "rewards": 1.0,
                "next_observations": [1.1, 2.1, 3.1],
                "terminated": False,
                "truncated": False,
                "policy_version": 0,
                "env_step": 1,
            }
        ]
    )
    info_hub = _StubInfoHub()
    runtime = OffPolicyRuntimeLoop(replay_buffer=ReplayBuffer(capacity=8, batch_size=2), collector=collector, learner=learner, info=info_hub)

    result = runtime.run(
        {
            "collection_schedule": {"mode": "step", "unit": "env_step", "amount": 1, "freeze_policy_during_collection": False, "warmup_env_steps": 1},
            "update_schedule": {"trigger_unit": "env_step", "updates_per_trigger": 1, "min_ready_size": 2, "policy_delay": 2},
            "publish_schedule": {"strategy": "on_actor_update", "on_actor_update_only": True},
            "checkpoint": {"enabled": True},
            "max_env_steps": 4,
        }
    )

    assert result["env_steps"] == 4
    assert len(collector.calls) == 4
    assert len(info_hub.training_events) == 3
    assert len(info_hub.checkpoint_events) >= 1
