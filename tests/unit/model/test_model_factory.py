from __future__ import annotations

from pathlib import Path

import torch

from rl_training_infra.model import ModelFactory, TorchPPOModel, TorchSACModel, TorchTD3Model


def test_model_factory_builds_expected_torch_algorithms() -> None:
    backend = {"name": "torch"}

    ppo_model = ModelFactory.build(
        {
            "algorithm": "ppo",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [8]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "value_head"},
        },
        backend,
    )
    sac_model = ModelFactory.build(
        {
            "algorithm": "sac",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [8]},
            "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
            "critic_head": {"kind": "twin_q", "action_dim": 2},
        },
        backend,
    )
    td3_model = ModelFactory.build(
        {
            "algorithm": "td3",
            "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [8]},
            "actor_head": {"kind": "deterministic_policy", "action_dim": 2},
            "critic_head": {"kind": "twin_q", "action_dim": 2},
        },
        backend,
    )

    assert isinstance(ppo_model, TorchPPOModel)
    assert isinstance(sac_model, TorchSACModel)
    assert isinstance(td3_model, TorchTD3Model)


def test_model_factory_round_trips_torch_checkpoints(tmp_path: Path) -> None:
    backend = {"name": "torch"}
    spec = {
        "algorithm": "ppo",
        "encoder": {"kind": "mlp", "input_dim": 3, "hidden_sizes": [8]},
        "actor_head": {"kind": "gaussian_policy", "action_dim": 2},
        "critic_head": {"kind": "value_head"},
    }
    model = ModelFactory.build(spec, backend)
    checkpoint_path = tmp_path / "ppo-model.pt"

    model.save_checkpoint(checkpoint_path, {"label": "roundtrip"})

    restored_model = ModelFactory.build(spec, backend)
    metadata = restored_model.load_checkpoint(checkpoint_path)

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[key])
    assert metadata["label"] == "roundtrip"
