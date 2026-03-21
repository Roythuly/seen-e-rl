import torch

from rl_training_infra.model import (
    DeterministicActor,
    GaussianActor,
    MLPEncoder,
    TorchPPOModel,
    TorchSACModel,
    TorchTD3Model,
    TwinQCritic,
    ValueHead,
)


def test_mlp_encoder_projects_batches_to_final_hidden_size():
    encoder = MLPEncoder(input_dim=4, hidden_sizes=[8, 6])

    encoded = encoder(torch.randn(3, 4))

    assert encoded.shape == (3, 6)


def test_gaussian_actor_returns_sample_log_prob_and_distribution_params():
    actor = GaussianActor(input_dim=6, action_dim=2, hidden_sizes=[5])

    outputs = actor.sample(torch.randn(3, 6))

    assert outputs["action"].shape == (3, 2)
    assert outputs["log_prob"].shape == (3,)
    assert outputs["distribution_params"]["mean"].shape == (3, 2)
    assert outputs["distribution_params"]["log_std"].shape == (3, 2)


def test_deterministic_actor_returns_action_batch():
    actor = DeterministicActor(input_dim=6, action_dim=2, hidden_sizes=[5])

    actions = actor(torch.randn(4, 6))

    assert actions.shape == (4, 2)


def test_value_head_returns_state_value_per_row():
    value_head = ValueHead(input_dim=6, hidden_sizes=[5])

    values = value_head(torch.randn(4, 6))

    assert values.shape == (4,)


def test_twin_q_critic_returns_two_q_value_streams():
    critic = TwinQCritic(input_dim=6, action_dim=2, hidden_sizes=[5])

    q_values = critic(torch.randn(4, 6), torch.randn(4, 2))

    assert q_values["q1"].shape == (4,)
    assert q_values["q2"].shape == (4,)


def test_torch_ppo_model_exposes_act_and_train_outputs_for_ppo_paths():
    torch.manual_seed(0)
    model = TorchPPOModel(
        encoder=MLPEncoder(input_dim=4, hidden_sizes=[8]),
        actor=GaussianActor(input_dim=8, action_dim=2),
        value_head=ValueHead(input_dim=8),
        policy_version=7,
    )
    observations = {"observations": torch.randn(5, 4)}

    act_outputs = model.forward_act(observations)
    train_outputs = model.forward_train(observations)

    assert act_outputs["action"].shape == (5, 2)
    assert act_outputs["log_prob"].shape == (5,)
    assert act_outputs["value_estimate"].shape == (5,)
    assert act_outputs["policy_version"] == 7
    assert train_outputs["policy"]["distribution_params"]["mean"].shape == (5, 2)
    assert train_outputs["value"]["state_values"].shape == (5,)


def test_torch_sac_model_returns_policy_q_and_alpha_outputs():
    torch.manual_seed(0)
    model = TorchSACModel(
        encoder=MLPEncoder(input_dim=4, hidden_sizes=[8]),
        actor=GaussianActor(input_dim=8, action_dim=2),
        critic=TwinQCritic(input_dim=8, action_dim=2),
        target_encoder=MLPEncoder(input_dim=4, hidden_sizes=[8]),
        target_critic=TwinQCritic(input_dim=8, action_dim=2),
        alpha=0.2,
        policy_version=11,
    )
    train_request = {
        "observations": torch.randn(5, 4),
        "actions": torch.randn(5, 2),
        "next_observations": torch.randn(5, 4),
    }

    act_outputs = model.forward_act({"observations": train_request["observations"]})
    train_outputs = model.forward_train(train_request)

    assert act_outputs["action"].shape == (5, 2)
    assert act_outputs["log_prob"].shape == (5,)
    assert act_outputs["policy_version"] == 11
    assert train_outputs["policy"]["distribution_params"]["mean"].shape == (5, 2)
    assert train_outputs["q"]["online"]["q1"].shape == (5,)
    assert train_outputs["q"]["online"]["q2"].shape == (5,)
    assert train_outputs["q"]["target"]["q1"].shape == (5,)
    assert train_outputs["q"]["target"]["q2"].shape == (5,)
    assert torch.isclose(train_outputs["aux"]["alpha"], torch.tensor(0.2))


def test_torch_td3_model_returns_policy_online_q_and_target_smoothing_outputs():
    torch.manual_seed(0)
    model = TorchTD3Model(
        encoder=MLPEncoder(input_dim=4, hidden_sizes=[8]),
        actor=DeterministicActor(input_dim=8, action_dim=2),
        critic=TwinQCritic(input_dim=8, action_dim=2),
        target_encoder=MLPEncoder(input_dim=4, hidden_sizes=[8]),
        target_actor=DeterministicActor(input_dim=8, action_dim=2),
        target_critic=TwinQCritic(input_dim=8, action_dim=2),
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_version=13,
    )
    train_request = {
        "observations": torch.randn(5, 4),
        "actions": torch.randn(5, 2),
        "next_observations": torch.randn(5, 4),
    }

    act_outputs = model.forward_act({"observations": train_request["observations"]})
    train_outputs = model.forward_train(train_request)

    assert act_outputs["action"].shape == (5, 2)
    assert act_outputs["policy_version"] == 13
    assert train_outputs["policy"]["actions"].shape == (5, 2)
    assert train_outputs["q"]["online"]["q1"].shape == (5,)
    assert train_outputs["q"]["online"]["q2"].shape == (5,)
    assert train_outputs["q"]["target"]["q1"].shape == (5,)
    assert train_outputs["q"]["target"]["q2"].shape == (5,)
    assert train_outputs["aux"]["target_policy_smoothing"]["smoothed_actions"].shape == (5, 2)
