from rl_training_infra.contracts import (
    build_checkpoint_manifest,
    build_eval_report,
    build_policy_snapshot,
    build_update_result,
    validate_contract_payload,
)


def test_policy_snapshot_builder_matches_minimum_schema():
    payload = build_policy_snapshot(
        run_id="run-1",
        policy_version=3,
        actor_ref="artifacts/policy.pt",
        backend="torch",
        algorithm="ppo",
        checkpoint_id="ckpt-3",
    )

    validate_contract_payload("policy_snapshot.schema.json", payload)
    assert payload["policy_version"] == 3


def test_checkpoint_manifest_builder_matches_minimum_schema():
    payload = build_checkpoint_manifest(
        checkpoint_id="ckpt-1",
        run_id="run-1",
        policy_version=1,
        path="artifacts/checkpoints/ckpt-1.pt",
        components=["actor", "critic", "optimizers"],
        backend="torch",
        algorithm="sac",
    )

    validate_contract_payload("checkpoint_manifest.schema.json", payload)
    assert payload["checkpoint_id"] == "ckpt-1"


def test_eval_report_builder_matches_minimum_schema():
    payload = build_eval_report(
        run_id="run-1",
        checkpoint_id="ckpt-2",
        policy_version=2,
        selector="latest",
        aggregate={"return_mean": 1.0},
        per_seed=[{"seed": 1, "return": 1.0}],
        algorithm="td3",
        backend="torch",
        env_id="Humanoid-v5",
    )

    validate_contract_payload("eval_report.schema.json", payload)
    assert payload["selector"] == "latest"


def test_update_result_builder_matches_minimum_schema():
    payload = build_update_result(
        run_id="run-1",
        policy_version=2,
        env_steps=1024,
        grad_steps=8,
        status="ok",
        published_policy=False,
        metrics={"policy_loss": 0.1},
        checkpoint_id="ckpt-2",
    )

    validate_contract_payload("update_result.schema.json", payload)
    assert payload["grad_steps"] == 8
