from __future__ import annotations

from importlib import import_module

from rl_training_infra.evaluator import (
    CheckpointSelector,
    CheckpointSelectorBase,
    CheckpointSelectorTemplate,
    Evaluator,
    EvaluatorBase,
    EvaluatorTemplate,
    EvalReportWriter,
    EvalReportWriterBase,
    EvalReportWriterTemplate,
)


def _checkpoint(checkpoint_id: str, policy_version: int, score: float, created_at: str) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint_id,
        "run_id": "run-1",
        "policy_version": policy_version,
        "path": f"/tmp/{checkpoint_id}.pt",
        "components": ["actor", "critic"],
        "backend": "torch",
        "algorithm": "ppo",
        "created_at": created_at,
        "score": score,
    }


def test_evaluator_package_preserves_template_and_runtime_exports() -> None:
    package = import_module("rl_training_infra.evaluator")
    templates = import_module("rl_training_infra.evaluator.templates")
    base = import_module("rl_training_infra.evaluator.base")
    impl = import_module("rl_training_infra.evaluator.impl")

    assert package.CheckpointSelectorTemplate is CheckpointSelectorTemplate
    assert package.EvaluatorTemplate is EvaluatorTemplate
    assert package.EvalReportWriterTemplate is EvalReportWriterTemplate
    assert templates.CheckpointSelectorTemplate is CheckpointSelectorTemplate
    assert templates.EvaluatorTemplate is EvaluatorTemplate
    assert templates.EvalReportWriterTemplate is EvalReportWriterTemplate
    assert base.CheckpointSelectorBase is CheckpointSelectorBase
    assert base.EvaluatorBase is EvaluatorBase
    assert base.EvalReportWriterBase is EvalReportWriterBase
    assert impl.CheckpointSelector is CheckpointSelector
    assert impl.Evaluator is Evaluator
    assert impl.EvalReportWriter is EvalReportWriter


def test_checkpoint_selector_supports_latest_best_and_milestone_selection() -> None:
    selector = CheckpointSelector(
        checkpoints=[
            _checkpoint("ckpt-1", policy_version=1, score=0.25, created_at="2026-03-21T00:00:01+00:00"),
            _checkpoint("ckpt-2", policy_version=2, score=0.90, created_at="2026-03-21T00:00:02+00:00"),
            _checkpoint("ckpt-3", policy_version=3, score=0.60, created_at="2026-03-21T00:00:03+00:00"),
        ]
    )

    latest = selector.select("latest")
    best = selector.select("best")
    milestone = selector.select("milestone", policy_version=2)

    assert latest["checkpoint_id"] == "ckpt-3"
    assert latest["selector"] == "latest"
    assert best["checkpoint_id"] == "ckpt-2"
    assert best["selector"] == "best"
    assert milestone["checkpoint_id"] == "ckpt-2"
    assert milestone["selector"] == "milestone"


def test_evaluator_assembles_multi_seed_eval_report() -> None:
    evaluator = Evaluator(
        seed_runner=lambda checkpoint_manifest, seed, env_spec: {
            "reward_mean": float(seed),
            "episode_count": 2,
            "env_id": env_spec["id"],
            "checkpoint_id": checkpoint_manifest["checkpoint_id"],
        }
    )

    report = evaluator.evaluate(
        checkpoint_manifest={
            "run_id": "run-1",
            "checkpoint_id": "ckpt-2",
            "policy_version": 2,
            "path": "/tmp/ckpt-2.pt",
            "components": ["actor", "critic"],
            "backend": "torch",
            "algorithm": "ppo",
            "created_at": "2026-03-21T00:00:02+00:00",
            "selector": "best",
        },
        seeds=[11, 13],
        env_spec={"id": "CartPole-v1", "seedable": True},
    )

    assert report["run_id"] == "run-1"
    assert report["checkpoint_id"] == "ckpt-2"
    assert report["policy_version"] == 2
    assert report["selector"] == "best"
    assert report["algorithm"] == "ppo"
    assert report["backend"] == "torch"
    assert report["env_id"] == "CartPole-v1"
    assert report["status"] == "ok"
    assert report["per_seed"] == [
        {"seed": 11, "reward_mean": 11.0, "episode_count": 2, "env_id": "CartPole-v1", "checkpoint_id": "ckpt-2"},
        {"seed": 13, "reward_mean": 13.0, "episode_count": 2, "env_id": "CartPole-v1", "checkpoint_id": "ckpt-2"},
    ]
    assert report["aggregate"]["seed_count"] == 2
    assert report["aggregate"]["reward_mean"] == 12.0
