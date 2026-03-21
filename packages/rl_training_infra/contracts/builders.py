from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_policy_snapshot(
    *,
    run_id: str,
    policy_version: int,
    actor_ref: str,
    backend: str,
    algorithm: str,
    checkpoint_id: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "policy_version": policy_version,
        "actor_ref": actor_ref,
        "published_at": _timestamp(),
        "backend": backend,
        "algorithm": algorithm,
        "checkpoint_id": checkpoint_id,
    }


def build_checkpoint_manifest(
    *,
    checkpoint_id: str,
    run_id: str,
    policy_version: int,
    path: str,
    components: list[str],
    backend: str,
    algorithm: str,
) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "policy_version": policy_version,
        "path": path,
        "components": components,
        "backend": backend,
        "algorithm": algorithm,
        "created_at": _timestamp(),
    }


def build_eval_report(
    *,
    run_id: str,
    checkpoint_id: str,
    policy_version: int,
    selector: str,
    aggregate: dict[str, Any],
    per_seed: list[dict[str, Any]],
    algorithm: str,
    backend: str,
    env_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "policy_version": policy_version,
        "selector": selector,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "algorithm": algorithm,
        "backend": backend,
        "env_id": env_id,
        "status": "ok",
        "created_at": _timestamp(),
    }
