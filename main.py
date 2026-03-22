from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from algorithms.ppo import build_ppo_algorithm
from algorithms.sac import build_sac_algorithm
from algorithms.td3 import build_td3_algorithm
from rl_training_infra.common import load_yaml


ALGORITHM_BUILDERS = {
    "ppo": build_ppo_algorithm,
    "sac": build_sac_algorithm,
    "td3": build_td3_algorithm,
}


def _resolve_algorithm_name(config: dict[str, object]) -> str:
    algo = config.get("algo")
    if not isinstance(algo, dict):
        raise ValueError("config must define algo.name")
    algorithm_name = algo.get("name")
    if not isinstance(algorithm_name, str) or not algorithm_name:
        raise ValueError("config must define algo.name")
    return algorithm_name


def _resolve_algorithm_builder(algorithm_name: str):
    builder = ALGORITHM_BUILDERS.get(algorithm_name)
    if builder is None:
        raise ValueError(f"unsupported algorithm: {algorithm_name}")
    return builder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template-based RL training entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train an algorithm from an experiment config.")
    train.add_argument("--config", required=True, help="Path to the experiment config YAML.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint or selector for an experiment.")
    evaluate.add_argument("--config", required=True, help="Path to the experiment config YAML.")
    evaluate.add_argument(
        "--selector",
        default="latest",
        choices=("latest", "best", "milestone"),
        help="Checkpoint selector to evaluate when no explicit checkpoint path is provided.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    try:
        algorithm_name = _resolve_algorithm_name(config)
        builder = _resolve_algorithm_builder(algorithm_name)
    except ValueError as exc:
        parser.error(str(exc))
    assembly = builder(config)

    if args.command == "train":
        result = assembly.train()
        print(json.dumps({"algorithm": algorithm_name, "env_steps": result["env_steps"], "updates": result["updates"]}))
        return 0
    if args.command == "evaluate":
        report = assembly.evaluate(selector=args.selector)
        print(json.dumps(report))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
