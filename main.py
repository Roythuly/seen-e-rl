from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PACKAGES = ROOT / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))


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
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
