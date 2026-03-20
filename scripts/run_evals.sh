#!/usr/bin/env bash
set -euo pipefail

echo "Evals are placeholders in v0.1 docs-first stage."
echo "Available packs:"
find evals/packs -maxdepth 2 -name 'cases.yaml' | sort
