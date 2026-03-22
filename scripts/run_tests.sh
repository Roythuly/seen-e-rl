#!/usr/bin/env bash
set -euo pipefail

python scripts/validate_docs.py
python scripts/validate_contracts.py
python scripts/validate_runtime_env.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q --tb=short
