#!/usr/bin/env bash
set -euo pipefail

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests -q
