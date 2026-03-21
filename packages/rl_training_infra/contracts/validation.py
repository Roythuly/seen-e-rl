from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "contracts" / "v0"


def load_schema(schema_name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))


def validate_contract_payload(schema_name: str, payload: dict[str, Any]) -> None:
    schema = load_schema(schema_name)
    Draft202012Validator(schema).validate(payload)
