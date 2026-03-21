import pytest

from rl_training_infra.contracts import validate_contract_payload


def test_validate_contract_payload_rejects_missing_required_field():
    with pytest.raises(Exception):
        validate_contract_payload("policy_snapshot.schema.json", {"run_id": "missing-fields"})
