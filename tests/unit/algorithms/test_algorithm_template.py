from algorithms.template import (
    build_algorithm,
    build_default_eval_spec,
    build_default_model_spec,
    build_default_runtime_spec,
)


def test_algorithm_template_exports_default_builders():
    model_spec = build_default_model_spec()
    runtime_spec = build_default_runtime_spec()
    eval_spec = build_default_eval_spec()

    assert model_spec["encoder"]["kind"] == "mlp"
    assert runtime_spec["collection_schedule"]["amount"] > 0
    assert eval_spec["selector"] == "latest"


def test_algorithm_template_requires_registry_and_config():
    assembly = build_algorithm({"name": "template"}, {"model": object()})

    assert assembly.algorithm_name == "template"
    assert assembly.module_registry["model"] is not None
