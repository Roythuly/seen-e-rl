from importlib import import_module

from rl_training_infra.model import ModelTemplate


def test_model_package_preserves_template_export():
    package = import_module("rl_training_infra.model")
    templates = import_module("rl_training_infra.model.templates")

    assert package.ModelTemplate is ModelTemplate
    assert templates.ModelTemplate is ModelTemplate


def test_model_package_exports_torch_model_building_blocks():
    package = import_module("rl_training_infra.model")

    assert hasattr(package, "MLPEncoder")
    assert hasattr(package, "GaussianActor")
    assert hasattr(package, "DeterministicActor")
    assert hasattr(package, "ValueHead")
    assert hasattr(package, "TwinQCritic")
    assert hasattr(package, "TorchPPOModel")
    assert hasattr(package, "TorchSACModel")
    assert hasattr(package, "TorchTD3Model")
