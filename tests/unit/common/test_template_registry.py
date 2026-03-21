from importlib import import_module


def test_template_modules_are_importable():
    model_templates = import_module("rl_training_infra.model.templates")
    sampler_templates = import_module("rl_training_infra.sampler.templates")
    trainer_templates = import_module("rl_training_infra.trainer.templates")
    info_templates = import_module("rl_training_infra.info.templates")
    evaluator_templates = import_module("rl_training_infra.evaluator.templates")

    assert hasattr(model_templates, "ModelTemplate")
    assert hasattr(sampler_templates, "ActorHandleTemplate")
    assert hasattr(trainer_templates, "LearnerTemplate")
    assert hasattr(info_templates, "InfoHubTemplate")
    assert hasattr(evaluator_templates, "EvaluatorTemplate")
