from __future__ import annotations

from algorithms.template import AlgorithmAssemblyTemplate, build_algorithm

from .defaults import build_default_ppo_experiment


def build_ppo_algorithm(module_registry):
    return build_algorithm(build_default_ppo_experiment()["algo"], module_registry)
