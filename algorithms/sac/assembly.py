from __future__ import annotations

from algorithms.template import build_algorithm

from .defaults import build_default_sac_experiment


def build_sac_algorithm(module_registry):
    return build_algorithm(build_default_sac_experiment()["algo"], module_registry)
