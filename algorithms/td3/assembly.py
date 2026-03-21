from __future__ import annotations

from algorithms.template import build_algorithm

from .defaults import build_default_td3_experiment


def build_td3_algorithm(module_registry):
    return build_algorithm(build_default_td3_experiment()["algo"], module_registry)
