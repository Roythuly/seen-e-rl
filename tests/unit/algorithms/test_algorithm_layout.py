from importlib import import_module


def test_each_algorithm_package_exposes_assembly_surfaces():
    for name in ("ppo", "sac", "td3"):
        import_module(f"algorithms.{name}.config")
        import_module(f"algorithms.{name}.assembly")
        import_module(f"algorithms.{name}.learner")
        import_module(f"algorithms.{name}.defaults")
