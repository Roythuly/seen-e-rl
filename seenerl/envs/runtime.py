"""Runtime management for Isaac Lab applications."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from typing import Any


_APP = None
_APP_SETTINGS: dict[str, Any] | None = None
_APP_REFCOUNT = 0
_PINK_CONFIGURATION_LIMIT_PATCHED = False
_USD_TO_URDF_CACHE_PATCHED = False


def _install_numpy_compat_shims() -> None:
    """Bridge NumPy 2.x private module locations expected by Isaac Sim deps."""
    import numpy as np

    if not hasattr(np, "_no_nep50_warning"):
        @contextmanager
        def _no_nep50_warning():
            yield

        np._no_nep50_warning = _no_nep50_warning  # type: ignore[attr-defined]

    module_aliases = {
        "numpy.lib.function_base": "numpy.lib._function_base_impl",
        "numpy.lib.twodim_base": "numpy.lib._twodim_base_impl",
    }
    for legacy_name, modern_name in module_aliases.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = importlib.import_module(modern_name)
        except ModuleNotFoundError:
            continue


def _needs_pink_configuration_limit_patch(
    env_id: str | None,
    task_imports: list[str] | None,
) -> bool:
    """Return whether the current Isaac Lab task uses Pink-backed pick-place IK."""
    task_imports = task_imports or []
    if any("pick_place" in module_name for module_name in task_imports):
        return True
    if env_id is None:
        return False
    return any(token in env_id for token in ("PickPlace", "NutPour", "ExhaustPipe"))


def maybe_patch_pink_configuration_limit(
    env_id: str | None,
    task_imports: list[str] | None = None,
) -> None:
    """Patch Pink's configuration-limit helper for Isaac Lab pick-place tasks.

    Some Isaac Lab manipulation environments invoke Pink through a pybind wrapper
    that exposes ``pinocchio.Model.hasConfigurationLimit()`` as ``std::vector<bool>``.
    On the current Isaac Sim stack this conversion can fail before the environment
    finishes construction. Pink only needs the effective valid-position mask, so we
    can derive it directly from the joint limits and avoid the problematic binding.
    """
    global _PINK_CONFIGURATION_LIMIT_PATCHED

    if _PINK_CONFIGURATION_LIMIT_PATCHED:
        return
    if not _needs_pink_configuration_limit_patch(env_id, task_imports):
        return

    try:
        import numpy as np
        import pink.limits.configuration_limit as configuration_limit
    except ImportError:
        return

    current_init = configuration_limit.ConfigurationLimit.__init__
    if getattr(current_init, "_seenerl_patched", False):
        _PINK_CONFIGURATION_LIMIT_PATCHED = True
        return

    def patched_init(self, model, config_limit_gain=0.5):
        assert 0.0 < config_limit_gain <= 1.0

        upper = np.asarray(model.upperPositionLimit)
        lower = np.asarray(model.lowerPositionLimit)
        has_configuration_limit = np.logical_and(
            upper < 1e20,
            upper > lower + 1e-10,
        )

        joints = [
            joint for joint in model.joints
            if joint.idx_q >= 0
            and has_configuration_limit[slice(joint.idx_q, joint.idx_q + joint.nq)].all()
        ]

        index_list: list[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))

        indices = np.asarray(index_list, dtype=np.int64)
        indices.setflags(write=False)
        projection_matrix = np.eye(model.nv)[indices] if len(indices) > 0 else None

        self.config_limit_gain = config_limit_gain
        self.indices = indices
        self.joints = joints
        self.model = model
        self.projection_matrix = projection_matrix

    patched_init._seenerl_patched = True  # type: ignore[attr-defined]
    configuration_limit.ConfigurationLimit.__init__ = patched_init
    _PINK_CONFIGURATION_LIMIT_PATCHED = True


def maybe_patch_usd_to_urdf_cache(
    env_id: str | None,
    task_imports: list[str] | None = None,
) -> None:
    """Reuse previously converted URDF artifacts for heavy Isaac Lab pick-place tasks."""
    global _USD_TO_URDF_CACHE_PATCHED

    if _USD_TO_URDF_CACHE_PATCHED:
        return
    if not _needs_pink_configuration_limit_patch(env_id, task_imports):
        return

    try:
        import isaaclab.controllers.utils as controller_utils
    except ImportError:
        return

    current_convert = controller_utils.convert_usd_to_urdf
    if getattr(current_convert, "_seenerl_patched", False):
        _USD_TO_URDF_CACHE_PATCHED = True
        return

    def patched_convert_usd_to_urdf(usd_path: str, output_path: str, force_conversion: bool = True):
        urdf_output_dir = os.path.join(output_path, "urdf")
        urdf_file_name = os.path.basename(usd_path).split(".")[0] + ".urdf"
        urdf_output_path = os.path.join(urdf_output_dir, urdf_file_name)
        urdf_meshes_output_dir = os.path.join(output_path, "meshes")

        if force_conversion and os.path.exists(urdf_output_path) and os.path.exists(urdf_meshes_output_dir):
            force_conversion = False

        return current_convert(usd_path, output_path, force_conversion=force_conversion)

    patched_convert_usd_to_urdf._seenerl_patched = True  # type: ignore[attr-defined]
    controller_utils.convert_usd_to_urdf = patched_convert_usd_to_urdf
    _USD_TO_URDF_CACHE_PATCHED = True


def ensure_isaaclab_app(headless: bool = True, enable_cameras: bool = False):
    """Launch Isaac Lab once per process and return the shared app handle."""
    global _APP, _APP_SETTINGS, _APP_REFCOUNT

    requested_settings = {
        "headless": headless,
        "enable_cameras": enable_cameras,
    }

    if _APP is None:
        _install_numpy_compat_shims()
        import numpy  # noqa: F401
        import numpy.lib.recfunctions  # noqa: F401
        import numpy.ma  # noqa: F401
        import isaacsim  # noqa: F401
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(**requested_settings)
        _APP = launcher.app
        _APP_SETTINGS = requested_settings
    else:
        if _APP_SETTINGS != requested_settings:
            raise RuntimeError(
                "Isaac Lab app is already running with different settings: "
                f"{_APP_SETTINGS} != {requested_settings}"
            )

    _APP_REFCOUNT += 1
    return _APP


def release_isaaclab_app() -> None:
    """Release a reference to the shared Isaac Lab app."""
    global _APP, _APP_SETTINGS, _APP_REFCOUNT

    if _APP is None:
        return

    _APP_REFCOUNT = max(_APP_REFCOUNT - 1, 0)
    if _APP_REFCOUNT == 0:
        _APP.close()
        _APP = None
        _APP_SETTINGS = None
