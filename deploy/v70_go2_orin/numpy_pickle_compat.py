"""Compatibility aliases for NumPy-2 checkpoint pickles on NumPy 1.x."""

from __future__ import annotations

import importlib
import sys


def install_numpy_pickle_compat() -> None:
    """Alias NumPy 2's ``numpy._core`` names without changing artifacts."""

    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass

    core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", core)
    for name in (
        "multiarray",
        "numeric",
        "_multiarray_umath",
        "fromnumeric",
        "shape_base",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)
