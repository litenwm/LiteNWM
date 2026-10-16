#!/usr/bin/env python3
"""Shape-compatible entrypoint for mechanism decomposition v1 raw draws."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_new_mechanism_decomposition_v1 as base  # noqa: E402


_original_metric_values = base.metric_values_waypoint_units


def metric_values_waypoint_units(candidates: np.ndarray, ground: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates)
    if value.ndim == 2 and value.shape[-1] == 8:
        value = value.reshape(len(value), 4, 2)
    if value.shape[-2:] != (4, 2):
        raise RuntimeError(f"unexpected raw draw shape: {value.shape}")
    return _original_metric_values(value, ground)


def main() -> None:
    base.metric_values_waypoint_units = metric_values_waypoint_units
    base.main()


if __name__ == "__main__":
    main()
