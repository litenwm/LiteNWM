#!/usr/bin/env python3
"""Run fixed full validation with empty baseline-corruption fields supported."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_highres_dual_ltwm_full_val_v1 as evaluation  # noqa: E402


def concatenate_allowing_empty_baseline(
    record: dict[str, list[np.ndarray] | list[str]],
) -> dict[str, np.ndarray | list[str]]:
    result: dict[str, np.ndarray | list[str]] = {}
    for key, value in record.items():
        if key == "routes":
            result[key] = value
        elif value:
            result[key] = np.concatenate(value, axis=0)
        else:
            result[key] = np.empty((0, 4), dtype=np.float32)
    return result


evaluation.concatenate_record = concatenate_allowing_empty_baseline


if __name__ == "__main__":
    evaluation.main()
