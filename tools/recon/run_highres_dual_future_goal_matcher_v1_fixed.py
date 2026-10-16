#!/usr/bin/env python3
"""Fixed entrypoint for complete tail-batch padding in the v1 matcher probe."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import run_highres_dual_future_goal_matcher_v1 as base  # noqa: E402


_original_collate = base.collate_indices


def _complete_collate(dataset, indices):
    values = np.asarray(indices, dtype=np.int64)
    if 0 < len(values) < base.BATCH_SIZE:
        repeats = (base.BATCH_SIZE + len(values) - 1) // len(values)
        values = np.tile(values, repeats)[: base.BATCH_SIZE]
    return _original_collate(dataset, values)


def main() -> None:
    base.collate_indices = _complete_collate
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
