#!/usr/bin/env python3
"""Deterministic-validation entrypoint for the single action-residual run."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(Path(__file__).resolve().parent),
]

import train_highres_dual_action_residual_v1 as base  # noqa: E402


_original_evaluate = base.evaluate


def deterministic_evaluate(model, loader, **kwargs):
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "epoch"):
        sampler.epoch = 0
    return _original_evaluate(model, loader, **kwargs)


def main() -> None:
    base.evaluate = deterministic_evaluate
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
