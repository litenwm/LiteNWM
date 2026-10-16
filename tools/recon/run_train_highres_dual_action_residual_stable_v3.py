#!/usr/bin/env python3
"""Deterministic 12-way action-retrieval entrypoint for the frozen run."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(Path(__file__).resolve().parent),
]

import train_highres_dual_action_residual_v1 as base  # noqa: E402


_original_evaluate = base.evaluate
_training_negative_actions = base.make_negative_actions


def _fixed_validation_negative_actions(actions):
    observed = [
        torch.roll(actions, shifts=offset, dims=0)
        for offset in range(1, 8)
    ]
    structured = _training_negative_actions(actions)[:, -4:]
    return torch.cat((torch.stack(observed, dim=1), structured), dim=1)


def deterministic_evaluate(model, loader, **kwargs):
    sampler = getattr(loader, "batch_sampler", None)
    if hasattr(sampler, "epoch"):
        sampler.epoch = 0
    base.make_negative_actions = _fixed_validation_negative_actions
    try:
        return _original_evaluate(model, loader, **kwargs)
    finally:
        base.make_negative_actions = _training_negative_actions


def main() -> None:
    base.evaluate = deterministic_evaluate
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
