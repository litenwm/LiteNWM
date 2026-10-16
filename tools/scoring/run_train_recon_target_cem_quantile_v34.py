#!/usr/bin/env python3
"""Run v34 while fitting only deployment-eligible existing-base promotions."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_recon_target_cem_quantile_v34.py")
    source = target.read_text(encoding="utf-8")
    old = "train_index=np.flatnonzero(cohort!=heldout)"
    new = "train_index=np.flatnonzero((cohort!=heldout) & valid.reshape(-1))"
    if source.count(old) != 1:
        raise RuntimeError("v34 eligible-training patch point changed")
    namespace = {"__name__": "__main__", "__file__": str(target)}
    exec(compile(source.replace(old, new), str(target), "exec"), namespace)


if __name__ == "__main__":
    main()
