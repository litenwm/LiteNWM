#!/usr/bin/env python3
"""Run v32 with finite encoding for invalid frozen CEM energies."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_recon_target_cem_replacement_v32.py")
    source = target.read_text(encoding="utf-8")
    old = "energy[:, :, None])"
    new = "np.nan_to_num(energy[:, :, None], nan=10.0, posinf=10.0, neginf=-10.0))"
    if source.count(old) != 1:
        raise RuntimeError("v32 energy patch point changed")
    namespace = {"__name__": "__main__", "__file__": str(target)}
    exec(compile(source.replace(old, new), str(target), "exec"), namespace)


if __name__ == "__main__":
    main()
