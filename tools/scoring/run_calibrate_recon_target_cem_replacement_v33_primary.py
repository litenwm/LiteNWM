#!/usr/bin/env python3
"""Run v33 with preregistered primary ATE/objective tail constraints."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("calibrate_recon_target_cem_replacement_v33.py")
    source = target.read_text(encoding="utf-8")
    old = 'all(value["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1.0e-9 for name in NAMES)'
    new = 'all(value["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1.0e-9 for name in ("ate", "objective"))'
    if source.count(old) != 1:
        raise RuntimeError("v33 primary-tail patch point changed")
    namespace = {"__name__": "__main__", "__file__": str(target)}
    exec(compile(source.replace(old, new), str(target), "exec"), namespace)


if __name__ == "__main__":
    main()
