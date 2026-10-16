#!/usr/bin/env python3
"""Run v61b with the repository root on the import path."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


if __name__ == "__main__":
    runpy.run_module(
        "tools.evaluation.crossfit_recon_cem_fixed_harm_v61b",
        run_name="__main__",
    )
