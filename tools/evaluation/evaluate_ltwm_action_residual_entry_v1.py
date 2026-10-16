#!/usr/bin/env python3
"""Mechanical entry point for the paired action-residual evaluation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools/recon")]

from tools.evaluation import evaluate_ltwm_action_residual_v1 as evaluation  # noqa: E402


evaluation.false = False


if __name__ == "__main__":
    evaluation.main()
