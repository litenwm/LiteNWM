#!/usr/bin/env python3
"""Project-root entry point for the frozen binding-corrected reranker."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import (  # noqa: E402
    run_scand_top10_latent_reranker_binding_corrected_v1 as corrected,
)


if __name__ == "__main__":
    corrected.runner.main()
