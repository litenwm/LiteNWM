#!/usr/bin/env python3
"""Execution entrypoint for the frozen 351+210 static-context table."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_scand_static_context_pair_table_v2 import main  # noqa: E402


if __name__ == "__main__":
    main()
