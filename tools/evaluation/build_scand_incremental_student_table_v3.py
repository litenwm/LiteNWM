#!/usr/bin/env python3
"""Path-safe entrypoint for the corrected spent incremental table builder."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import build_scand_incremental_student_table_v2 as runner


if __name__ == "__main__":
    runner.main()
