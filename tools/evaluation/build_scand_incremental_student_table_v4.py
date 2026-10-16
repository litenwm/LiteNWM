#!/usr/bin/env python3
"""Legacy-manifest compatible entrypoint for the spent incremental table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import build_scand_incremental_student_table_v1 as base
from tools.evaluation import build_scand_incremental_student_table_v2 as runner


_load_jsonl = base.load_jsonl


def compatible_jsonl(path):
    rows = _load_jsonl(path)
    result = []
    for row in rows:
        if "robot" not in row and "bag_id" in row:
            bag = str(row["bag_id"])
            row = {**row, "robot": "Jackal" if "_Jackal_" in bag else "Spot" if "_Spot_" in bag else "unknown"}
        result.append(row)
    return result


if __name__ == "__main__":
    base.load_jsonl = compatible_jsonl
    runner.main()
