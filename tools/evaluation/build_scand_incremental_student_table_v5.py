#!/usr/bin/env python3
"""Final schema-compatible entrypoint for the spent incremental table."""

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
    for value in rows:
        row = dict(value)
        if "robot" not in row and "bag_id" in row:
            bag = str(row["bag_id"])
            row["robot"] = "Jackal" if "_Jackal_" in bag else "Spot" if "_Spot_" in bag else "unknown"
        # v1's fallback expression is eagerly evaluated even when record_key
        # exists.  This sentinel is never used for identity when record_key is
        # present; it only bridges historical manifest schemas safely.
        if "record_key" in row and "curr_time" not in row:
            row["curr_time"] = 0
        result.append(row)
    return result


if __name__ == "__main__":
    base.load_jsonl = compatible_jsonl
    runner.main()
