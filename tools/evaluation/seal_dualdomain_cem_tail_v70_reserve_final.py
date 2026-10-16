#!/usr/bin/env python3
"""Execute the v70 reserve seal and bind every executable source used later."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import seal_dualdomain_cem_tail_v70_reserve as core  # noqa: E402


_write_json_atomic = core.write_json_atomic


def write_json_atomic(path: Path, value) -> None:
    value["scripts"]["pair_binder_core"] = value["scripts"]["pair_binder"]
    value["scripts"]["pair_binder"] = core.artifact(
        ROOT / "tools/evaluation/bind_dualdomain_cem_tail_v70_reserve_pair_sealed.py"
    )
    value["scripts"]["seal_orchestrator"] = core.artifact(Path(__file__))
    _write_json_atomic(path, value)


def main() -> None:
    core.write_json_atomic = write_json_atomic
    core.main()


if __name__ == "__main__":
    main()
