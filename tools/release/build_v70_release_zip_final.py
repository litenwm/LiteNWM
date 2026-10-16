#!/usr/bin/env python3
"""Final compatibility-closed invocation of build_v70_release_zip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_v70_release_zip as base


DEFAULT_OUTPUT = base.ROOT / "artifacts/v70_release_20260723_final"

base.DATA_TREES = base.DATA_TREES + (
    "data/evaluation/scand/strict_test_h4_100_seed17",
)
base.V70_PROVENANCE_FILES = base.V70_PROVENANCE_FILES + (
    (
        "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v66_candidate_level_calibration_correction.json"
    ),
)

_write_release_docs = base.write_release_docs


def write_release_docs(staging: Path, contract: dict) -> None:
    _write_release_docs(staging, contract)
    path = staging / "README_V70_RELEASE.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "    python tools/evaluation/check_v70_training_overlap.py",
        "    PYTHONPATH=src python tools/evaluation/check_v70_training_overlap.py",
    )
    path.write_text(text, encoding="utf-8")


base.write_release_docs = write_release_docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(base.build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
