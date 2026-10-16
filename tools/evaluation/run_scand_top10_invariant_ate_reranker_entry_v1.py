#!/usr/bin/env python3
"""Mechanical entry point that creates the new artifact directory after freeze checks."""

from __future__ import annotations

from pathlib import Path
import sys

from tools.evaluation import run_scand_top10_invariant_ate_reranker_v1 as runner


_ORIGINAL_READ_JSON = runner.read_json


def _output_dir() -> Path:
    flag = sys.argv.index("--output-dir")
    return Path(sys.argv[flag + 1]).resolve()


def _read_json(path: Path):
    value = _ORIGINAL_READ_JSON(path)
    output = _output_dir()
    runner.require(not output.exists(), f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    runner.read_json = _ORIGINAL_READ_JSON
    return value


runner.read_json = _read_json


if __name__ == "__main__":
    runner.main()
