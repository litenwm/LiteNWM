#!/usr/bin/env python3
"""Apply the sealed pre-outcome string-array comparison erratum to pair binding."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import bind_dualdomain_cem_tail_v70_reserve_pair_sealed as sealed  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


ERRATUM = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/"
    "reserve_confirmation/sealed_preopen_audit_erratum.json"
)


def main() -> None:
    erratum = json.loads(ERRATUM.read_text(encoding="utf-8"))
    if erratum.get("status") != "preoutcome_audit_implementation_erratum_sealed":
        raise RuntimeError("preopen audit erratum is not sealed")
    if erratum["replacement_pair_binder"]["sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("replacement pair binder differs from sealed erratum")
    seal_path = Path(erratum["original_seal"]["path"])
    if erratum["original_seal"]["sha256"] != file_sha256(seal_path):
        raise RuntimeError("original v70 reserve seal changed after erratum")

    core = sealed.core
    original_array_equal = core.np.array_equal

    def array_equal(left, right, equal_nan=False):
        left_array = np.asanyarray(left)
        right_array = np.asanyarray(right)
        numeric = np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
            right_array.dtype, np.number
        )
        return original_array_equal(
            left_array,
            right_array,
            equal_nan=bool(equal_nan and numeric),
        )

    core.np.array_equal = array_equal
    original_write_json_atomic = core.write_json_atomic

    def write_json_atomic(path, value):
        value["preoutcome_audit_implementation_erratum"] = str(ERRATUM.resolve())
        value["preoutcome_audit_implementation_erratum_sha256"] = file_sha256(ERRATUM)
        original_write_json_atomic(path, value)

    core.write_json_atomic = write_json_atomic
    sealed.main()


if __name__ == "__main__":
    main()
