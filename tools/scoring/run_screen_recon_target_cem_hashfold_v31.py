#!/usr/bin/env python3
"""Run v31 with identity-bound copies of legacy CEM frozen-score files."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def prepare(root: Path) -> None:
    for letter in "abcdef":
        iteration = root / f"bank_{letter}/cem_run_v30/iteration_02"
        feature_path = iteration / "cem_features.npz"
        raw_path = iteration / "frozen_scores.npz"
        output = iteration / "frozen_scores_record_v31.npz"
        audit = iteration / "frozen_scores_record_v31.audit.json"
        if output.exists():
            continue
        with np.load(feature_path, allow_pickle=False) as payload:
            record = payload["record_key"].astype(str)
        with np.load(raw_path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        write_npz(output, record_key=record, **arrays)
        write_json(audit, {
            "schema": "cem_frozen_scores_identity_binding_v31",
            "source": str(raw_path),
            "source_sha256": file_sha256(raw_path),
            "feature": str(feature_path),
            "feature_sha256": file_sha256(feature_path),
            "output": str(output),
            "output_sha256": file_sha256(output),
            "outcomes_accessed": False,
        })


def main() -> None:
    root = REPO_ROOT / "outputs/cem_development_v1/recon_target_calibration_v20"
    prepare(root)
    target = Path(__file__).with_name("screen_recon_target_cem_hashfold_v31.py")
    source = target.read_text(encoding="utf-8")
    old = 'score_path = bank / "cem_run_v30/iteration_02/frozen_scores.npz"'
    new = 'score_path = bank / "cem_run_v30/iteration_02/frozen_scores_record_v31.npz"'
    if source.count(old) != 1:
        raise RuntimeError("v31 score-path patch point changed")
    namespace = {"__name__": "__main__", "__file__": str(target)}
    exec(compile(source.replace(old, new), str(target), "exec"), namespace)


if __name__ == "__main__":
    main()
