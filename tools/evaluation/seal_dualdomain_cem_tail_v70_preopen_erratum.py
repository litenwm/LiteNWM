#!/usr/bin/env python3
"""Append-only seal for the pre-outcome string-array audit implementation fix."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import write_json_atomic  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


CONFIRMATION = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/"
    "reserve_confirmation"
)
SEAL = CONFIRMATION / "sealed_manifest.json"
OUTPUT = CONFIRMATION / "sealed_preopen_audit_erratum.json"
OLD_BINDER = ROOT / "tools/evaluation/bind_dualdomain_cem_tail_v70_reserve_pair_sealed.py"
NEW_BINDER = ROOT / "tools/evaluation/bind_dualdomain_cem_tail_v70_reserve_pair_erratum.py"


def artifact(path: Path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "v70_sealed_before_reserve_outcomes":
        raise RuntimeError("original reserve seal is not active")
    forbidden = (
        CONFIRMATION / "preopen_reserve_pair_audit.json",
        CONFIRMATION / "evaluation/one_time_outcome_open_started.json",
        CONFIRMATION / "evaluation/ground_truth_once.npz",
        CONFIRMATION / "evaluation/result.json",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError("cannot seal pre-outcome erratum after reserve access: " + ", ".join(existing))
    value = {
        "schema": "dualdomain_cem_tail_v70_preopen_audit_erratum",
        "status": "preoutcome_audit_implementation_erratum_sealed",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_seal": artifact(SEAL),
        "failed_pair_binder": artifact(OLD_BINDER),
        "replacement_pair_binder": artifact(NEW_BINDER),
        "erratum_builder": artifact(Path(__file__)),
        "failure": {
            "type": "TypeError",
            "operation": "numpy.array_equal on two independent string record_key arrays with equal_nan=True",
            "stage": "outcome-free paired preopen audit",
        },
        "change_scope": "use equal_nan only for numeric arrays; use exact equality for string identity arrays",
        "model_changed": False,
        "thresholds_changed": False,
        "runner_changed": False,
        "selected_actions_changed": False,
        "acceptance_protocol_changed": False,
        "one_time_evaluator_changed": False,
        "reserve_outcomes_accessed": False,
        "retuning_allowed": False,
    }
    write_json_atomic(OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
