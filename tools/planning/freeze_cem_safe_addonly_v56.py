#!/usr/bin/env python3
"""Freeze fixed-safe-pool plus CEM-safe add-only selection before outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cem-freeze", type=Path, required=True)
    parser.add_argument("--cem-contract", type=Path, required=True)
    parser.add_argument("--cem-selected", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    cem_freeze = json.loads(args.cem_freeze.read_text(encoding="utf-8"))
    contract = json.loads(args.cem_contract.read_text(encoding="utf-8"))
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    if (
        cem_freeze.get("schema") != "universal_raw_d_cem_safe_elite_v55"
        or cem_freeze.get("labels_opened")
        or cem_freeze.get("outcomes_used_by_cem")
        or cem_freeze.get("state_gate_used")
        or contract.get("outcomes_accessed")
        or contract.get("labels_opened")
        or contract.get("state_gate_used")
        or preregistration["selection_rule"][
            "fixed_pool_action_replacement_allowed"
        ]
    ):
        raise RuntimeError("add-only pre-outcome contract is invalid")
    if file_sha256(args.cem_freeze) != contract["prelabel_freeze_sha256"]:
        raise RuntimeError("CEM freeze changed after contract")
    if file_sha256(args.cem_selected) != cem_freeze["selected_actions_sha256"]:
        raise RuntimeError("CEM selection changed after freeze")
    with np.load(args.cem_selected, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        cem_false = payload["universal_d_cem_false_selected_index"].astype(
            np.int64
        )
        cem_true_local = payload["universal_d_cem_true_selected_index"].astype(
            np.int64
        )
        final_safe = payload["final_cem_safe"].astype(bool)
    row = np.arange(len(record))
    if not np.all(
        (cem_true_local == 0) | final_safe[row, cem_true_local]
    ):
        raise RuntimeError("CEM-selected action did not pass final safe gate")
    added = (cem_false == 0) & (cem_true_local != 0)
    hybrid = cem_false.copy()
    hybrid[added] = 16 + cem_true_local[added]
    if not np.array_equal(hybrid[cem_false != 0], cem_false[cem_false != 0]):
        raise RuntimeError("add-only rule replaced a fixed-pool promotion")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_index.npz"
    write_npz_atomic(
        selected_path,
        record_key=record,
        base_k17_feature_waypoint_units=base,
        final_cem_feature_waypoint_units=final,
        safe_cem_false_selected_index=cem_false,
        final_cem_safe_selected_local_index=cem_true_local,
        hybrid_combined_selected_index=hybrid,
        cem_added_window=added,
    )
    freeze_path = args.output_dir / "freeze.json"
    freeze = {
        "schema": "recon_cem_safe_addonly_v56_freeze",
        "status": "hybrid_addonly_selection_frozen_before_outcomes",
        "evidence_role": "A_to_T_spent_development_confirmation",
        "blind_confirmation": False,
        "production_safe_claim_allowed": False,
        "cem": True,
        "state_gate_used": False,
        "selection_rule": (
            "Keep every safe CEM=false fixed-pool promotion unchanged. Only "
            "where CEM=false abstains, add the final-iteration CEM candidate if "
            "it passed the same frozen candidate-level safe gate."
        ),
        "fixed_pool_action_replacement_allowed": False,
        "attribution": (
            "Only cem_added_window is attributable to CEM marginal coverage; "
            "retained fixed-pool successes are not counted as CEM gains."
        ),
        "states": len(record),
        "fixed_pool_promotions": int((cem_false != 0).sum()),
        "cem_safe_candidates_selected": int((cem_true_local != 0).sum()),
        "cem_added_windows": int(added.sum()),
        "hybrid_promotions": int((hybrid != 0).sum()),
        "cem_freeze": str(args.cem_freeze),
        "cem_freeze_sha256": file_sha256(args.cem_freeze),
        "cem_contract": str(args.cem_contract),
        "cem_contract_sha256": file_sha256(args.cem_contract),
        "cem_selected": str(args.cem_selected),
        "cem_selected_sha256": file_sha256(args.cem_selected),
        "preregistration": str(args.preregistration),
        "preregistration_sha256": file_sha256(args.preregistration),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "labels_opened": False,
        "outcomes_accessed": False,
        "reserve_accessed": False,
    }
    write_json_atomic(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
