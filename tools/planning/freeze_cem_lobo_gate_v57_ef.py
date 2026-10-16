#!/usr/bin/env python3
"""Freeze the complete E/F v57 evaluation identity before opening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--reserve-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    calibration_path = args.policy_root / "calibration.json"
    prereg_path = args.policy_root / "selector_preregistration.json"
    calibration = read_json(calibration_path)
    prereg = read_json(prereg_path)
    reserve = read_json(args.reserve_freeze)
    if (
        calibration.get("E_F_outcomes_accessed")
        or prereg.get("E_F_outcomes_accessed")
        or prereg.get("post_outcome_tuning_allowed")
        or prereg.get("reserve_access_allowed")
        or reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
    ):
        raise RuntimeError("E/F or reserve outcome contract is not sealed")

    cohorts: dict[str, object] = {}
    route_sets: dict[str, set[str]] = {}
    for bank in ("bank_e", "bank_f"):
        run_dir = args.root / bank / "cem_run_v57_lobo_gate"
        selector_dir = args.root / bank / "cem_safe_addonly_v57"
        contract_path = run_dir / "preutility_contract.json"
        selector_freeze_path = selector_dir / "freeze.json"
        contract = read_json(contract_path)
        selector = read_json(selector_freeze_path)
        if (
            contract.get("labels_opened")
            or contract.get("outcomes_accessed")
            or contract.get("E_F_outcomes_accessed")
            or contract.get("reserve_accessed")
            or selector.get("labels_opened")
            or selector.get("outcomes_accessed")
            or selector.get("reserve_accessed")
            or selector.get("fixed_pool_action_replacement_allowed")
        ):
            raise RuntimeError(f"{bank} pre-outcome freeze is invalid")
        if contract["calibration_sha256"] != file_sha256(calibration_path):
            raise RuntimeError(f"{bank} calibration binding changed")
        if selector["cem_contract_sha256"] != file_sha256(contract_path):
            raise RuntimeError(f"{bank} contract binding changed")
        selected_path = Path(str(selector["selected_index"]))
        if selector["selected_index_sha256"] != file_sha256(selected_path):
            raise RuntimeError(f"{bank} selected index changed")
        routes = set(contract["paired_comparison"]["routes"])
        if len(routes) != int(contract["paired_comparison"]["route_count"]):
            raise RuntimeError(f"{bank} route identity is inconsistent")
        route_sets[bank] = routes
        cohorts[bank] = {
            "states": selector["states"],
            "routes": len(routes),
            "seed": contract["paired_comparison"]["seed"],
            "preutility_contract": str(contract_path),
            "preutility_contract_sha256": file_sha256(contract_path),
            "prelabel_freeze": contract["prelabel_freeze"],
            "prelabel_freeze_sha256": contract["prelabel_freeze_sha256"],
            "cem_selected_actions": contract["selected_actions"],
            "cem_selected_actions_sha256": contract[
                "selected_actions_sha256"
            ],
            "addonly_freeze": str(selector_freeze_path),
            "addonly_freeze_sha256": file_sha256(selector_freeze_path),
            "selected_index": str(selected_path),
            "selected_index_sha256": selector["selected_index_sha256"],
            "fixed_pool_promotions": selector["fixed_pool_promotions"],
            "cem_added_windows": selector["cem_added_windows"],
            "hybrid_promotions": selector["hybrid_promotions"],
            "iteration_audit": contract["iterations"],
        }
    overlap = sorted(route_sets["bank_e"] & route_sets["bank_f"])
    if overlap:
        raise RuntimeError(f"E/F route overlap: {overlap}")

    total_added = sum(
        int(cohorts[bank]["cem_added_windows"]) for bank in cohorts
    )
    gain_rule = prereg["substantial_gain_rule"]
    report = {
        "schema": "recon_cem_lobo_gate_v57_EF_preoutcome_freeze",
        "status": "E_F_candidates_policy_and_selected_index_frozen_before_outcomes",
        "evidence_role": "one_time_spent_development_evaluation",
        "candidate_gate": calibration["frozen_gate"],
        "formula": calibration["formula"],
        "calibration": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "selector_preregistration": str(prereg_path),
        "selector_preregistration_sha256": file_sha256(prereg_path),
        "cohorts": cohorts,
        "E_F_route_overlap": overlap,
        "total_states": sum(int(cohorts[b]["states"]) for b in cohorts),
        "total_routes": sum(int(cohorts[b]["routes"]) for b in cohorts),
        "total_addonly_windows_label_blind": total_added,
        "label_blind_max_coverage_gain": total_added / 200.0,
        "substantial_gain_rule": gain_rule,
        "substantial_added_count_already_impossible_label_blind": (
            total_added < int(gain_rule["added_windows_min"])
        ),
        "population": 16,
        "elite": 4,
        "shortfall_rule": "safe_count below 4 keeps proposal unchanged",
        "fixed_pool_action_replacement_allowed": False,
        "state_gate_used": False,
        "E_F_labels_opened": False,
        "E_F_outcomes_accessed": False,
        "post_E_F_tuning_allowed": False,
        "reserve_freeze": str(args.reserve_freeze),
        "reserve_freeze_sha256": file_sha256(args.reserve_freeze),
        "reserve_labels_opened": False,
        "reserve_outcomes_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
