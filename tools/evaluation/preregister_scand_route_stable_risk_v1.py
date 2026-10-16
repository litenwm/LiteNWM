#!/usr/bin/env python3
"""Freeze the route-stable SCAND proposal-tail risk experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
OUTPUT = BASE / "route_stable_risk_v1"
PREREG = OUTPUT / "preregistration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if PREREG.exists():
        raise FileExistsError(PREREG)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        "pilot_preregistration": BASE / "preregistration.json",
        "pilot_labels": BASE / "pilot_candidate_labels.npz",
        "pilot_manifest": BASE / "pilot_manifest.windows.jsonl",
        "pilot_collection_manifest": BASE / "proposal_tail_sidecar/collection_manifest.json",
        "pilot_oof_result": BASE / "oof_upperbound_v1/result.json",
        "frontier_result": BASE / "risk_coverage_frontier_v1/result.json",
        "frontier_preregistration": BASE / "risk_coverage_frontier_v1/preregistration.json",
        "formal_v70_result": ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json",
    }
    contract = {
        "schema": "scand_route_stable_risk_preregistration_v1",
        "status": "frozen_before_route_stable_outer_predictions",
        "scope": "adaptive spent old-train development experiment; not independent evidence",
        "hypothesis": "state-normalized relative/rank features plus worst-route group-DRO can improve high-confidence tail stability even when full-feature AUROC is already high",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "forbidden": {
            "reserve_access": True,
            "unused_old_train_bag_access": True,
            "fresh_outer_old_test_or_scorer_cal_access": True,
            "v70_cem_safety_archive_change": True,
            "feature_pattern_or_hyperparameter_change_after_outer": True,
            "outer_label_threshold_selection": True,
            "automatic_validation_or_promotion": True,
        },
        "feature_contract": {
            "source": "audited 184 deployment-time pilot features",
            "include_patterns": [
                "relative_formal_*",
                "within_state_rank_*",
                "local_density_*",
                "proposal_source_*",
                "safety_std_[0-3]",
                "seed_[1,2,4,5,7,8]",
                "support_distance_0",
            ],
            "transformation": "for every selected feature, concatenate statewise physical+support-pool z-score clipped to [-5,5] and percentile rank scaled to [-1,1]",
            "online_forbidden": ["route_id", "bag_id", "record_id", "truth", "absolute image identity"],
        },
        "model": {
            "name": "state_normalized_group_dro_linear_multitask_ensemble",
            "members": 3,
            "seeds": [20267801, 20267802, 20267803],
            "architecture": "single affine gain/dual/harm head",
            "epochs": 160,
            "optimizer": "Adam lr=0.002 weight_decay=0.01",
            "loss": "route-level mean of gain Huber + balanced dual BCE + balanced harm BCE",
            "loss_weights": {"gain": 0.5, "dual": 1.0, "harm": 1.0},
            "group_dro_eta": 0.05,
            "training_pool": "physical+support q0.999 challengers, formal-v70 excluded",
        },
        "evaluation": {
            "outer_folds": 5,
            "inner_route_oof": True,
            "route_and_bag_overlap": 0,
            "risk_score": "min(dual ensemble LCB, 1-harm ensemble UCB), sigma=1",
            "gain_lcb_floor_m": 0.002,
            "coverage_grid": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.35],
            "inner_choice": "largest coverage passing every frozen risk/utility check",
            "outer_labels": "evaluation only",
        },
        "success_gate": {
            "nested_replacements_nonzero": True,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "mean_objective_and_ate_improve": True,
            "replacement_routes_min": 3,
            "recon_unchanged": True,
        },
        "stop_rule": "if nested success gate fails, stop this representation family; do not tune thresholds or add another model",
        "if_pass": "freeze only as development candidate, then ask before any one-time unused-old-train validation",
    }
    PREREG.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(PREREG), "sha256": sha256_file(PREREG)}, indent=2))


if __name__ == "__main__":
    main()
