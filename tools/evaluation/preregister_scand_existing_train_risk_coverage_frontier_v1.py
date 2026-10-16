#!/usr/bin/env python3
"""Freeze the spent-development SCAND risk/coverage diagnostic contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
OUTPUT = BASE / "risk_coverage_frontier_v1"
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
        "pilot_result": BASE / "oof_upperbound_v1/result.json",
        "pilot_audit": BASE / "oof_upperbound_v1/audit.json",
        "pilot_oof_predictions": BASE / "oof_upperbound_v1/outer_oof_predictions.npz",
        "pilot_labels": BASE / "pilot_candidate_labels.npz",
        "pilot_manifest": BASE / "pilot_manifest.windows.jsonl",
        "pilot_collection_manifest": BASE / "proposal_tail_sidecar/collection_manifest.json",
        "formal_v70_result": ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json",
    }
    contract = {
        "schema": "scand_existing_train_risk_coverage_frontier_preregistration_v1",
        "status": "frozen_before_frontier_computation",
        "scope": "spent old-train development diagnostic only; never deployment or independent evidence",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "forbidden": {
            "reserve_access": True,
            "unused_old_train_bag_access": True,
            "fresh_outer_or_old_test_access": True,
            "model_or_feature_change": True,
            "cem_or_v70_change": True,
            "threshold_relaxation_after_outer_results": True,
            "promotion_from_this_diagnostic": True,
        },
        "models": [
            "route_balanced_pairwise_linear_ensemble",
            "group_dro_listwise_multitask_ensemble",
        ],
        "risk_score": {
            "ensemble_sigma": 1.0,
            "gain_floor_m": 0.002,
            "candidate_score": "min(p_dual_mean-sigma*p_dual_std, 1-(p_harm_mean+sigma*p_harm_std))",
            "candidate_choice": "per state max candidate_score; tie-break max gain_lcb then stable candidate index",
            "gain_lcb": "gain_mean-sigma*gain_std",
            "base_pool": "physical+support q0.999, excluding formal-v70 candidate",
        },
        "coverage_grid_fraction_of_all_states": [
            0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.35
        ],
        "posthoc_curve": {
            "role": "diagnostic_upper_bound_only",
            "threshold": "global OOF state-confidence quantile at each frozen coverage grid point",
            "outer_labels_may_not_select_a_deployable_threshold": True,
        },
        "nested_curve": {
            "outer_folds": 5,
            "inner_grouping": "route and bag zero overlap",
            "model_definition": "exact pilot/v77 model families and seeds; refit only to reconstruct inner-OOF predictions",
            "inner_choice": "largest frozen coverage point satisfying all inner risk and utility checks; deterministic smaller coverage tie-break",
            "outer_role": "evaluation only",
        },
        "inner_checks": {
            "replacement_count_nonzero": True,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "mean_objective_delta_max_m": 0.0,
            "mean_ate_delta_max_m": 0.0,
        },
        "diagnostic_verdicts": {
            "stable_high_precision_region": "nested aggregate nonzero, dual LCB>=0.85, harm UCB<=0.15, objective and ATE improve, replacements span >=3 routes",
            "posthoc_only_region": "posthoc passes but nested does not",
            "risk_representation_insufficient": "no posthoc point with >=5% coverage passes, or nested chooses no threshold in >=4/5 folds",
        },
        "small_sample_rule": "a high empirical rate that fails Wilson solely from too few replacements is reported as insufficient sample, not representation failure",
        "next_step": "only stable_high_precision_region permits a separately frozen one-time development validation on unused old-train bags",
    }
    PREREG.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(PREREG), "sha256": sha256_file(PREREG)}, indent=2))


if __name__ == "__main__":
    main()
