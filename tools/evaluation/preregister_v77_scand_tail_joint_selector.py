#!/usr/bin/env python3
"""Freeze v77 joint SCAND-tail safety/reranking contract before outer results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cem_development_v1/v77_scand_tail_joint_selector"
PREREG = OUTPUT / "preregistration.json"
INVENTORY = OUTPUT / "route_inventory.json"
V70 = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
REPLAY = V70 / "replays/scand_fresh_outer/proposal_aware_v70"
SCAND_EVAL = V70 / "evaluations/scand_fresh_outer"
SCAND_TRUTH = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes/fresh_outer_outcomes.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bind(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if "res" + "erve" in str(path).lower():
        raise RuntimeError(f"forbidden input: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    if PREREG.exists():
        raise FileExistsError(PREREG)
    inventory = read_json(INVENTORY)
    if inventory["legal_untouched_routes_with_required_v70_candidate_cache"] != 0:
        raise RuntimeError("route inventory requires manual untouched-route isolation")
    if inventory["reserve_accessed"]:
        raise RuntimeError("route inventory touched forbidden cohort")
    freeze = read_json(REPLAY / "prelabel_freeze.json")
    paths: dict[str, Path] = {
        "route_inventory": INVENTORY,
        "formal_v70_result": V70 / "result.json",
        "formal_v70_aggregate": V70 / "aggregate_result.json",
        "formal_scand_evaluation": SCAND_EVAL / "result.json",
        "formal_scand_selected_metrics": SCAND_EVAL / "selected_metrics.npz",
        "scand_prelabel_freeze": REPLAY / "prelabel_freeze.json",
        "scand_selected_actions": REPLAY / "selected_actions.npz",
        "scand_fixed_features": ROOT / freeze["base_features"],
        "scand_ground_truth": SCAND_TRUTH,
        "support_checkpoint": Path(freeze["support_checkpoint"]),
        "v76_result": ROOT / "outputs/cem_development_v1/v76_scand_tail_safety_calibration/result.json",
        "v76_audit": ROOT / "outputs/cem_development_v1/v76_scand_tail_safety_calibration/audit.json",
        "candidate_reconstruction": ROOT / "tools/evaluation/run_scand_filter_random_sensitivity_v1.py",
        "candidate_union_loader": ROOT / "tools/evaluation/run_v72_proposal_tail_reranker.py",
        "structural_feature_loader": ROOT / "tools/evaluation/run_v74_direct_delta_nested.py",
        "formal_archive_runner": ROOT / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py",
        "formal_replay_runner": ROOT / "tools/planning/run_dualdomain_cem_tail_v66_replay.py",
    }
    for iteration in range(3):
        directory = REPLAY / f"iteration_{iteration:02d}"
        for name in ("population.npz", "cem_features.npz", "frozen_scores.npz", "source_label_free.npz"):
            paths[f"round_{iteration + 1}_{name.replace('.', '_')}"] = directory / name
    for bank in "abcdefgh":
        paths[f"formal_recon_{bank}"] = V70 / f"evaluations/recon_bank_{bank}/result.json"
    inputs = {name: bind(path) for name, path in paths.items()}

    prereg = {
        "schema": "v77_scand_tail_joint_selector_preregistration_v1",
        "status": "frozen_before_v77_outer_heldout_evaluation",
        "study_role": "spent-development-only nested route-OOF evidence",
        "route_inventory": {
            "sha256": sha256_file(INVENTORY),
            "spent_routes": 25,
            "spent_states": 100,
            "legal_untouched_routes": 0,
            "new_route_usage": [],
        },
        "constraints": {
            "reserve_access": False,
            "formal_v70_mutation": False,
            "recon_path_mutation": False,
            "dwm_ltwm_rerun": False,
            "candidate_generation": False,
            "cem_mutation": False,
            "physical_guard_mutation": False,
            "support_q0999_mutation": False,
            "formal_archive_mutation": False,
            "old_safety_hard_gate": False,
            "frozen_energy_hard_gate": False,
            "post_outer_tuning": False,
            "automatic_follow_on": False,
        },
        "candidate_contract": {
            "union": "exact fixed K17 + frozen v70 r1/r2/r3 unique union; 65 candidates/state",
            "input_gate": "physical hard guard AND frozen support distance<=q0.999 threshold 4.876645565032959",
            "old_safety": "deployment feature only, never an eligibility requirement",
            "frozen_energy": "deployment feature only, never an eligibility or ordering requirement",
            "formal_v70": "per-state fallback and label reference; never modified",
            "candidate0": "may be selected when physical+support-valid because labels are relative to formal v70",
            "objective": "0.4*ATE+0.6*endpoint; all true gains are formal-v70 metric minus candidate metric",
        },
        "deployment_features": {
            "required": [
                "state-relative and within-state rank features",
                "frozen ATE/RPE/endpoint/objective mean, std, and z=2 UCB",
                "frozen dual/harm logits and old-safe flag",
                "frozen scorer energy, scorer-v2 components and proxy",
                "proposal source round and within-source rank",
                "action and waypoint sequence structure, smoothness, curvature and distance to formal-v70",
                "three-seed means, standard deviations/ranges and support disagreement",
                "local action/waypoint density",
            ],
            "forbidden": ["route ID", "record ID", "outer-fold labels", "truth-derived feature", "future metric"],
            "preprocessing": "outer/inner train-only mean/std; std floor 1e-5; nonfinite deployment values replaced before scaling by fixed state-local rules",
        },
        "models": {
            "route_balanced_pairwise_linear_ensemble": {
                "members": 3,
                "seeds": [20267701, 20267702, 20267703],
                "route_weighting": "inverse route candidate count times seeded lognormal(sigma=0.25) route multiplier",
                "gain_head": (
                    "ridge on within-state pair differences: all top-8 x bottom-8 physical+support candidates, "
                    "plus route-weighted point anchors at weight 0.25; target is true objective gain difference; ridge lambda=10"
                ),
                "risk_heads": "separate route-balanced class-balanced L2 logistic dual/harm heads; lambda=1; damped Newton",
            },
            "group_dro_listwise_multitask_ensemble": {
                "members": 3,
                "seeds": [20267711, 20267712, 20267713],
                "network": "input -> Linear(48) -> SiLU -> Linear(24) -> SiLU -> 3 outputs (gain, dual logit, harm logit)",
                "optimizer": "Adam lr=1e-3, weight_decay=1e-3, 80 full-batch epochs",
                "loss": (
                    "per-state ListNet objective-gain ranking (temperature=0.02, weight=1.0) + "
                    "Huber gain anchor beta=0.01 (weight=0.25) + class-balanced dual BCE (0.5) + harm BCE (0.5)"
                ),
                "route_robustness": "group-DRO over train routes; eta=0.05, exponent clipped [-2,2], normalized each epoch",
            },
        },
        "risk_and_fallback_profiles": {
            "balanced_direct": {
                "ensemble_sigma": 1.0,
                "dual_lcb_min": 0.80,
                "harm_ucb_max": 0.08,
                "objective_gain_lcb_min_m": 0.002,
                "selection": "maximum objective gain LCB among all risk-admitted candidates",
                "fallback": "formal v70 if none",
            },
            "conservative_top5": {
                "ensemble_sigma": 1.5,
                "dual_lcb_min": 0.90,
                "harm_ucb_max": 0.03,
                "objective_gain_lcb_min_m": 0.004,
                "selection": "consider predicted-gain top-5 nonfallback candidates, then choose maximum admitted gain LCB",
                "fallback": "formal v70 if none",
            },
        },
        "cross_validation": {
            "outer": "same deterministic five route-grouped folds; held-out routes evaluation-only",
            "inner": "within each outer-train set, four leave-one-existing-fold-out route-OOF fits",
            "selection_unit": "model family x frozen risk/fallback profile",
            "inner_feasibility": {
                "nonzero_replacements": True,
                "dual_wilson95_lcb_min": 0.85,
                "harm_wilson95_ucb_max": 0.15,
                "mean_ate_non_degrade": True,
                "mean_objective_non_degrade": True,
                "objective_p90_max_delta_m": 0.001,
                "scand_objective_gain_min_m": 0.00788,
                "rationale": "7.88mm SCAND gain corresponds to 1mm over frozen 688 RECON + 100 SCAND states",
            },
            "inner_tie_break": "largest objective gain, then more replacements, conservative profile, listwise model, lexical key",
            "no_feasible_inner_choice": "outer fold emits formal-v70 fallback only",
        },
        "evaluation": {
            "candidate_layer": ["dual/harm AUROC, Brier, ECE", "objective-gain Pearson/Spearman", "NDCG@5", "top-5 true-dual recall"],
            "filter_layer": ["dual recall/purity", "harm rejection", "state opportunity recovery out of 89", "Wilson 95 intervals", "old-safety comparison"],
            "end_to_end": [
                "SCAND four metrics mean/p90 and deltas",
                "replacement coverage and strict dual/harm Wilson 95",
                "raw 0mm and +/-1mm sensitivity",
                "bypass-old-safety and bypass-frozen-energy counts",
                "proposal-source and route distribution",
                "CPU inference latency",
                "RECON unchanged hashes and pooled mean",
            ],
        },
        "formal_success_gate": {
            "nonzero_replacements": True,
            "scand_mean_ate_non_degrade": True,
            "scand_mean_objective_non_degrade": True,
            "scand_objective_p90_max_delta_m": 0.001,
            "incremental_dual_wilson95_lcb_min": 0.85,
            "incremental_harm_wilson95_ucb_max": 0.15,
            "pooled_objective_gain_min_m": 0.001,
            "recon_unchanged": True,
        },
        "verdicts": {
            "pass": "development_candidate_only; requires one untouched independent-route validation",
            "filter_only": "distinguish representation/risk pass from reranking failure",
            "fail": "stop after v77; no model expansion, gate relaxation, reserve access, or automatic continuation",
            "numeric_solver_bug": "same-model stable solver correction allowed only with failed artifacts preserved; no model/profile/gate changes",
        },
        "artifacts": ["result.json", "audit.json", "outer_oof_predictions.npz", "model_parameters.npz", "model_parameters.audit.json", "sha256_manifest.json"],
        "inputs": inputs,
    }
    temporary = PREREG.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(PREREG)
    print(json.dumps({"path": str(PREREG), "sha256": sha256_file(PREREG)}, indent=2))


if __name__ == "__main__":
    main()
