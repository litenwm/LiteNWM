#!/usr/bin/env python3
"""Freeze the v76 SCAND-tail safety-calibration contract before outer OOF evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cem_development_v1/v76_scand_tail_safety_calibration"
PREREG = OUTPUT / "preregistration.json"
V70_ROOT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
REPLAY = V70_ROOT / "replays/scand_fresh_outer/proposal_aware_v70"
EVALUATION = V70_ROOT / "evaluations/scand_fresh_outer"
ATTRIBUTION = ROOT / "outputs/cem_development_v1/scand_filter_attribution_v1"
SCAND_TRUTH = (
    ROOT
    / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes"
    / "fresh_outer_outcomes.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def bind(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    forbidden = "res" + "erve"
    if forbidden in str(resolved).lower():
        raise RuntimeError(f"forbidden cohort path: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def main() -> None:
    if PREREG.exists():
        raise RuntimeError(f"refusing to overwrite frozen preregistration: {PREREG}")

    freeze = read_json(REPLAY / "prelabel_freeze.json")
    scand_inputs: dict[str, Path] = {
        "formal_v70_result": V70_ROOT / "result.json",
        "formal_v70_aggregate_result": V70_ROOT / "aggregate_result.json",
        "scand_formal_evaluation": EVALUATION / "result.json",
        "scand_formal_selected_metrics": EVALUATION / "selected_metrics.npz",
        "scand_prelabel_freeze": REPLAY / "prelabel_freeze.json",
        "scand_selected_actions": REPLAY / "selected_actions.npz",
        "scand_fixed_k17_features": ROOT / freeze["base_features"],
        "scand_ground_truth": SCAND_TRUTH,
        "support_checkpoint": Path(freeze["support_checkpoint"]),
        "filter_attribution_preregistration": ATTRIBUTION / "preregistration.json",
        "filter_attribution_result": ATTRIBUTION / "result.json",
        "filter_attribution_audit": ATTRIBUTION / "audit.json",
        "candidate_reconstruction_runner": ROOT
        / "tools/evaluation/run_scand_filter_random_sensitivity_v1.py",
        "filter_attribution_runner": ROOT
        / "tools/evaluation/run_scand_filter_attribution_v1.py",
        "union_loader": ROOT / "tools/evaluation/run_v72_proposal_tail_reranker.py",
        "formal_archive_runner": ROOT
        / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py",
        "formal_replay_runner": ROOT
        / "tools/planning/run_dualdomain_cem_tail_v66_replay.py",
    }
    for iteration in range(3):
        prefix = f"scand_round_{iteration + 1}"
        directory = REPLAY / f"iteration_{iteration:02d}"
        for stem in (
            "cem_features.npz",
            "frozen_scores.npz",
            "population.npz",
            "source_label_free.npz",
            "cem_features.audit.json",
        ):
            scand_inputs[f"{prefix}_{stem.replace('.', '_')}"] = directory / stem

    recon_inputs = {
        f"formal_recon_bank_{bank}_evaluation": (
            V70_ROOT / f"evaluations/recon_bank_{bank}/result.json"
        )
        for bank in "abcdefgh"
    }
    inputs = {
        name: bind(path) for name, path in {**scand_inputs, **recon_inputs}.items()
    }

    contract = {
        "schema": "v76_scand_tail_safety_calibration_preregistration_v1",
        "status": "frozen_before_v76_outer_heldout_evaluation",
        "study_role": "spent-development-only; not independent validation or production certification",
        "objective": (
            "Test whether simple SCAND-only calibration of the frozen v70 safety outputs "
            "recovers held-out-route direction without weakening the physical/support/CEM/scorer/archive contracts."
        ),
        "constraints": {
            "reserve_access": False,
            "reserve_replay": False,
            "formal_v70_mutation": False,
            "recon_path_mutation": False,
            "dwm_ltwm_rerun": False,
            "candidate_generation": False,
            "cem_mutation": False,
            "candidate_pool_mutation": False,
            "physical_guard_mutation": False,
            "support_q0999_mutation": False,
            "scorer_energy_mutation": False,
            "archive_logic_mutation": False,
            "post_outer_threshold_tuning": False,
            "automatic_follow_on_modeling": False,
        },
        "frozen_system": {
            "domain_changed": "SCAND only",
            "recon_behavior": "byte-for-byte binding to the eight existing formal v70 evaluation results",
            "states": 100,
            "routes_expected": 25,
            "candidate_union": "exact-deduplicated fixed K17 plus frozen v70 rounds r1/r2/r3; 65 candidates/state",
            "cem_iterations": 3,
            "population_per_iteration": 16,
            "physical_guard": "unchanged hard constraint",
            "support": {"quantile": 0.999, "distance_threshold": 4.876645565032959},
            "formal_safety_reference": "fixed-pool candidate0 incumbent",
            "formal_safety_margins_m": {"ate": -0.01, "objective": -0.01},
            "formal_v70_fallback": True,
            "end_to_end_rank": "unchanged frozen v70 scorer energy and cross-round archive energy",
        },
        "labels": {
            "calibration_target": (
                "candidate metric minus fixed-pool candidate0 metric; negative is improvement; "
                "objective=0.4*ATE+0.6*endpoint"
            ),
            "filter_and_end_to_end_reference": "formal v70 selected action in the same state",
            "strict_dual": "ATE gain > 1e-7 m AND objective gain > 1e-7 m",
            "strict_harm": "ATE gain < -1e-7 m OR objective gain < -1e-7 m",
            "neutral_1mm": "sensitivity report only; never used for formal admission or success",
        },
        "cross_validation": {
            "outer": "five route-grouped folds over SCAND; held-out routes are evaluation-only",
            "inner": (
                "for each outer fold, four-fold leave-one-existing-fold-out route cross-fitting on outer-train routes; "
                "all family/profile selection uses only the concatenated inner-OOF predictions"
            ),
            "route_overlap_allowed": False,
            "deployment_forbidden_features": ["route_id", "record_id", "ground_truth", "outer_fold_labels"],
            "calibration_training_domain": "outer-train SCAND only",
        },
        "families": {
            "location_scale_platt": {
                "mean": (
                    "separate ATE and objective robust affine fits y=a+b*x using fixed 10-step Huber IRLS; "
                    "Huber c=1.345*MAD and ridge=1e-6"
                ),
                "scale": (
                    "separate residual scale multiplier median(|residual|/max(std,1e-4))/0.67448975, "
                    "clipped to [0.25,4.0]"
                ),
                "ucb": "calibrated_mean + Normal.ppf(q)*calibrated_std",
                "probabilities": "separate two-parameter Platt calibration of frozen p_dual and p_harm",
            },
            "source_uncertainty_conformal_platt": {
                "mean": "frozen mean retained",
                "bins": (
                    "fixed/r1/r2/r3 source x low/high combined uncertainty; split at outer-train-only median; "
                    "a bin needs >=100 candidates and >=5 routes, otherwise global fallback"
                ),
                "ucb": (
                    "mean plus inverse-route-frequency weighted finite-sample empirical q-quantile of y-mean residual"
                ),
                "probabilities": "separate two-parameter Platt calibration of frozen p_dual and p_harm",
            },
        },
        "risk_profiles": {
            "balanced": {"ucb_quantile": 0.90, "p_dual_min": 0.60, "p_harm_max": 0.10},
            "conservative": {"ucb_quantile": 0.95, "p_dual_min": 0.75, "p_harm_max": 0.05},
        },
        "admission_and_selection": {
            "candidate_safety": (
                "physical AND support AND calibrated ATE UCB<-0.01 m AND calibrated objective UCB<-0.01 m "
                "AND calibrated p_dual>=profile threshold AND calibrated p_harm<=profile threshold"
            ),
            "fallback": "formal v70 selected action",
            "replacement": (
                "among admitted candidates with finite frozen energy strictly lower than formal v70 archive_energy, "
                "choose minimum frozen energy; otherwise fallback"
            ),
            "inner_feasibility": {
                "nonzero_replacements": True,
                "incremental_dual_wilson_95_lcb_min": 0.85,
                "incremental_harm_wilson_95_ucb_max": 0.15,
                "mean_ate_non_degradation": True,
                "mean_objective_non_degradation": True,
                "objective_p90_max_delta_m": 0.001,
                "mean_objective_gain_min_m": 0.001,
            },
            "inner_tie_break": (
                "highest objective gain, then more replacements, then conservative over balanced, "
                "then family name lexical order"
            ),
            "no_inner_feasible_policy": "nested-selected outer fold makes zero replacement; diagnostic families remain reported",
        },
        "reporting": {
            "calibration": [
                "ATE/objective residual bias and MAE",
                "UCB empirical coverage/violation",
                "dual/harm Brier, ECE, AUROC",
                "route and proposal-source breakdown",
            ],
            "filter": [
                "physical+support pool admitted dual/harm candidates",
                "state coverage and opportunity coverage",
                "dual recall/purity and harm rejection",
                "original-safety comparison and Wilson 95 intervals",
            ],
            "end_to_end": [
                "SCAND and pooled ATE/RPE/endpoint/objective mean and p90",
                "replacement coverage, incremental dual/harm with Wilson 95",
                "1mm neutral-band sensitivity",
                "route-level outcomes",
                "frozen RECON metrics and hashes",
            ],
            "artifacts": [
                "result.json",
                "audit.json",
                "outer_oof_predictions.npz",
                "calibration_parameters.json",
                "sha256_manifest.json",
            ],
        },
        "formal_success_gate": {
            "nested_selected_nonzero_replacements": True,
            "scand_mean_ate_non_degradation": True,
            "scand_mean_objective_non_degradation": True,
            "scand_objective_p90_max_delta_m": 0.001,
            "incremental_dual_wilson_95_lcb_min": 0.85,
            "incremental_harm_wilson_95_ucb_max": 0.15,
            "pooled_state_weighted_objective_gain_min_m": 0.001,
            "pooled_definition": "eight frozen RECON A-H cohorts plus SCAND, 788 states total, state weighted",
            "filter_only_outcome": "safety_calibration_only_pass/end_to_end_fail",
        },
        "observability": {
            "cached_draws_only": True,
            "pre_persistence_sampling_draws": "unobservable and excluded",
            "no_counterfactual_candidate_fabrication": True,
        },
        "failure_rule": "stop after v76; do not add models, relax gates, or continue automatically",
        "inputs": inputs,
    }
    write_json_atomic(PREREG, contract)
    print(json.dumps({"path": str(PREREG), "sha256": sha256_file(PREREG)}, indent=2))


if __name__ == "__main__":
    main()
