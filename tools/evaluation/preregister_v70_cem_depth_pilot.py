#!/usr/bin/env python3
"""Preregister the v70 depth-3/4/5 spent-cohort pilot before new rounds."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import write_json_atomic  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


V70_DIR = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
OUTPUT_DIR = ROOT / "outputs/cem_development_v1/v70_cem_depth_ablation_v71_pilot"
PREREG = OUTPUT_DIR / "preregistration.json"
V70_RESULT = V70_DIR / "result.json"
BASE_RESULT = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
RESERVE_AUDIT = V70_DIR / "reserve_confirmation/evaluation/one_time_outcome_open_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(OUTPUT_DIR)
    reserve_audit = read_json(RESERVE_AUDIT)
    if reserve_audit.get("status") != "reserve_opened_exactly_once_and_reported_without_retuning":
        raise RuntimeError("reserve spent status is unclear; depth pilot must stop")
    if not reserve_audit.get("reserve_spent") or reserve_audit.get("retuning_allowed"):
        raise RuntimeError("reserve is not irreversibly spent")

    v70 = read_json(V70_RESULT)
    if v70.get("status") != "frozen_after_v69_spent_failure_before_v70_spent_replay":
        raise RuntimeError("formal v70 candidate is not frozen")
    if v70.get("reserve_accessed"):
        raise RuntimeError("v70 training result reports reserve use")
    expected_policies = {
        "recon": {
            "z": 0.5,
            "margin_ate": 0.01,
            "margin_objective": 0.01,
            "dual_threshold": 0.9,
            "harm_threshold": 0.2,
        },
        "scand": {
            "z": 2.0,
            "margin_ate": 0.01,
            "margin_objective": 0.01,
            "dual_threshold": 0.5,
            "harm_threshold": 0.1,
        },
    }
    for domain, policy in expected_policies.items():
        if v70["calibration"][domain]["policy"] != policy:
            raise RuntimeError(f"v70 {domain} policy changed")

    cohort_specs = {
        "recon_bank_g": {
            "freeze": V70_DIR / "replays/recon_bank_g/proposal_aware_v70/prelabel_freeze.json",
            "evaluation": V70_DIR / "evaluations/recon_bank_g/result.json",
            "trajectory_root": Path("/home/rainbow/nwm/data/recon"),
            "continuation_dir": OUTPUT_DIR / "continuations/recon_bank_g",
        },
        "scand_fresh_outer": {
            "freeze": V70_DIR / "replays/scand_fresh_outer/proposal_aware_v70/prelabel_freeze.json",
            "evaluation": V70_DIR / "evaluations/scand_fresh_outer/result.json",
            "trajectory_root": Path("/home/rainbow/nwm/data/scand"),
            "continuation_dir": OUTPUT_DIR / "continuations/scand_fresh_outer",
        },
    }
    cohorts: dict[str, Any] = {}
    for cohort, spec in cohort_specs.items():
        freeze = read_json(spec["freeze"])
        evaluation = read_json(spec["evaluation"])
        if freeze.get("schema") != "dualdomain_cem_tail_v70_archive_prelabel_replay":
            raise RuntimeError(f"{cohort} depth-3 replay is not v70")
        if freeze.get("config", {}).get("iterations") != 3:
            raise RuntimeError(f"{cohort} baseline depth is not 3")
        if freeze.get("reserve_accessed") or freeze.get("labels_opened"):
            raise RuntimeError(f"{cohort} depth-3 replay reports forbidden access")
        if freeze.get("policy") != expected_policies[freeze["domain"]]:
            raise RuntimeError(f"{cohort} policy differs from frozen v70")
        if not freeze.get("cross_round_safe_archive"):
            raise RuntimeError(f"{cohort} cross-round archive is disabled")
        if evaluation.get("status") != "spent_development_outcomes_evaluated_without_retuning":
            raise RuntimeError(f"{cohort} is not an already-spent diagnostic cohort")
        outcome_path = Path(evaluation["ground_truth_source"])
        if file_sha256(outcome_path) != evaluation["ground_truth_source_sha256"]:
            raise RuntimeError(f"{cohort} spent outcome source changed")
        selected_path = Path(freeze["selected_actions"])
        cohorts[cohort] = {
            "domain": freeze["domain"],
            "states": freeze["states"],
            "depth3_freeze": str(spec["freeze"].resolve()),
            "depth3_freeze_sha256": file_sha256(spec["freeze"]),
            "depth3_selected_actions": str(selected_path.resolve()),
            "depth3_selected_actions_sha256": file_sha256(selected_path),
            "spent_evaluation": str(spec["evaluation"].resolve()),
            "spent_evaluation_sha256": file_sha256(spec["evaluation"]),
            "spent_ground_truth_source": str(outcome_path.resolve()),
            "spent_ground_truth_source_sha256": file_sha256(outcome_path),
            "manifest": artifact(Path(freeze["manifest"])),
            "candidate_pool": artifact(Path(freeze["candidate_pool"])),
            "base_features": artifact(Path(freeze["base_features"])),
            "latent_cache": artifact(Path(freeze["latent_cache"])),
            "trajectory_root": str(spec["trajectory_root"].resolve()),
            "continuation_dir": str(spec["continuation_dir"].resolve()),
            "depth3_config": freeze["config"],
            "policy": freeze["policy"],
            "dual_temperature": freeze["dual_temperature"],
            "harm_temperature": freeze["harm_temperature"],
        }

    ltwm = [
        artifact(ROOT / f"checkpoints/project/ltwm_expansion/seed{seed}/best.pt")
        for seed in (1701, 1702, 1703)
    ]
    continuation_runner = ROOT / "tools/planning/run_v70_cem_depth_continuation_pilot.py"
    evaluator = ROOT / "tools/evaluation/evaluate_v70_cem_depth_pilot.py"
    prereg_builder = Path(__file__)
    value = {
        "schema": "v70_cem_depth_3_4_5_pilot_preregistration",
        "status": "frozen_before_depth4_or_depth5_candidate_evaluation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": "strict v70 CEM-depth ablation on spent pilot cohorts only",
        "only_allowed_variable": "cem_iterations",
        "compared_depths": [3, 4, 5],
        "depth3_role": "reuse existing frozen replay and outcome result bit-for-bit",
        "depth4_role": "append only iteration index 3 from reconstructed q3/archive",
        "depth5_role": "append only iteration indices 3 and 4",
        "cached_round_policy": {
            "ltwm_or_scorer_recomputation_for_iterations_0_to_2": False,
            "rng_state_reconstruction": "dry replay sampling only, verified bit-exact against cached populations",
            "candidate_evaluation_repeated_for_iterations_0_to_2": False,
        },
        "frozen_cem_contract": {
            "cem_iterations": [3, 4, 5],
            "population": 16,
            "set_cardinality_with_incumbent": 17,
            "elite_count": 4,
            "elite_fraction": 0.25,
            "proposal_scale": 0.7,
            "covariance_floor_waypoint_units": 0.015,
            "support_quantile": 0.999,
            "random_seed": 20260730,
            "candidate_level_safety_before_elite": True,
            "safe_shortfall_rule": "retain previous proposal unchanged",
            "cross_round_safe_archive": True,
            "universal_valid_contract": "physical_guard AND spacing_support<=q0.999",
            "final_selection_rule": "lowest frozen scorer energy among all safe candidates archived across rounds; incumbent on no-safe",
        },
        "frozen_model_contract": {
            "checkpoint_version": "v70_broad_hashfold_archive",
            "recon_policy": expected_policies["recon"],
            "scand_policy": expected_policies["scand"],
            "recon_temperatures": {"dual": 3.0, "harm": 3.0},
            "scand_temperatures": {"dual": 1.5, "harm": 1.5},
            "ltwm_seeds": [1701, 1702, 1703],
            "ltwm_checkpoints": ltwm,
            "model_or_threshold_changes_allowed": False,
        },
        "model_bindings": {
            "v70_result": artifact(V70_RESULT),
            "original_arm_d_result": artifact(BASE_RESULT),
            "full_deployment_checkpoints": v70["checkpoints"]["full"],
        },
        "cohorts": cohorts,
        "recorded_diagnostics": [
            "coverage dual harm",
            "absolute ate/rpe/endpoint/objective mean and p90",
            "per-round valid safe elite update fallback counts",
            "proposal variance and Gaussian entropy",
            "archive source round counts",
            "scorer preference versus true objective gain Spearman correlation",
            "safe-candidate and round-local true gain/harm",
            "scorer exploitation and benefit saturation flags",
            "end-to-end added-round latency and candidate evaluation volume",
        ],
        "pilot_gate": {
            "safety_each_domain": "dual_precision >= 0.85 and harm_rate <= 0.15",
            "recon_nondegradation": "all four absolute means and all four p90 <= depth3 + 1e-7 m",
            "coverage_trigger": "coverage(depth) - coverage(depth3) >= 0.02 in both domains",
            "stable_error_trigger": "all four mean/p90 nondegraded and aggregate mean and p90 sums strictly improve in both domains",
            "two_domain_direction_rule": "coverage trigger or stable-error trigger must hold in both domains",
            "qualify_rule": "both-domain safety AND RECON nondegradation AND two-domain direction rule",
            "tie_break": "choose shallowest qualifying depth",
            "failure_rule": "retain formal v70 depth3 and do not expand to A-H",
        },
        "source_bindings": {
            "continuation_runner": artifact(continuation_runner),
            "evaluator": artifact(evaluator),
            "preregistration_builder": artifact(prereg_builder),
        },
        "reserve_spent_audit": artifact(RESERVE_AUDIT),
        "reserve_may_be_used": False,
        "reserve_may_be_replayed": False,
        "reserve_may_select_iterations": False,
        "go2_may_be_used": False,
        "formal_v70_candidate_may_be_overwritten": False,
        "A_to_H_may_be_run_before_pilot_report": False,
        "independent_generalization_claim_allowed": False,
        "new_independent_cohort_required_for_confirmation": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_json_atomic(PREREG, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
