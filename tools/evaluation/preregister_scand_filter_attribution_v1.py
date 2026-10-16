#!/usr/bin/env python3
"""Freeze the read-only SCAND candidate-filter attribution protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cem_development_v1/scand_filter_attribution_v1"
PREREG = OUTPUT / "preregistration.json"
REPLAY = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive"
    / "replays/scand_fresh_outer/proposal_aware_v70"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    if PREREG.exists():
        raise FileExistsError(PREREG)
    ground = (
        ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/outcomes"
        / "fresh_outer_outcomes.npz"
    )
    inputs = {
        "ground_truth": ground,
        "prelabel_freeze": REPLAY / "prelabel_freeze.json",
        "selected_actions": REPLAY / "selected_actions.npz",
        "base_features": (
            ROOT
            / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/features"
            / "scand_fresh_outer_h4_100_seed20260720_features.npz"
        ),
        "support_checkpoint": (
            ROOT
            / "outputs/universal_dualdomain_raw_v2/training_v2/support"
            / "dualdomain_spacing_envelopes.npz"
        ),
        "v70_result": (
            ROOT
            / "outputs/cem_development_v1"
            / "dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
        ),
        "runner_v70": ROOT / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py",
        "runner_v66": ROOT / "tools/planning/run_dualdomain_cem_tail_v66_replay.py",
        "runner_safe_elite": ROOT / "tools/planning/run_cem_safe_elite_v55.py",
        "runner_sampling": ROOT / "tools/planning/run_cem_development_v1.py",
        "universal_contract": ROOT / "src/scoring/universal_dualdomain_raw_v2.py",
    }
    for iteration in range(3):
        directory = REPLAY / f"iteration_{iteration:02d}"
        inputs[f"round_{iteration + 1}_population"] = directory / "population.npz"
        inputs[f"round_{iteration + 1}_features"] = directory / "cem_features.npz"
        inputs[f"round_{iteration + 1}_scores"] = directory / "frozen_scores.npz"
    forbidden = "res" + "erve"
    for name, path in inputs.items():
        if forbidden in str(path).lower():
            raise RuntimeError(f"forbidden path in preregistration: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)

    value = {
        "schema": "scand_candidate_filter_attribution_preregistration_v1",
        "status": "frozen_before_candidate_truth_attribution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "spent_scand_offline_filter_attribution_only",
        "scope": {
            "cohort": "scand_fresh_outer",
            "states": 100,
            "runtime_slots_per_state": 68,
            "expected_exact_unique_candidates_per_state": 65,
            "formal_v70_action_is_fallback": True,
            "target_audit_claim": {
                "states_with_any_raw_true_dual_in_full_unique_pool": 89,
                "states_with_any_true_dual_after_formal_valid_safe_contract": 22,
                "states_losing_all_dual_opportunities_before_eligible": 67,
            },
        },
        "constraints": {
            "new_training": False,
            "v76_training": False,
            "candidate_generation": False,
            "dwm_ltwm_rerun": False,
            "formal_v70_mutation": False,
            "cem_archive_mutation": False,
            "threshold_mutation": False,
            "automatic_continuation": False,
            "truth_based_deployment_selection": False,
            "reserve_access": False,
        },
        "label_contract": {
            "metrics_m": ["ate", "rpe", "endpoint", "objective"],
            "objective": "0.4*ATE + 0.6*endpoint",
            "raw_dual": "ATE and objective each improve by >1e-7 m versus formal v70",
            "raw_harm": "ATE or objective worsens by >1e-7 m versus formal v70",
            "neutral_band_m": 0.001,
            "challenger_denominator": "64 per state after excluding the formal v70 action",
        },
        "runtime_contract": {
            "analytical_union": (
                "fixed K17 plus three round K17 pools; exact-action dedup is an "
                "analysis normalization, not a deployment eligibility gate"
            ),
            "upstream_sampler": (
                "sample_population rejects physical/support-invalid draws before cache; "
                "rejected draws and their outcomes are unobservable"
            ),
            "observable_filter_order": [
                "physical_guard",
                "spacing_support_le_q0.999",
                "candidate_safety_aggregate",
            ],
            "candidate_safety_simultaneous_checks": [
                "ate_ucb_lt_-0.01m",
                "objective_ucb_lt_-0.01m",
                "dual_probability_ge_0.5",
                "harm_probability_le_0.1",
            ],
            "safety_component_diagnostic_precedence_only": [
                "ate_ucb",
                "objective_ucb",
                "dual_probability",
                "harm_probability",
            ],
            "no_boolean_cross_seed_consensus_gate": True,
            "safe_count_ge_4": (
                "controls q update only; it does not prevent a safe candidate entering "
                "the cross-round archive"
            ),
            "archive": (
                "lowest frozen energy safe candidate updates archive only if energy is "
                "lower than the prior archive energy"
            ),
        },
        "waterfalls": {
            "candidate_first_reject": ["physical", "support", "safety", "eligible"],
            "state_first_loss": (
                "first stage at which a state that had raw dual candidates has none left"
            ),
            "overlap": (
                "report every simultaneously failed physical/support/safety component; "
                "safety subcomponent precedence is attribution-only"
            ),
        },
        "counterfactuals": {
            "cumulative_stages": [
                "full_unique_pool",
                "physical",
                "physical_support",
                "physical_support_safety",
            ],
            "single_filter_restores": [
                "restore_physical_diagnostic_only",
                "restore_support",
                "restore_safety_aggregate",
                "restore_ate_ucb_only",
                "restore_objective_ucb_only",
                "restore_dual_probability_only",
                "restore_harm_probability_only",
            ],
            "oracles": [
                "unconstrained_min_true_objective_with_formal_fallback",
                "true_dual_only_min_objective_with_formal_fallback",
            ],
            "no_new_proposals": True,
        },
        "stratification": {
            "keys": [
                "route",
                "source_fixed_r1_r2_r3",
                "source_candidate_index",
                "frozen_energy_rank_within_source_pool",
                "support_relative_margin",
                "ate_seed_std_range",
                "rtp_seed_std_range",
                "support_seed_vote_count",
                "action_linear_magnitude",
                "action_yaw_magnitude",
                "waypoint_turn_angle",
            ],
            "fixed_support_relative_margin_bins": [
                "fail:<0",
                "near:[0,0.02]",
                "mid:(0.02,0.10]",
                "far:>0.10",
            ],
            "route_dominance": "report top routes and maximum share of lost states",
        },
        "filter_value_reporting": {
            "dual_recall": True,
            "harm_rejection_rate": True,
            "pass_precision_among_dual_or_harm": True,
            "dual_rejected": True,
            "harm_rejected": True,
            "hard_physical_separate_from_statistical_filters": True,
        },
        "observability_rules": {
            "missing_history_is_unobservable": True,
            "no_invented_thresholds_or_margins": True,
            "sampling_reject_truth": "unobservable",
            "new_proposals_after_counterfactual_q_update": "unobservable_without_forbidden_rerun",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(inputs.items())
        },
        "terminal_rule": "write result/audit, report whether a next step is justified, then stop",
    }
    write_json_atomic(PREREG, value)
    print(json.dumps({"path": str(PREREG), "sha256": sha256_file(PREREG)}))


if __name__ == "__main__":
    main()
