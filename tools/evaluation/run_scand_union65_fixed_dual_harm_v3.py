#!/usr/bin/env python3
"""Final adaptive prototype with one fixed conservative dual-minus-harm policy.

Unlike v2, this runner performs no outcome-label-based policy search inside
each outer fold.  Inner route-OOF predictions are used only to calibrate the
precommitted 40% score-distribution threshold.  The score, uncertainty beta,
harm limit, target coverage, model, and seeds are all fixed in the contract.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools" / "evaluation"),
]

from planning.proposal_tail_sidecar_v1 import sha256_file  # noqa: E402
import run_scand_union65_safe_multitask_v1 as base  # noqa: E402
from run_scand_union65_safe_dual_harm_v2 import (  # noqa: E402
    conservative_dual_harm_candidates,
    dual_harm_utility_score,
)


BASE_RUNNER_PATH = Path(base.__file__).resolve()
V2_RUNNER_PATH = ROOT / "tools" / "evaluation" / "run_scand_union65_safe_dual_harm_v2.py"
ORIGINAL_WRITE_JSON_ATOMIC = base.write_json_atomic


def fixed_inner_policy(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    contract: dict[str, Any],
    smoke: bool,
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    del smoke, outer_fold
    fixed = contract["fixed_policy"]
    beta = float(fixed["uncertainty_beta"])
    harm_limit = float(fixed["harm_probability_limit"])
    coverage = float(fixed["target_inner_oof_coverage"])
    _, advantage, _ = conservative_dual_harm_candidates(
        mean,
        std,
        data,
        states,
        beta,
        harm_limit,
        {
            **contract,
            "policy_grid": {
                "candidate_requirements": fixed["candidate_requirements"]
            },
        },
    )
    threshold = base.threshold_for_coverage(advantage, coverage)
    policy = {
        "fallback": False,
        "uncertainty_beta": beta,
        "harm_probability_limit": harm_limit,
        "target_inner_coverage": coverage,
        "advantage_threshold_m": threshold,
        "reason": "fixed_policy_inner_oof_score_quantile_only_no_outcome_label_search",
    }
    # apply_policy calls the patched candidate function and expects policy_grid.
    compatible_contract = {
        **contract,
        "policy_grid": {
            "candidate_requirements": fixed["candidate_requirements"]
        },
    }
    selected = base.apply_policy(
        mean, std, data, states, policy, compatible_contract
    )
    stats = base.augmented_stats(data, states, selected)
    audit = {
        **policy,
        "outcome_labels_used_to_choose_policy": False,
        "replacements": stats["replacements"],
        "coverage": stats["coverage"],
        "objective_gain_m_audit_only": stats["objective_gain_m"],
        "ate_gain_m_audit_only": stats["ate_gain_m"],
        "harm_rate_audit_only": stats["harm_rate"],
        "positive_objective_route_fraction_audit_only": stats[
            "positive_route_fraction"
        ],
        "positive_ate_route_fraction_audit_only": stats[
            "positive_ate_route_fraction"
        ],
    }
    return policy, stats, [audit]


def v3_conservative_candidates(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    beta: float,
    harm_limit: float,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed = contract["fixed_policy"]
    compatible_contract = {
        **contract,
        "policy_grid": {
            "candidate_requirements": fixed["candidate_requirements"]
        },
    }
    return conservative_dual_harm_candidates(
        mean,
        std,
        data,
        states,
        beta,
        harm_limit,
        compatible_contract,
    )


def provenance_write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    value["schema"] = "scand_union65_fixed_dual_harm_nested_route_oof_result_v3"
    if value["decision"] != "smoke_only":
        value["decision"] = (
            "go_freeze_exact_v3_selector_for_one_time_sealed_validation"
            if value["prototype_go"]
            else "no_go_fixed_dual_harm_v3_not_reliably_above_formal_v70"
        )
    value["adaptive_development_provenance"] = {
        "hypothesis_source": (
            "fixed beta/coverage policy selected after v2 showed fold-specific "
            "policy search worsened ATE p90"
        ),
        "inner_outcome_labels_choose_policy": False,
        "confirmatory_claim_allowed": False,
        "required_next_step_if_go": (
            "freeze exact artifacts and evaluate once on unopened sealed validation"
        ),
    }
    value["inputs"]["base_runner"] = {
        "path": str(BASE_RUNNER_PATH),
        "sha256": sha256_file(BASE_RUNNER_PATH),
    }
    value["inputs"]["v2_score_runner"] = {
        "path": str(V2_RUNNER_PATH),
        "sha256": sha256_file(V2_RUNNER_PATH),
    }
    ORIGINAL_WRITE_JSON_ATOMIC(path, value)


def main() -> None:
    base.conservative_candidates = v3_conservative_candidates
    base.utility_score = dual_harm_utility_score
    base.choose_inner_policy = fixed_inner_policy
    base.write_json_atomic = provenance_write_json_atomic
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
