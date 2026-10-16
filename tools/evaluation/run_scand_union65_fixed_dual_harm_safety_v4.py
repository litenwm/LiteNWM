#!/usr/bin/env python3
"""Safety-tightened final prototype: v3 with harm UCB capped at 0.45."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools" / "evaluation"),
]

from planning.proposal_tail_sidecar_v1 import sha256_file  # noqa: E402
import run_scand_union65_safe_multitask_v1 as base  # noqa: E402
from run_scand_union65_safe_dual_harm_v2 import dual_harm_utility_score  # noqa: E402
from run_scand_union65_fixed_dual_harm_v3 import (  # noqa: E402
    fixed_inner_policy,
    v3_conservative_candidates,
)


BASE_RUNNER_PATH = Path(base.__file__).resolve()
V2_RUNNER_PATH = ROOT / "tools" / "evaluation" / "run_scand_union65_safe_dual_harm_v2.py"
V3_RUNNER_PATH = ROOT / "tools" / "evaluation" / "run_scand_union65_fixed_dual_harm_v3.py"
ORIGINAL_WRITE_JSON_ATOMIC = base.write_json_atomic


def provenance_write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    value["schema"] = (
        "scand_union65_fixed_dual_harm_safety_nested_route_oof_result_v4"
    )
    if value["decision"] != "smoke_only":
        value["decision"] = (
            "go_freeze_exact_v4_selector_for_one_time_sealed_validation"
            if value["prototype_go"]
            else "no_go_fixed_dual_harm_safety_v4_not_reliably_above_formal_v70"
        )
    value["adaptive_development_provenance"] = {
        "hypothesis_source": (
            "v3 passed every precommitted mean/generalization check but worsened "
            "ATE p90 by 1.85 mm; v4 changes only harm UCB limit 0.50 to 0.45"
        ),
        "inner_outcome_labels_choose_policy": False,
        "confirmatory_claim_allowed": False,
        "required_next_step_if_go": (
            "freeze exact artifacts and evaluate once on unopened sealed validation"
        ),
        "further_spent_development_search_allowed_if_no_go": False,
    }
    value["inputs"]["base_runner"] = {
        "path": str(BASE_RUNNER_PATH),
        "sha256": sha256_file(BASE_RUNNER_PATH),
    }
    value["inputs"]["v2_score_runner"] = {
        "path": str(V2_RUNNER_PATH),
        "sha256": sha256_file(V2_RUNNER_PATH),
    }
    value["inputs"]["v3_fixed_policy_runner"] = {
        "path": str(V3_RUNNER_PATH),
        "sha256": sha256_file(V3_RUNNER_PATH),
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
