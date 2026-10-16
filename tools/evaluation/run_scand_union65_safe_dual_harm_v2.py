#!/usr/bin/env python3
"""Adaptive v2: rank union65 actions by conservative predicted dual-minus-harm.

Training, nested route splitting, inner-only policy selection, artifacts, and
go/no-go checks are inherited from the frozen v1 runner.  This wrapper changes
only the candidate score that the v1 OOF diagnosis identified as promising.
It deliberately records that provenance and remains a spent-development
prototype until a separately frozen, one-time validation succeeds.
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


BASE_RUNNER_PATH = Path(base.__file__).resolve()
ORIGINAL_WRITE_JSON_ATOMIC = base.write_json_atomic


def conservative_dual_harm_candidates(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    beta: float,
    harm_limit: float,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the highest conservative dual-minus-harm challenger per state."""
    requirements = contract["policy_grid"]["candidate_requirements"]
    mask = base.choice_mask(data, states)
    formal = data["formal_v70_index"][states].astype(np.int64)
    harm_ucb = np.clip(mean[..., 2] + beta * std[..., 2], 0.0, 1.0)
    score = (
        mean[..., 3]
        - mean[..., 2]
        - beta * (std[..., 3] + std[..., 2])
    )
    challenger_mask = (
        mask
        & (harm_ucb <= harm_limit)
        & (
            score
            > float(requirements["conservative_dual_minus_harm_advantage"])
        )
    )
    challenger_mask[np.arange(len(states)), formal] = False
    masked = np.where(challenger_mask, score, -np.inf)
    candidate = np.argmax(masked, axis=1).astype(np.int64)
    advantage = masked[np.arange(len(states)), candidate].astype(np.float32)
    eligible = challenger_mask.any(axis=1)
    candidate[~eligible] = formal[~eligible]
    advantage[~eligible] = -np.inf
    return candidate, advantage, eligible


def dual_harm_utility_score(
    mean: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    del contract
    return (mean[..., 3] - mean[..., 2]).astype(np.float32)


def provenance_write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    value["schema"] = "scand_union65_safe_dual_harm_nested_route_oof_result_v2"
    if value["decision"] != "smoke_only":
        value["decision"] = (
            "go_freeze_dual_harm_selector_for_one_time_sealed_validation"
            if value["prototype_go"]
            else "no_go_dual_harm_selector_not_reliably_above_formal_v70"
        )
    value["adaptive_development_provenance"] = {
        "hypothesis_source": (
            "dual-minus-harm score selected after inspecting v1 spent-development "
            "outer-OOF diagnostics"
        ),
        "confirmatory_claim_allowed": False,
        "required_next_step_if_go": (
            "freeze exact artifacts and evaluate once on unopened sealed validation"
        ),
    }
    value["inputs"]["base_runner"] = {
        "path": str(BASE_RUNNER_PATH),
        "sha256": sha256_file(BASE_RUNNER_PATH),
    }
    ORIGINAL_WRITE_JSON_ATOMIC(path, value)


def main() -> None:
    # The imported main resolves these names dynamically in its own module.
    base.conservative_candidates = conservative_dual_harm_candidates
    base.utility_score = dual_harm_utility_score
    base.write_json_atomic = provenance_write_json_atomic
    # Ensure the recorded primary runner is this v2 wrapper, not the base module.
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
