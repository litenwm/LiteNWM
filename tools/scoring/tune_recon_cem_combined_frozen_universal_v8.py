#!/usr/bin/env python3
"""Tune a spent-development gate over frozen universal-D base+CEM scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.planning.run_cem_development_v1 import domain_policy  # noqa: E402
from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    evaluate_policy,
    load_combined_cohort,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, nargs=3, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_score(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {
            key: payload[key]
            for key in (
                "mean",
                "std",
                "dual_probability",
                "harm_probability",
                "rank_logit",
                "valid",
            )
        }
        if "record_key" in payload.files:
            result["record_key"] = payload["record_key"]
        return result


def combined_prediction(
    base_path: Path,
    final_path: Path,
    expected_record: np.ndarray,
) -> dict[str, np.ndarray]:
    base = load_score(base_path)
    final = load_score(final_path)
    if not np.array_equal(
        base["record_key"].astype(str), expected_record
    ):
        raise RuntimeError("combined frozen score identity mismatch")
    result: dict[str, np.ndarray] = {}
    for key in (
        "mean",
        "std",
        "dual_probability",
        "harm_probability",
        "rank_logit",
        "valid",
    ):
        result[key] = np.concatenate(
            (base[key], final[key][:, 1:]), axis=1
        )
    return result


def select(
    prediction: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    allowed = prediction["valid"].astype(bool) & candidate_mask
    allowed[:, 0] = False
    rank = np.where(allowed, prediction["rank_logit"], -np.inf)
    challenger = rank.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0)
    row = np.arange(len(challenger))
    z = float(policy["z"])
    ucb_ate = (
        prediction["mean"][row, challenger, 0]
        + z * prediction["std"][row, challenger, 0]
    )
    ucb_objective = (
        prediction["mean"][row, challenger, 3]
        + z * prediction["std"][row, challenger, 3]
    )
    promote = (
        exists
        & (ucb_ate < -float(policy["margin_ate"]))
        & (
            ucb_objective
            < -float(policy["margin_objective"])
        )
        & (
            prediction["dual_probability"][row, challenger]
            >= float(policy["dual_threshold"])
        )
        & (
            prediction["harm_probability"][row, challenger]
            <= float(policy["harm_threshold"])
        )
    )
    return np.where(promote, challenger, 0).astype(np.int64)


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    cohorts = []
    predictions = []
    for value, base_score in zip(v4["inputs"], args.base_scores):
        cohort = load_combined_cohort(
            name=value["name"],
            final_feature_path=Path(value["features"]),
            final_score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        prediction = combined_prediction(
            base_score,
            Path(value["scores"]),
            cohort["record_key"],
        )
        cohorts.append(cohort)
        predictions.append(prediction)
    prediction = {
        key: np.concatenate([value[key] for value in predictions])
        for key in predictions[0]
    }
    cohort_name = np.concatenate(
        [
            np.full(len(value["route"]), value["name"], dtype="<U32")
            for value in cohorts
        ]
    )
    false_mask = np.zeros((len(cohort_name), 33), dtype=bool)
    false_mask[:, :17] = True
    true_mask = np.ones((len(cohort_name), 33), dtype=bool)

    training = json.loads(
        args.training_result.read_text(encoding="utf-8")
    )
    frozen_policy, _, _ = domain_policy(training, "recon")
    base_only_frozen = select(prediction, false_mask, frozen_policy)
    expected_base = np.concatenate(
        [value["baseline"] for value in cohorts]
    )
    if not np.array_equal(base_only_frozen, expected_base):
        mismatch = int((base_only_frozen != expected_base).sum())
        raise RuntimeError(
            f"recomputed frozen base selection mismatch: {mismatch}"
        )

    policies: list[dict[str, Any]] = []
    for z in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        for margin_ate in (0.0, 0.005, 0.010, 0.020, 0.030):
            for margin_objective in (
                0.0,
                0.005,
                0.010,
                0.020,
                0.030,
            ):
                for dual_threshold in np.linspace(0.50, 0.95, 10):
                    for harm_threshold in (
                        0.05,
                        0.10,
                        0.15,
                        0.20,
                        0.30,
                        0.40,
                    ):
                        policy = {
                            "z": z,
                            "margin_ate": margin_ate,
                            "margin_objective": margin_objective,
                            "dual_threshold": float(dual_threshold),
                            "harm_threshold": harm_threshold,
                        }
                        false_selected = select(
                            prediction, false_mask, policy
                        )
                        true_selected = select(
                            prediction, true_mask, policy
                        )
                        evaluation = evaluate_policy(
                            false_selected=false_selected,
                            true_selected=true_selected,
                            cohorts=cohorts,
                            cohort_name=cohort_name,
                        )
                        policies.append(
                            {
                                "policy": policy,
                                "false_selected": false_selected,
                                "true_selected": true_selected,
                                **evaluation,
                            }
                        )
    feasible = [value for value in policies if value["strict_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda value: (
                value["coverage_sum"],
                value["cem_selected_total"],
                -value["penalty"],
            ),
        )
        tier = "strict_all_three_spent_cohorts"
    else:
        best = min(
            policies,
            key=lambda value: (
                value["penalty"],
                -value["coverage_sum"],
                -value["cem_selected_total"],
            ),
        )
        tier = "best_available_spent_development"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = args.output_dir / "selection.npz"
    write_npz(
        selection_path,
        record_key=np.concatenate(
            [value["record_key"] for value in cohorts]
        ),
        cohort=cohort_name,
        cem_false_selected_index=best["false_selected"],
        cem_true_selected_index=best["true_selected"],
    )
    report = {
        "schema": "recon_cem_combined_frozen_universal_v8",
        "status": "spent_development_policy_search_complete",
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": int(len(cohort_name)),
        "base_frozen_selection_exactly_reproduced": True,
        "frozen_policy": frozen_policy,
        "policy_search": {
            "tier": tier,
            "strict_feasible_policies": len(feasible),
            "searched_policies": len(policies),
            "policy": best["policy"],
            "per_cohort": best["per_cohort"],
            "strict_safety_all_cohorts": best[
                "strict_safety_all_cohorts"
            ],
            "quality_all_cohorts": best["quality_all_cohorts"],
            "strict_feasible": best["strict_feasible"],
            "penalty": best["penalty"],
            "coverage_sum": best["coverage_sum"],
            "cem_selected_total": best["cem_selected_total"],
        },
        "selection": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "base_scores": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in args.base_scores
        ],
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": file_sha256(result_path),
                "policy_search": report["policy_search"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
