#!/usr/bin/env python3
"""Tune separate base-candidate and conservative CEM risk tiers."""

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

from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    evaluate_policy,
    load_combined_cohort,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.tune_recon_cem_combined_frozen_universal_v8 import (  # noqa: E402
    combined_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, nargs=3, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_twotier(
    prediction: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    is_cem = np.zeros((1, 33), dtype=bool)
    is_cem[:, 17:] = True
    z = np.where(
        is_cem,
        float(policy["cem_z"]),
        float(policy["base_z"]),
    )
    dual_threshold = np.where(
        is_cem,
        float(policy["cem_dual_threshold"]),
        float(policy["base_dual_threshold"]),
    )
    harm_threshold = np.where(
        is_cem,
        float(policy["cem_harm_threshold"]),
        float(policy["base_harm_threshold"]),
    )
    objective_margin = np.where(
        is_cem,
        float(policy["cem_objective_margin"]),
        float(policy["base_objective_margin"]),
    )
    ucb_ate = prediction["mean"][..., 0] + z * prediction["std"][..., 0]
    ucb_objective = (
        prediction["mean"][..., 3] + z * prediction["std"][..., 3]
    )
    allowed = (
        prediction["valid"].astype(bool)
        & candidate_mask
        & (ucb_ate < 0.0)
        & (ucb_objective < -objective_margin)
        & (prediction["dual_probability"] >= dual_threshold)
        & (prediction["harm_probability"] <= harm_threshold)
    )
    allowed[:, 0] = False
    rank = prediction["rank_logit"].astype(np.float64)
    rank_center = np.nanmedian(
        np.where(prediction["valid"], rank, np.nan), axis=1, keepdims=True
    )
    rank_scale = np.nanstd(
        np.where(prediction["valid"], rank, np.nan), axis=1, keepdims=True
    )
    score = (rank - rank_center) / np.maximum(rank_scale, 1.0e-3)
    score -= float(policy["cem_replacement_penalty"]) * is_cem
    score = np.where(allowed, score, -np.inf)
    challenger = score.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    return np.where(exists, challenger, 0).astype(np.int64)


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
        predictions.append(
            combined_prediction(
                base_score,
                Path(value["scores"]),
                cohort["record_key"],
            )
        )
        cohorts.append(cohort)
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

    policies: list[dict[str, Any]] = []
    for base_z in (0.5, 1.0, 1.5, 2.0):
        for base_dual in (0.40, 0.50, 0.60, 0.70, 0.80):
            for base_harm in (0.05, 0.075, 0.10, 0.125, 0.15):
                for base_margin in (0.0, 0.005, 0.010):
                    for cem_z in (1.5, 2.0, 2.5):
                        for cem_dual in (0.50, 0.60, 0.70):
                            for cem_harm in (0.025, 0.05, 0.075):
                                for cem_penalty in (0.5, 1.0, 1.5):
                                    policy = {
                                        "base_z": base_z,
                                        "base_dual_threshold": base_dual,
                                        "base_harm_threshold": base_harm,
                                        "base_objective_margin": base_margin,
                                        "cem_z": cem_z,
                                        "cem_dual_threshold": cem_dual,
                                        "cem_harm_threshold": cem_harm,
                                        "cem_objective_margin": 0.005,
                                        "cem_replacement_penalty": cem_penalty,
                                        "selection_order": (
                                            "two_tier_risk_prefilter_then_rank"
                                        ),
                                    }
                                    false_selected = select_twotier(
                                        prediction, false_mask, policy
                                    )
                                    true_selected = select_twotier(
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
        "schema": "recon_cem_combined_twotier_v11",
        "status": "spent_development_policy_search_complete",
        "evidence_role": "development_only_three_spent_cohorts",
        "cem": True,
        "states": int(len(cohort_name)),
        "online_causal": True,
        "future_route_windows_used": False,
        "candidate_type_only_conditioning": True,
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
            {"path": str(path), "sha256": file_sha256(path)}
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
