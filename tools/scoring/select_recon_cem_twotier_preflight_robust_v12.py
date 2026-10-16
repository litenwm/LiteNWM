#!/usr/bin/env python3
"""Select a spent-safe v11 policy robust to a label-blind preflight shift."""

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
)
from tools.scoring.tune_recon_cem_combined_frozen_universal_v8 import (  # noqa: E402
    combined_prediction,
)
from tools.scoring.tune_recon_cem_combined_twotier_v11 import (  # noqa: E402
    select_twotier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, nargs=3, required=True)
    parser.add_argument("--preflight-base-scores", type=Path, required=True)
    parser.add_argument("--preflight-cem-scores", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--preflight-freeze", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    preflight_freeze = json.loads(
        args.preflight_freeze.read_text(encoding="utf-8")
    )
    if (
        preflight_freeze.get("outcomes_accessed")
        or preflight_freeze.get("labels_opened")
    ):
        raise RuntimeError("preflight labels were already opened")
    preflight_rows = [
        json.loads(line)
        for line in args.preflight_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    preflight_record = np.asarray(
        [
            f"{row['trajectory_id']}:{int(row['current_t'])}"
            for row in preflight_rows
        ]
    )
    preflight_prediction = combined_prediction(
        args.preflight_base_scores,
        args.preflight_cem_scores,
        preflight_record,
    )
    preflight_false_mask = np.zeros((100, 33), dtype=bool)
    preflight_false_mask[:, :17] = True
    preflight_true_mask = np.ones((100, 33), dtype=bool)

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

    feasible: list[dict[str, Any]] = []
    searched = 0
    for base_z in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        for base_dual in (0.20, 0.30, 0.40, 0.50):
            for base_harm in (0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
                for base_margin in (0.0,):
                    for cem_z in (1.0, 1.5, 2.0):
                        for cem_dual in (0.40, 0.50, 0.60):
                            for cem_harm in (0.05, 0.075, 0.10, 0.15):
                                for cem_penalty in (0.5, 1.0, 1.5, 2.0):
                                    searched += 1
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
                                    if not evaluation["strict_feasible"]:
                                        continue
                                    preflight_false = select_twotier(
                                        preflight_prediction,
                                        preflight_false_mask,
                                        policy,
                                    )
                                    preflight_true = select_twotier(
                                        preflight_prediction,
                                        preflight_true_mask,
                                        policy,
                                    )
                                    false_count = int(
                                        (preflight_false != 0).sum()
                                    )
                                    true_count = int(
                                        (preflight_true != 0).sum()
                                    )
                                    if (
                                        true_count < 40
                                        or true_count < false_count
                                        or not (preflight_true > 16).any()
                                    ):
                                        continue
                                    spent_coverages = [
                                        item["cem_true"]["coverage"]
                                        for item in evaluation[
                                            "per_cohort"
                                        ].values()
                                    ]
                                    feasible.append(
                                        {
                                            "policy": policy,
                                            "development": evaluation,
                                            "preflight_false_promotions": (
                                                false_count
                                            ),
                                            "preflight_true_promotions": (
                                                true_count
                                            ),
                                            "preflight_cem_selected": int(
                                                (preflight_true > 16).sum()
                                            ),
                                            "minimum_coverage_across_spent_and_preflight": min(
                                                *spent_coverages,
                                                true_count / 100.0,
                                            ),
                                        }
                                    )
    if not feasible:
        raise RuntimeError(
            "no spent-safe policy reaches label-blind preflight coverage"
        )
    best = max(
        feasible,
        key=lambda value: (
            value["minimum_coverage_across_spent_and_preflight"],
            value["preflight_true_promotions"],
            value["development"]["coverage_sum"],
            value["preflight_cem_selected"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "recon_cem_twotier_preflight_robust_v12",
        "status": "policy_frozen_after_label_blind_coverage_stress_test",
        "evidence_role": (
            "development_only; preflight covariates spent, outcomes sealed"
        ),
        "cem": True,
        "policy": best["policy"],
        "searched_policies": searched,
        "spent_strict_feasible_policies_with_preflight_coverage": len(
            feasible
        ),
        "development": best["development"],
        "preflight": {
            "manifest": str(args.preflight_manifest),
            "manifest_sha256": file_sha256(args.preflight_manifest),
            "freeze": str(args.preflight_freeze),
            "freeze_sha256": file_sha256(args.preflight_freeze),
            "cem_false_promotions": best[
                "preflight_false_promotions"
            ],
            "cem_true_promotions": best[
                "preflight_true_promotions"
            ],
            "cem_selected": best["preflight_cem_selected"],
            "outcomes_accessed": False,
            "labels_opened": False,
            "status": "label_blind_preflight_spent_do_not_open",
        },
        "minimum_coverage_across_spent_and_preflight": best[
            "minimum_coverage_across_spent_and_preflight"
        ],
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
