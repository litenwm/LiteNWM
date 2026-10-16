#!/usr/bin/env python3
"""Tune a calibration-reference plus past-only causal CEM risk gate."""

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


def candidate_and_risk(
    prediction: dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    policy: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    valid = prediction["valid"].astype(bool) & candidate_mask
    valid[:, 0] = False
    rank = prediction["rank_logit"].astype(np.float64)
    rank_center = np.nanmedian(
        np.where(prediction["valid"], rank, np.nan), axis=1, keepdims=True
    )
    rank_scale = np.nanstd(
        np.where(prediction["valid"], rank, np.nan), axis=1, keepdims=True
    )
    rank_z = (rank - rank_center) / np.maximum(rank_scale, 1.0e-3)
    is_cem = np.zeros((1, prediction["mean"].shape[1]), dtype=np.float64)
    if is_cem.shape[1] == 33:
        is_cem[:, 17:] = 1.0
    candidate_score = (
        rank_z
        + float(policy["candidate_safety_weight"])
        * (
            prediction["dual_probability"]
            - prediction["harm_probability"]
        )
        - float(policy["cem_replacement_penalty"]) * is_cem
    )
    candidate_score = np.where(valid, candidate_score, -np.inf)
    challenger = candidate_score.argmax(axis=1).astype(np.int64)
    if not valid.any(axis=1).all():
        raise RuntimeError("causal gate has a state without valid challenger")
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
    risk = (
        np.maximum(ucb_ate, ucb_objective)
        + float(policy["harm_weight"])
        * prediction["harm_probability"][row, challenger]
        - float(policy["dual_weight"])
        * prediction["dual_probability"][row, challenger]
        - float(policy["risk_rank_weight"])
        * rank_z[row, challenger]
        + float(policy["risk_cem_penalty"]) * is_cem[0, challenger]
    )
    return challenger, risk.astype(np.float64)


def causal_gate(
    challenger: np.ndarray,
    risk: np.ndarray,
    reference_risk: np.ndarray,
    *,
    quantile: float,
    window: int,
    warmup: int,
) -> np.ndarray:
    selected = np.zeros(len(challenger), dtype=np.int64)
    history: list[float] = []
    reference_threshold = float(np.quantile(reference_risk, quantile))
    for index, value in enumerate(risk):
        if len(history) < warmup:
            threshold = reference_threshold
        else:
            threshold = float(
                np.quantile(
                    np.asarray(history[-window:], dtype=np.float64),
                    quantile,
                )
            )
        if float(value) <= threshold:
            selected[index] = int(challenger[index])
        history.append(float(value))
    return selected


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
    rows = [
        json.loads(line)
        for line in args.preflight_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    preflight_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
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
    false_mask = np.zeros((300, 33), dtype=bool)
    false_mask[:, :17] = True
    true_mask = np.ones((300, 33), dtype=bool)

    feasible: list[dict[str, Any]] = []
    searched = 0
    for candidate_safety_weight in (0.0, 1.0, 2.0):
        for cem_replacement_penalty in (0.5, 1.0):
            for z in (1.0, 2.0):
                for harm_weight in (0.25, 0.50, 1.0):
                    for dual_weight in (0.10, 0.25, 0.50):
                        for risk_rank_weight in (0.05, 0.10):
                            for risk_cem_penalty in (0.0, 0.25):
                                for quantile in (0.40, 0.45, 0.50):
                                    for window in (20, 40):
                                        searched += 1
                                        policy = {
                                            "candidate_safety_weight": (
                                                candidate_safety_weight
                                            ),
                                            "cem_replacement_penalty": (
                                                cem_replacement_penalty
                                            ),
                                            "z": z,
                                            "harm_weight": harm_weight,
                                            "dual_weight": dual_weight,
                                            "risk_rank_weight": (
                                                risk_rank_weight
                                            ),
                                            "risk_cem_penalty": (
                                                risk_cem_penalty
                                            ),
                                            "quantile": quantile,
                                            "window": window,
                                            "warmup": 10,
                                            "gate": (
                                                "calibration_reference_then_"
                                                "past_only_running_quantile"
                                            ),
                                        }
                                        false_challenger, false_risk = (
                                            candidate_and_risk(
                                                prediction,
                                                false_mask,
                                                policy,
                                            )
                                        )
                                        true_challenger, true_risk = (
                                            candidate_and_risk(
                                                prediction,
                                                true_mask,
                                                policy,
                                            )
                                        )
                                        false_selected = np.zeros(
                                            300, dtype=np.int64
                                        )
                                        true_selected = np.zeros(
                                            300, dtype=np.int64
                                        )
                                        for cohort in cohorts:
                                            mask = cohort_name == cohort["name"]
                                            false_selected[mask] = causal_gate(
                                                false_challenger[mask],
                                                false_risk[mask],
                                                false_risk,
                                                quantile=quantile,
                                                window=window,
                                                warmup=10,
                                            )
                                            true_selected[mask] = causal_gate(
                                                true_challenger[mask],
                                                true_risk[mask],
                                                true_risk,
                                                quantile=quantile,
                                                window=window,
                                                warmup=10,
                                            )
                                        evaluation = evaluate_policy(
                                            false_selected=false_selected,
                                            true_selected=true_selected,
                                            cohorts=cohorts,
                                            cohort_name=cohort_name,
                                        )
                                        if not evaluation[
                                            "strict_feasible"
                                        ]:
                                            continue
                                        pf_false_challenger, pf_false_risk = (
                                            candidate_and_risk(
                                                preflight_prediction,
                                                preflight_false_mask,
                                                policy,
                                            )
                                        )
                                        pf_true_challenger, pf_true_risk = (
                                            candidate_and_risk(
                                                preflight_prediction,
                                                preflight_true_mask,
                                                policy,
                                            )
                                        )
                                        preflight_false = causal_gate(
                                            pf_false_challenger,
                                            pf_false_risk,
                                            false_risk,
                                            quantile=quantile,
                                            window=window,
                                            warmup=10,
                                        )
                                        preflight_true = causal_gate(
                                            pf_true_challenger,
                                            pf_true_risk,
                                            true_risk,
                                            quantile=quantile,
                                            window=window,
                                            warmup=10,
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
                                                    (
                                                        preflight_true > 16
                                                    ).sum()
                                                ),
                                            }
                                        )
    if not feasible:
        raise RuntimeError(
            "no causal quantile policy passes spent safety and preflight coverage"
        )
    best = max(
        feasible,
        key=lambda value: (
            min(
                item["cem_true"]["coverage"]
                for item in value["development"]["per_cohort"].values()
            ),
            value["preflight_true_promotions"],
            value["development"]["coverage_sum"],
            value["preflight_cem_selected"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "recon_cem_causal_quantile_v13",
        "status": "causal_policy_frozen_after_label_blind_preflight",
        "evidence_role": (
            "development_only; preflight covariates spent, outcomes sealed"
        ),
        "cem": True,
        "policy": best["policy"],
        "searched_policies": searched,
        "feasible_policies": len(feasible),
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
        "online_causal": True,
        "future_route_windows_used": False,
        "reference_risk_source": (
            "three spent RECON development cohorts; score distribution only"
        ),
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
