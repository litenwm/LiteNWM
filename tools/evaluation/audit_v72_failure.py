#!/usr/bin/env python3
"""Read-only failure attribution for the completed v72 OOF result."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


def binary_auc(score: np.ndarray, target: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    positive = int(target.sum())
    negative = int((~target).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score), dtype=np.float64)
    rank[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((rank[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def local_metrics(data, prediction, state_mask):
    candidate_mask = state_mask[:, None] & data["valid"]
    row = np.arange(len(data["record_key"]))
    formal_score = prediction["score"][row, data["formal_index"]]
    predicted_safe = (
        candidate_mask
        & (prediction["p_dual"] >= 0.5)
        & (prediction["p_harm"] <= 0.5)
        & (prediction["score"] < formal_score[:, None])
    )
    predicted_safe[row, data["formal_index"]] = False
    dual = data["dual_vs_formal"].astype(bool)
    harm = data["harm_vs_formal"].astype(bool)
    count = int(predicted_safe.sum())
    true_dual = int((predicted_safe & dual).sum())
    true_harm = int((predicted_safe & harm).sum())
    return {
        "valid_candidate_rows": int(candidate_mask.sum()),
        "predicted_safe_candidate_rows": count,
        "predicted_safe_fraction": float(count / max(int(candidate_mask.sum()), 1)),
        "predicted_safe_dual_precision": float(true_dual / count) if count else None,
        "predicted_safe_harm_rate": float(true_harm / count) if count else None,
        "dual_auc": binary_auc(
            prediction["p_dual"][candidate_mask], dual[candidate_mask]
        ),
        "harm_auc": binary_auc(
            prediction["p_harm"][candidate_mask], harm[candidate_mask]
        ),
        "dual_brier": float(np.mean((prediction["p_dual"][candidate_mask] - dual[candidate_mask]) ** 2)),
        "harm_brier": float(np.mean((prediction["p_harm"][candidate_mask] - harm[candidate_mask]) ** 2)),
        "true_dual_prevalence": float(dual[candidate_mask].mean()),
        "true_harm_prevalence": float(harm[candidate_mask].mean()),
    }


def main() -> None:
    data, _ = v72.load_data()
    prediction_file = v72.load_npz(v72.OUTPUT / "oof_predictions.npz")
    result = v72.read_json(v72.OUTPUT / "result.json")
    analysis = {
        "schema": "v72_route_oof_failure_attribution_v1",
        "result_sha256": v72.sha256_file(v72.OUTPUT / "result.json"),
        "models": {},
        "thresholds_changed": False,
        "model_retrained": False,
        "reserve_accessed": False,
    }
    for name in v72.MODEL_NAMES:
        prediction = {
            key: prediction_file[f"{name}__{key}"]
            for key in ("score", "p_dual", "p_harm")
        }
        unconstrained = v72.select_unconstrained(prediction["score"], data["valid"])
        local = {
            "unconstrained_rank_selection": v72.selection_summaries(data, unconstrained),
            "safety": {},
            "ranking_by_domain": {},
        }
        for domain, key in ((None, "pooled"), ("recon", "recon_a_to_h"), ("scand", "fresh_scand")):
            state_mask = np.ones(len(data["record_key"]), dtype=bool) if domain is None else data["domain"].astype(str) == domain
            local["safety"][key] = local_metrics(data, prediction, state_mask)
            local["ranking_by_domain"][key] = v72.ranking_metrics(
                prediction["score"][state_mask],
                data["true_metrics"][state_mask, :, 3],
                data["valid"][state_mask],
            )
        analysis["models"][name] = local
    analysis["attribution"] = {
        "feature_unlearnable": False,
        "route_oof_domain_shift": True,
        "safety_probability_gate_overpermissive": True,
        "evidence": [
            "Pointwise ridge improves RECON but its unconstrained selection worsens fresh SCAND.",
            "All fixed-threshold safety policies replace far more formal-v70 actions than their held-out dual precision supports.",
            "The MLP raises true-top1 recall@5 but does not make top1 ranking or dual/harm probabilities accurate enough for safe deployment.",
        ],
        "next_research_if_new_authority_is_granted": "training-fold-only domain-conditional probability calibration or conformal lower bounds, with the architecture and v70 proposal cache fixed",
    }
    v72.write_json(v72.OUTPUT / "failure_analysis.json", analysis)
    print(v72.OUTPUT / "failure_analysis.json")


if __name__ == "__main__":
    main()
