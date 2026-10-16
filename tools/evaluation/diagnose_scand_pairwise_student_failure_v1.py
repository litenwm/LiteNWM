#!/usr/bin/env python3
"""Read-only failure diagnosis for the spent pairwise student OOF result."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


BASE = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1"
TABLE = BASE / "spent_runtime_pair_table.npz"
PRED = BASE / "pairwise_student_v1/outer_oof_predictions.npz"
RESULT = BASE / "pairwise_student_v1/result.json"
OUTPUT = BASE / "pairwise_student_v1/failure_diagnostic.json"


def auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    positive = int(label.sum())
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    cursor = 0
    while cursor < len(score):
        end = cursor + 1
        while end < len(score) and sorted_score[end] == sorted_score[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def ece(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        local = (probability >= edges[index]) & (probability < edges[index + 1] if index + 1 < bins else probability <= edges[index + 1])
        if local.any():
            total += local.mean() * abs(float(probability[local].mean() - label[local].mean()))
    return total


def metrics(mask: np.ndarray, pred: np.ndarray, table: dict[str, np.ndarray]) -> dict:
    dual = table["dual_vs_formal_v70"][mask]
    harm = table["harm_vs_formal_v70"][mask]
    helpful = table["helpful_vs_formal_v70"][mask]
    gain = table["objective_gain_vs_formal_v70_m"][mask]
    phelp, pdual, pharm = pred[mask, 2], pred[mask, 3], pred[mask, 4]
    pgain = pred[mask, 0]
    return {
        "rows": int(mask.sum()),
        "dual_rate": float(dual.mean()),
        "harm_rate": float(harm.mean()),
        "dual_auroc": auc(pdual, dual),
        "harm_auroc": auc(pharm, harm),
        "helpful_auroc": auc(phelp, helpful),
        "dual_brier": float(np.mean((pdual - dual) ** 2)),
        "harm_brier": float(np.mean((pharm - harm) ** 2)),
        "dual_ece10": ece(pdual, dual),
        "harm_ece10": ece(pharm, harm),
        "objective_gain_pearson": float(np.corrcoef(pgain, gain)[0, 1]),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with np.load(TABLE, allow_pickle=False) as payload:
        table = {key: payload[key].copy() for key in payload.files}
    with np.load(PRED, allow_pickle=False) as payload:
        predfile = {key: payload[key].copy() for key in payload.files}
    pred = predfile["predictions"].astype(np.float32)
    formal = table["formal_v70_index"].astype(np.int64)
    pool = table["physical_support_pool"].astype(bool).copy()
    pool[np.arange(len(pool)), formal] = False
    if not np.array_equal(table["record_key"].astype(str), predfile["record_key"].astype(str)):
        raise RuntimeError("record mismatch")
    by_cohort = {}
    for name in np.unique(table["cohort"].astype(str)):
        state = table["cohort"].astype(str) == name
        by_cohort[name] = metrics(pool & state[:, None], pred, table)
    by_fold = {}
    for fold in range(5):
        state = predfile["outer_fold"] == fold
        by_fold[str(fold)] = metrics(pool & state[:, None], pred, table)
    # Pure diagnostic top-one policy without acceptance; never used for upgrade.
    selected = formal.copy()
    for state in range(len(formal)):
        candidates = np.flatnonzero(pool[state])
        if len(candidates):
            selected[state] = int(candidates[np.argmax(pred[state, candidates, 0])])
    row = np.arange(len(formal))
    top_dual = table["dual_vs_formal_v70"][row, selected]
    top_harm = table["harm_vs_formal_v70"][row, selected]
    top_gain = table["objective_gain_vs_formal_v70_m"][row, selected]
    # Top-5 dual opportunity recall by predicted risk/gain composite.
    dual_state = 0
    recalled = 0
    score = pred[..., 0] + 0.02 * pred[..., 3] - 0.02 * pred[..., 4]
    for state in range(len(formal)):
        candidates = np.flatnonzero(pool[state])
        if not table["dual_vs_formal_v70"][state, candidates].any():
            continue
        dual_state += 1
        top = candidates[np.argsort(score[state, candidates], kind="stable")[-5:]]
        recalled += int(table["dual_vs_formal_v70"][state, top].any())
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    diagnostic = {
        "schema": "scand_pairwise_student_failure_diagnostic_v1",
        "status": "complete_posthoc_diagnostic_not_for_threshold_or_upgrade",
        "overall_candidate_layer": metrics(pool, pred, table),
        "by_cohort": by_cohort,
        "by_outer_fold": by_fold,
        "ungated_predicted_gain_top1": {
            "states": len(formal),
            "dual_empirical": float(top_dual.mean()),
            "harm_empirical": float(top_harm.mean()),
            "all_state_mean_objective_gain_m": float(top_gain.mean()),
        },
        "top5_dual_state_recall": recalled / dual_state,
        "dual_opportunity_states": dual_state,
        "inner_profile_ranges": {
            profile: {
                "replacement_min": min(result["inner_outer_audit"][outer]["profiles"][profile]["replacements"] for outer in result["inner_outer_audit"]),
                "replacement_max": max(result["inner_outer_audit"][outer]["profiles"][profile]["replacements"] for outer in result["inner_outer_audit"]),
                "dual_empirical_min": min(result["inner_outer_audit"][outer]["profiles"][profile]["dual_empirical"] for outer in result["inner_outer_audit"]),
                "dual_empirical_max": max(result["inner_outer_audit"][outer]["profiles"][profile]["dual_empirical"] for outer in result["inner_outer_audit"]),
                "harm_empirical_min": min(result["inner_outer_audit"][outer]["profiles"][profile]["harm_empirical"] for outer in result["inner_outer_audit"]),
                "harm_empirical_max": max(result["inner_outer_audit"][outer]["profiles"][profile]["harm_empirical"] for outer in result["inner_outer_audit"]),
            }
            for profile in ("strict_090_010", "balanced_085_015", "harm_tight_080_010")
        },
        "causal_interpretation": {
            "primary": "risk-tail probability is not route-stable; stricter predicted probability does not consistently enrich dual-safe replacements",
            "secondary": "predicted objective gain retains useful average ranking signal, but benefit and safety are decoupled across held routes",
            "not_a_threshold_mismatch_claim": True,
            "current_model_or_outer_threshold_further_tuning_authorized": False,
            "next_material_representation_to_audit": "cached static visual/state context omitted by v72-v77 and the 351-feature student",
        },
        "inputs": {"table_sha256": sha256_file(TABLE), "predictions_sha256": sha256_file(PRED), "result_sha256": sha256_file(RESULT)},
        "sealed_validation_opened": False,
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.write_text(canonical_json(diagnostic) + "\n", encoding="utf-8")
    print(canonical_json({"status": diagnostic["status"], "output": str(OUTPUT), "sha256": sha256_file(OUTPUT), "overall": diagnostic["overall_candidate_layer"], "top1": diagnostic["ungated_predicted_gain_top1"], "top5_dual_recall": diagnostic["top5_dual_state_recall"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(canonical_json({"status": "fail_closed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
