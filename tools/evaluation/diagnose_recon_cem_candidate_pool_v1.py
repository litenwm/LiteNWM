#!/usr/bin/env python3
"""Diagnose whether RECON CEM failures come from proposal or ranking quality."""

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

from tools.scoring.train_recon_cem_candidate_calibrator_v4 import (  # noqa: E402
    absolute_summary,
    file_sha256,
    load_cohort,
    safety,
    sigmoid,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rank_candidates(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    rank_score = -score if higher_is_better else score
    rank_score = np.where(valid, rank_score, np.inf)
    selected = rank_score.argmin(axis=1)
    exists = np.isfinite(
        rank_score[np.arange(len(rank_score)), selected]
    )
    return np.where(exists, selected + 1, 0).astype(np.int64)


def cohort_diagnostics(
    *,
    cohort: dict[str, Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    delta = cohort["delta"]
    valid = cohort["valid"]
    dual = cohort["dual"].astype(bool)
    harm = cohort["harm"].astype(bool)
    true_utility = (
        delta[..., 0]
        + 0.25 * delta[..., 1]
        + 0.25 * np.maximum(delta[..., 2], 0.0)
        + 0.50 * delta[..., 3]
    )
    learned_mean = prediction["mean"]
    learned_utility = (
        learned_mean[..., 0]
        + 0.25 * learned_mean[..., 1]
        + 0.25 * np.maximum(learned_mean[..., 2], 0.0)
        + 0.50 * learned_mean[..., 3]
    )
    learned_dual = sigmoid(prediction["dual_logit"])
    learned_harm = sigmoid(prediction["harm_logit"])
    methods = {
        "learned_utility": rank_candidates(
            learned_utility, valid, higher_is_better=False
        ),
        "learned_pdual": rank_candidates(
            learned_dual, valid, higher_is_better=True
        ),
        "learned_pharm": rank_candidates(
            learned_harm, valid, higher_is_better=False
        ),
        "learned_dual_minus_harm": rank_candidates(
            learned_dual - learned_harm,
            valid,
            higher_is_better=True,
        ),
        "oracle_utility": rank_candidates(
            true_utility, valid, higher_is_better=False
        ),
        "oracle_dual_then_utility": rank_candidates(
            np.where(dual, true_utility, np.inf),
            valid,
            higher_is_better=False,
        ),
    }
    method_metrics = {}
    for name, selected in methods.items():
        method_metrics[name] = {
            "safety": safety(selected, delta),
            "absolute": absolute_summary(
                selected, cohort["absolute_final"]
            ),
        }
    return {
        "states": int(len(delta)),
        "valid_candidate_fraction": float(valid.mean()),
        "states_with_any_valid": float(valid.any(axis=1).mean()),
        "states_with_any_dual_valid": float(
            (valid & dual).any(axis=1).mean()
        ),
        "states_with_any_harm_free_valid": float(
            (valid & ~harm).any(axis=1).mean()
        ),
        "candidate_dual_rate_within_valid": float(
            dual[valid].mean()
        ),
        "candidate_harm_rate_within_valid": float(
            harm[valid].mean()
        ),
        "ranking_methods_no_gate": method_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v4 = json.loads(args.v4_result.read_text(encoding="utf-8"))
    cohorts = [
        load_cohort(
            name=value["name"],
            feature_path=Path(value["features"]),
            score_path=Path(value["scores"]),
            result_path=Path(value["result"]),
        )
        for value in v4["inputs"]
    ]
    with np.load(v4["oof_predictions"], allow_pickle=False) as payload:
        all_prediction = {
            "mean": payload["mean"].astype(np.float32),
            "dual_logit": payload["dual_logit"].astype(np.float32),
            "harm_logit": payload["harm_logit"].astype(np.float32),
        }
        names = payload["cohort"].astype(str)
    report: dict[str, Any] = {
        "schema": "recon_cem_candidate_pool_diagnostic_v1",
        "evidence_role": "development_only_three_spent_cohorts",
        "source_v4_result": str(args.v4_result),
        "source_v4_result_sha256": file_sha256(args.v4_result),
        "cohorts": {},
        "next_confirmation_accessed": False,
        "formal_claim_allowed": False,
    }
    for cohort in cohorts:
        mask = names == cohort["name"]
        prediction = {
            key: value[mask] for key, value in all_prediction.items()
        }
        report["cohorts"][cohort["name"]] = cohort_diagnostics(
            cohort=cohort,
            prediction=prediction,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
