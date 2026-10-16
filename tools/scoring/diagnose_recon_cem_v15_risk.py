#!/usr/bin/env python3
"""Diagnose cohort-held-out v15 risk ranking without creating a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def top_summary(
    score: np.ndarray,
    selected: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    count: int,
) -> dict[str, float | int]:
    order = np.argsort(-score, kind="stable")[:count]
    row = np.arange(len(selected))[order]
    index = selected[order]
    dual_count = int(dual[row, index].sum())
    harm_count = int(harm[row, index].sum())
    return {
        "promotions": int(count),
        "dual_precision": float(dual_count / count),
        "dual_wilson_lower_95": float(
            wilson_lower(dual_count, count)
        ),
        "harm_rate": float(harm_count / count),
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    with np.load(report["loco_predictions"], allow_pickle=False) as payload:
        prediction = {key: payload[key] for key in payload.files}
    absolute_values = []
    for value in report["inputs"]:
        result = json.loads(Path(value["result"]).read_text(encoding="utf-8"))
        with np.load(result["outcomes"], allow_pickle=False) as payload:
            absolute_values.append(
                payload["combined_absolute_metrics"].astype(np.float32)
            )
    absolute = np.concatenate(absolute_values)
    delta = absolute[:, 1:] - absolute[:, :1]
    dual = (
        (delta[..., 0] < -1.0e-7)
        & (delta[..., 3] < -1.0e-7)
    )
    harm = (
        (delta[..., 0] > 1.0e-7)
        | (delta[..., 3] > 1.0e-7)
    )
    valid = prediction["valid"].astype(bool)
    cohort = prediction["cohort"].astype(str)
    modes = []
    for pool_name, pool_mask in (
        (
            "base16",
            np.concatenate(
                (
                    np.ones((len(valid), 16), dtype=bool),
                    np.zeros((len(valid), 16), dtype=bool),
                ),
                axis=1,
            ),
        ),
        ("combined32", np.ones_like(valid)),
    ):
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            dual_lower = (
                prediction["dual_mean"]
                - z * prediction["dual_std"]
            )
            harm_upper = (
                prediction["harm_mean"]
                + z * prediction["harm_std"]
            )
            delta_upper = (
                prediction["mean"]
                + z * prediction["mean_std"]
            )
            worst_delta = np.maximum(
                delta_upper[..., 0], delta_upper[..., 3]
            )
            for delta_weight in (0.0, 1.0, 2.0, 4.0, 8.0):
                for harm_weight in (0.5, 1.0, 2.0, 4.0):
                    candidate_score = (
                        dual_lower
                        - harm_weight * harm_upper
                        - delta_weight * np.maximum(worst_delta, 0.0)
                    )
                    candidate_score = np.where(
                        valid & pool_mask, candidate_score, -np.inf
                    )
                    selected = candidate_score.argmax(axis=1)
                    row = np.arange(len(selected))
                    state_score = candidate_score[row, selected]
                    per_cohort = {}
                    for name in np.unique(cohort):
                        mask = cohort == name
                        per_cohort[name] = {
                            "top40_route_future_diagnostic_only": top_summary(
                                state_score[mask],
                                selected[mask],
                                dual[mask],
                                harm[mask],
                                40,
                            ),
                            "top50_route_future_diagnostic_only": top_summary(
                                state_score[mask],
                                selected[mask],
                                dual[mask],
                                harm[mask],
                                50,
                            ),
                        }
                    worst_top40_dual = min(
                        value[
                            "top40_route_future_diagnostic_only"
                        ]["dual_precision"]
                        for value in per_cohort.values()
                    )
                    worst_top40_harm = max(
                        value[
                            "top40_route_future_diagnostic_only"
                        ]["harm_rate"]
                        for value in per_cohort.values()
                    )
                    modes.append(
                        {
                            "pool": pool_name,
                            "z": z,
                            "delta_weight": delta_weight,
                            "harm_weight": harm_weight,
                            "worst_top40_dual": worst_top40_dual,
                            "worst_top40_harm": worst_top40_harm,
                            "per_cohort": per_cohort,
                        }
                    )
    best = max(
        modes,
        key=lambda value: (
            value["worst_top40_dual"],
            -value["worst_top40_harm"],
        ),
    )
    output = {
        "schema": "recon_cem_v15_risk_diagnostic",
        "status": "spent_development_diagnostic_complete",
        "future_route_ranking_used": True,
        "deployment_policy_allowed": False,
        "purpose": (
            "upper-bound diagnosis only; full-cohort top-k is not causal "
            "and cannot qualify or freeze a policy"
        ),
        "best_candidate_risk_ordering": best,
        "searched_modes": len(modes),
        "source": str(args.result),
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
