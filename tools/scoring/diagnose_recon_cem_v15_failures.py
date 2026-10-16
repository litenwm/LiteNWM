#!/usr/bin/env python3
"""Report v15 fold Pareto, calibration, route, and action failures."""

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
from tools.scoring.train_recon_cem_combined_pool_v7 import (  # noqa: E402
    resolve_base_feature,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--risk-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    risk_report = json.loads(
        args.risk_diagnostic.read_text(encoding="utf-8")
    )
    best = risk_report["best_candidate_risk_ordering"]
    with np.load(report["loco_predictions"], allow_pickle=False) as payload:
        prediction = {key: payload[key] for key in payload.files}

    absolute_values = []
    action_values = []
    candidate_values = []
    for value in report["inputs"]:
        result_path = Path(value["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with np.load(result["outcomes"], allow_pickle=False) as payload:
            absolute_values.append(
                payload["combined_absolute_metrics"].astype(np.float32)
            )
        base_path = resolve_base_feature(result_path)
        with np.load(base_path, allow_pickle=False) as base:
            base_action = base["actions_h4"].astype(np.float32)
            base_candidate = base["candidates"].astype(np.float32)
        with np.load(value["features"], allow_pickle=False) as cem:
            cem_action = cem["actions_h4"].astype(np.float32)
            cem_candidate = cem["candidates"].astype(np.float32)
        action_values.append(
            np.concatenate(
                (base_action[:, 1:], cem_action[:, 1:]), axis=1
            )
        )
        candidate_values.append(
            np.concatenate(
                (base_candidate[:, 1:], cem_candidate[:, 1:]), axis=1
            )
        )

    absolute = np.concatenate(absolute_values)
    actions = np.concatenate(action_values)
    candidates = np.concatenate(candidate_values)
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
    route = prediction["route_group"].astype(str)

    z = float(best["z"])
    dual_lower = (
        prediction["dual_mean"] - z * prediction["dual_std"]
    )
    harm_upper = (
        prediction["harm_mean"] + z * prediction["harm_std"]
    )
    delta_upper = prediction["mean"] + z * prediction["mean_std"]
    worst_delta = np.maximum(
        delta_upper[..., 0], delta_upper[..., 3]
    )
    candidate_score = (
        dual_lower
        - float(best["harm_weight"]) * harm_upper
        - float(best["delta_weight"]) * np.maximum(worst_delta, 0.0)
    )
    if best["pool"] == "base16":
        candidate_score[:, 16:] = -np.inf
    candidate_score = np.where(valid, candidate_score, -np.inf)
    selected = candidate_score.argmax(axis=1)
    row = np.arange(len(selected))
    state_score = candidate_score[row, selected]
    selected_delta = delta[row, selected]
    selected_dual = dual[row, selected]
    selected_harm = harm[row, selected]
    selected_actions = actions[row, selected]
    selected_candidates = candidates[row, selected]
    incumbent_actions = np.concatenate(
        [value[:, :1] for value in action_values], axis=0
    )[:, 0]
    incumbent_candidates = np.concatenate(
        [value[:, :1] for value in candidate_values], axis=0
    )[:, 0]
    action_delta_norm = np.linalg.norm(
        selected_actions - incumbent_actions, axis=(1, 2)
    )
    endpoint_delta = np.linalg.norm(
        selected_candidates[:, -1] - incumbent_candidates[:, -1],
        axis=1,
    )
    max_abs_yaw = np.abs(selected_actions[..., 2]).max(axis=1)

    folds = {}
    for name in np.unique(cohort):
        mask = cohort == name
        order = np.argsort(-state_score[mask], kind="stable")
        local_score = state_score[mask][order]
        local_dual = selected_dual[mask][order]
        local_harm = selected_harm[mask][order]
        local_delta = selected_delta[mask][order]
        local_route = route[mask][order]
        local_action_delta = action_delta_norm[mask][order]
        local_endpoint_delta = endpoint_delta[mask][order]
        local_yaw = max_abs_yaw[mask][order]

        pareto = []
        for count in range(10, 101, 10):
            dual_count = int(local_dual[:count].sum())
            harm_count = int(local_harm[:count].sum())
            pareto.append(
                {
                    "coverage": count / 100.0,
                    "dual_precision": dual_count / count,
                    "dual_wilson_lower_95": float(
                        wilson_lower(dual_count, count)
                    ),
                    "harm_rate": harm_count / count,
                }
            )
        calibration = []
        for start in range(0, 100, 10):
            stop = start + 10
            calibration.append(
                {
                    "rank_start": start,
                    "rank_stop": stop,
                    "score_mean": float(local_score[start:stop].mean()),
                    "dual_rate": float(local_dual[start:stop].mean()),
                    "harm_rate": float(local_harm[start:stop].mean()),
                }
            )
        top_harm = local_harm[:40]
        harmful_magnitude = np.maximum(
            local_delta[:40, 0], local_delta[:40, 3]
        )[top_harm]
        harmful_routes, harmful_counts = np.unique(
            local_route[:40][top_harm], return_counts=True
        )
        route_order = np.argsort(-harmful_counts, kind="stable")
        folds[name] = {
            "coverage_dual_pareto_future_rank_diagnostic_only": pareto,
            "score_calibration_deciles_future_rank_diagnostic_only": (
                calibration
            ),
            "top40_failure": {
                "harmful_promotions": int(top_harm.sum()),
                "harm_magnitude_m": {
                    "mean": (
                        float(harmful_magnitude.mean())
                        if len(harmful_magnitude)
                        else 0.0
                    ),
                    "p90": (
                        float(np.quantile(harmful_magnitude, 0.90))
                        if len(harmful_magnitude)
                        else 0.0
                    ),
                    "fraction_at_most_0.005m": (
                        float((harmful_magnitude <= 0.005).mean())
                        if len(harmful_magnitude)
                        else 0.0
                    ),
                    "fraction_at_most_0.010m": (
                        float((harmful_magnitude <= 0.010).mean())
                        if len(harmful_magnitude)
                        else 0.0
                    ),
                },
                "action_geometry": {
                    "harm_action_delta_l2_mean": float(
                        local_action_delta[:40][top_harm].mean()
                    ),
                    "harm_endpoint_delta_m_mean": float(
                        local_endpoint_delta[:40][top_harm].mean()
                    ),
                    "harm_max_abs_yaw_mean": float(
                        local_yaw[:40][top_harm].mean()
                    ),
                    "safe_action_delta_l2_mean": float(
                        local_action_delta[:40][~top_harm].mean()
                    ),
                    "safe_endpoint_delta_m_mean": float(
                        local_endpoint_delta[:40][~top_harm].mean()
                    ),
                    "safe_max_abs_yaw_mean": float(
                        local_yaw[:40][~top_harm].mean()
                    ),
                },
                "route_concentration": [
                    {
                        "route_group": str(harmful_routes[index]),
                        "harmful_promotions": int(harmful_counts[index]),
                    }
                    for index in route_order[:10]
                ],
            },
        }

    output = {
        "schema": "recon_cem_v15_failure_diagnostic",
        "status": "spent_development_failure_diagnostic_complete",
        "deployment_policy_allowed": False,
        "future_route_ranking_used": True,
        "folds": folds,
        "robot_diagnostic": {
            "available": False,
            "reason": (
                "RECON artifacts expose no deployable robot token; route "
                "identity cannot be used as a robot proxy"
            ),
        },
        "feature_family_decision": {
            "continue_threshold_tuning": False,
            "reason": (
                "even non-causal per-fold top40 ranking misses the fixed "
                "dual/harm gates"
            ),
            "next_identifiable_feature": (
                "spatial history-goal DINOv3 token alignment from the "
                "already frozen latent caches"
            ),
        },
        "source_result": str(args.result),
        "source_risk_diagnostic": str(args.risk_diagnostic),
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
