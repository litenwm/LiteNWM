#!/usr/bin/env python3
"""Open cem-cal outcomes only after CEM actions/selections were SHA-frozen."""

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

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth as recon_ground_truth,
)
from tools.evaluation.build_scand_inner_k17_source_v1 import (  # noqa: E402
    local_ground_truth as scand_ground_truth,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def deployment_checks(
    cem: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, bool]:
    names = ("ate", "rpe", "endpoint", "objective")
    return {
        "dual_precision_not_below_cem_false": (
            cem["dual_precision"] is not None
            and reference["dual_precision"] is not None
            and cem["dual_precision"] + 1.0e-12 >= reference["dual_precision"]
        ),
        "wilson_not_below_cem_false": (
            cem["dual_wilson_lower_95"] is not None
            and reference["dual_wilson_lower_95"] is not None
            and cem["dual_wilson_lower_95"] + 1.0e-12
            >= reference["dual_wilson_lower_95"]
        ),
        "harm_not_above_cem_false": (
            cem["harm_rate_per_promotion"] is not None
            and reference["harm_rate_per_promotion"] is not None
            and cem["harm_rate_per_promotion"]
            <= reference["harm_rate_per_promotion"] + 1.0e-12
        ),
        "coverage_not_below_cem_false": (
            cem["promotion_fraction"] + 1.0e-12
            >= reference["promotion_fraction"]
        ),
        "all_means_better_than_cem_false": all(
            cem["absolute"][name]["mean_m"]
            < reference["absolute"][name]["mean_m"]
            for name in names
        ),
        "all_p90_nondegraded_vs_cem_false": all(
            cem["absolute"][name]["p90_m"]
            <= reference["absolute"][name]["p90_m"] + 1.0e-12
            for name in names
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_replicates < 1000:
        raise ValueError("clustered bootstrap needs at least 1000 replicates")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "cem_true_actions_and_selections_frozen_before_outcomes"
        or freeze.get("evidence_role") != "development_only_cem_cal"
        or freeze.get("cem") is not True
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_cem")
    ):
        raise RuntimeError("CEM prelabel freeze contract is invalid")
    if file_sha256(args.selected) != freeze["selected_actions_sha256"]:
        raise RuntimeError("CEM selected actions changed after freeze")
    if file_sha256(args.manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("cem-cal manifest changed after freeze")

    rows = read_rows(args.manifest)
    domain = str(freeze["domain"])
    if domain not in {"recon", "scand"} or len(rows) != 100:
        raise RuntimeError("CEM domain/state contract changed")
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    ground_truth_fn = recon_ground_truth if domain == "recon" else scand_ground_truth
    ground_truth = np.stack(
        [
            ground_truth_fn(
                args.trajectory_root
                / str(row["trajectory_id"])
                / "traj_data.pkl",
                int(row["current_t"]),
            )
            for row in rows
        ]
    ).astype(np.float32)
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route_key = "route_id" if domain == "recon" else "route_group"
    route = np.asarray([str(row[route_key]) for row in rows])

    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected_record):
            raise RuntimeError("selection/manifest identity mismatch")
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)
        cem_false = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
        cem_true_local = payload[
            "universal_d_cem_true_selected_index"
        ].astype(np.int64)
    if base.shape != (100, 17, 4, 2) or final.shape != (100, 17, 4, 2):
        raise RuntimeError("CEM candidate pool shape changed")
    if not np.allclose(base[:, 0], final[:, 0]):
        raise RuntimeError("CEM incumbent changed")

    combined = np.concatenate((base, final[:, 1:]), axis=1)
    labels = metric_labels(combined, ground_truth, 0.25, 0.4, 0.6)
    absolute = labels["values"]
    cem_true = np.where(cem_true_local == 0, 0, 16 + cem_true_local)
    oracle = absolute[..., 3].argmin(axis=1).astype(np.int64)
    methods = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        "universal_d_cem_false": cem_false,
        "universal_d_cem_true": cem_true,
        "posthoc_combined_pool_oracle": oracle,
    }
    metrics = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index * 100,
        )
        for index, (name, selected) in enumerate(methods.items())
    }
    checks = deployment_checks(
        metrics["universal_d_cem_true"],
        metrics["universal_d_cem_false"],
    )
    checks["development_goal_pass"] = all(checks.values())

    outcome_path = args.output.with_suffix(".outcomes.npz")
    write_npz(
        outcome_path,
        record_key=expected_record,
        trajectory=trajectory,
        route_group=route,
        ground_truth_local_m=ground_truth,
        combined_absolute_metrics=absolute,
        native_scorer_v2_selected_index=native,
        universal_d_cem_false_selected_index=cem_false,
        universal_d_cem_true_combined_index=cem_true,
    )
    report = {
        "schema": "universal_raw_d_cem_development_outcomes_v1",
        "status": "cem_cal_development_evaluated_after_prelabeled_freeze",
        "evidence_role": "development_only_cem_cal_spent_after_this_evaluation",
        "domain": domain,
        "states": 100,
        "trajectories": int(len(np.unique(trajectory))),
        "route_groups": int(len(np.unique(route))),
        "cem": True,
        "formal_claim_allowed": False,
        "scand_fresh_outer_available": False if domain == "scand" else None,
        "selection_freeze": str(args.freeze),
        "selection_freeze_sha256": file_sha256(args.freeze),
        "selected_actions_sha256": file_sha256(args.selected),
        "manifest_sha256": file_sha256(args.manifest),
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "methods": metrics,
        "cem_true_vs_false_checks": checks,
        "labels_opened_only_after_selection_freeze": True,
        "outcomes_used_inside_cem": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "cem_cal_now_spent": True,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "domain": domain,
                "evidence_role": report["evidence_role"],
                "methods": {
                    name: {
                        "promotions": value["promotions"],
                        "coverage": value["promotion_fraction"],
                        "dual_precision": value["dual_precision"],
                        "wilson": value["dual_wilson_lower_95"],
                        "harm_rate": value["harm_rate_per_promotion"],
                        "absolute": value["absolute"],
                    }
                    for name, value in metrics.items()
                },
                "cem_true_vs_false_checks": checks,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
