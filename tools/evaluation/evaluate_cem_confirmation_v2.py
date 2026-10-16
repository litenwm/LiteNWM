#!/usr/bin/env python3
"""Evaluate a prelabel-frozen hybrid CEM selector on cem-cal confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRICS = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--hybrid-selected", type=Path, required=True)
    parser.add_argument("--selected-actions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "confirmation_selector_and_actions_frozen_before_outcomes"
        or freeze.get("evidence_role")
        != "route_trajectory_disjoint_cem_cal_confirmation"
        or freeze.get("domain") != "recon"
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_selector_on_confirmation")
    ):
        raise RuntimeError("hybrid confirmation prelabel freeze is invalid")
    if file_sha256(args.hybrid_selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("hybrid selected index changed after freeze")
    if file_sha256(args.selected_actions) != freeze["selected_actions_sha256"]:
        raise RuntimeError("CEM action pool changed after freeze")
    if file_sha256(args.manifest) != json.loads(
        Path(freeze["confirmation_audit"]).read_text(encoding="utf-8")
    )["output_sha256"]:
        raise RuntimeError("confirmation manifest changed after freeze")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError("confirmation state count changed")
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    ground_truth = np.stack(
        [
            local_ground_truth(
                args.trajectory_root
                / str(row["trajectory_id"])
                / "traj_data.pkl",
                int(row["current_t"]),
            )
            for row in rows
        ]
    ).astype(np.float32)
    with np.load(args.selected_actions, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("CEM action/manifest identity mismatch")
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)
        raw_cem = payload[
            "universal_d_cem_true_selected_index"
        ].astype(np.int64)
    with np.load(args.hybrid_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("hybrid selection/manifest identity mismatch")
        cem_false = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
        hybrid = payload[
            "cem_hybrid_v2_selected_combined_index"
        ].astype(np.int64)
        replacement = payload["cem_replacement_allowed"].astype(bool)
    combined = np.concatenate((base, final[:, 1:]), axis=1)
    absolute = metric_labels(
        combined, ground_truth, 0.25, 0.4, 0.6
    )["values"]
    raw_cem_combined = np.where(raw_cem == 0, 0, 16 + raw_cem)
    methods = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        "universal_d_cem_false": cem_false,
        "universal_d_raw_cem_true_diagnostic": raw_cem_combined,
        "cem_hybrid_v2": hybrid,
    }
    metrics = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.seed + index * 100,
        )
        for index, (name, selected) in enumerate(methods.items())
    }
    baseline = metrics["universal_d_cem_false"]
    cem = metrics["cem_hybrid_v2"]
    safety = {
        "coverage_not_lower": (
            cem["promotion_fraction"] + 1.0e-12
            >= baseline["promotion_fraction"]
        ),
        "dual_precision_not_lower": (
            cem["dual_precision"] is not None
            and baseline["dual_precision"] is not None
            and cem["dual_precision"] + 1.0e-12 >= baseline["dual_precision"]
        ),
        "wilson_not_lower": (
            cem["dual_wilson_lower_95"] is not None
            and baseline["dual_wilson_lower_95"] is not None
            and cem["dual_wilson_lower_95"] + 1.0e-12
            >= baseline["dual_wilson_lower_95"]
        ),
        "harm_not_higher": (
            cem["harm_rate_per_promotion"] is not None
            and baseline["harm_rate_per_promotion"] is not None
            and cem["harm_rate_per_promotion"]
            <= baseline["harm_rate_per_promotion"] + 1.0e-12
        ),
    }
    quality = {
        "all_means_not_higher": all(
            cem["absolute"][name]["mean_m"]
            <= baseline["absolute"][name]["mean_m"] + 1.0e-12
            for name in METRICS
        ),
        "ate_endpoint_objective_means_strictly_better": all(
            cem["absolute"][name]["mean_m"]
            < baseline["absolute"][name]["mean_m"] - 1.0e-7
            for name in ("ate", "endpoint", "objective")
        ),
        "all_p90_not_higher": all(
            cem["absolute"][name]["p90_m"]
            <= baseline["absolute"][name]["p90_m"] + 1.0e-12
            for name in METRICS
        ),
    }
    checks = {
        **safety,
        **quality,
        "development_level_good_result": all(safety.values())
        and all(quality.values()),
    }
    outcome_path = args.output.with_suffix(".outcomes.npz")
    write_npz(
        outcome_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        ground_truth_local_m=ground_truth,
        combined_absolute_metrics=absolute,
        cem_hybrid_v2_selected_combined_index=hybrid,
        cem_replacement_allowed=replacement,
    )
    report = {
        "schema": "cem_hybrid_v2_confirmation_outcomes",
        "status": "confirmation_evaluated_once_after_full_freeze",
        "evidence_role": "development_level_route_trajectory_disjoint_confirmation",
        "formal_outer_claim_allowed": False,
        "domain": "recon",
        "states": 100,
        "trajectories": int(len(np.unique(trajectory))),
        "route_groups": int(len(np.unique(route))),
        "cem": True,
        "methods": metrics,
        "checks": checks,
        "hybrid_changed_windows": int(replacement.sum()),
        "freeze": str(args.freeze),
        "freeze_sha256": file_sha256(args.freeze),
        "hybrid_selected_sha256": file_sha256(args.hybrid_selected),
        "selected_actions_sha256": file_sha256(args.selected_actions),
        "manifest_sha256": file_sha256(args.manifest),
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "labels_opened_only_after_full_freeze": True,
        "confirmation_now_spent": True,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "checks": checks,
                "hybrid_changed_windows": int(replacement.sum()),
                "methods": {
                    name: {
                        "promotions": value["promotions"],
                        "coverage": value["promotion_fraction"],
                        "dual": value["dual_precision"],
                        "wilson": value["dual_wilson_lower_95"],
                        "harm": value["harm_rate_per_promotion"],
                        "absolute": value["absolute"],
                    }
                    for name, value in metrics.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
