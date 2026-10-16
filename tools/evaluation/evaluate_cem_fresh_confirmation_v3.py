#!/usr/bin/env python3
"""Evaluate prelabel-frozen add-only CEM-v3 on fresh RECON test routes."""

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


METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--selected-actions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "fresh_cem_v3_actions_and_selection_frozen_before_outcomes"
        or freeze.get("evidence_role")
        != "fresh_route_disjoint_cem_confirmation"
        or freeze.get("domain") != "recon"
        or freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("outcomes_used_by_selector_on_confirmation")
        or not freeze.get("fixed_pool_action_never_replaced")
    ):
        raise RuntimeError("fresh CEM-v3 prelabel freeze is invalid")
    if file_sha256(args.selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("fresh CEM-v3 selected index changed")
    if file_sha256(args.selected_actions) != freeze["selected_actions_sha256"]:
        raise RuntimeError("fresh CEM-v3 action pool changed")
    manifest_audit = json.loads(
        Path(freeze["manifest_audit"]).read_text(encoding="utf-8")
    )
    if manifest_audit["manifest_sha256"] != file_sha256(args.manifest):
        raise RuntimeError("fresh CEM-v3 manifest changed")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != 100
        or len({row["trajectory_id"] for row in rows}) != 25
        or len({row["route_id"] for row in rows}) != 25
    ):
        raise RuntimeError("fresh CEM-v3 route/state contract changed")
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
            raise RuntimeError("fresh CEM action/manifest mismatch")
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)
        raw_cem = payload[
            "universal_d_cem_true_selected_index"
        ].astype(np.int64)
    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("fresh CEM-v3 selector/manifest mismatch")
        cem_false = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
        cem_v3 = payload[
            "cem_addonly_v3_selected_combined_index"
        ].astype(np.int64)
        added = payload["cem_addition_allowed"].astype(bool)
    if np.any(cem_v3[cem_false != 0] != cem_false[cem_false != 0]):
        raise RuntimeError("add-only selector replaced a fixed-pool action")

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
        "cem_addonly_v3": cem_v3,
    }
    metrics = {
        name: method_metrics(
            selected_index,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.seed + index * 100,
        )
        for index, (name, selected_index) in enumerate(methods.items())
    }
    baseline = metrics["universal_d_cem_false"]
    cem = metrics["cem_addonly_v3"]
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
            for name in METRIC_NAMES
        ),
        "ate_endpoint_objective_means_strictly_better": all(
            cem["absolute"][name]["mean_m"]
            < baseline["absolute"][name]["mean_m"] - 1.0e-7
            for name in ("ate", "endpoint", "objective")
        ),
        "all_p90_not_higher": all(
            cem["absolute"][name]["p90_m"]
            <= baseline["absolute"][name]["p90_m"] + 1.0e-12
            for name in METRIC_NAMES
        ),
        "at_least_one_cem_action_added": int(added.sum()) > 0,
    }
    checks = {
        **safety,
        **quality,
        "fresh_cem_confirmation_good_result": all(safety.values())
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
        universal_d_cem_false_selected_index=cem_false,
        cem_addonly_v3_selected_combined_index=cem_v3,
        cem_addition_allowed=added,
    )
    report = {
        "schema": "cem_addonly_v3_fresh_confirmation_outcomes",
        "status": "fresh_confirmation_evaluated_once_after_full_freeze",
        "evidence_role": "fresh_route_disjoint_cem_confirmation",
        "formal_fixed_pool_qualification_claim": False,
        "domain": "recon",
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
        "cem": True,
        "cem_added_windows": int(added.sum()),
        "methods": metrics,
        "checks": checks,
        "freeze": str(args.freeze),
        "freeze_sha256": file_sha256(args.freeze),
        "selected_index_sha256": file_sha256(args.selected),
        "selected_actions_sha256": file_sha256(args.selected_actions),
        "manifest_sha256": file_sha256(args.manifest),
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "labels_opened_only_after_full_freeze": True,
        "fresh_confirmation_now_spent": True,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "checks": checks,
                "cem_added_windows": int(added.sum()),
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
