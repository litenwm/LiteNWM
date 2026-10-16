#!/usr/bin/env python3
"""Evaluate prelabel-frozen universal-v2 choices on fresh RECON H4-100."""

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
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


ROOT = (
    REPO_ROOT
    / "outputs/universal_dualdomain_raw_v2/recon_fresh_outer_v2"
)
METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "prelabel_freeze/prelabel_freeze.json",
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=ROOT / "prelabel_freeze/selected_index.npz",
    )
    parser.add_argument(
        "--labeled-source",
        type=Path,
        default=ROOT / "fresh_outer_source_labeled.npz",
    )
    parser.add_argument(
        "--labeled-source-audit",
        type=Path,
        default=ROOT / "fresh_outer_source_labeled.audit.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "result.json"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    return parser.parse_args()


def deployment_checks(
    metrics: dict[str, Any],
    incumbent: dict[str, Any],
    native: dict[str, Any],
) -> dict[str, bool]:
    safety = {
        "coverage_at_least_0.40": metrics["promotion_fraction"] >= 0.40,
        "dual_precision_at_least_0.90": (
            metrics["dual_precision"] is not None
            and metrics["dual_precision"] >= 0.90
        ),
        "wilson_lower_at_least_0.80": (
            metrics["dual_wilson_lower_95"] is not None
            and metrics["dual_wilson_lower_95"] >= 0.80
        ),
        "harm_rate_at_most_0.10": (
            metrics["harm_rate_per_promotion"] is not None
            and metrics["harm_rate_per_promotion"] <= 0.10
        ),
    }
    quality = {
        "all_means_better_than_incumbent": all(
            metrics["absolute"][name]["mean_m"]
            < incumbent["absolute"][name]["mean_m"]
            for name in METRIC_NAMES
        ),
        "all_means_better_than_native_scorer_v2": all(
            metrics["absolute"][name]["mean_m"]
            < native["absolute"][name]["mean_m"]
            for name in METRIC_NAMES
        ),
        "all_p90_nondegraded_vs_incumbent": all(
            metrics["absolute"][name]["p90_m"]
            <= incumbent["absolute"][name]["p90_m"] + 1.0e-9
            for name in METRIC_NAMES
        ),
        "all_p90_nondegraded_vs_native_scorer_v2": all(
            metrics["absolute"][name]["p90_m"]
            <= native["absolute"][name]["p90_m"] + 1.0e-9
            for name in METRIC_NAMES
        ),
    }
    return {
        **safety,
        **quality,
        "all_safety_constraints_pass": all(safety.values()),
        "all_quality_constraints_pass": all(quality.values()),
        "overall_pass": all(safety.values()) and all(quality.values()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_replicates < 1000:
        raise ValueError("clustered bootstrap needs at least 1000 replicates")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != (
        "all_fixed_pool_selections_frozen_before_future_pose_labels"
    ):
        raise RuntimeError("prelabel selection freeze is incomplete")
    if (
        freeze.get("labels_opened")
        or freeze.get("future_pose_outcomes_opened")
        or freeze.get("historically_spent")
        or freeze.get("cem")
    ):
        raise RuntimeError("fresh-outer prelabel provenance is invalid")
    if freeze.get("evidence_role") != "fresh route-disjoint outer H4-100":
        raise RuntimeError("fresh evidence role changed")
    if file_sha256(args.selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("selected indices changed after prelabel freeze")

    label_audit = json.loads(
        args.labeled_source_audit.read_text(encoding="utf-8")
    )
    if (
        label_audit.get("labels_opened") is not True
        or label_audit.get("partition") != "fresh-outer"
        or label_audit.get("states") != 100
        or label_audit.get("spent_recon100_trajectory_overlap") != 0
        or label_audit.get("candidate_sha256")
        != freeze["candidate_pool_sha256"]
    ):
        raise RuntimeError("fresh label artifact audit is invalid")
    if label_audit["output_sha256"] != file_sha256(args.labeled_source):
        raise RuntimeError("fresh labeled source changed")

    with np.load(args.labeled_source, allow_pickle=False) as payload:
        candidates = payload["candidates"].astype(np.float32)
        ground_truth = payload["ground_truth_local_m"].astype(np.float32)
        record_key = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
    if candidates.shape != (100, 17, 4, 2):
        raise RuntimeError("fresh labeled candidate shape changed")
    labels = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)
    absolute = labels["values"]
    if absolute.shape != (100, 17, 4):
        raise RuntimeError("fresh absolute metric contract changed")

    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("selection/labeled-source identity mismatch")
        if not np.array_equal(payload["trajectory"].astype(str), trajectory):
            raise RuntimeError("selection trajectory identity mismatch")
        if not np.array_equal(payload["route_group"].astype(str), route):
            raise RuntimeError("selection route identity mismatch")
        selected = {
            key: payload[key].astype(np.int64)
            for key in payload.files
            if key
            not in ("record_key", "trajectory", "route_group")
        }

    oracle = absolute[..., 3].argmin(axis=1).astype(np.int64)
    methods: dict[str, np.ndarray] = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": selected.pop("native_scorer_v2"),
        **selected,
        "posthoc_objective_pool_oracle": oracle,
    }
    metrics: dict[str, Any] = {
        name: method_metrics(
            index,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + method_index * 10,
        )
        for method_index, (name, index) in enumerate(methods.items())
    }
    for name in (
        "legacy_balanced_support_plus_raw",
        "universal_b",
        "universal_c",
        "universal_d",
        "safe_rank_failed_ablation",
    ):
        metrics[name]["deployment_checks"] = deployment_checks(
            metrics[name],
            metrics["incumbent"],
            metrics["native_scorer_v2"],
        )

    outcome_path = args.output.parent / "fresh_outer_outcomes.npz"
    if outcome_path.exists():
        raise FileExistsError(outcome_path)
    write_npz(
        outcome_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        absolute_metrics=absolute,
        objective_oracle_selected_index=oracle,
    )
    report = {
        "schema": "universal_dualdomain_raw_v2_recon_fresh_outer",
        "status": "fresh_route_disjoint_outer_evaluated_after_prelabel_freeze",
        "evidence_role": "fresh route-disjoint outer H4-100",
        "states": 100,
        "trajectories": int(len(np.unique(trajectory))),
        "route_groups": int(len(np.unique(route))),
        "selection_freeze": str(args.freeze),
        "selection_freeze_sha256": file_sha256(args.freeze),
        "selected_index_sha256": file_sha256(args.selected),
        "labeled_source_sha256": file_sha256(args.labeled_source),
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "methods": metrics,
        "success_contract": {
            "dual_precision": 0.90,
            "dual_wilson_lower_95": 0.80,
            "minimum_promotion_fraction": 0.40,
            "maximum_harm_rate_per_promotion": 0.10,
            "means": "all four lower than incumbent and native scorer-v2",
            "p90": "all four no higher than incumbent and native scorer-v2",
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "labels_opened_only_after_selection_freeze": True,
        "historically_spent": False,
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(args.output, report)
    compact = {
        name: {
            "promotions": value["promotions"],
            "coverage": value["promotion_fraction"],
            "dual_precision": value["dual_precision"],
            "wilson": value["dual_wilson_lower_95"],
            "harm_rate": value["harm_rate_per_promotion"],
            "absolute": value["absolute"],
            "deployment_checks": value.get("deployment_checks"),
        }
        for name, value in metrics.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
