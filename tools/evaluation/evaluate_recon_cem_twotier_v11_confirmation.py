#!/usr/bin/env python3
"""Evaluate a fully frozen v11 RECON CEM confirmation exactly once."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("schema")
        != "recon_cem_twotier_v11_confirmation_freeze"
        or freeze.get("status")
        != "selections_frozen_before_confirmation_outcomes"
        or freeze.get("outcomes_accessed")
        or freeze.get("labels_opened")
        or freeze.get("cem_true_promotions", 0) < 40
    ):
        raise RuntimeError("v11 confirmation freeze is not eligible")
    if file_sha256(args.manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("v11 confirmation manifest changed")
    selected_path = Path(freeze["selected_actions"])
    if file_sha256(selected_path) != freeze["selected_actions_sha256"]:
        raise RuntimeError("v11 confirmation selection changed")
    for key, sha_key in (
        ("base_features", "base_features_sha256"),
        ("cem_features", "cem_features_sha256"),
        ("base_scores", "base_scores_sha256"),
        ("cem_scores", "cem_scores_sha256"),
        ("candidate_pool", "candidate_pool_sha256"),
        ("development_result", "development_result_sha256"),
    ):
        if file_sha256(Path(freeze[key])) != freeze[sha_key]:
            raise RuntimeError(f"v11 frozen artifact changed: {key}")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != 100
        or len({row["trajectory_id"] for row in rows}) != 25
        or len({row["route_group"] for row in rows}) != 25
    ):
        raise RuntimeError("v11 confirmation route/state contract changed")
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_group"]) for row in rows])
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
    with np.load(freeze["base_features"], allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("v11 base feature identity mismatch")
        base = payload["candidates"].astype(np.float32)
    with np.load(freeze["cem_features"], allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("v11 CEM feature identity mismatch")
        final = payload["candidates"].astype(np.float32)
    combined = np.concatenate((base, final[:, 1:]), axis=1)
    absolute = metric_labels(
        combined, ground_truth, 0.25, 0.4, 0.6
    )["values"]
    with np.load(selected_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("v11 selection identity mismatch")
        v11_false = payload["cem_false_selected_index"].astype(np.int64)
        v11_true = payload["cem_true_selected_index"].astype(np.int64)
    cem_selected_actions = json.loads(
        Path(freeze["cem_freeze"]).read_text(encoding="utf-8")
    )["selected_actions"]
    with np.load(cem_selected_actions, allow_pickle=False) as payload:
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)
        original_false = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
        raw_cem = payload[
            "universal_d_cem_true_selected_index"
        ].astype(np.int64)
    raw_cem_combined = np.where(raw_cem == 0, 0, 16 + raw_cem)
    oracle = absolute[..., 0].argmin(axis=1).astype(np.int64)
    methods = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        "original_universal_d_cem_false": original_false,
        "original_universal_d_raw_cem_true": raw_cem_combined,
        "v11_same_selector_cem_false": v11_false,
        "v11_cem_true": v11_true,
        "posthoc_combined_pool_ate_oracle": oracle,
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
    baseline = metrics["v11_same_selector_cem_false"]
    cem = metrics["v11_cem_true"]
    safety_gate = {
        "coverage_at_least_0.40": cem["promotion_fraction"] >= 0.40,
        "dual_at_least_0.90": (
            cem["dual_precision"] is not None
            and cem["dual_precision"] >= 0.90
        ),
        "wilson_at_least_0.80": (
            cem["dual_wilson_lower_95"] is not None
            and cem["dual_wilson_lower_95"] >= 0.80
        ),
        "harm_at_most_0.10": (
            cem["harm_rate_per_promotion"] is not None
            and cem["harm_rate_per_promotion"] <= 0.10
        ),
        "coverage_not_lower_than_cem_false": (
            cem["promotion_fraction"] >= baseline["promotion_fraction"]
        ),
        "dual_not_lower_than_cem_false": (
            cem["dual_precision"] is not None
            and baseline["dual_precision"] is not None
            and cem["dual_precision"] >= baseline["dual_precision"]
        ),
        "wilson_not_lower_than_cem_false": (
            cem["dual_wilson_lower_95"] is not None
            and baseline["dual_wilson_lower_95"] is not None
            and cem["dual_wilson_lower_95"]
            >= baseline["dual_wilson_lower_95"]
        ),
        "harm_not_higher_than_cem_false": (
            cem["harm_rate_per_promotion"] is not None
            and baseline["harm_rate_per_promotion"] is not None
            and cem["harm_rate_per_promotion"]
            <= baseline["harm_rate_per_promotion"]
        ),
    }
    quality_gate = {
        "all_means_not_worse": all(
            cem["absolute"][name]["mean_m"]
            <= baseline["absolute"][name]["mean_m"] + 1.0e-12
            for name in METRIC_NAMES
        ),
        "all_p90_not_worse": all(
            cem["absolute"][name]["p90_m"]
            <= baseline["absolute"][name]["p90_m"] + 1.0e-12
            for name in METRIC_NAMES
        ),
        "at_least_one_mean_strictly_better": any(
            cem["absolute"][name]["mean_m"]
            < baseline["absolute"][name]["mean_m"] - 1.0e-7
            for name in METRIC_NAMES
        ),
    }
    passed = all(safety_gate.values()) and all(quality_gate.values())
    outcome_path = args.output.with_suffix(".outcomes.npz")
    write_npz(
        outcome_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        ground_truth_local_m=ground_truth,
        combined_absolute_metrics=absolute,
        **{f"{name}_selected_index": value for name, value in methods.items()},
    )
    report = {
        "schema": "recon_cem_twotier_v11_confirmation_result",
        "status": (
            "development_confirmation_passed"
            if passed
            else "development_confirmation_failed"
        ),
        "evidence_role": "new_route_disjoint_development_confirmation",
        "domain": "recon",
        "cem": True,
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
        "freeze": str(args.freeze),
        "freeze_sha256": file_sha256(args.freeze),
        "manifest_sha256": file_sha256(args.manifest),
        "labels_opened_only_after_full_freeze": True,
        "confirmation_now_spent": True,
        "methods": metrics,
        "safety_gate": safety_gate,
        "quality_gate": quality_gate,
        "development_confirmation_pass": passed,
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "formal_outer_claim_allowed": False,
        "scand_fresh_confirmation_available": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
