#!/usr/bin/env python3
"""Open fresh-outer pose outcomes only after all selections are frozen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifest.windows.jsonl"
    )
    parser.add_argument(
        "--candidate", type=Path, default=ROOT / "fresh_outer_k17.npz"
    )
    parser.add_argument(
        "--prelabel-complete",
        type=Path,
        default=ROOT / "prelabel_freeze/prelabel_freeze_complete.json",
    )
    parser.add_argument(
        "--six-arm-selected",
        type=Path,
        default=ROOT / "prelabel_freeze/selected_index.npz",
    )
    parser.add_argument(
        "--native-selected",
        type=Path,
        default=ROOT / "prelabel_freeze/native_v2_selected_index.npz",
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outcomes")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def local_ground_truth(path: Path, timestep: int) -> np.ndarray:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    position = np.asarray(raw["position"], dtype=np.float64)[timestep : timestep + 5, :2]
    yaw = np.asarray(raw["yaw"], dtype=np.float64).reshape(-1)[timestep : timestep + 5]
    if position.shape != (5, 2) or yaw.shape != (5,):
        raise RuntimeError(f"trajectory shorter than H4: {path}:{timestep}")
    angle = float(yaw[0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return ((position - position[0]).dot(rotation))[1:].astype(np.float32)


def absolute_metrics(candidates: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    candidates_m = np.asarray(candidates, dtype=np.float64) * 0.38
    truth = np.asarray(ground_truth, dtype=np.float64)
    error = candidates_m - truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    rpe = np.sqrt(
        np.mean(
            np.sum(
                (
                    np.diff(candidates_m, axis=2)
                    - np.diff(truth, axis=1)[:, None]
                )
                ** 2,
                axis=-1,
            ),
            axis=-1,
        )
    )
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def clustered_ci(
    delta: np.ndarray,
    cluster: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    names = np.asarray(cluster).astype(str)
    unique = sorted(np.unique(names).tolist())
    cluster_mean = np.stack([delta[names == value].mean(axis=0) for value in unique])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(replicates, len(unique)))
    sampled = cluster_mean[draw].mean(axis=1)
    return {
        name: {
            "mean_delta_m": float(delta[:, index].mean()),
            "ci95_low_m": float(np.quantile(sampled[:, index], 0.025)),
            "ci95_high_m": float(np.quantile(sampled[:, index], 0.975)),
        }
        for index, name in enumerate(METRIC_NAMES)
    }


def method_metrics(
    selected: np.ndarray,
    values: np.ndarray,
    trajectory: np.ndarray,
    route: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    index = np.asarray(selected, dtype=np.int64)
    if index.shape != (len(values),) or np.any(index < 0) or np.any(index >= values.shape[1]):
        raise RuntimeError("selected index contract failed")
    row = np.arange(len(index))
    chosen = values[row, index]
    incumbent = values[:, 0]
    delta = chosen - incumbent
    promote = index != 0
    promotions = int(promote.sum())
    ate_success = int((delta[promote, 0] < -1.0e-7).sum())
    objective_success = int((delta[promote, 3] < -1.0e-7).sum())
    dual_success = int(
        ((delta[promote, 0] < -1.0e-7) & (delta[promote, 3] < -1.0e-7)).sum()
    )
    harmed = promote & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    absolute = {
        name: {
            "mean_m": float(chosen[:, metric].mean()),
            "p90_m": float(np.quantile(chosen[:, metric], 0.90)),
        }
        for metric, name in enumerate(METRIC_NAMES)
    }
    return {
        "states": int(len(index)),
        "promotions": promotions,
        "promotion_fraction": float(promotions / len(index)),
        "ate_precision": ate_success / promotions if promotions else None,
        "objective_precision": (
            objective_success / promotions if promotions else None
        ),
        "dual_precision": dual_success / promotions if promotions else None,
        "dual_wilson_lower_95": wilson_lower(dual_success, promotions),
        "harm_count": int(harmed.sum()),
        "harm_rate_per_promotion": (
            float(harmed.sum() / promotions) if promotions else None
        ),
        "mean_harm_magnitude_m": (
            float(np.maximum(delta[harmed, 0], delta[harmed, 3]).mean())
            if harmed.any()
            else 0.0
        ),
        "max_harm_magnitude_m": (
            float(np.maximum(delta[harmed, 0], delta[harmed, 3]).max())
            if harmed.any()
            else 0.0
        ),
        "absolute": absolute,
        "mean_delta_vs_incumbent_m": {
            name: float(delta[:, metric].mean())
            for metric, name in enumerate(METRIC_NAMES)
        },
        "trajectory_clustered_paired_bootstrap_ci": clustered_ci(
            delta,
            trajectory,
            replicates=replicates,
            seed=seed,
        ),
        "route_clustered_paired_bootstrap_ci": clustered_ci(
            delta,
            route,
            replicates=replicates,
            seed=seed + 1,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bootstrap_replicates < 1000:
        raise ValueError("clustered bootstrap needs at least 1000 replicates")
    complete = json.loads(args.prelabel_complete.read_text(encoding="utf-8"))
    if complete.get("status") != "all_fixed_pool_selections_frozen_before_pose_labels":
        raise RuntimeError("prelabel freeze is incomplete")
    if complete.get("labels_opened") or complete.get("future_pose_outcomes_opened"):
        raise RuntimeError("prelabel freeze claims labels were already opened")
    if complete.get("historically_spent") or complete.get("cem"):
        raise RuntimeError("fresh outer provenance is invalid")
    if file_sha256(args.candidate) != complete["candidate_pool_sha256"]:
        raise RuntimeError("candidate pool changed after prelabel freeze")
    if file_sha256(args.six_arm_selected) != complete[
        "six_arm_selected_index_sha256"
    ]:
        raise RuntimeError("six-arm selections changed after prelabel freeze")
    if file_sha256(args.native_selected) != complete[
        "native_v2_selected_index_sha256"
    ]:
        raise RuntimeError("native-v2 selections changed after prelabel freeze")

    rows = read_manifest(args.manifest)
    if len(rows) != 100:
        raise RuntimeError("fresh outer state count changed")
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_group"]) for row in rows])
    if len(np.unique(trajectory)) != 25 or len(np.unique(route)) != 25:
        raise RuntimeError("fresh outer clustering contract changed")
    with np.load(args.candidate, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("candidate/manifest identity mismatch")
        candidates = payload["candidates_waypoint_units"].astype(np.float32)
    with np.load(args.six_arm_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("six-arm selection/manifest identity mismatch")
        six_arm = {
            key: payload[key].astype(np.int64)
            for key in payload.files
            if "__" in key and not key.endswith("__meta_score")
        }
    with np.load(args.native_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("native selection/manifest identity mismatch")
        native = payload["native_v2_selected_index"].astype(np.int64)

    # This is the first and only point at which fresh-outer future pose outcomes
    # are opened.
    ground_truth = np.empty((100, 4, 2), dtype=np.float32)
    for index, row in enumerate(rows):
        ground_truth[index] = local_ground_truth(
            args.trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
            int(row["current_t"]),
        )
    values = absolute_metrics(candidates, ground_truth)
    labels = metric_labels(candidates, ground_truth, 0.38, 0.4, 0.6)
    objective_oracle = values[..., 3].argmin(axis=1).astype(np.int64)
    methods: dict[str, np.ndarray] = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        **six_arm,
        "posthoc_objective_pool_oracle": objective_oracle,
    }
    primary_freeze = json.loads(
        (ROOT / "prelabel_freeze/prelabel_freeze.json").read_text(encoding="utf-8")
    )
    result_methods = {}
    for method_index, (name, selected) in enumerate(methods.items()):
        metrics = method_metrics(
            selected,
            values,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + method_index * 10,
        )
        if name in six_arm:
            regime, arm = name.split("__", 1)
            metrics["support"] = primary_freeze["selection_audit"][regime][arm][
                "support"
            ]
        result_methods[name] = metrics
    args.output_dir.mkdir(parents=True)
    outcome_path = args.output_dir / "fresh_outer_outcomes.npz"
    write_npz(
        outcome_path,
        record_key=record_key,
        trajectory=trajectory,
        route_group=route,
        ground_truth_local_m=ground_truth,
        absolute_metrics=values,
        targets=labels["targets"],
        objective_oracle_selected_index=objective_oracle,
    )
    report = {
        "schema": "scand_fresh_outer_h4_100_result_v1",
        "status": "completed_first_label_open_fixed_pool",
        "evidence_role": "fresh outer; no current/history trajectory or route reuse found before freeze",
        "states": 100,
        "trajectories": 25,
        "route_groups": 25,
        "bags": 25,
        "windows_per_trajectory": 4,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "prelabel_complete": str(args.prelabel_complete),
        "prelabel_complete_sha256": file_sha256(args.prelabel_complete),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "six_arm_selected_index_sha256": file_sha256(args.six_arm_selected),
        "native_v2_selected_index_sha256": file_sha256(args.native_selected),
        "outcome_artifact": str(outcome_path),
        "outcome_artifact_sha256": file_sha256(outcome_path),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "methods": result_methods,
        "labels_opened": True,
        "future_pose_outcomes_opened": True,
        "opened_only_after_all_selection_sha_freeze": True,
        "spent_strict100_outcomes_used": False,
        "cem_cal_accessed": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output_dir / "result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
