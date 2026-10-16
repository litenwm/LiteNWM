#!/usr/bin/env python3
"""Validate a rotation-only camera geometry proxy across routes.

This diagnostic corrects a deliberately conservative metric in the first
self-calibration pilots.  A fixed camera-to-body rotation axis is a
route-level quantity, whereas individual essential-matrix decompositions are
noisy.  The script therefore fits a centered shared-focal model on development
routes, estimates one robust aggregate axis per route, and evaluates the
aggregate axis plus rotation-angle transfer on a held route.  It does not
estimate translation extrinsics or claim full camera calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import run_external_camera_reduced_selfcal_pilot_v1 as reduced_pilot
import run_external_camera_rotation_selfcal_pilot_v1 as full_pilot
import run_raw_resolution_paired_pilot_v1 as raw_pilot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_rotation_proxy_validation_v1/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=float, default=4.0)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--minimum-yaw-deg", type=float, default=1.5)
    parser.add_argument("--minimum-matches", type=int, default=40)
    parser.add_argument("--minimum-inliers", type=int, default=24)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026072406)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def signed_axes_and_weights(
    intrinsic: np.ndarray,
    pairs: list[full_pilot.Pair],
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axes = []
    weights = []
    for pair in pairs:
        rotation, _ = full_pilot.selected_rotation(intrinsic, pair)
        vector = cv2.Rodrigues(rotation)[0].reshape(3)
        if pair.delta_yaw < 0:
            vector = -vector
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-8:
            continue
        axis = vector / norm
        if float(np.dot(axis, reference)) < 0.0:
            axis = -axis
        axes.append(axis)
        weights.append(float(pair.inliers))
    full_pilot.require(bool(axes), "no rotation axes for bootstrap")
    return np.stack(axes), np.asarray(weights, dtype=np.float64)


def aggregate_axis(
    axes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    value = np.average(axes, axis=0, weights=weights)
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def bootstrap_axis_transfer(
    intrinsic: np.ndarray,
    pairs: list[full_pilot.Pair],
    train_axis: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    axes, weights = signed_axes_and_weights(intrinsic, pairs, train_axis)
    held_axis = aggregate_axis(axes, weights)
    errors = []
    sample_size = len(axes)
    for _ in range(replicates):
        indices = rng.integers(0, sample_size, size=sample_size)
        sampled_axis = aggregate_axis(axes[indices], weights[indices])
        errors.append(axis_angle_deg(sampled_axis, train_axis))
    errors_array = np.asarray(errors, dtype=np.float64)
    return {
        "held_aggregate_axis": held_axis,
        "held_to_train_axis_deg": axis_angle_deg(held_axis, train_axis),
        "bootstrap_replicates": replicates,
        "bootstrap_median_deg": float(np.median(errors_array)),
        "bootstrap_p95_deg": float(np.quantile(errors_array, 0.95)),
        "bootstrap_probability_at_most_15deg": float(
            np.mean(errors_array <= 15.0)
        ),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output.resolve()
    full_pilot.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    full_pilot.require(
        args.bootstrap_replicates >= 200,
        "need at least 200 bootstrap replicates",
    )
    cv2.setNumThreads(max(1, min(8, cv2.getNumberOfCPUs())))
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    all_pairs: dict[str, list[full_pilot.Pair]] = {}
    bag_audits = {}
    for spec in raw_pilot.DEFAULT_BAGS:
        pairs, audit = full_pilot.fundamental_pairs(
            spec,
            sample_rate=args.sample_rate,
            frame_gap=args.frame_gap,
            width=args.image_width,
            height=args.image_height,
            minimum_yaw=math.radians(args.minimum_yaw_deg),
            minimum_matches=args.minimum_matches,
            minimum_inliers=args.minimum_inliers,
            minimum_inlier_ratio=args.minimum_inlier_ratio,
        )
        all_pairs[spec.trajectory_id] = pairs
        bag_audits[spec.trajectory_id] = audit
        print(
            json.dumps(
                {
                    "stage": "matched_route",
                    "trajectory": spec.trajectory_id,
                    "split": spec.split,
                    "accepted_pairs": len(pairs),
                }
            ),
            flush=True,
        )

    train_specs = [
        spec for spec in raw_pilot.DEFAULT_BAGS if spec.split == "train"
    ]
    val_specs = [
        spec for spec in raw_pilot.DEFAULT_BAGS if spec.split == "val"
    ]
    train_pairs = [
        pair for spec in train_specs for pair in all_pairs[spec.trajectory_id]
    ]
    val_pairs = [
        pair for spec in val_specs for pair in all_pairs[spec.trajectory_id]
    ]
    full_pilot.require(len(train_pairs) >= 12, "insufficient train pairs")
    full_pilot.require(len(val_pairs) >= 12, "insufficient held pairs")

    joint = reduced_pilot.fit_reduced_intrinsic(
        train_pairs,
        args.image_width,
        args.image_height,
    )
    joint_axis = full_pilot.estimate_axis(
        joint["intrinsic"],
        train_pairs,
    )
    held_transfer = bootstrap_axis_transfer(
        joint["intrinsic"],
        val_pairs,
        joint_axis,
        args.bootstrap_replicates,
        rng,
    )
    held_joint = full_pilot.evaluate_geometry(
        joint["intrinsic"],
        joint_axis,
        val_pairs,
    )

    canonical_intrinsic = reduced_pilot.centered_intrinsic(
        math.log(1.0),
        args.image_width,
        args.image_height,
    )
    canonical_axis = full_pilot.estimate_axis(
        canonical_intrinsic,
        train_pairs,
    )
    held_canonical = full_pilot.evaluate_geometry(
        canonical_intrinsic,
        canonical_axis,
        val_pairs,
    )

    per_train_route = {}
    route_focals = []
    route_axes = []
    per_route_angle_transfer_passes = []
    for spec in train_specs:
        route_pairs = all_pairs[spec.trajectory_id]
        fitted = reduced_pilot.fit_reduced_intrinsic(
            route_pairs,
            args.image_width,
            args.image_height,
        )
        axis = full_pilot.estimate_axis(
            fitted["intrinsic"],
            route_pairs,
        )
        held = full_pilot.evaluate_geometry(
            fitted["intrinsic"],
            axis,
            val_pairs,
        )
        improvement = 1.0 - (
            held["median_rotation_angle_error_deg"]
            / max(
                held_canonical["median_rotation_angle_error_deg"],
                1.0e-8,
            )
        )
        transfer = bootstrap_axis_transfer(
            fitted["intrinsic"],
            val_pairs,
            axis,
            args.bootstrap_replicates,
            rng,
        )
        focal = float(fitted["intrinsic"][0, 0])
        route_focals.append(focal)
        route_axes.append(axis)
        per_route_angle_transfer_passes.append(
            held["median_rotation_angle_error_deg"] <= 3.0
            and improvement >= 0.20
        )
        per_train_route[spec.trajectory_id] = {
            "train_pairs": len(route_pairs),
            "focal_px": focal,
            "axis_camera": axis,
            "held_evaluation": held,
            "held_angle_improvement_over_canonical": improvement,
            "held_aggregate_axis_transfer": transfer,
        }

    focal_route_relative_difference = abs(
        route_focals[0] - route_focals[1]
    ) / max(0.5 * (route_focals[0] + route_focals[1]), 1.0e-8)
    train_route_axis_difference = axis_angle_deg(
        route_axes[0],
        route_axes[1],
    )
    joint_angle_improvement = 1.0 - (
        held_joint["median_rotation_angle_error_deg"]
        / max(
            held_canonical["median_rotation_angle_error_deg"],
            1.0e-8,
        )
    )
    checks = {
        "minimum_pairs_each_train_route": all(
            len(all_pairs[spec.trajectory_id]) >= 8
            for spec in train_specs
        ),
        "minimum_held_pairs": len(val_pairs) >= 12,
        "optimizer_not_on_bound": (
            joint["minimum_parameter_distance_to_bound"] > 1.0e-3
        ),
        "cross_route_focal_stability": (
            focal_route_relative_difference <= 0.20
        ),
        "cross_route_aggregate_axis_at_most_15deg": (
            train_route_axis_difference <= 15.0
        ),
        "held_aggregate_axis_at_most_15deg": (
            held_transfer["held_to_train_axis_deg"] <= 15.0
        ),
        "held_axis_bootstrap_p95_at_most_15deg": (
            held_transfer["bootstrap_p95_deg"] <= 15.0
        ),
        "held_rotation_median_at_most_3deg": (
            held_joint["median_rotation_angle_error_deg"] <= 3.0
        ),
        "held_angle_improves_canonical_by_20pct": (
            joint_angle_improvement >= 0.20
        ),
        "each_train_route_transfers_rotation_angle": all(
            per_route_angle_transfer_passes
        ),
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_ROTATION_ONLY_GEOMETRY_PROXY"
            if go
            else "NO_GO_ROTATION_ONLY_GEOMETRY_PROXY"
        ),
        "decision": (
            "test this fixed rotation proxy only as a latent-warp prior"
            if go
            else "bag-only geometry is not stable enough for a latent-warp prior"
        ),
        "scope": {
            "full_camera_calibration": False,
            "translation_extrinsics": False,
            "rotation_only_proxy": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
        },
        "bags": bag_audits,
        "joint_train_fit": {
            "pairs": len(train_pairs),
            "focal_px": float(joint["intrinsic"][0, 0]),
            "focal_over_width": float(
                joint["intrinsic"][0, 0] / args.image_width
            ),
            "principal_point_fixed": [
                0.5 * args.image_width,
                0.5 * args.image_height,
            ],
            "axis_camera": joint_axis,
            "fit": {
                key: value
                for key, value in joint.items()
                if key not in {"parameters", "intrinsic"}
            },
        },
        "held_route": {
            "pairs": len(val_pairs),
            "fitted_geometry": held_joint,
            "canonical_pixel_geometry": held_canonical,
            "median_angle_error_relative_improvement": (
                joint_angle_improvement
            ),
            "aggregate_axis_transfer": held_transfer,
        },
        "per_train_route_transfer": per_train_route,
        "stability": {
            "focal_route_relative_difference": (
                focal_route_relative_difference
            ),
            "train_route_aggregate_axis_difference_deg": (
                train_route_axis_difference
            ),
        },
        "checks": checks,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "formal_v70_modified": False,
        "frozen_highres_dual_modified": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, result)
    print(
        json.dumps(
            jsonable(
                {
                    "status": result["status"],
                    "checks": checks,
                    "joint_train_fit": result["joint_train_fit"],
                    "held_route": result["held_route"],
                    "stability": result["stability"],
                    "result": str(result_path),
                    "result_sha256": sha256_file(result_path),
                    "runtime_seconds": result["runtime_seconds"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
