#!/usr/bin/env python3
"""Held-route test of the observable part of external-camera self-calibration.

Planar robot motion does not reliably identify a free principal point and
aspect ratio.  This follow-up therefore fixes the principal point at the
known center crop, assumes square pixels, and estimates only one effective
focal length from development routes.  The fitted geometry is evaluated once
on the same held route without loading outcome labels.
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
from scipy.optimize import least_squares

import run_external_camera_rotation_selfcal_pilot_v1 as full_pilot
import run_raw_resolution_paired_pilot_v1 as raw_pilot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_reduced_selfcal_pilot_v1/result.json"
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
    parser.add_argument("--seed", type=int, default=2026072405)
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


def centered_intrinsic(
    log_focal_ratio: float,
    width: int,
    height: int,
) -> np.ndarray:
    focal = math.exp(float(log_focal_ratio)) * width
    return np.asarray(
        (
            (focal, 0.0, 0.5 * width),
            (0.0, focal, 0.5 * height),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def reduced_pair_residuals(
    parameters: np.ndarray,
    pair: full_pilot.Pair,
    width: int,
    height: int,
) -> np.ndarray:
    intrinsic = centered_intrinsic(parameters[0], width, height)
    essential = intrinsic.T @ pair.fundamental @ intrinsic
    singular = np.linalg.svd(essential, compute_uv=False)
    denominator = max(float(singular[0] + singular[1]), 1.0e-12)
    equal_singular = 2.0 * float(singular[0] - singular[1]) / denominator
    rank_two = float(singular[2]) / max(float(singular[1]), 1.0e-12)
    rotations = full_pilot.rotation_candidates(essential)
    target = abs(pair.delta_yaw)
    angle_error = min(
        abs(full_pilot.rotation_angle(rotation) - target)
        for rotation in rotations
    )
    weight = float(np.clip(math.sqrt(pair.inliers / 40.0), 0.7, 2.0))
    return weight * np.asarray(
        (
            2.0 * equal_singular,
            2.0 * rank_two,
            angle_error / math.radians(3.0),
        ),
        dtype=np.float64,
    )


def fit_reduced_intrinsic(
    pairs: list[full_pilot.Pair],
    width: int,
    height: int,
) -> dict[str, Any]:
    full_pilot.require(bool(pairs), "cannot fit without pairs")
    lower = np.asarray((math.log(0.30),), dtype=np.float64)
    upper = np.asarray((math.log(3.00),), dtype=np.float64)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                reduced_pair_residuals(parameters, pair, width, height)
                for pair in pairs
            ]
        )

    candidates = []
    for focal_ratio in (0.35, 0.50, 0.75, 1.00, 1.50, 2.00):
        fitted = least_squares(
            residual,
            np.asarray((math.log(focal_ratio),), dtype=np.float64),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=300,
        )
        candidates.append(fitted)
    selected = min(candidates, key=lambda value: float(value.cost))
    distance_to_bounds = np.minimum(
        selected.x - lower,
        upper - selected.x,
    )
    return {
        "parameters": selected.x,
        "intrinsic": centered_intrinsic(selected.x[0], width, height),
        "cost": float(selected.cost),
        "optimality": float(selected.optimality),
        "nfev": int(selected.nfev),
        "success": bool(selected.success),
        "message": str(selected.message),
        "minimum_parameter_distance_to_bound": float(
            distance_to_bounds.min()
        ),
        "starts": len(candidates),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output.resolve()
    full_pilot.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    cv2.setNumThreads(max(1, min(8, cv2.getNumberOfCPUs())))
    np.random.seed(args.seed)
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
    full_pilot.require(
        len(train_pairs) >= 12,
        "insufficient training rotation pairs",
    )
    full_pilot.require(
        len(val_pairs) >= 12,
        "insufficient held-route rotation pairs",
    )

    joint = fit_reduced_intrinsic(
        train_pairs,
        args.image_width,
        args.image_height,
    )
    joint_axis = full_pilot.estimate_axis(joint["intrinsic"], train_pairs)
    route_fits = {}
    route_focals = []
    route_axes = []
    for spec in train_specs:
        pairs = all_pairs[spec.trajectory_id]
        full_pilot.require(
            len(pairs) >= 6,
            f"insufficient pairs: {spec.trajectory_id}",
        )
        fitted = fit_reduced_intrinsic(
            pairs,
            args.image_width,
            args.image_height,
        )
        axis = full_pilot.estimate_axis(fitted["intrinsic"], pairs)
        focal = float(fitted["intrinsic"][0, 0])
        route_focals.append(focal)
        route_axes.append(axis)
        route_fits[spec.trajectory_id] = {
            "focal_px": focal,
            "focal_over_width": focal / args.image_width,
            "axis_camera": axis,
            "fit": {
                key: value
                for key, value in fitted.items()
                if key not in {"parameters", "intrinsic"}
            },
            "self_evaluation": full_pilot.evaluate_geometry(
                fitted["intrinsic"],
                axis,
                pairs,
            ),
        }

    canonical_intrinsic = centered_intrinsic(
        math.log(1.0),
        args.image_width,
        args.image_height,
    )
    canonical_axis = full_pilot.estimate_axis(
        canonical_intrinsic,
        train_pairs,
    )
    held_joint = full_pilot.evaluate_geometry(
        joint["intrinsic"],
        joint_axis,
        val_pairs,
    )
    held_canonical = full_pilot.evaluate_geometry(
        canonical_intrinsic,
        canonical_axis,
        val_pairs,
    )
    focal_route_relative_difference = abs(
        route_focals[0] - route_focals[1]
    ) / max(0.5 * (route_focals[0] + route_focals[1]), 1.0e-8)
    axis_route_difference_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(np.dot(route_axes[0], route_axes[1])),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    angle_improvement = 1.0 - (
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
        "cross_route_axis_difference_at_most_15deg": (
            axis_route_difference_deg <= 15.0
        ),
        "held_rotation_median_at_most_3deg": (
            held_joint["median_rotation_angle_error_deg"] <= 3.0
        ),
        "held_axis_median_at_most_15deg": (
            held_joint["median_axis_error_deg"] <= 15.0
        ),
        "held_angle_improves_canonical_by_20pct": (
            angle_improvement >= 0.20
        ),
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_REDUCED_ROTATION_GEOMETRY_OBSERVABLE"
            if go
            else "NO_GO_REDUCED_ROTATION_GEOMETRY_UNSTABLE"
        ),
        "decision": (
            "use centered shared-focal rotation geometry in a latent-warp pilot"
            if go
            else "do not infer usable camera geometry from these bags alone"
        ),
        "protocol": {
            "image_size": [args.image_width, args.image_height],
            "sample_rate_hz": args.sample_rate,
            "frame_gap": args.frame_gap,
            "minimum_yaw_deg": args.minimum_yaw_deg,
            "fit": (
                "one shared effective focal length; crop center fixed; "
                "square pixels; odometry yaw used on development routes only"
            ),
            "translation_extrinsics_estimated": False,
            "full_camera_calibration_claimed": False,
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
            "train_evaluation": full_pilot.evaluate_geometry(
                joint["intrinsic"],
                joint_axis,
                train_pairs,
            ),
        },
        "per_train_route_fit": route_fits,
        "held_route": {
            "pairs": len(val_pairs),
            "fitted_geometry": held_joint,
            "canonical_pixel_geometry": held_canonical,
            "median_angle_error_relative_improvement": angle_improvement,
        },
        "stability": {
            "focal_route_relative_difference": (
                focal_route_relative_difference
            ),
            "axis_route_difference_deg": axis_route_difference_deg,
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
            {
                "status": result["status"],
                "checks": checks,
                "joint_train_fit": result["joint_train_fit"],
                "held_route": result["held_route"],
                "stability": result["stability"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
