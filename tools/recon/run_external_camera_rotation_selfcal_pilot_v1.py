#!/usr/bin/env python3
"""Cross-route observability test for external-camera rotational self-calibration.

The pilot fits one effective pinhole intrinsic matrix from uncalibrated
fundamental matrices and odometry yaw magnitudes on two development bags.  It
then tests rotation-angle and camera-frame yaw-axis consistency on a held
route.  Translation extrinsics are intentionally not estimated because planar
monocular motion does not make the full camera-to-body transform reliably
observable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import run_raw_resolution_paired_pilot_v1 as raw_pilot  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_rotation_selfcal_pilot_v1/result.json"
)


@dataclass(frozen=True)
class Pair:
    trajectory_id: str
    first: int
    second: int
    fundamental: np.ndarray
    delta_yaw: float
    translation_m: float
    matches: int
    inliers: int
    inlier_ratio: float


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
    parser.add_argument("--seed", type=int, default=2026072404)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def wrap_to_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def preprocess(image: Image.Image, width: int, height: int) -> np.ndarray:
    cropped = raw_pilot.center_crop_4_3(image.convert("RGB"))
    resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)


def fundamental_pairs(
    spec: raw_pilot.BagSpec,
    *,
    sample_rate: float,
    frame_gap: int,
    width: int,
    height: int,
    minimum_yaw: float,
    minimum_matches: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
) -> tuple[list[Pair], dict[str, Any]]:
    segment, source_audit = raw_pilot.read_longest_forward_segment(
        spec.path,
        sample_rate,
    )
    images = [preprocess(row[0], width, height) for row in segment]
    positions = np.stack([row[1] for row in segment]).astype(np.float64)
    yaw = np.unwrap(np.asarray([row[2] for row in segment], dtype=np.float64))
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.025)
    features = [sift.detectAndCompute(image, None) for image in images]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    attempted = 0
    yaw_eligible = 0
    rejected_matches = 0
    rejected_fundamental = 0
    rejected_inliers = 0
    pairs = []
    for first in range(0, len(images) - frame_gap):
        second = first + frame_gap
        delta_yaw = wrap_to_pi(float(yaw[second] - yaw[first]))
        if abs(delta_yaw) < minimum_yaw:
            continue
        yaw_eligible += 1
        keypoints_first, descriptors_first = features[first]
        keypoints_second, descriptors_second = features[second]
        if descriptors_first is None or descriptors_second is None:
            rejected_matches += 1
            continue
        attempted += 1
        neighbors = matcher.knnMatch(
            descriptors_first,
            descriptors_second,
            k=2,
        )
        accepted = [
            first_match
            for first_match, second_match in neighbors
            if first_match.distance < 0.75 * second_match.distance
        ]
        if len(accepted) < minimum_matches:
            rejected_matches += 1
            continue
        points_first = np.float32(
            [keypoints_first[match.queryIdx].pt for match in accepted]
        )
        points_second = np.float32(
            [keypoints_second[match.trainIdx].pt for match in accepted]
        )
        fundamental, mask = cv2.findFundamentalMat(
            points_first,
            points_second,
            cv2.USAC_MAGSAC,
            1.0,
            0.999,
            10000,
        )
        if fundamental is None or fundamental.shape != (3, 3) or mask is None:
            rejected_fundamental += 1
            continue
        inliers = int(np.asarray(mask).reshape(-1).astype(bool).sum())
        inlier_ratio = inliers / len(accepted)
        if inliers < minimum_inliers or inlier_ratio < minimum_inlier_ratio:
            rejected_inliers += 1
            continue
        fundamental = np.asarray(fundamental, dtype=np.float64)
        norm = float(np.linalg.norm(fundamental))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            rejected_fundamental += 1
            continue
        pairs.append(
            Pair(
                trajectory_id=spec.trajectory_id,
                first=first,
                second=second,
                fundamental=fundamental / norm,
                delta_yaw=delta_yaw,
                translation_m=float(
                    np.linalg.norm(positions[second] - positions[first])
                ),
                matches=len(accepted),
                inliers=inliers,
                inlier_ratio=float(inlier_ratio),
            )
        )
    return pairs, {
        **source_audit,
        "trajectory_id": spec.trajectory_id,
        "split": spec.split,
        "bag": str(spec.path),
        "bag_sha256": sha256_file(spec.path),
        "frames": len(images),
        "frame_gap": frame_gap,
        "yaw_eligible_pairs": yaw_eligible,
        "feature_pairs_attempted": attempted,
        "accepted_pairs": len(pairs),
        "rejected_matches": rejected_matches,
        "rejected_fundamental": rejected_fundamental,
        "rejected_inliers": rejected_inliers,
        "median_matches": (
            float(np.median([pair.matches for pair in pairs])) if pairs else None
        ),
        "median_inliers": (
            float(np.median([pair.inliers for pair in pairs])) if pairs else None
        ),
        "median_inlier_ratio": (
            float(np.median([pair.inlier_ratio for pair in pairs]))
            if pairs
            else None
        ),
        "median_abs_yaw_deg": (
            float(np.degrees(np.median([abs(pair.delta_yaw) for pair in pairs])))
            if pairs
            else None
        ),
        "median_translation_m": (
            float(np.median([pair.translation_m for pair in pairs]))
            if pairs
            else None
        ),
    }


def intrinsic_from_parameters(
    parameters: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    common_focal = math.exp(float(parameters[0])) * width
    aspect = math.exp(float(parameters[1]))
    fx = common_focal * math.sqrt(aspect)
    fy = common_focal / math.sqrt(aspect)
    cx = width * (0.5 + float(parameters[2]))
    cy = height * (0.5 + float(parameters[3]))
    return np.asarray(
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def rotation_candidates(essential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, second, _ = cv2.decomposeEssentialMat(essential)
    return np.asarray(first), np.asarray(second)


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def pair_residuals(
    parameters: np.ndarray,
    pair: Pair,
    width: int,
    height: int,
) -> np.ndarray:
    intrinsic = intrinsic_from_parameters(parameters, width, height)
    essential = intrinsic.T @ pair.fundamental @ intrinsic
    singular = np.linalg.svd(essential, compute_uv=False)
    denominator = max(float(singular[0] + singular[1]), 1.0e-12)
    equal_singular = 2.0 * float(singular[0] - singular[1]) / denominator
    rank_two = float(singular[2]) / max(float(singular[1]), 1.0e-12)
    rotations = rotation_candidates(essential)
    target = abs(pair.delta_yaw)
    angle_error = min(
        abs(rotation_angle(rotation) - target) for rotation in rotations
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


def fit_intrinsic(
    pairs: list[Pair],
    width: int,
    height: int,
) -> dict[str, Any]:
    require(bool(pairs), "cannot fit without pairs")
    lower = np.asarray(
        (math.log(0.30), math.log(0.70), -0.15, -0.15),
        dtype=np.float64,
    )
    upper = np.asarray(
        (math.log(3.00), math.log(1.40), 0.15, 0.15),
        dtype=np.float64,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                pair_residuals(parameters, pair, width, height)
                for pair in pairs
            ]
        )

    starts = (
        (0.50, 1.0),
        (0.75, 1.0),
        (1.00, 1.0),
        (1.50, 1.0),
        (2.00, 1.0),
        (1.00, 1.15),
        (1.00, 0.85),
    )
    candidates = []
    for focal_ratio, aspect in starts:
        initial = np.asarray(
            (math.log(focal_ratio), math.log(aspect), 0.0, 0.0),
            dtype=np.float64,
        )
        fitted = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=400,
        )
        candidates.append(fitted)
    selected = min(candidates, key=lambda value: float(value.cost))
    intrinsic = intrinsic_from_parameters(selected.x, width, height)
    distance_to_bounds = np.minimum(selected.x - lower, upper - selected.x)
    return {
        "parameters": selected.x,
        "intrinsic": intrinsic,
        "cost": float(selected.cost),
        "optimality": float(selected.optimality),
        "nfev": int(selected.nfev),
        "success": bool(selected.success),
        "message": str(selected.message),
        "minimum_parameter_distance_to_bound": float(distance_to_bounds.min()),
        "starts": len(starts),
    }


def selected_rotation(
    intrinsic: np.ndarray,
    pair: Pair,
) -> tuple[np.ndarray, float]:
    essential = intrinsic.T @ pair.fundamental @ intrinsic
    rotations = rotation_candidates(essential)
    target = abs(pair.delta_yaw)
    rotation = min(
        rotations,
        key=lambda value: abs(rotation_angle(value) - target),
    )
    return rotation, rotation_angle(rotation)


def estimate_axis(
    intrinsic: np.ndarray,
    pairs: list[Pair],
) -> np.ndarray:
    axes = []
    weights = []
    for pair in pairs:
        rotation, _ = selected_rotation(intrinsic, pair)
        rotation_vector = cv2.Rodrigues(rotation)[0].reshape(3)
        if pair.delta_yaw < 0:
            rotation_vector = -rotation_vector
        norm = float(np.linalg.norm(rotation_vector))
        if norm > 1.0e-8:
            axes.append(rotation_vector / norm)
            weights.append(float(pair.inliers))
    require(bool(axes), "rotation axes unavailable")
    reference = axes[0]
    aligned = [
        axis if float(np.dot(axis, reference)) >= 0.0 else -axis
        for axis in axes
    ]
    mean = np.average(np.stack(aligned), axis=0, weights=np.asarray(weights))
    return mean / np.linalg.norm(mean)


def evaluate_geometry(
    intrinsic: np.ndarray,
    axis: np.ndarray,
    pairs: list[Pair],
) -> dict[str, Any]:
    angle_errors = []
    axis_errors = []
    equal_residuals = []
    rank_residuals = []
    selected_angles = []
    for pair in pairs:
        essential = intrinsic.T @ pair.fundamental @ intrinsic
        singular = np.linalg.svd(essential, compute_uv=False)
        denominator = max(float(singular[0] + singular[1]), 1.0e-12)
        equal_residuals.append(
            2.0 * float(abs(singular[0] - singular[1])) / denominator
        )
        rank_residuals.append(
            float(singular[2]) / max(float(singular[1]), 1.0e-12)
        )
        rotation, angle = selected_rotation(intrinsic, pair)
        selected_angles.append(angle)
        angle_errors.append(abs(angle - abs(pair.delta_yaw)))
        vector = cv2.Rodrigues(rotation)[0].reshape(3)
        if pair.delta_yaw < 0:
            vector = -vector
        vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
        cosine = float(np.clip(abs(np.dot(vector, axis)), -1.0, 1.0))
        axis_errors.append(math.acos(cosine))
    angle_deg = np.degrees(np.asarray(angle_errors))
    axis_deg = np.degrees(np.asarray(axis_errors))
    return {
        "pairs": len(pairs),
        "median_rotation_angle_error_deg": float(np.median(angle_deg)),
        "p90_rotation_angle_error_deg": float(np.quantile(angle_deg, 0.90)),
        "median_axis_error_deg": float(np.median(axis_deg)),
        "p90_axis_error_deg": float(np.quantile(axis_deg, 0.90)),
        "median_equal_singular_residual": float(np.median(equal_residuals)),
        "median_rank_two_residual": float(np.median(rank_residuals)),
        "median_recovered_rotation_deg": float(
            np.degrees(np.median(selected_angles))
        ),
        "median_odom_yaw_deg": float(
            np.degrees(np.median([abs(pair.delta_yaw) for pair in pairs]))
        ),
    }


def intrinsic_summary(intrinsic: np.ndarray, width: int, height: int) -> dict[str, float]:
    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "fx_over_width": float(intrinsic[0, 0] / width),
        "fy_over_height": float(intrinsic[1, 1] / height),
    }


def main() -> None:
    args = parse_args()
    result_path = args.output.resolve()
    require(not result_path.exists(), f"refusing to overwrite {result_path}")
    require(args.frame_gap >= 1, "frame gap must be positive")
    require(args.image_width > 0 and args.image_height > 0, "invalid image size")
    cv2.setNumThreads(max(1, min(8, cv2.getNumberOfCPUs())))
    np.random.seed(args.seed)
    started = time.perf_counter()
    all_pairs: dict[str, list[Pair]] = {}
    bag_audits = {}
    for spec in raw_pilot.DEFAULT_BAGS:
        pairs, audit = fundamental_pairs(
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
                    "median_inliers": audit["median_inliers"],
                    "median_abs_yaw_deg": audit["median_abs_yaw_deg"],
                }
            ),
            flush=True,
        )
    train_specs = [spec for spec in raw_pilot.DEFAULT_BAGS if spec.split == "train"]
    val_specs = [spec for spec in raw_pilot.DEFAULT_BAGS if spec.split == "val"]
    train_pairs = [
        pair for spec in train_specs for pair in all_pairs[spec.trajectory_id]
    ]
    val_pairs = [
        pair for spec in val_specs for pair in all_pairs[spec.trajectory_id]
    ]
    require(len(train_pairs) >= 12, "insufficient training rotation pairs")
    require(len(val_pairs) >= 12, "insufficient held-route rotation pairs")
    joint = fit_intrinsic(train_pairs, args.image_width, args.image_height)
    joint_axis = estimate_axis(joint["intrinsic"], train_pairs)
    route_fits = {}
    for spec in train_specs:
        pairs = all_pairs[spec.trajectory_id]
        require(len(pairs) >= 6, f"insufficient pairs: {spec.trajectory_id}")
        fitted = fit_intrinsic(pairs, args.image_width, args.image_height)
        axis = estimate_axis(fitted["intrinsic"], pairs)
        route_fits[spec.trajectory_id] = {
            "intrinsic": intrinsic_summary(
                fitted["intrinsic"],
                args.image_width,
                args.image_height,
            ),
            "axis_camera": axis,
            "fit": {
                key: value
                for key, value in fitted.items()
                if key not in {"parameters", "intrinsic"}
            },
            "self_evaluation": evaluate_geometry(
                fitted["intrinsic"],
                axis,
                pairs,
            ),
        }
    canonical_intrinsic = np.asarray(
        (
            (args.image_width, 0.0, 0.5 * args.image_width),
            (0.0, args.image_width, 0.5 * args.image_height),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    canonical_axis = estimate_axis(canonical_intrinsic, train_pairs)
    held_joint = evaluate_geometry(joint["intrinsic"], joint_axis, val_pairs)
    held_canonical = evaluate_geometry(
        canonical_intrinsic,
        canonical_axis,
        val_pairs,
    )
    first_intrinsic = route_fits[train_specs[0].trajectory_id]["intrinsic"]
    second_intrinsic = route_fits[train_specs[1].trajectory_id]["intrinsic"]
    focal_route_relative_difference = max(
        abs(first_intrinsic["fx"] - second_intrinsic["fx"])
        / max(0.5 * (first_intrinsic["fx"] + second_intrinsic["fx"]), 1.0e-8),
        abs(first_intrinsic["fy"] - second_intrinsic["fy"])
        / max(0.5 * (first_intrinsic["fy"] + second_intrinsic["fy"]), 1.0e-8),
    )
    principal_route_difference_fraction = math.hypot(
        first_intrinsic["cx"] - second_intrinsic["cx"],
        first_intrinsic["cy"] - second_intrinsic["cy"],
    ) / math.hypot(args.image_width, args.image_height)
    angle_improvement = 1.0 - (
        held_joint["median_rotation_angle_error_deg"]
        / max(held_canonical["median_rotation_angle_error_deg"], 1.0e-8)
    )
    checks = {
        "minimum_pairs_each_train_route": all(
            len(all_pairs[spec.trajectory_id]) >= 8 for spec in train_specs
        ),
        "minimum_held_pairs": len(val_pairs) >= 12,
        "optimizer_not_on_bound": (
            joint["minimum_parameter_distance_to_bound"] > 1.0e-3
        ),
        "cross_route_focal_stability": focal_route_relative_difference <= 0.20,
        "cross_route_principal_stability": (
            principal_route_difference_fraction <= 0.08
        ),
        "held_rotation_median_at_most_3deg": (
            held_joint["median_rotation_angle_error_deg"] <= 3.0
        ),
        "held_axis_median_at_most_15deg": (
            held_joint["median_axis_error_deg"] <= 15.0
        ),
        "held_angle_improves_canonical_by_20pct": angle_improvement >= 0.20,
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_ROTATIONAL_SELFCAL_OBSERVABLE"
            if go
            else "NO_GO_ROTATIONAL_SELFCAL_UNSTABLE"
        ),
        "decision": (
            "fit effective rotation geometry before LTWM warp pilot"
            if go
            else "do not use bag-only self-calibration as camera geometry"
        ),
        "protocol": {
            "image_size": [args.image_width, args.image_height],
            "sample_rate_hz": args.sample_rate,
            "frame_gap": args.frame_gap,
            "minimum_yaw_deg": args.minimum_yaw_deg,
            "feature": "SIFT ratio 0.75",
            "fundamental": "OpenCV USAC_MAGSAC threshold 1px",
            "fit": (
                "shared zero-skew effective pinhole K using essential singular "
                "constraints plus odometry-yaw rotation magnitude"
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
            "intrinsic": intrinsic_summary(
                joint["intrinsic"],
                args.image_width,
                args.image_height,
            ),
            "axis_camera": joint_axis,
            "fit": {
                key: value
                for key, value in joint.items()
                if key not in {"parameters", "intrinsic"}
            },
            "train_evaluation": evaluate_geometry(
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
            "focal_route_relative_difference": focal_route_relative_difference,
            "principal_route_difference_fraction_of_diagonal": (
                principal_route_difference_fraction
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
            {
                "status": result["status"],
                "checks": checks,
                "joint_intrinsic": result["joint_train_fit"]["intrinsic"],
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
