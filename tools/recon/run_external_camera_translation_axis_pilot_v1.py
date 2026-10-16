#!/usr/bin/env python3
"""Held-route observability test for the camera-frame body-forward axis.

The effective focal length and body-yaw axis come from the already frozen
rotation-only proxy.  For each forward-motion image pair, essential-matrix
cheirality recovers the camera-center translation direction.  Odom-derived
lateral heading is removed around the fixed yaw axis, yielding an estimate of
the body-forward axis in camera coordinates.  The aggregate axis is fitted on
two development routes and evaluated once on the original held route.

Only extrinsic orientation is tested.  Translation lever arm and metric depth
are not estimated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

import run_external_camera_rotation_selfcal_pilot_v1 as rotation_pilot
import run_raw_resolution_paired_pilot_v1 as raw_pilot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_rotation_proxy_validation_v1/result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "external_camera_translation_axis_pilot_v1/result.json"
)


@dataclass(frozen=True)
class TranslationAxisPair:
    trajectory_id: str
    first: int
    second: int
    forward_axis_camera: np.ndarray
    camera_center_axis: np.ndarray
    body_translation_angle: float
    body_translation_m: float
    inliers: int
    cheirality_inliers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=float, default=4.0)
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--minimum-translation-m", type=float, default=0.20)
    parser.add_argument("--minimum-matches", type=int, default=40)
    parser.add_argument("--minimum-inliers", type=int, default=24)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026072407)
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


def wrap_to_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def axis_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return cv2.Rodrigues(
        np.asarray(axis, dtype=np.float64).reshape(3, 1) * angle
    )[0]


def weighted_axis(
    axes: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if reference is None:
        reference = axes[0]
    aligned = axes.copy()
    opposite = np.dot(aligned, reference) < 0.0
    aligned[opposite] *= -1.0
    mean = np.average(aligned, axis=0, weights=weights)
    mean /= max(float(np.linalg.norm(mean)), 1.0e-12)
    return mean, float(opposite.mean())


def translation_pairs(
    spec: raw_pilot.BagSpec,
    *,
    sample_rate: float,
    frame_gap: int,
    width: int,
    height: int,
    intrinsic: np.ndarray,
    yaw_axis: np.ndarray,
    minimum_translation: float,
    minimum_matches: int,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
) -> tuple[list[TranslationAxisPair], dict[str, Any]]:
    segment, source_audit = raw_pilot.read_longest_forward_segment(
        spec.path,
        sample_rate,
    )
    images = [
        rotation_pilot.preprocess(value[0], width, height)
        for value in segment
    ]
    positions = np.stack([value[1] for value in segment]).astype(np.float64)
    yaw = np.asarray([value[2] for value in segment], dtype=np.float64)
    sift = cv2.SIFT_create(nfeatures=2500)
    features = [sift.detectAndCompute(image, None) for image in images]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    attempted = 0
    rejected_motion = 0
    rejected_matches = 0
    rejected_fundamental = 0
    rejected_cheirality = 0
    for first in range(0, len(images) - frame_gap):
        second = first + frame_gap
        delta_global = positions[second] - positions[first]
        cosine = math.cos(yaw[first])
        sine = math.sin(yaw[first])
        dx = cosine * delta_global[0] + sine * delta_global[1]
        dy = -sine * delta_global[0] + cosine * delta_global[1]
        translation = math.hypot(dx, dy)
        if translation < minimum_translation or dx <= 0.0:
            rejected_motion += 1
            continue
        attempted += 1
        keypoints_first, descriptors_first = features[first]
        keypoints_second, descriptors_second = features[second]
        if descriptors_first is None or descriptors_second is None:
            rejected_matches += 1
            continue
        knn = matcher.knnMatch(
            descriptors_first,
            descriptors_second,
            k=2,
        )
        accepted = [
            first_match
            for first_match, second_match in knn
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
        if fundamental is None or mask is None or fundamental.shape != (3, 3):
            rejected_fundamental += 1
            continue
        inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        if (
            inliers < minimum_inliers
            or inliers / len(accepted) < minimum_inlier_ratio
        ):
            rejected_fundamental += 1
            continue
        inlier_first = points_first[inlier_mask]
        inlier_second = points_second[inlier_mask]
        essential = intrinsic.T @ np.asarray(fundamental) @ intrinsic
        cheirality, recovered_rotation, recovered_translation, _ = (
            cv2.recoverPose(
                essential,
                inlier_first,
                inlier_second,
                intrinsic,
            )
        )
        if cheirality < minimum_inliers:
            rejected_cheirality += 1
            continue
        camera_center = (
            -np.asarray(recovered_rotation).T
            @ np.asarray(recovered_translation).reshape(3)
        )
        camera_center /= max(
            float(np.linalg.norm(camera_center)),
            1.0e-12,
        )
        planar_center = camera_center - np.dot(
            camera_center,
            yaw_axis,
        ) * yaw_axis
        norm = float(np.linalg.norm(planar_center))
        if norm <= 1.0e-6:
            rejected_cheirality += 1
            continue
        planar_center /= norm
        body_angle = math.atan2(dy, dx)
        forward_axis = (
            axis_rotation(yaw_axis, -body_angle) @ planar_center
        )
        forward_axis -= np.dot(forward_axis, yaw_axis) * yaw_axis
        forward_axis /= max(
            float(np.linalg.norm(forward_axis)),
            1.0e-12,
        )
        pairs.append(
            TranslationAxisPair(
                trajectory_id=spec.trajectory_id,
                first=first,
                second=second,
                forward_axis_camera=forward_axis,
                camera_center_axis=camera_center,
                body_translation_angle=body_angle,
                body_translation_m=translation,
                inliers=inliers,
                cheirality_inliers=int(cheirality),
            )
        )
    return pairs, {
        **source_audit,
        "trajectory_id": spec.trajectory_id,
        "split": spec.split,
        "bag": str(spec.path),
        "bag_sha256": sha256_file(spec.path),
        "frames": len(segment),
        "pairs_attempted": attempted,
        "accepted_pairs": len(pairs),
        "rejected_motion": rejected_motion,
        "rejected_matches": rejected_matches,
        "rejected_fundamental": rejected_fundamental,
        "rejected_cheirality": rejected_cheirality,
        "median_translation_m": (
            float(np.median([value.body_translation_m for value in pairs]))
            if pairs
            else None
        ),
        "median_abs_body_translation_angle_deg": (
            float(
                np.degrees(
                    np.median(
                        [
                            abs(value.body_translation_angle)
                            for value in pairs
                        ]
                    )
                )
            )
            if pairs
            else None
        ),
        "median_inliers": (
            float(np.median([value.inliers for value in pairs]))
            if pairs
            else None
        ),
        "median_cheirality_inliers": (
            float(
                np.median([value.cheirality_inliers for value in pairs])
            )
            if pairs
            else None
        ),
    }


def aggregate(
    pairs: list[TranslationAxisPair],
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    axes = np.stack([pair.forward_axis_camera for pair in pairs])
    weights = np.asarray(
        [pair.cheirality_inliers for pair in pairs],
        dtype=np.float64,
    )
    axis, opposite_fraction = weighted_axis(axes, weights, reference)
    deviations = np.asarray(
        [axis_angle_deg(value, axis) for value in axes],
        dtype=np.float64,
    )
    return {
        "axis": axis,
        "opposite_alignment_fraction": opposite_fraction,
        "pair_axis_median_deviation_deg": float(
            np.median(deviations)
        ),
        "pair_axis_p90_deviation_deg": float(
            np.quantile(deviations, 0.90)
        ),
    }


def bootstrap_transfer(
    pairs: list[TranslationAxisPair],
    train_axis: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    axes = np.stack([pair.forward_axis_camera for pair in pairs])
    axes[np.dot(axes, train_axis) < 0.0] *= -1.0
    weights = np.asarray(
        [pair.cheirality_inliers for pair in pairs],
        dtype=np.float64,
    )
    held_axis, _ = weighted_axis(axes, weights, train_axis)
    errors = []
    for _ in range(replicates):
        indices = rng.integers(0, len(axes), size=len(axes))
        sampled, _ = weighted_axis(
            axes[indices],
            weights[indices],
            train_axis,
        )
        errors.append(axis_angle_deg(sampled, train_axis))
    errors = np.asarray(errors)
    return {
        "held_axis": held_axis,
        "held_to_train_axis_deg": axis_angle_deg(
            held_axis,
            train_axis,
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_median_deg": float(np.median(errors)),
        "bootstrap_p95_deg": float(np.quantile(errors, 0.95)),
        "bootstrap_probability_at_most_15deg": float(
            np.mean(errors <= 15.0)
        ),
    }


def main() -> None:
    args = parse_args()
    geometry_path = args.geometry.resolve()
    result_path = args.output.resolve()
    rotation_pilot.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    rotation_pilot.require(
        geometry_path.is_file(),
        f"missing geometry result: {geometry_path}",
    )
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    rotation_pilot.require(
        geometry["status"] == "GO_ROTATION_ONLY_GEOMETRY_PROXY",
        "rotation proxy did not pass",
    )
    rotation_pilot.require(
        args.bootstrap_replicates >= 200,
        "need at least 200 bootstrap replicates",
    )
    yaw_axis = np.asarray(
        geometry["joint_train_fit"]["axis_camera"],
        dtype=np.float64,
    )
    yaw_axis /= np.linalg.norm(yaw_axis)
    intrinsic = np.asarray(
        (
            (
                float(geometry["joint_train_fit"]["focal_px"]),
                0.0,
                0.5 * args.image_width,
            ),
            (
                0.0,
                float(geometry["joint_train_fit"]["focal_px"]),
                0.5 * args.image_height,
            ),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    cv2.setNumThreads(max(1, min(8, cv2.getNumberOfCPUs())))
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    all_pairs = {}
    audits = {}
    for spec in raw_pilot.DEFAULT_BAGS:
        pairs, audit = translation_pairs(
            spec,
            sample_rate=args.sample_rate,
            frame_gap=args.frame_gap,
            width=args.image_width,
            height=args.image_height,
            intrinsic=intrinsic,
            yaw_axis=yaw_axis,
            minimum_translation=args.minimum_translation_m,
            minimum_matches=args.minimum_matches,
            minimum_inliers=args.minimum_inliers,
            minimum_inlier_ratio=args.minimum_inlier_ratio,
        )
        all_pairs[spec.trajectory_id] = pairs
        audits[spec.trajectory_id] = audit
        print(
            json.dumps(
                {
                    "stage": "translation_axis_route",
                    "trajectory": spec.trajectory_id,
                    "split": spec.split,
                    "accepted_pairs": len(pairs),
                    "median_translation_m": audit[
                        "median_translation_m"
                    ],
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
    rotation_pilot.require(
        len(train_pairs) >= 12,
        "insufficient training translation pairs",
    )
    rotation_pilot.require(
        len(val_pairs) >= 12,
        "insufficient held translation pairs",
    )
    joint = aggregate(train_pairs)
    route = {
        spec.trajectory_id: aggregate(
            all_pairs[spec.trajectory_id],
            joint["axis"],
        )
        for spec in train_specs
    }
    route_axes = [
        np.asarray(route[spec.trajectory_id]["axis"])
        for spec in train_specs
    ]
    train_route_difference = axis_angle_deg(
        route_axes[0],
        route_axes[1],
    )
    held = aggregate(val_pairs, joint["axis"])
    held_transfer = bootstrap_transfer(
        val_pairs,
        joint["axis"],
        args.bootstrap_replicates,
        rng,
    )
    lateral_axis = np.cross(yaw_axis, joint["axis"])
    lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1.0e-12)
    orientation = np.stack(
        (joint["axis"], lateral_axis, yaw_axis),
        axis=1,
    )
    determinant = float(np.linalg.det(orientation))
    checks = {
        "minimum_pairs_each_train_route": all(
            len(all_pairs[spec.trajectory_id]) >= 8
            for spec in train_specs
        ),
        "minimum_held_pairs": len(val_pairs) >= 12,
        "train_route_axis_difference_at_most_15deg": (
            train_route_difference <= 15.0
        ),
        "held_axis_difference_at_most_15deg": (
            held_transfer["held_to_train_axis_deg"] <= 15.0
        ),
        "held_axis_bootstrap_p95_at_most_15deg": (
            held_transfer["bootstrap_p95_deg"] <= 15.0
        ),
        "held_opposite_alignment_fraction_at_most_20pct": (
            held["opposite_alignment_fraction"] <= 0.20
        ),
        "orientation_is_right_handed": determinant >= 0.99,
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_TRANSLATION_ORIENTATION_PROXY"
            if go
            else "NO_GO_TRANSLATION_ORIENTATION_PROXY"
        ),
        "decision": (
            "depth-aware SE3 latent transport is observable enough for a pilot"
            if go
            else "do not build translation latent transport without calibration"
        ),
        "scope": {
            "camera_to_body_rotation_proxy": True,
            "translation_lever_arm_estimated": False,
            "metric_depth_estimated": False,
            "full_camera_calibration": False,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
        },
        "inputs": {
            "rotation_geometry": {
                "path": str(geometry_path),
                "sha256": sha256_file(geometry_path),
            },
            "intrinsic": intrinsic,
            "yaw_axis_camera": yaw_axis,
        },
        "bags": audits,
        "joint_train": {
            "pairs": len(train_pairs),
            **joint,
            "lateral_axis_camera": lateral_axis,
            "camera_from_body_rotation": orientation,
            "rotation_determinant": determinant,
        },
        "per_train_route": route,
        "held_route": {
            "pairs": len(val_pairs),
            **held,
            "transfer": held_transfer,
        },
        "stability": {
            "train_route_axis_difference_deg": (
                train_route_difference
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
                    "joint_train": result["joint_train"],
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
