#!/usr/bin/env python3
"""Generate a conservative, review-only single-goal arrival candidate.

The tool never promotes ``arrival_calibrated``. It emits a candidate route
with that field forced to false, plus evidence and rejection reasons for
manual review and a later held-out standing validation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("far", "approach", "stop", "overshoot", "lookalike")
DANGEROUS_NEGATIVE_LABELS = ("far", "approach", "lookalike")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--candidate-route", required=True, type=Path)
    parser.add_argument("--min-per-label", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(destination)


def verify_record_hash(record: dict[str, Any]) -> bool:
    expected = str(record.get("record_sha256", ""))
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == expected


def load_samples(path: Path) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        sample = json.loads(line)
        if not isinstance(sample, dict):
            raise ValueError(f"sample {line_number} is not an object")
        if sample.get("schema") != "v70_arrival_sample_v1":
            raise ValueError(f"sample {line_number} has wrong schema")
        if sample.get("operator_label") not in LABELS:
            raise ValueError(f"sample {line_number} has invalid label")
        if not sample.get("operator_confirmed_standing_stationary"):
            raise ValueError(f"sample {line_number} was not standing")
        if not verify_record_hash(sample):
            raise ValueError(f"sample {line_number} hash mismatch")
        temporal = float(sample["nomad_temporal_distance"])
        cosine = float(sample["dinov3_current_goal_cosine_distance"])
        if not math.isfinite(temporal) or not math.isfinite(cosine):
            raise ValueError(f"sample {line_number} is non-finite")
        if temporal < 0.0 or cosine < 0.0:
            raise ValueError(f"sample {line_number} has negative distance")
        samples.append(sample)
    if not samples:
        raise RuntimeError("arrival dataset is empty")
    return samples


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def metric(sample: dict[str, Any], key: str) -> float:
    return float(sample[key])


def within(
    sample: dict[str, Any],
    temporal_limit: float,
    cosine_limit: float,
) -> bool:
    return (
        metric(sample, "nomad_temporal_distance") <= temporal_limit
        and metric(sample, "dinov3_current_goal_cosine_distance")
        <= cosine_limit
    )


def summarize_label(samples: list[dict[str, Any]]) -> dict[str, Any]:
    temporal = [
        metric(sample, "nomad_temporal_distance")
        for sample in samples
    ]
    cosine = [
        metric(sample, "dinov3_current_goal_cosine_distance")
        for sample in samples
    ]
    return {
        "count": len(samples),
        "nomad_temporal_distance": {
            "min": min(temporal),
            "median": statistics.median(temporal),
            "p75": percentile(temporal, 0.75),
            "max": max(temporal),
            "mad": median_absolute_deviation(temporal),
        },
        "dinov3_current_goal_cosine_distance": {
            "min": min(cosine),
            "median": statistics.median(cosine),
            "p75": percentile(cosine, 0.75),
            "max": max(cosine),
            "mad": median_absolute_deviation(cosine),
        },
    }


def load_route(path: Path) -> tuple[dict[str, Any], Path]:
    route_path = path.expanduser().resolve()
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if route.get("schema") != "v70_image_goal_route_v1":
        raise RuntimeError("unexpected route schema")
    goals = route.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise RuntimeError("calibrator supports exactly one goal")
    image = Path(str(goals[0]["image"])).expanduser()
    if not image.is_absolute():
        image = route_path.parent / image
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    return route, image


def calibrate(
    samples: list[dict[str, Any]],
    route: dict[str, Any],
    goal_image: Path,
    min_per_label: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    by_label = {
        label: [
            sample
            for sample in samples
            if sample["operator_label"] == label
        ]
        for label in LABELS
    }
    counts = Counter(sample["operator_label"] for sample in samples)
    reasons: list[str] = []
    for label in LABELS:
        if counts[label] < min_per_label:
            reasons.append(
                f"insufficient_{label}:{counts[label]}/{min_per_label}"
            )

    goal_hashes = {str(sample["goal_sha256"]) for sample in samples}
    route_goal_hash = sha256_file(goal_image)
    if goal_hashes != {route_goal_hash}:
        reasons.append("dataset_goal_hash_does_not_match_route")
    observation_keys = [
        tuple(str(value) for value in sample["observation_sha256"])
        for sample in samples
    ]
    if len(set(observation_keys)) != len(observation_keys):
        reasons.append("duplicate_observation_history")

    stop_samples = by_label["stop"]
    if stop_samples:
        temporal_enter = max(
            metric(sample, "nomad_temporal_distance")
            for sample in stop_samples
        )
        cosine_enter = max(
            metric(
                sample,
                "dinov3_current_goal_cosine_distance",
            )
            for sample in stop_samples
        )
        temporal_margin = max(
            0.15,
            2.0
            * median_absolute_deviation(
                [
                    metric(sample, "nomad_temporal_distance")
                    for sample in stop_samples
                ]
            ),
        )
        cosine_margin = max(
            0.005,
            2.0
            * median_absolute_deviation(
                [
                    metric(
                        sample,
                        "dinov3_current_goal_cosine_distance",
                    )
                    for sample in stop_samples
                ]
            ),
        )
        temporal_exit = temporal_enter + temporal_margin
        cosine_exit = cosine_enter + cosine_margin
    else:
        temporal_enter = cosine_enter = 0.0
        temporal_exit = cosine_exit = 0.0

    dangerous = [
        sample
        for label in DANGEROUS_NEGATIVE_LABELS
        for sample in by_label[label]
    ]
    false_enter = [
        sample for sample in dangerous
        if within(sample, temporal_enter, cosine_enter)
    ]
    false_exit = [
        sample for sample in dangerous
        if within(sample, temporal_exit, cosine_exit)
    ]
    missed_stop = [
        sample for sample in stop_samples
        if not within(sample, temporal_enter, cosine_enter)
    ]
    if false_enter:
        reasons.append(f"dangerous_enter_false_positives:{len(false_enter)}")
    if false_exit:
        reasons.append(f"dangerous_exit_false_positives:{len(false_exit)}")
    if missed_stop:
        reasons.append(f"missed_stop_samples:{len(missed_stop)}")

    approach_temporal = [
        metric(sample, "nomad_temporal_distance")
        for sample in by_label["approach"]
    ]
    approach_threshold = max(
        temporal_exit,
        percentile(approach_temporal, 0.75)
        if approach_temporal
        else temporal_exit,
    )
    overshoot_enter = sum(
        within(sample, temporal_enter, cosine_enter)
        for sample in by_label["overshoot"]
    )
    overshoot_exit = sum(
        within(sample, temporal_exit, cosine_exit)
        for sample in by_label["overshoot"]
    )

    ready = not reasons
    thresholds = {
        "approach_temporal_distance": approach_threshold,
        "temporal_distance_enter": temporal_enter,
        "temporal_distance_exit": temporal_exit,
        "cosine_distance_enter": cosine_enter,
        "cosine_distance_exit": cosine_exit,
        "required_confirmations": max(
            4,
            int(route["goals"][0].get("required_confirmations", 4)),
        ),
    }
    report = {
        "schema": "v70_arrival_calibration_report_v1",
        "dataset_samples": len(samples),
        "goal_image": str(goal_image),
        "goal_sha256": route_goal_hash,
        "minimum_per_label": min_per_label,
        "label_summary": {
            label: summarize_label(values) if values else {"count": 0}
            for label, values in by_label.items()
        },
        "candidate_thresholds": thresholds,
        "diagnostics": {
            "dangerous_enter_false_positives": len(false_enter),
            "dangerous_exit_false_positives": len(false_exit),
            "stop_recall": (
                1.0 - len(missed_stop) / len(stop_samples)
                if stop_samples
                else 0.0
            ),
            "overshoot_enter_count": overshoot_enter,
            "overshoot_exit_count": overshoot_exit,
            "overshoot_total": len(by_label["overshoot"]),
        },
        "ready_for_manual_review": ready,
        "arrival_calibrated_was_not_promoted": True,
        "rejection_reasons": reasons,
        "next_required_step": (
            "held-out standing stationary shadow validation, then "
            "explicit human promotion in a separately reviewed route"
        ),
    }
    candidate_route: dict[str, Any] | None = None
    if ready:
        candidate_route = json.loads(json.dumps(route))
        goal = candidate_route["goals"][0]
        goal.update(thresholds)
        goal["arrival_calibrated"] = False
        candidate_route["calibration_candidate"] = {
            "schema": "v70_arrival_candidate_v1",
            "goal_sha256": route_goal_hash,
            "dataset_samples": len(samples),
            "manual_review_required": True,
            "held_out_validation_required": True,
        }
    return report, candidate_route


def main() -> int:
    args = parse_args()
    if args.min_per_label < 3:
        raise ValueError("--min-per-label must be at least 3")
    samples = load_samples(args.dataset)
    route, goal_image = load_route(args.route_manifest)
    report, candidate = calibrate(
        samples,
        route,
        goal_image,
        args.min_per_label,
    )
    atomic_write_json(args.report, report)
    if candidate is not None:
        atomic_write_json(args.candidate_route, candidate)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_manual_review"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
