#!/usr/bin/env python3
"""Build offline-only truth labels aligned to a proposal-tail candidate table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    canonical_json,
    reject_forbidden_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-one-time-validation", action="store_true")
    return parser.parse_args()


def metric_values(candidates: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float64) * 0.25
    truth = np.asarray(ground_truth, dtype=np.float64)
    error = value - truth[:, None]
    squared = np.sum(error * error, axis=-1)
    ate = np.sqrt(np.mean(squared, axis=-1))
    endpoint = np.sqrt(squared[..., -1])
    relative = np.diff(value, axis=2) - np.diff(truth, axis=1)[:, None]
    rpe = np.sqrt(np.mean(np.sum(relative * relative, axis=-1), axis=-1))
    objective = 0.4 * ate + 0.6 * endpoint
    return np.stack((ate, rpe, endpoint, objective), axis=-1).astype(np.float32)


def main() -> None:
    args = parse_args()
    reject_forbidden_path(args.collection)
    reject_forbidden_path(args.truth)
    manifest = json.loads((args.collection / "collection_manifest.json").read_text(encoding="utf-8"))
    role = manifest["cohort_role"]
    marker = args.collection / "one_time_validation_labels_opened.marker.json"
    if role == "one_time_validation":
        if not args.confirm_one_time_validation:
            raise RuntimeError("one-time validation requires explicit confirmation")
        if marker.exists():
            raise RuntimeError("one-time validation labels were already opened")
        marker.write_text(
            canonical_json(
                {"status": "opening_started_fail_closed", "truth_sha256": sha256_file(args.truth)}
            )
            + "\n",
            encoding="utf-8",
        )
    table = np.load(args.collection / "candidate_table.npz", allow_pickle=False)
    truth = np.load(args.truth, allow_pickle=False)
    if "ground_truth_local_m" not in truth.files:
        raise RuntimeError("truth file lacks ground_truth_local_m")
    record = table["record_key"].astype(str)
    truth_record = truth["record_key"].astype(str)
    index = {value: position for position, value in enumerate(truth_record)}
    if any(value not in index for value in record):
        raise RuntimeError("truth is missing collection record keys")
    order = np.asarray([index[value] for value in record], dtype=np.int64)
    ground = truth["ground_truth_local_m"][order].astype(np.float32)
    if ground.shape != (len(record), 4, 2):
        raise RuntimeError(f"ground truth shape mismatch: {ground.shape}")
    metrics = metric_values(table["candidates_feature_waypoint_units"], ground)
    if not np.allclose(metrics[..., 3], 0.4 * metrics[..., 0] + 0.6 * metrics[..., 2], atol=1.0e-7):
        raise RuntimeError("objective formula audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=record,
            candidate_hash=table["candidate_hash"],
            ground_truth_local_m=ground,
            true_metrics_m=metrics,
            metric_names=np.asarray(("ate", "rpe", "endpoint", "objective")),
            spent_smoke=np.asarray(role == "spent_smoke"),
            online_usable=np.asarray(False),
        )
    audit = {
        "schema": "proposal_tail_offline_labels_v1",
        "status": "spent_shape_alignment_smoke_only" if role == "spent_smoke" else "offline_labels_built",
        "cohort_role": role,
        "states": len(record),
        "candidate_shape": list(metrics.shape),
        "objective_formula": "0.4*ATE+0.6*endpoint",
        "unit": "meter",
        "collection_manifest_sha256": sha256_file(args.collection / "collection_manifest.json"),
        "truth_sha256": sha256_file(args.truth),
        "output_sha256": sha256_file(args.output),
        "online_label_access_allowed": False,
        "training_evidence_allowed": role != "spent_smoke",
        "reserve_accessed": False,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    if role == "one_time_validation":
        marker.write_text(canonical_json({"labels_sha256": audit["output_sha256"]}) + "\n", encoding="utf-8")
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
