#!/usr/bin/env python3
"""Open and evaluate a prelabel-frozen v29 RECON confirmation exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels
from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path, required=True)
    parser.add_argument("--native-freeze", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checks(value: dict, incumbent: dict, native: dict) -> dict[str, bool]:
    safety = {
        "coverage_at_least_0.40": value["promotion_fraction"] >= 0.40,
        "dual_precision_at_least_0.90": value["dual_precision"] is not None and value["dual_precision"] >= 0.90,
        "wilson_lower_at_least_0.80": value["dual_wilson_lower_95"] is not None and value["dual_wilson_lower_95"] >= 0.80,
        "harm_rate_at_most_0.10": value["harm_rate_per_promotion"] is not None and value["harm_rate_per_promotion"] <= 0.10,
    }
    quality = {
        "all_means_better_than_incumbent": all(value["absolute"][name]["mean_m"] < incumbent["absolute"][name]["mean_m"] for name in METRIC_NAMES),
        "all_means_better_than_native_v2": all(value["absolute"][name]["mean_m"] < native["absolute"][name]["mean_m"] for name in METRIC_NAMES),
        "all_p90_nondegraded_vs_incumbent": all(value["absolute"][name]["p90_m"] <= incumbent["absolute"][name]["p90_m"] + 1.0e-9 for name in METRIC_NAMES),
        "all_p90_nondegraded_vs_native_v2": all(value["absolute"][name]["p90_m"] <= native["absolute"][name]["p90_m"] + 1.0e-9 for name in METRIC_NAMES),
    }
    return {**safety, **quality, "all_safety_constraints_pass": all(safety.values()), "all_quality_constraints_pass": all(quality.values()), "overall_pass": all(safety.values()) and all(quality.values())}


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    labeled_path = args.confirmation_root / "source_labeled.npz"
    audit_path = args.confirmation_root / "outcome_open.audit.json"
    if result_path.exists() or labeled_path.exists() or audit_path.exists():
        raise FileExistsError(result_path)
    if args.bootstrap_replicates < 1000:
        raise ValueError("clustered bootstrap needs at least 1000 replicates")
    freeze = json.loads(args.selection_freeze.read_text(encoding="utf-8"))
    native_freeze = json.loads(args.native_freeze.read_text(encoding="utf-8"))
    if freeze["status"] != "selected_index_frozen_before_confirmation_outcomes" or freeze.get("labels_opened") or freeze.get("outcomes_accessed") or freeze.get("cem"):
        raise RuntimeError("v29 selection freeze is invalid")
    if native_freeze["status"] != "native_v2_selected_before_confirmation_outcomes" or native_freeze.get("labels_opened") or native_freeze.get("outcomes_accessed"):
        raise RuntimeError("native selection freeze is invalid")
    selected_path = Path(freeze["selected_index"])
    native_path = Path(native_freeze["selected_index"])
    if file_sha256(selected_path) != freeze["artifacts"]["selected_index_sha256"]:
        raise RuntimeError("v29 selected index changed after freeze")
    if file_sha256(native_path) != native_freeze["selected_index_sha256"]:
        raise RuntimeError("native selected index changed after freeze")
    manifest_path = args.confirmation_root / "manifest.windows.jsonl"
    source_path = args.confirmation_root / "source_label_free.npz"
    if file_sha256(manifest_path) != freeze["artifacts"]["manifest_sha256"] or file_sha256(source_path) != freeze["artifacts"]["source_label_free_sha256"]:
        raise RuntimeError("confirmation artifacts changed after freeze")

    rows = read_rows(manifest_path)
    with np.load(source_path, allow_pickle=False) as payload:
        source_arrays = {key: payload[key].copy() for key in payload.files}
        record = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
    expected = np.asarray([f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows])
    if len(rows) != 100 or not np.array_equal(record, expected) or len(np.unique(trajectory)) != 25 or len(np.unique(route)) != 25:
        raise RuntimeError("confirmation identity contract changed")
    with np.load(selected_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record) or not np.array_equal(payload["route_group"].astype(str), route):
            raise RuntimeError("v29 selection identity mismatch")
        selected = payload["selected_index"].astype(np.int64)
    with np.load(native_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record):
            raise RuntimeError("native selection identity mismatch")
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)

    # This is deliberately the first outcome access for this confirmation set.
    ground_truth = np.stack([
        local_ground_truth(args.trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl", int(row["current_t"]))
        for row in rows
    ]).astype(np.float32)
    labels = metric_labels(source_arrays["candidates"], ground_truth, 0.25, 0.4, 0.6)
    absolute = labels["values"].astype(np.float32)
    write_npz(labeled_path, **source_arrays, targets=labels["targets"], absolute_metrics=absolute, ground_truth_local_m=ground_truth)
    oracle = absolute[..., 3].argmin(axis=1).astype(np.int64)
    methods_index = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        "hashfold_v29_cem_false": selected,
        "posthoc_objective_pool_oracle": oracle,
    }
    methods = {
        name: method_metrics(index, absolute, trajectory, route, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed + method_number * 10)
        for method_number, (name, index) in enumerate(methods_index.items())
    }
    methods["hashfold_v29_cem_false"]["deployment_checks"] = checks(methods["hashfold_v29_cem_false"], methods["incumbent"], methods["native_scorer_v2"])
    audit = {
        "schema": "recon_target_hashfold_confirmation_v29_outcome_open",
        "status": "confirmation_outcomes_opened_once_after_all_selections_frozen",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "states": 100, "trajectories": 25, "routes": 25,
        "selection_freeze_sha256": file_sha256(args.selection_freeze),
        "native_freeze_sha256": file_sha256(args.native_freeze),
        "selected_index_sha256": file_sha256(selected_path),
        "native_selected_index_sha256": file_sha256(native_path),
        "source_label_free_sha256": file_sha256(source_path),
        "source_labeled_sha256": file_sha256(labeled_path),
        "labels_opened": True, "outcomes_accessed": True, "cem": False,
    }
    write_json(audit_path, audit)
    report = {
        "schema": "recon_target_hashfold_confirmation_v29_result",
        "status": "fresh_route_disjoint_confirmation_evaluated_after_selection_freeze",
        "evidence_role": "fresh route-disjoint RECON H4-100 confirmation",
        "methods": methods,
        "selection_freeze": str(args.selection_freeze),
        "selection_freeze_sha256": file_sha256(args.selection_freeze),
        "native_freeze": str(args.native_freeze),
        "native_freeze_sha256": file_sha256(args.native_freeze),
        "labeled_source_sha256": file_sha256(labeled_path),
        "success_contract": {"coverage": 0.40, "dual_precision": 0.90, "wilson_lower": 0.80, "harm_rate": 0.10},
        "fixed_pool_only": True, "cem": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(result_path, report)
    compact = {name: {"promotions": value["promotions"], "coverage": value["promotion_fraction"], "dual": value["dual_precision"], "wilson": value["dual_wilson_lower_95"], "harm": value["harm_rate_per_promotion"], "absolute": value["absolute"], "checks": value.get("deployment_checks")} for name, value in methods.items()}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
