#!/usr/bin/env python3
"""Paired comparison of final tri-domain V70 and the frozen old dual-domain V70."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("ate_nwm", "rte_nwm", "fde_nwm")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["record_key"], row["method"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-root",
        type=Path,
        default=Path("/root/autodl-tmp/dwm_runs/v70_tridomain_fair_eval_triseed_20260728/sacson100_old_v70"),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1/fair_eval_final_relaxed_v1/sacson100"),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    old_result_path = args.old_root / "metrics/results.json"
    new_result_path = args.new_root / "metrics/results.json"
    old_csv_path = args.old_root / "metrics/per_state_metrics.csv"
    new_csv_path = args.new_root / "metrics/per_state_metrics.csv"
    output_path = args.new_root / "comparison_to_old_dualdomain_v70.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    old_result = load_json(old_result_path)
    new_result = load_json(new_result_path)
    if old_result["manifest_sha256"] != new_result["manifest_sha256"]:
        raise RuntimeError("old and new results do not use the same SACSoN-100 manifest")
    if old_result["states"] != 100 or new_result["states"] != 100:
        raise RuntimeError("expected exactly 100 states in both results")

    old_rows = read_rows(old_csv_path)
    new_rows = read_rows(new_csv_path)
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for k in (16, 32):
        old_method = f"old_v70_triseed_k{k}"
        new_method = f"new_v70_triseed_k{k}"
        old_keys = {key for key, method in old_rows if method == old_method}
        new_keys = {key for key, method in new_rows if method == new_method}
        if old_keys != new_keys or len(old_keys) != 100:
            raise RuntimeError(f"paired state keys differ for K{k}")
        keys = sorted(old_keys)
        row = {}
        for metric in METRICS:
            old_values = np.asarray([float(old_rows[(key, old_method)][metric]) for key in keys], dtype=np.float64)
            new_values = np.asarray([float(new_rows[(key, new_method)][metric]) for key in keys], dtype=np.float64)
            delta = old_values - new_values
            samples = rng.integers(0, len(keys), size=(args.bootstrap, len(keys)))
            boot = delta[samples].mean(axis=1)
            old_mean = float(old_values.mean())
            new_mean = float(new_values.mean())
            row[metric] = {
                "old_mean": old_mean,
                "new_mean": new_mean,
                "paired_delta_new_better_positive": float(delta.mean()),
                "paired_delta_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "relative_improvement_percent": (old_mean - new_mean) / old_mean * 100.0,
            }
        for metric in ("success_fde_1step", "success_fde_2step"):
            old_values = np.asarray([float(old_rows[(key, old_method)][metric]) for key in keys], dtype=np.float64)
            new_values = np.asarray([float(new_rows[(key, new_method)][metric]) for key in keys], dtype=np.float64)
            delta = new_values - old_values
            samples = rng.integers(0, len(keys), size=(args.bootstrap, len(keys)))
            boot = delta[samples].mean(axis=1)
            row[metric] = {
                "old_mean": float(old_values.mean()),
                "new_mean": float(new_values.mean()),
                "change_percentage_points": float(delta.mean() * 100.0),
                "paired_delta_ci95_percentage_points": [
                    float(np.quantile(boot, 0.025) * 100.0),
                    float(np.quantile(boot, 0.975) * 100.0),
                ],
            }
        comparisons[f"K{k}"] = row

    payload = {
        "schema": "sacson100_final_tridomain_vs_old_dualdomain_v70_paired_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "states": 100,
        "manifest_sha256": new_result["manifest_sha256"],
        "old_model_contract": "old dual-domain LTWM/scorer ensemble; SACSoN unseen during training",
        "new_model_contract": "final proposal-aware tri-domain V70 with tri-domain LTWM ensemble",
        "old_results_path": str(old_result_path),
        "old_results_sha256": digest(old_result_path),
        "new_results_path": str(new_result_path),
        "new_results_sha256": digest(new_result_path),
        "direction": "positive paired delta and positive relative improvement mean final V70 is better",
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed},
        "comparisons": comparisons,
    }
    tmp = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
