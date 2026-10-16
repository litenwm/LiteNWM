#!/usr/bin/env python3
"""Verify sidecar on/off numerical equivalence and raw-draw alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def equal_arrays(left: Path, right: Path, *, ignore: set[str] | None = None) -> dict[str, Any]:
    ignore = ignore or set()
    a, b = load(left), load(right)
    keys = sorted((set(a) | set(b)) - ignore)
    missing = sorted(set(a) ^ set(b))
    unequal = []
    max_abs: dict[str, float] = {}
    for key in keys:
        if key not in a or key not in b:
            continue
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            unequal.append(key)
            continue
        if np.array_equal(a[key], b[key]):
            continue
        unequal.append(key)
        if np.issubdtype(a[key].dtype, np.number):
            max_abs[key] = float(np.max(np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))))
    return {
        "left": str(left),
        "right": str(right),
        "keys": keys,
        "ignored": sorted(ignore),
        "missing": missing,
        "unequal": unequal,
        "max_abs": max_abs,
        "exact": not missing and not unequal,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.sidecar / "collection_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("raw_draws_observable"):
        raise RuntimeError("sidecar collection is incomplete")
    comparisons = {
        "selected_actions": equal_arrays(args.off / "selected_actions.npz", args.on / "selected_actions.npz"),
    }
    draw_audit = {}
    sidecar_pool_audit = {}
    for iteration in range(3):
        name = f"iteration_{iteration:02d}"
        for filename in ("population.npz", "source_label_free.npz", "frozen_scores.npz"):
            comparisons[f"{name}/{filename}"] = equal_arrays(args.off / name / filename, args.on / name / filename)
        comparisons[f"{name}/cem_features.npz"] = equal_arrays(
            args.off / name / "cem_features.npz",
            args.on / name / "cem_features.npz",
            ignore={"metadata"},
        )
        formal_population = load(args.on / name / "population.npz")["candidates_feature_waypoint_units"]
        formal_scores = load(args.on / name / "frozen_scores.npz")
        side = load(args.sidecar / name / "candidates_and_scores.npz")
        score_map = {
            "energy": "energy",
            "mean": "metric_mean",
            "std": "metric_std",
            "dual_probability": "dual_probability",
            "harm_probability": "harm_probability",
            "valid": "valid",
            "safe": "safe",
            "elite_index_for_update": "elite_index_for_update",
            "proposal_updated": "proposal_updated",
            "safe_count": "safe_count",
        }
        pool_checks = {
            "candidates_exact": bool(np.array_equal(formal_population, side["candidates_feature_waypoint_units"])),
        }
        for formal_key, side_key in score_map.items():
            pool_checks[f"{formal_key}_exact"] = bool(np.array_equal(formal_scores[formal_key], side[side_key]))
        sidecar_pool_audit[name] = pool_checks

        draws = load(args.sidecar / name / "raw_draws.npz")
        final_index = draws["final_candidate_index"].astype(np.int64)
        accepted = final_index > 0
        state = draws["state_index"].astype(np.int64)[accepted]
        index = final_index[accepted]
        mapped = formal_population[state, index]
        observed = draws["draw_clipped"][accepted].reshape(mapped.shape)
        draw_audit[name] = {
            "raw_draws": int(len(final_index)),
            "accepted_and_mapped": int(accepted.sum()),
            "rejected_draws": int((~draws["accepted_mask"].astype(bool)).sum()),
            "candidate_mapping_exact": bool(np.array_equal(mapped, observed)),
            "draw_sequence_unique_per_state": bool(
                all(
                    len(np.unique(draws["draw_sequence"][draws["state_index"] == value]))
                    == int((draws["state_index"] == value).sum())
                    for value in np.unique(draws["state_index"])
                )
            ),
        }
    comparisons_exact = all(value["exact"] for value in comparisons.values())
    pools_exact = all(all(value.values()) for value in sidecar_pool_audit.values())
    draws_exact = all(
        value["candidate_mapping_exact"] and value["draw_sequence_unique_per_state"]
        for value in draw_audit.values()
    )
    result = {
        "schema": "scand_existing_train_pilot_logger_equivalence_v1",
        "status": "pass" if comparisons_exact and pools_exact and draws_exact else "fail",
        "selected_actions_sha256_off": sha256_file(args.off / "selected_actions.npz"),
        "selected_actions_sha256_on": sha256_file(args.on / "selected_actions.npz"),
        "comparisons": comparisons,
        "sidecar_pool_alignment": sidecar_pool_audit,
        "raw_draw_alignment": draw_audit,
        "rng_candidate_order_archive_selected_action_unchanged": comparisons_exact,
        "formal_behavior_influence": False,
        "labels_opened": False,
        "reserve_accessed": False,
    }
    if result["status"] != "pass":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha256_file(args.output), "draws": draw_audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
