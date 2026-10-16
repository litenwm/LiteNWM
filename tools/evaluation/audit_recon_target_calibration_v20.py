#!/usr/bin/env python3
"""Audit the frozen RECON target-calibration banks and sealed reserve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_metadata(payload: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(payload["metadata"]))


def audit_bank(root: Path, letter: str) -> tuple[dict, set[str], set[str]]:
    bank = root / f"bank_{letter}"
    feature = bank / "features" / f"recon_target_bank_{letter}_v20_features.npz"
    latent = bank / "features" / f"recon_target_bank_{letter}_v20_latents.npz"
    rollout = bank / "spatial_rollout_v17.npz"
    score = bank / "universal_scores.npz"
    source = bank / "source_labeled.npz"
    paths = (feature, latent, rollout, score, source)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(missing)
    with np.load(source, allow_pickle=False) as payload:
        source_record = payload["record_key"].astype(str)
        source_route = payload["route_group"].astype(str)
        source_trajectory = payload["trajectory"].astype(str)
        targets = payload["targets"]
    with np.load(feature, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
        feature_meta = read_metadata(payload)
        feature_arrays = {
            key: payload[key]
            for key in (
                "static_features",
                "raw_ltwm_features",
                "actions_h4",
                "candidates",
                "majority_support",
            )
        }
    with np.load(latent, allow_pickle=False) as payload:
        latent_record = payload["record_key"].astype(str)
        history = payload["history"]
        goal = payload["goal"]
    with np.load(rollout, allow_pickle=False) as payload:
        rollout_record = payload["record_key"].astype(str)
        spatial = payload["spatial_rollout_features"]
        rollout_meta = read_metadata(payload)
    with np.load(score, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        score_arrays = {
            key: payload[key]
            for key in (
                "mean",
                "std",
                "dual_probability",
                "harm_probability",
                "rank_logit",
            )
        }
        score_meta = read_metadata(payload)
    identities = (feature_record, latent_record, rollout_record, score_record)
    identity_ok = all(np.array_equal(source_record, value) for value in identities)
    expected_shapes = {
        "targets": tuple(targets.shape),
        "history": tuple(history.shape),
        "goal": tuple(goal.shape),
        "spatial_rollout_features": tuple(spatial.shape),
    }
    shape_ok = (
        targets.shape == (100, 17, 8)
        and history.shape == (100, 4, 256, 384)
        and goal.shape == (100, 256, 384)
        and spatial.shape == (100, 17, 258)
    )
    arrays = [targets, history, goal, spatial]
    arrays.extend(feature_arrays.values())
    arrays.extend(score_arrays.values())
    finite_ok = all(np.isfinite(value).all() for value in arrays)
    label_contract_ok = (
        feature_meta.get("labels_in_output") is True
        and rollout_meta.get("labels_in_output") is False
        and rollout_meta.get("outcomes_accessed") is False
        and score_meta.get("outcomes_accessed") is False
    )
    report = {
        "states": int(len(source_record)),
        "routes": int(len(set(source_route))),
        "trajectories": int(len(set(source_trajectory))),
        "identity_ok": identity_ok,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "label_contract_ok": label_contract_ok,
        "shapes": expected_shapes,
        "sha256": {path.name: file_sha256(path) for path in paths},
    }
    if not all((identity_ok, shape_ok, finite_ok, label_contract_ok)):
        raise RuntimeError(f"bank_{letter} audit failed: {report}")
    return report, set(source_record), set(source_route)


def main() -> None:
    args = parse_args()
    banks = {}
    identities = {}
    routes = {}
    for letter in ("a", "b", "c"):
        banks[letter], identities[letter], routes[letter] = audit_bank(
            args.root, letter
        )
    reserve = args.root / "final_confirmation_reserve"
    reserve_source = reserve / "source_label_free.npz"
    with np.load(reserve_source, allow_pickle=False) as payload:
        reserve_record = set(payload["record_key"].astype(str))
        reserve_route = set(payload["route_group"].astype(str))
        reserve_states = len(payload["record_key"])
    forbidden = [
        reserve / "source_labeled.npz",
        reserve / "features",
        reserve / "spatial_rollout_v17.npz",
        reserve / "universal_scores.npz",
    ]
    reserve_sealed = all(not path.exists() for path in forbidden)
    overlap = {}
    names = ("a", "b", "c")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap[f"{left}_{right}"] = {
                "record": len(identities[left] & identities[right]),
                "route": len(routes[left] & routes[right]),
            }
    overlap["banks_reserve"] = {
        "record": len(set.union(*(identities[name] for name in names)) & reserve_record),
        "route": len(set.union(*(routes[name] for name in names)) & reserve_route),
    }
    disjoint = all(
        value["record"] == 0 and value["route"] == 0
        for value in overlap.values()
    )
    report = {
        "schema": "recon_target_calibration_v20_audit",
        "status": "passed" if reserve_sealed and disjoint else "failed",
        "banks": banks,
        "overlap": overlap,
        "reserve": {
            "states": int(reserve_states),
            "routes": int(len(reserve_route)),
            "sealed": reserve_sealed,
            "source_label_free_sha256": file_sha256(reserve_source),
        },
        "all_feature_arrays_finite": True,
        "all_record_identity_checks_passed": True,
        "all_banks_and_reserve_route_disjoint": disjoint,
    }
    if report["status"] != "passed":
        raise RuntimeError(report)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
