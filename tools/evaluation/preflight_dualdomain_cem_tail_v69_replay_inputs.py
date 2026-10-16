#!/usr/bin/env python3
"""Validate all frozen v69 replay inputs without opening any outcome arrays."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)


def specifications():
    for bank in "abcdef":
        base = (
            ROOT
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{bank}"
        )
        yield {
            "cohort": f"recon_bank_{bank}",
            "domain": "recon",
            "manifest": base / "manifest.windows.jsonl",
            "candidate": base / "k17.npz",
            "features": base
            / "features_cem_v30"
            / f"recon_target_bank_{bank}_cem_v30_features.npz",
            "latents": base
            / "features_cem_v30"
            / f"recon_target_bank_{bank}_cem_v30_latents.npz",
            "trajectory_root": Path("/home/rainbow/nwm/data/recon"),
        }
    for bank in "gh":
        base = (
            ROOT
            / "outputs/cem_development_v1/recon_cem_boundary_v60"
            / f"bank_{bank}_v60"
        )
        yield {
            "cohort": f"recon_bank_{bank}",
            "domain": "recon",
            "manifest": base / "manifest.windows.jsonl",
            "candidate": base / "k17.npz",
            "features": base / "features" / f"recon_bank_{bank}_v60_features.npz",
            "latents": base / "features" / f"recon_bank_{bank}_v60_latents.npz",
            "trajectory_root": Path("/home/rainbow/nwm/data/recon"),
        }
    base = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
    yield {
        "cohort": "scand_fresh_outer",
        "domain": "scand",
        "manifest": base / "manifest.windows.jsonl",
        "candidate": base / "fresh_outer_k17.npz",
        "features": base
        / "features"
        / "scand_fresh_outer_h4_100_seed20260720_features.npz",
        "latents": base
        / "features"
        / "scand_fresh_outer_h4_100_seed20260720_latents.npz",
        "trajectory_root": Path("/home/rainbow/nwm/data/scand"),
    }


def validate(spec):
    for key in ("manifest", "candidate", "features", "latents"):
        spec[key].resolve(strict=True)
    rows = [
        json.loads(line)
        for line in spec["manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate = load_npz(spec["candidate"])
    features = load_features(spec["features"], labels_expected=False)
    latents = load_npz(spec["latents"])
    expected = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    identity_ok = np.array_equal(
        candidate["record_key"].astype(str), expected
    ) and np.array_equal(features["record_key"].astype(str), expected)
    bridge = np.float32(1.0 if spec["domain"] == "recon" else 0.38 / 0.25)
    incumbent = candidate["candidates_waypoint_units"][:, 0].astype(np.float32) * bridge
    reference = candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge
    base_candidates = np.concatenate((incumbent[:, None], reference), axis=1)
    bridge_ok = np.array_equal(
        base_candidates, features["candidates"].astype(np.float32)
    )
    trajectory_ids = sorted({str(row["trajectory_id"]) for row in rows})
    missing = [
        value
        for value in trajectory_ids
        if not (spec["trajectory_root"] / value / "traj_data.pkl").is_file()
    ]
    result = {
        "cohort": spec["cohort"],
        "states": len(rows),
        "trajectories": len(trajectory_ids),
        "identity_ok": bool(identity_ok),
        "candidate_bridge_ok": bool(bridge_ok),
        "latent_keys": sorted(latents),
        "missing_trajectory_data": len(missing),
    }
    if not identity_ok or not bridge_ok or missing:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    results = [validate(spec) for spec in specifications()]
    print(json.dumps({
        "status": "all_replay_inputs_valid_without_opening_outcomes",
        "outcomes_opened": False,
        "reserve_accessed": False,
        "cohorts": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
