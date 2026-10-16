#!/usr/bin/env python3
"""Freeze the label-free six-route development pilot and image view."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


SOURCE = ROOT / "data/proposal_tail_collection_v1/incoming/development_routefit10_v4/manifest.windows.jsonl"
READINESS = ROOT / "data/proposal_tail_collection_v1/incoming/readiness.routefit10_v4.json"
VALIDATION = ROOT / "data/proposal_tail_collection_v1/incoming/one_time_validation_routefit10_v4/manifest.windows.jsonl"
FORMAL_V70 = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
OUTPUT = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v1"


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    routes = sorted({row["route_group"] for row in source_rows})
    trajectories = sorted({row["trajectory_id"] for row in source_rows})
    if len(source_rows) != 240 or len(routes) != 6 or len(trajectories) != 6:
        raise RuntimeError("development input cardinality changed")
    if set(routes) & {json.loads(line)["route_group"] for line in VALIDATION.read_text(encoding="utf-8").splitlines() if line.strip()}:
        raise RuntimeError("development/validation route overlap")
    view = OUTPUT / "trajectory_view"
    rows = []
    linked: set[tuple[str, int]] = set()
    for source in source_rows:
        trajectory = str(source["trajectory_id"])
        directory = view / trajectory
        directory.mkdir(parents=True, exist_ok=True)
        frame_map = {}
        for timestep, path in zip(source["history_timesteps"], source["context_image_paths"]):
            frame_map[int(timestep)] = Path(path)
        frame_map[int(source["current_t"])] = Path(source["current_image_path"])
        frame_map[int(source["goal_t"])] = Path(source["goal_image_path"])
        for timestep, target in frame_map.items():
            key = (trajectory, timestep)
            link = directory / f"{timestep}.jpg"
            if key not in linked:
                if not target.is_file():
                    raise RuntimeError(f"missing source image: {target}")
                link.symlink_to(target.resolve())
                linked.add(key)
        current = int(source["current_t"])
        history = list(map(int, source["history_timesteps"]))
        goal = int(source["goal_t"])
        rows.append({
            "bag_id": source["bag_id"],
            "cohort": "scand_new_routefit10_v4_development_pilot_v1",
            "context_image_paths": [str((directory / f"{t}.jpg").resolve()) for t in history],
            "current_image_path": str((directory / f"{current}.jpg").resolve()),
            "current_t": current,
            "dataset": "scand",
            "day": source["collection_day"],
            "goal_image_path": str((directory / f"{goal}.jpg").resolve()),
            "goal_t": goal,
            "history_timesteps": history,
            "partition": "train-fit",
            "route_group": source["route_group"],
            "route_id": source["route_id"],
            "selection_label_free": True,
            "trajectory_id": trajectory,
            "window_uid": source["window_uid"],
        })
    manifest = OUTPUT / "pilot_manifest.windows.jsonl"
    write_new(manifest, "".join(canonical_json(row) + "\n" for row in rows))
    prereg = {
        "schema": "scand_new_routefit10_v4_development_pilot_preregistration_v1",
        "status": "frozen_before_candidate_generation_and_outcomes",
        "scope": "new public development routes; underpowered six-route pilot; not deployment or independent validation evidence",
        "inputs": {
            "source_manifest": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
            "readiness": {"path": str(READINESS), "sha256": sha256_file(READINESS)},
            "sealed_validation_manifest": {"path": str(VALIDATION), "sha256": sha256_file(VALIDATION), "labels_opened": False},
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "formal_v70": {"path": str(FORMAL_V70), "sha256": sha256_file(FORMAL_V70)},
        },
        "selection": {
            "routes": 6,
            "states": 240,
            "states_per_route": dict(sorted(Counter(row["route_group"] for row in rows).items())),
            "windows_per_route": 40,
            "camera_contract": "external RGB only; body-fisheye-only routes fail closed",
            "labels_or_pose_opened": False,
        },
        "frozen_v70": {
            "cem_iterations": 3,
            "population": 16,
            "elite": 4,
            "proposal_scale": 0.70,
            "support_quantile": 0.999,
            "safety_thresholds_unchanged": True,
            "archive_unchanged": True,
            "sidecar_decision_isolated": True,
        },
        "staged_gate": {
            "proposal_oracle_before_training": "physical+support union must provide positive objective space and dual opportunity across at least three routes",
            "route_oof": "outer leave-one-route-out; training labels never score their own route",
            "candidate_auroc_min": 0.75,
            "risk_certified_replacements_nonzero": True,
            "dual_wilson_lcb_min": 0.85,
            "harm_wilson_ucb_max": 0.15,
            "mean_ate_and_objective_must_improve": True,
            "zero_replacement_is_failure": True,
        },
        "stop_conditions": [
            "no meaningful physical+support oracle space",
            "route-stable risk AUROC remains below 0.75",
            "no nonzero risk-certified replacement",
            "camera, timestamp, pose, or action alignment audit fails",
        ],
        "one_time_validation_labels_opened": False,
        "reserve_accessed": False,
    }
    prereg_path = OUTPUT / "preregistration.json"
    write_new(prereg_path, json.dumps(prereg, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema": "scand_new_routefit10_v4_development_pilot_prepare_audit_v1",
        "status": "complete_label_free",
        "states": len(rows),
        "routes": routes,
        "trajectories": trajectories,
        "linked_images": len(linked),
        "manifest_sha256": sha256_file(manifest),
        "preregistration_sha256": sha256_file(prereg_path),
        "sealed_validation_manifest_sha256": sha256_file(VALIDATION),
        "one_time_validation_labels_opened": False,
        "reserve_accessed": False,
    }
    audit_path = OUTPUT / "prepare.audit.json"
    write_new(audit_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**audit, "audit_sha256": sha256_file(audit_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
