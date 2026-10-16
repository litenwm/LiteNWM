#!/usr/bin/env python3
"""Freeze a truly unused RGB-only SCAND fresh-outer H4-100 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.strict_inner_partitions import parse_scand_trajectory_id  # noqa: E402
from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json  # noqa: E402


TARGET_TRAJECTORIES = 25
WINDOWS_PER_TRAJECTORY = 4
TARGET_WINDOWS = TARGET_TRAJECTORIES * WINDOWS_PER_TRAJECTORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[REPO_ROOT / "outputs", Path("/home/rainbow/ndwm/DWM/reports")],
    )
    parser.add_argument(
        "--current-output-root",
        type=Path,
        default=REPO_ROOT / "outputs/scand_domain_adaptation_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/manifest.windows.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> str:
    return hashlib.sha256(
        "|".join((str(seed), *(str(value) for value in values))).encode()
    ).hexdigest()


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prior_npz_files(roots: list[Path], current_root: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for directory, _, files in os.walk(root):
            base = Path(directory)
            for name in files:
                if not name.endswith(".npz"):
                    continue
                path = (base / name).resolve()
                if path in seen or inside(path, current_root):
                    continue
                seen.add(path)
                result.append(path)
    return sorted(result)


def npz_trajectories(path: Path) -> set[str]:
    trajectories: set[str] = set()
    with np.load(path, allow_pickle=False) as payload:
        if "record_key" in payload.files:
            for value in payload["record_key"].astype(str).reshape(-1):
                trajectory, separator, _ = value.rpartition(":")
                if separator:
                    trajectories.add(trajectory)
        for key in ("trajectory", "trajectory_id", "traj_name"):
            if key in payload.files:
                trajectories.update(payload[key].astype(str).reshape(-1).tolist())
    return trajectories


def known_used_trajectories(args: argparse.Namespace) -> tuple[set[str], dict[str, Any]]:
    used: set[str] = set()
    sources: dict[str, Any] = {}
    for partition in ("train-fit", "scorer-cal", "cem-cal", "internal-holdout"):
        path = REPO_ROOT / f"data/inner_manifests/scand/{partition}.windows.jsonl"
        values = {str(row["trajectory_id"]) for row in read_jsonl(path)}
        used.update(values)
        sources[f"inner_{partition}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "trajectories": len(values),
        }
    strict_path = (
        REPO_ROOT / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json"
    )
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_values = {str(row["trajectory_id"]) for row in strict["records"]}
    used.update(strict_values)
    sources["spent_strict100"] = {
        "path": str(strict_path),
        "sha256": sha256(strict_path),
        "trajectories": len(strict_values),
    }
    latent_path = REPO_ROOT / "data/latent_metadata/scand/splits.json"
    latent = json.loads(latent_path.read_text(encoding="utf-8"))
    latent_values = set(latent.get("train", [])) | set(latent.get("val", []))
    used.update(latent_values)
    sources["ltwm_dinov3_train_val"] = {
        "path": str(latent_path),
        "sha256": sha256(latent_path),
        "trajectories": len(latent_values),
    }
    artifact_files = prior_npz_files(args.scan_root, args.current_output_root)
    failures = []
    artifact_values: set[str] = set()
    identity_files = 0
    for path in artifact_files:
        try:
            values = npz_trajectories(path)
        except Exception as error:
            failures.append({"path": str(path), "error": repr(error)})
            continue
        if values:
            identity_files += 1
            artifact_values.update(values)
    used.update(artifact_values)
    sources["prior_artifacts"] = {
        "npz_files_scanned": len(artifact_files),
        "identity_files": identity_files,
        "trajectories": len(artifact_values),
        "scan_failures": failures,
    }
    return used, sources


def numeric_images(path: Path) -> set[int]:
    result = set()
    for image in path.glob("*.jpg"):
        if image.stem.isdigit():
            result.add(int(image.stem))
    return result


def valid_currents(images: set[int]) -> list[int]:
    return sorted(
        current
        for current in images
        if current >= 3
        and all(value in images for value in range(current - 3, current + 1))
        and current + 4 in images
    )


def union_sha(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def choose_trajectories(
    candidates: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    routes: dict[str, list[dict[str, Any]]] = {}
    for value in candidates:
        routes.setdefault(value["route_group"], []).append(value)
    for route, values in routes.items():
        values.sort(
            key=lambda item: stable_key(
                seed, "fresh_outer_trajectory", route, item["trajectory_id"]
            )
        )
    route_order = sorted(
        routes,
        key=lambda route: stable_key(seed, "fresh_outer_route", route),
    )
    selected: list[dict[str, Any]] = []
    used_bags: set[str] = set()
    # First pass maximizes physical-bag diversity; second pass fills if needed.
    for require_new_bag in (True, False):
        progressed = True
        while len(selected) < TARGET_TRAJECTORIES and progressed:
            progressed = False
            for route in route_order:
                choice = next(
                    (
                        value
                        for value in routes[route]
                        if value not in selected
                        and (not require_new_bag or value["bag_id"] not in used_bags)
                    ),
                    None,
                )
                if choice is None:
                    continue
                selected.append(choice)
                used_bags.add(choice["bag_id"])
                progressed = True
                if len(selected) == TARGET_TRAJECTORIES:
                    break
    return selected


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    universe_path = args.output.with_suffix(".excluded_trajectories.json")
    if args.output.exists() or audit_path.exists() or universe_path.exists():
        raise FileExistsError(args.output)
    used, sources = known_used_trajectories(args)
    used_routes = {
        parse_scand_trajectory_id(value)["route_group"]
        for value in used
        if value.startswith("random_mdps_")
    }
    candidates = []
    raw_trajectories = 0
    for path in sorted(args.trajectory_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            metadata = parse_scand_trajectory_id(path.name)
        except ValueError:
            continue
        raw_trajectories += 1
        if path.name in used or metadata["route_group"] in used_routes:
            continue
        currents = valid_currents(numeric_images(path))
        if len(currents) < WINDOWS_PER_TRAJECTORY:
            continue
        candidates.append(
            {
                "trajectory_id": path.name,
                **metadata,
                "valid_currents": currents,
            }
        )
    selected = choose_trajectories(candidates, args.seed)
    enough = len(selected) == TARGET_TRAJECTORIES
    rows: list[dict[str, Any]] = []
    if enough:
        for trajectory in selected:
            currents = trajectory["valid_currents"]
            positions = np.linspace(
                0, len(currents) - 1, WINDOWS_PER_TRAJECTORY
            ).round().astype(int)
            chosen_currents = [currents[index] for index in positions]
            if len(set(chosen_currents)) != WINDOWS_PER_TRAJECTORY:
                raise RuntimeError("fresh outer temporal window selection duplicated")
            for temporal_bin, current in enumerate(chosen_currents):
                uid = hashlib.sha256(
                    f"scand-fresh-outer-v1|{args.seed}|{trajectory['trajectory_id']}|{current}".encode()
                ).hexdigest()[:24]
                rows.append(
                    {
                        "partition": "fresh-outer",
                        "trajectory_id": trajectory["trajectory_id"],
                        "current_t": int(current),
                        "history_timesteps": list(range(current - 3, current + 1)),
                        "goal_t": int(current + 4),
                        "route_group": trajectory["route_group"],
                        "route_id": trajectory["route_group"],
                        "bag_id": trajectory["bag_id"],
                        "robot": trajectory["robot"],
                        "survey": trajectory["survey"],
                        "temporal_bin": temporal_bin,
                        "window_uid": uid,
                        "selection_seed": args.seed,
                        "selection_source": "RGB filenames only; no pose/outcome access",
                    }
                )
        rows.sort(
            key=lambda row: stable_key(
                args.seed,
                "fresh_outer_window_order",
                row["trajectory_id"],
                row["current_t"],
            )
        )
        write_jsonl(args.output, rows)
    excluded = {
        "schema": "scand_fresh_outer_excluded_trajectory_universe_v1",
        "used_trajectory_count": len(used),
        "used_route_count": len(used_routes),
        "used_trajectory_union_sha256": union_sha(used),
        "used_route_union_sha256": union_sha(used_routes),
        "used_trajectories": sorted(used),
        "used_routes": sorted(used_routes),
    }
    write_json(universe_path, excluded)
    audit = {
        "schema": "scand_fresh_outer_manifest_freeze_v1",
        "status": "manifest_frozen_rgb_only" if enough else "insufficient_fresh_data",
        "seed": args.seed,
        "target_windows": TARGET_WINDOWS,
        "target_trajectories": TARGET_TRAJECTORIES,
        "windows_per_trajectory": WINDOWS_PER_TRAJECTORY,
        "raw_trajectory_directories": raw_trajectories,
        "excluded_used_trajectories": len(used),
        "excluded_used_routes": len(used_routes),
        "fresh_candidate_trajectories": len(candidates),
        "fresh_candidate_routes": len({value["route_group"] for value in candidates}),
        "selected_trajectories": len(selected),
        "selected_routes": len({value["route_group"] for value in selected}),
        "selected_bags": len({value["bag_id"] for value in selected}),
        "manifest": str(args.output),
        "manifest_sha256": sha256(args.output) if enough else None,
        "excluded_universe": str(universe_path),
        "excluded_universe_sha256": sha256(universe_path),
        "used_sources": sources,
        "train_cal_cem_route_overlap": [],
        "prior_artifact_trajectory_overlap": [],
        "labels_opened": False,
        "pose_outcomes_opened": False,
        "rgb_only": True,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not enough:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
