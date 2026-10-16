#!/usr/bin/env python3
"""Freeze an unused route-disjoint RGB-only RECON fresh H4-100 manifest."""

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

from scoring.strict_inner_partitions import recon_route_id  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


TARGET_ROUTES = 25
WINDOWS_PER_ROUTE = 4
TARGET_WINDOWS = TARGET_ROUTES * WINDOWS_PER_ROUTE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/recon")
    )
    parser.add_argument(
        "--split-lock",
        type=Path,
        default=REPO_ROOT / "data/split_lock/split_lock.json",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[REPO_ROOT / "outputs", Path("/home/rainbow/ndwm/DWM/reports")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "outputs/universal_dualdomain_raw_v2/recon_fresh_outer_v2/"
        "manifest.windows.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def stable_key(seed: int, *values: object) -> str:
    return hashlib.sha256(
        "|".join((str(seed), *(str(value) for value in values))).encode()
    ).hexdigest()


def union_sha(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def npz_trajectories(path: Path) -> set[str]:
    result: set[str] = set()
    with np.load(path, allow_pickle=False) as payload:
        if "record_key" in payload.files:
            for value in payload["record_key"].astype(str).reshape(-1):
                trajectory, separator, _ = value.rpartition(":")
                if separator:
                    result.add(trajectory)
        for key in ("trajectory", "trajectory_id", "traj_name"):
            if key in payload.files:
                result.update(payload[key].astype(str).reshape(-1).tolist())
    return result


def used_identity(scan_roots: list[Path]) -> tuple[set[str], dict[str, Any]]:
    trajectories: set[str] = set()
    scanned = 0
    identity_files = 0
    failures = []
    seen: set[Path] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        for directory, _, files in os.walk(root):
            for name in files:
                if not name.endswith(".npz"):
                    continue
                path = (Path(directory) / name).resolve()
                if path in seen:
                    continue
                seen.add(path)
                scanned += 1
                try:
                    values = npz_trajectories(path)
                except Exception as error:
                    failures.append({"path": str(path), "error": repr(error)})
                    continue
                if values:
                    identity_files += 1
                    trajectories.update(values)
    for partition in ("train-fit", "scorer-cal", "cem-cal", "internal-holdout"):
        path = REPO_ROOT / f"data/inner_manifests/recon/{partition}.windows.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trajectories.add(str(json.loads(line)["trajectory_id"]))
    return trajectories, {
        "npz_files_scanned": scanned,
        "identity_files": identity_files,
        "scan_failures": failures,
    }


def numeric_images(path: Path) -> set[int]:
    return {
        int(image.stem)
        for image in path.glob("*.jpg")
        if image.stem.isdigit()
    }


def valid_currents(images: set[int]) -> list[int]:
    return sorted(
        current
        for current in images
        if current >= 3
        and all(value in images for value in range(current - 3, current + 1))
        and current + 4 in images
    )


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


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    excluded_path = args.output.with_suffix(".excluded_identity.json")
    if args.output.exists() or audit_path.exists() or excluded_path.exists():
        raise FileExistsError(args.output)
    split_lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    test_trajectories = {
        str(value)
        for value in split_lock["recon"]["splits"]["test"]
    }
    test_routes = {recon_route_id(value) for value in test_trajectories}
    used_trajectories, scan_audit = used_identity(args.scan_root)
    used_routes = {recon_route_id(value) for value in used_trajectories}

    candidates: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(args.trajectory_root.iterdir()):
        if not path.is_dir() or path.name not in test_trajectories:
            continue
        route = recon_route_id(path.name)
        if path.name in used_trajectories or route in used_routes:
            continue
        currents = valid_currents(numeric_images(path))
        if len(currents) < WINDOWS_PER_ROUTE:
            continue
        if not (path / "traj_data.pkl").is_file():
            continue
        candidates.setdefault(route, []).append(
            {
                "trajectory_id": path.name,
                "route_id": route,
                "valid_currents": currents,
            }
        )
    route_order = sorted(
        candidates,
        key=lambda route: stable_key(args.seed, "recon-fresh-route", route),
    )
    selected = []
    for route in route_order[:TARGET_ROUTES]:
        choices = sorted(
            candidates[route],
            key=lambda value: stable_key(
                args.seed,
                "recon-fresh-trajectory",
                route,
                value["trajectory_id"],
            ),
        )
        selected.append(choices[0])
    if len(selected) != TARGET_ROUTES:
        raise RuntimeError(
            f"only {len(selected)} unused test routes are H4-ready"
        )

    rows = []
    for value in selected:
        currents = value["valid_currents"]
        positions = np.linspace(
            0, len(currents) - 1, WINDOWS_PER_ROUTE
        ).round().astype(int)
        chosen = [currents[index] for index in positions]
        if len(set(chosen)) != WINDOWS_PER_ROUTE:
            raise RuntimeError("fresh temporal selection duplicated")
        for temporal_bin, current in enumerate(chosen):
            uid = hashlib.sha256(
                (
                    f"recon-fresh-outer-v2|{args.seed}|"
                    f"{value['trajectory_id']}|{current}"
                ).encode()
            ).hexdigest()[:24]
            rows.append(
                {
                    "dataset": "recon",
                    "partition": "fresh-outer",
                    "trajectory_id": value["trajectory_id"],
                    "current_t": int(current),
                    "history_timesteps": list(range(current - 3, current + 1)),
                    "goal_t": int(current + 4),
                    "route_id": value["route_id"],
                    "route_group": value["route_id"],
                    "temporal_bin": temporal_bin,
                    "window_uid": uid,
                    "selection_seed": args.seed,
                    "selection_source": (
                        "split-lock test + RGB filenames only; no future pose access"
                    ),
                }
            )
    rows.sort(
        key=lambda row: stable_key(
            args.seed,
            "recon-fresh-window-order",
            row["trajectory_id"],
            row["current_t"],
        )
    )
    if len(rows) != TARGET_WINDOWS:
        raise RuntimeError("fresh RECON outer is not H4-100")
    write_jsonl(args.output, rows)
    excluded = {
        "schema": "recon_fresh_outer_v2_excluded_identity",
        "used_trajectories": sorted(used_trajectories),
        "used_routes": sorted(used_routes),
        "used_trajectory_union_sha256": union_sha(used_trajectories),
        "used_route_union_sha256": union_sha(used_routes),
    }
    write_json(excluded_path, excluded)
    audit = {
        "schema": "recon_fresh_outer_manifest_freeze_v2",
        "status": "manifest_frozen_rgb_only",
        "seed": args.seed,
        "states": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "route_groups": len({row["route_id"] for row in rows}),
        "split_lock": str(args.split_lock),
        "split_lock_sha256": file_sha256(args.split_lock),
        "test_trajectories": len(test_trajectories),
        "test_routes": len(test_routes),
        "excluded_used_trajectories": len(used_trajectories),
        "excluded_used_routes": len(used_routes),
        "available_unused_h4_routes": len(candidates),
        "manifest": str(args.output),
        "manifest_sha256": file_sha256(args.output),
        "excluded_identity": str(excluded_path),
        "excluded_identity_sha256": file_sha256(excluded_path),
        "selected_trajectory_overlap": sorted(
            {row["trajectory_id"] for row in rows}.intersection(used_trajectories)
        ),
        "selected_route_overlap": sorted(
            {row["route_id"] for row in rows}.intersection(used_routes)
        ),
        "artifact_scan": scan_audit,
        "labels_opened": False,
        "future_pose_opened": False,
        "rgb_only": True,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
