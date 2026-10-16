#!/usr/bin/env python3
"""Read-only second-level audit of RECON data and frozen blocker artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROUTE_SUFFIX = re.compile(r"_r\d+$")
TRAJECTORY_PATTERN = re.compile(r"^jackal_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\d+_r\d+$")


def route_id(value: str) -> str:
    return ROUTE_SUFFIX.sub("", value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def identities_from_npz(path: Path) -> set[str]:
    result: set[str] = set()
    with np.load(path, allow_pickle=False) as payload:
        if "record_key" in payload.files:
            for value in payload["record_key"].astype(str).reshape(-1):
                trajectory, separator, _ = value.rpartition(":")
                if separator and TRAJECTORY_PATTERN.match(trajectory):
                    result.add(trajectory)
        for key in ("trajectory", "trajectory_id", "traj_name"):
            if key in payload.files:
                for value in payload[key].astype(str).reshape(-1):
                    if TRAJECTORY_PATTERN.match(value):
                        result.add(value)
    return result


def scan_used(scan_roots: list[Path]) -> tuple[set[str], dict[str, Any]]:
    trajectories: set[str] = set()
    npz_count = 0
    jsonl_count = 0
    identity_files = 0
    failures: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        for directory, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in {".git", "__pycache__"}]
            for name in files:
                path = (Path(directory) / name).resolve()
                if path in seen:
                    continue
                if name.endswith(".npz"):
                    seen.add(path)
                    npz_count += 1
                    try:
                        found = identities_from_npz(path)
                    except Exception as error:  # corrupt/legacy files are reported.
                        failures.append({"path": str(path), "error": repr(error)})
                        continue
                    if found:
                        identity_files += 1
                        trajectories.update(found)
                elif name.endswith(".windows.jsonl") or name == "manifest.windows.jsonl":
                    seen.add(path)
                    jsonl_count += 1
                    try:
                        found = {
                            str(row.get("trajectory_id", ""))
                            for row in json_rows(path)
                            if TRAJECTORY_PATTERN.match(str(row.get("trajectory_id", "")))
                        }
                    except Exception as error:
                        failures.append({"path": str(path), "error": repr(error)})
                        continue
                    if found:
                        identity_files += 1
                        trajectories.update(found)
    return trajectories, {
        "roots": [str(path) for path in scan_roots],
        "npz_files": npz_count,
        "window_jsonl_files": jsonl_count,
        "identity_files": identity_files,
        "failures": failures,
    }


def directory_identities(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.name for path in root.iterdir()
        if path.is_dir() and TRAJECTORY_PATTERN.match(path.name)
    }


def numeric_images(path: Path) -> set[int]:
    return {int(item.stem) for item in path.glob("*.jpg") if item.stem.isdigit()}


def valid_currents(path: Path) -> list[int]:
    images = numeric_images(path)
    return sorted(
        current for current in images
        if current >= 3
        and all(value in images for value in range(current - 3, current + 1))
        and current + 4 in images
    )


def manifest_routes(paths: list[Path]) -> tuple[set[str], set[str]]:
    trajectories: set[str] = set()
    routes: set[str] = set()
    for path in paths:
        for row in json_rows(path):
            trajectory = str(row["trajectory_id"])
            trajectories.add(trajectory)
            routes.add(str(row.get("route_id", route_id(trajectory))))
    return trajectories, routes


def txt_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if TRAJECTORY_PATTERN.match(line.strip())}


def split_summary(splits: dict[str, list[str]]) -> dict[str, Any]:
    route_sets = {name: {route_id(value) for value in values} for name, values in splits.items()}
    names = sorted(route_sets)
    overlaps = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps[f"{left}__{right}"] = len(route_sets[left].intersection(route_sets[right]))
    return {
        "trajectory_counts": {name: len(values) for name, values in splits.items()},
        "route_counts": {name: len(route_sets[name]) for name in splits},
        "route_overlap": overlaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/hard_blocker_reaudit_v54.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = REPO / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/state_gate_screen/result.json"
    blocker_path = REPO / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/hard_blocker_inventory.json"
    reserve = REPO / "outputs/cem_development_v1/recon_target_calibration_v20/final_confirmation_reserve"
    reserve_manifest = reserve / "manifest.windows.jsonl"
    reserve_audit_path = reserve / "manifest.windows.audit.json"
    reserve_freeze_path = reserve / "selection_v26/freeze.json"
    reserve_audit = json.loads(reserve_audit_path.read_text(encoding="utf-8"))
    reserve_freeze = json.loads(reserve_freeze_path.read_text(encoding="utf-8"))

    strict_path = REPO / "data/split_lock/recon_strict_splits.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_sets = {name: set(values) for name, values in strict.items()}
    strict_test_routes = {route_id(value) for value in strict_sets["test"]}

    canonical_root = Path("/home/rainbow/nwm/data/recon")
    roots = [
        canonical_root,
        Path("/home/rainbow/rrnwm/data/recon"),
        Path("/home/rainbow/DWM/recon_data/recon"),
        Path("/home/rainbow/DWM/data/raw/recon"),
    ]
    root_names = {str(root): directory_identities(root) for root in roots}
    canonical_names = root_names[str(canonical_root)]
    root_summary = {}
    for root in roots:
        names = root_names[str(root)]
        root_summary[str(root)] = {
            "exists": root.is_dir(),
            "trajectories": len(names),
            "routes": len({route_id(value) for value in names}),
            "new_trajectory_ids_vs_canonical": sorted(names - canonical_names),
            "missing_trajectory_ids_vs_canonical_count": len(canonical_names - names),
        }

    bank_manifest_paths = []
    for parent in (
        REPO / "outputs/cem_development_v1/recon_target_calibration_v20",
        REPO / "outputs/cem_development_v1/recon_target_development_v40",
        REPO / "outputs/cem_development_v1/recon_target_development_v44",
        REPO / "outputs/cem_development_v1/recon_target_development_v46",
    ):
        for path in parent.glob("bank_*/manifest.windows.jsonl"):
            bank = path.parent.name.removeprefix("bank_")
            if len(bank) == 1 and "a" <= bank <= "t":
                bank_manifest_paths.append(path)
    bank_manifest_paths.extend(path for path in (REPO / "outputs/cem_development_v1/recon_target_confirmation_v29/manifest.windows.jsonl", REPO / "outputs/cem_development_v1/recon_target_cem_confirmation_v35/manifest.windows.jsonl") if path.is_file())
    bank_manifest_paths.sort()
    at_trajectories, at_routes = manifest_routes(bank_manifest_paths)
    reserve_trajectories, reserve_routes = manifest_routes([reserve_manifest])

    scan_roots = [
        REPO,
        Path("/home/rainbow/ndwm/DWM/reports"),
        Path("/home/rainbow/rrnwm/eval_output"),
        Path("/home/rainbow/DWM/reports"),
    ]
    used_trajectories, scan_audit = scan_used(scan_roots)
    used_routes = {route_id(value) for value in used_trajectories}
    used_test_routes = strict_test_routes.intersection(used_routes)
    historical_test_routes = used_test_routes - at_routes - reserve_routes
    unused_test_routes = strict_test_routes - used_test_routes

    unused_h4 = {}
    for trajectory in sorted(strict_sets["test"]):
        route = route_id(trajectory)
        if route not in unused_test_routes:
            continue
        path = canonical_root / trajectory
        currents = valid_currents(path) if (path / "traj_data.pkl").is_file() else []
        if len(currents) >= 4:
            unused_h4.setdefault(route, []).append({
                "trajectory_id": trajectory,
                "h4_ready_currents": len(currents),
            })

    nwm_split_root = Path("/home/rainbow/nwm/data_splits/recon")
    rr_split_root = Path("/home/rainbow/rrnwm/data_splits/recon")
    text_split_nwm = {
        "train": sorted(txt_names(nwm_split_root / "train/traj_names.txt")),
        "test": sorted(txt_names(nwm_split_root / "test/traj_names.txt")),
    }
    text_split_rr = {
        "train": sorted(txt_names(rr_split_root / "train/traj_names.txt")),
        "test": sorted(txt_names(rr_split_root / "test/traj_names.txt")),
    }
    server_split_path = rr_split_root / "server_full/splits.json"
    server_split = json.loads(server_split_path.read_text(encoding="utf-8"))
    legacy_benchmark_path = rr_split_root / "server_full_nwm_unseen_h4_100/manifest.json"
    legacy_benchmark = json.loads(legacy_benchmark_path.read_text(encoding="utf-8"))
    legacy_trajectories = {str(row["trajectory_id"]) for row in legacy_benchmark["records"]}
    legacy_routes = {route_id(value) for value in legacy_trajectories}

    report = {
        "schema": "recon_hard_blocker_second_level_reaudit_v54",
        "mode": "read_only_no_reserve_outcomes_no_model_training",
        "artifact_sha256": {
            str(result_path.resolve()): sha256(result_path),
            str(blocker_path.resolve()): sha256(blocker_path),
            str(reserve_manifest.resolve()): sha256(reserve_manifest),
            str(reserve_audit_path.resolve()): sha256(reserve_audit_path),
            str(reserve_freeze_path.resolve()): sha256(reserve_freeze_path),
        },
        "reserve_reverification": {
            "labels_opened_manifest_audit": reserve_audit.get("labels_opened"),
            "future_pose_opened_manifest_audit": reserve_audit.get("future_pose_opened"),
            "labels_opened_selection_freeze": reserve_freeze.get("labels_opened"),
            "outcomes_accessed_selection_freeze": reserve_freeze.get("outcomes_accessed"),
            "source_labeled_exists": (reserve / "source_labeled.npz").exists(),
            "outcome_named_files": sorted(
                str(path.relative_to(reserve)) for path in reserve.rglob("*")
                if path.is_file() and any(token in path.name.lower() for token in ("outcome", "labeled", "pose_target"))
            ),
        },
        "canonical_data": {
            "root": str(canonical_root),
            "trajectory_directories": len(canonical_names),
            "route_ids": len({route_id(value) for value in canonical_names}),
            "strict_split_sha256": sha256(strict_path),
            "strict_split": split_summary(strict),
            "strict_union_trajectories": len(set().union(*strict_sets.values())),
            "strict_union_routes": len({route_id(value) for values in strict_sets.values() for value in values}),
            "root_not_in_strict_split": sorted(canonical_names - set().union(*strict_sets.values())),
            "strict_split_missing_from_root": sorted(set().union(*strict_sets.values()) - canonical_names),
        },
        "data_roots": root_summary,
        "history_decomposition_within_strict_test": {
            "strict_test_trajectories": len(strict_sets["test"]),
            "strict_test_routes": len(strict_test_routes),
            "historical_other_routes": len(historical_test_routes),
            "at_routes": len(at_routes),
            "reserve_routes": len(reserve_routes),
            "unused_routes": len(unused_test_routes),
            "partition_disjoint": not (
                historical_test_routes.intersection(at_routes)
                or historical_test_routes.intersection(reserve_routes)
                or at_routes.intersection(reserve_routes)
                or unused_test_routes.intersection(used_test_routes)
            ),
            "unused_route_ids": sorted(unused_test_routes),
            "unused_h4_ready_routes": len(unused_h4),
            "unused_h4_ready_trajectories": sum(len(value) for value in unused_h4.values()),
            "unused_h4_ready_detail": unused_h4,
            "max_states_at_4_per_route": 4 * len(unused_h4),
        },
        "used_identity_scan": {
            **scan_audit,
            "trajectories": len(used_trajectories),
            "routes": len(used_routes),
            "strict_test_trajectories": len(strict_sets["test"].intersection(used_trajectories)),
            "strict_test_routes": len(used_test_routes),
        },
        "alternative_splits": {
            "nwm_text": split_summary(text_split_nwm),
            "rrnwm_text": split_summary(text_split_rr),
            "server_full": split_summary(server_split),
            "server_full_sha256": sha256(server_split_path),
            "server_full_union_trajectories": len(set().union(*(set(v) for v in server_split.values()))),
            "server_full_new_ids_vs_canonical": sorted(set().union(*(set(v) for v in server_split.values())) - canonical_names),
            "legal_under_current_route_disjoint_split_lock": False,
            "reason": "alternative trajectory-level splits have route overlap and cannot replace the frozen route-disjoint split",
        },
        "legacy_server_full_nwm_unseen_h4_100": {
            "manifest": str(legacy_benchmark_path),
            "manifest_sha256": sha256(legacy_benchmark_path),
            "trajectories": len(legacy_trajectories),
            "routes": len(legacy_routes),
            "trajectory_overlap_with_used": len(legacy_trajectories.intersection(used_trajectories)),
            "route_overlap_with_used": len(legacy_routes.intersection(used_routes)),
            "unused_routes": sorted(legacy_routes - used_routes),
            "new_trajectory_ids_vs_canonical": sorted(legacy_trajectories - canonical_names),
            "is_new_data_root": False,
        },
        "at_manifest_count": len(bank_manifest_paths),
        "at_trajectories": len(at_trajectories),
        "at_routes": len(at_routes),
        "reserve_trajectories": len(reserve_trajectories),
        "reserve_routes": len(reserve_routes),
    }
    if report["reserve_reverification"] != {
        "labels_opened_manifest_audit": False,
        "future_pose_opened_manifest_audit": False,
        "labels_opened_selection_freeze": False,
        "outcomes_accessed_selection_freeze": False,
        "source_labeled_exists": False,
        "outcome_named_files": [],
    }:
        raise RuntimeError("reserve seal re-verification failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "canonical": report["canonical_data"],
        "decomposition": report["history_decomposition_within_strict_test"],
        "scan": report["used_identity_scan"],
        "data_roots": report["data_roots"],
        "alternative_splits": report["alternative_splits"],
        "legacy_benchmark": report["legacy_server_full_nwm_unseen_h4_100"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
