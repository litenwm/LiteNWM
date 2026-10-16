#!/usr/bin/env python3
"""Freeze a leakage-audited RECON/SCAND/SACSoN H4 successor split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-split-lock", type=Path, required=True)
    parser.add_argument("--adapter-parent-split-lock", type=Path, required=True)
    parser.add_argument("--scand-root", type=Path, required=True)
    parser.add_argument("--scand100-test-list", type=Path, required=True)
    parser.add_argument("--sacson-root", type=Path, required=True)
    parser.add_argument("--sacson-rrnwm-train-list", type=Path, required=True)
    parser.add_argument("--sacson-rrnwm-test-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--scand-val-route-fraction", type=float, default=0.10)
    parser.add_argument("--sacson-val-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-h4-frames", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable(seed: int, *values: str) -> str:
    return hashlib.sha256(
        "|".join((str(seed), *map(str, values))).encode("utf-8")
    ).hexdigest()


def read_names(path: Path, *, allow_duplicates: bool = False) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)) and not allow_duplicates:
        raise ValueError(f"duplicate trajectory identities in {path}")
    return list(dict.fromkeys(values)) if allow_duplicates else values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_names(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def scand_route(trajectory_id: str) -> str:
    parts = trajectory_id.split("_")
    if len(parts) != 11 or parts[:2] != ["random", "mdps"]:
        raise ValueError(f"unsupported SCAND trajectory ID: {trajectory_id}")
    return "__".join(sorted((parts[4].strip().lower(), parts[5].strip().lower())))


def sacson_session(trajectory_id: str) -> str:
    session = re.sub(r"_\d+_\d+$", "", trajectory_id)
    if session == trajectory_id:
        raise ValueError(f"unsupported SACSoN trajectory ID: {trajectory_id}")
    return session


def scan_root(root: Path, minimum_frames: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    eligible: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        images = list(path.glob("*.jpg"))
        frames = len(images)
        pkl = path / "traj_data.pkl"
        reason = None
        if not pkl.is_file():
            reason = "missing_traj_data_pickle"
        elif frames < minimum_frames:
            reason = "fewer_than_two_frames_not_encodable"
        record = {
            "trajectory_id": path.name,
            "num_frames": frames,
            "h4_windows": max(frames - 7, 0),
            "source_dir": str(path),
        }
        if reason is None:
            eligible[path.name] = record
        else:
            excluded.append({**record, "reason": reason})
    return eligible, excluded


def assert_exact_disjoint(splits: dict[str, list[str]], dataset: str) -> None:
    sets = {name: set(splits[name]) for name in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"{dataset} {left}/{right} overlap: {sorted(overlap)[:5]}")


def assert_scand_route_disjoint(splits: dict[str, list[str]]) -> dict[str, int]:
    routes = {split: {scand_route(value) for value in values} for split, values in splits.items()}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = routes[left] & routes[right]
            if overlap:
                raise ValueError(f"SCAND route leakage {left}/{right}: {sorted(overlap)}")
    return {split: len(routes[split]) for split in SPLITS}


def choose_extra_scand_val_routes(
    candidates: list[str],
    route_records: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    target_frames: float,
    target_windows: float,
    existing_frames: int,
    existing_windows: int,
    seed: int,
) -> set[str]:
    if count <= 0:
        return set()
    if count > len(candidates):
        raise ValueError("not enough novel SCAND routes for requested validation target")

    stats = {
        route: {
            "frames": sum(int(row["num_frames"]) for row in route_records[route]),
            "windows": sum(int(row["h4_windows"]) for row in route_records[route]),
        }
        for route in candidates
    }

    def objective(selected: set[str]) -> float:
        frames = existing_frames + sum(stats[route]["frames"] for route in selected)
        windows = existing_windows + sum(stats[route]["windows"] for route in selected)
        return (
            ((frames - target_frames) / max(target_frames, 1.0)) ** 2
            + ((windows - target_windows) / max(target_windows, 1.0)) ** 2
        ) / 2.0

    selected: set[str] = set()
    while len(selected) < count:
        selected.add(
            min(
                (route for route in candidates if route not in selected),
                key=lambda route: (
                    objective(selected | {route}),
                    stable(seed, "scand_val_greedy", route),
                ),
            )
        )

    while True:
        current = objective(selected)
        swaps = []
        for left in sorted(selected):
            for right in sorted(set(candidates) - selected):
                proposal = (selected - {left}) | {right}
                swaps.append(
                    (
                        objective(proposal),
                        stable(seed, "scand_val_swap", left, right),
                        left,
                        right,
                    )
                )
        if not swaps:
            break
        best = min(swaps)
        if best[0] >= current - 1.0e-12:
            break
        selected.remove(best[2])
        selected.add(best[3])
    return selected


def build_scand(
    active: dict[str, Any],
    root: Path,
    test_list: Path,
    *,
    minimum_frames: int,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    eligible, excluded = scan_root(root, 2)
    eligible = {trajectory_id: record for trajectory_id, record in eligible.items() if trajectory_id.count(chr(95)) == 10}
    test_ids = read_names(test_list, allow_duplicates=True)
    missing_test = sorted(set(test_ids) - set(eligible))
    if missing_test:
        raise FileNotFoundError(f"SCAND100 H4 identities missing locally: {missing_test}")
    fixed_test_routes = {scand_route(value) for value in test_ids}
    if len(fixed_test_routes) != 7:
        raise ValueError(f"expected seven SCAND100 routes, got {len(fixed_test_routes)}")

    inherited = {
        str(value): split
        for split in SPLITS
        for value in active["scand"]["splits"][split]
    }
    missing_inherited = sorted(set(inherited) - set(eligible))
    if missing_inherited:
        raise FileNotFoundError(f"active SCAND identities absent from full root: {missing_inherited}")

    route_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory_id, record in eligible.items():
        route_records[scand_route(trajectory_id)].append(record)
    if len(route_records) != 77:
        raise ValueError(f"full SCAND route universe changed: {len(route_records)} != 77")

    inherited_route_assignment: dict[str, str] = {}
    for trajectory_id, split in inherited.items():
        route = scand_route(trajectory_id)
        previous = inherited_route_assignment.setdefault(route, split)
        if previous != split:
            raise ValueError(f"active SCAND route crosses splits: {route}")
    for route in fixed_test_routes:
        previous = inherited_route_assignment.get(route)
        if previous is not None and previous != "test":
            raise ValueError(f"SCAND100 route was previously development data: {route}")
    inherited_test_routes = {
        route for route, split in inherited_route_assignment.items() if split == "test"
    }
    if inherited_test_routes != fixed_test_routes:
        raise ValueError(
            "SCAND100 route seal differs from active test routes: "
            f"active_only={sorted(inherited_test_routes-fixed_test_routes)} "
            f"scand100_only={sorted(fixed_test_routes-inherited_test_routes)}"
        )

    dev_routes = sorted(set(route_records) - fixed_test_routes)
    desired_val_routes = max(1, round(len(dev_routes) * val_fraction))
    inherited_val_routes = {
        route for route, split in inherited_route_assignment.items() if split == "val"
    }
    inherited_train_routes = {
        route for route, split in inherited_route_assignment.items() if split == "train"
    }
    novel_dev_routes = sorted(
        set(dev_routes) - inherited_val_routes - inherited_train_routes
    )
    extra_count = max(0, desired_val_routes - len(inherited_val_routes))
    total_dev_frames = sum(
        int(row["num_frames"]) for route in dev_routes for row in route_records[route]
    )
    total_dev_windows = sum(
        int(row["h4_windows"]) for route in dev_routes for row in route_records[route]
    )
    inherited_val_frames = sum(
        int(row["num_frames"])
        for route in inherited_val_routes
        for row in route_records[route]
    )
    inherited_val_windows = sum(
        int(row["h4_windows"])
        for route in inherited_val_routes
        for row in route_records[route]
    )
    extra_val_routes = choose_extra_scand_val_routes(
        novel_dev_routes,
        route_records,
        count=extra_count,
        target_frames=total_dev_frames * val_fraction,
        target_windows=total_dev_windows * val_fraction,
        existing_frames=inherited_val_frames,
        existing_windows=inherited_val_windows,
        seed=seed,
    )
    assignment = dict(inherited_route_assignment)
    for route in route_records:
        if route in fixed_test_routes:
            split = "test"
        elif route in inherited_val_routes or route in extra_val_routes:
            split = "val"
        else:
            split = "train"
        previous = assignment.setdefault(route, split)
        if previous != split:
            raise ValueError(f"SCAND inherited assignment moved: {route} {previous}->{split}")

    splits = {
        split: sorted(
            trajectory_id
            for trajectory_id in eligible
            if assignment[scand_route(trajectory_id)] == split
        )
        for split in SPLITS
    }
    assert_exact_disjoint(splits, "scand")
    route_counts = assert_scand_route_disjoint(splits)
    for trajectory_id, split in inherited.items():
        if trajectory_id not in splits[split]:
            raise ValueError(f"SCAND active identity moved: {trajectory_id}")
    for split in SPLITS:
        if split == "test" and not set(test_ids).issubset(splits[split]):
            raise ValueError("SCAND100 identities are not fully sealed in test")

    route_rows = []
    for route in sorted(route_records):
        rows = route_records[route]
        route_rows.append(
            {
                "route_group": route,
                "split": assignment[route],
                "inherited_split": inherited_route_assignment.get(route),
                "scand100_outer_test_route": route in fixed_test_routes,
                "trajectories": len(rows),
                "frames": sum(int(row["num_frames"]) for row in rows),
                "h4_windows": sum(int(row["h4_windows"]) for row in rows),
            }
        )
    audit = {
        "status": "PASS",
        "allocation_seed": seed,
        "policy": "preserve_v70_membership_extend_route_isolated_dev_seal_scand100_routes",
        "eligible_trajectories": len(eligible),
        "excluded": excluded,
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "route_counts": route_counts,
        "fixed_scand100_h4_identities": len(test_ids),
        "fixed_outer_test_routes": sorted(fixed_test_routes),
        "newly_sealed_test_route_trajectories": len(splits["test"]) - len(test_ids),
        "desired_val_route_fraction": val_fraction,
        "desired_val_route_count": desired_val_routes,
        "inherited_val_routes": sorted(inherited_val_routes),
        "extra_val_routes": sorted(extra_val_routes),
        "route_assignment": route_rows,
    }
    return splits, audit


def build_sacson(
    root: Path,
    rrnwm_train_path: Path,
    rrnwm_test_path: Path,
    *,
    minimum_frames: int,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    eligible, excluded = scan_root(root, 2)
    rrnwm_train = read_names(rrnwm_train_path)
    rrnwm_test = read_names(rrnwm_test_path)
    if set(rrnwm_train) & set(rrnwm_test):
        raise ValueError("RRNWM SACSoN train/test identities overlap")
    known = set(rrnwm_train) | set(rrnwm_test)
    unexpected = sorted(set(eligible) - known)
    if unexpected:
        raise ValueError(f"uploaded SACSoN identities absent from RRNWM split: {unexpected[:10]}")

    available_original_train = sorted(set(rrnwm_train) & set(eligible))
    available_test = sorted(set(rrnwm_test) & set(eligible))
    by_session: dict[str, list[str]] = defaultdict(list)
    for trajectory_id in available_original_train:
        by_session[sacson_session(trajectory_id)].append(trajectory_id)
    val: set[str] = set()
    for session, values in sorted(by_session.items()):
        ordered = sorted(values, key=lambda value: stable(seed, "sacson_val", session, value))
        count = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
        val.update(ordered[:count])
    splits = {
        "train": sorted(set(available_original_train) - val),
        "val": sorted(val),
        "test": available_test,
    }
    assert_exact_disjoint(splits, "sacson")
    if not set(splits["train"]) | set(splits["val"]) <= set(rrnwm_train):
        raise ValueError("SACSoN development data escaped the RRNWM train membership")
    if set(splits["test"]) - set(rrnwm_test):
        raise ValueError("SACSoN test data escaped the RRNWM test membership")

    full_root_ids = {item.name for item in root.iterdir() if item.is_dir()}
    audit = {
        "status": "PASS",
        "allocation_seed": seed,
        "policy": "preserve_rrnwm_exact_test_stratified_hash_val_within_original_train",
        "rrnwm": {
            "train_declared": len(rrnwm_train),
            "test_declared": len(rrnwm_test),
            "train_sha256": sha256(rrnwm_train_path),
            "test_sha256": sha256(rrnwm_test_path),
            "exact_overlap": 0,
        },
        "uploaded_root_trajectories": len(full_root_ids),
        "encodable_trajectories": len(eligible),
        "excluded": excluded,
        "missing_from_uploaded_root": {
            "rrnwm_train": sorted(set(rrnwm_train) - full_root_ids),
            "rrnwm_test": sorted(set(rrnwm_test) - full_root_ids),
        },
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "session_counts": {
            split: len({sacson_session(value) for value in splits[split]})
            for split in SPLITS
        },
        "session_overlap_note": (
            "RRNWM's frozen test is trajectory-random within the same 27 sessions; "
            "exact identity is sealed, but session identity is intentionally shared."
        ),
        "val_fraction_within_available_rrnwm_train": val_fraction,
    }
    return splits, audit


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.minimum_h4_frames != 8:
        raise ValueError("this successor protocol is fixed at four history plus four future frames")
    active = json.loads(args.active_split_lock.read_text(encoding="utf-8"))
    adapter_parent = json.loads(args.adapter_parent_split_lock.read_text(encoding="utf-8"))
    active_sha = sha256(args.active_split_lock)
    adapter_parent_sha = sha256(args.adapter_parent_split_lock)
    if active["recon"]["splits"] != adapter_parent["recon"]["splits"]:
        raise ValueError("RECON membership changed between adapter parent and active v70 lock")

    scand_splits, scand_audit = build_scand(
        active,
        args.scand_root,
        args.scand100_test_list,
        minimum_frames=args.minimum_h4_frames,
        val_fraction=args.scand_val_route_fraction,
        seed=args.seed,
    )
    sacson_splits, sacson_audit = build_sacson(
        args.sacson_root,
        args.sacson_rrnwm_train_list,
        args.sacson_rrnwm_test_list,
        minimum_frames=args.minimum_h4_frames,
        val_fraction=args.sacson_val_fraction,
        seed=args.seed,
    )
    recon_splits = {
        split: sorted(map(str, active["recon"]["splits"][split]))
        for split in SPLITS
    }
    assert_exact_disjoint(recon_splits, "recon")

    split_lock = {
        "schema_version": 3,
        "experiment": "ltwm_v70_successor_tridomain_h4_v1_20260727",
        "policy": {
            "history_frames": 4,
            "future_frames": 4,
            "recon": "unchanged_v70_parent_membership",
            "scand": "preserve_v70_membership_expand_all_routes_seal_scand100_route_groups",
            "sacson": "preserve_rrnwm_exact_test_split_val_only_from_original_train",
            "outer_tests_never_encoded_for_training": True,
        },
        "lineage": {
            "adapter_parent_split_lock": str(args.adapter_parent_split_lock),
            "adapter_parent_split_lock_sha256": adapter_parent_sha,
            "warm_start_parent_split_lock": str(args.active_split_lock),
            "warm_start_parent_split_lock_sha256": active_sha,
        },
        "recon": {"splits": recon_splits},
        "scand": {
            "splits": scand_splits,
            "route_assignment": scand_audit["route_assignment"],
            "outer_test_protocol": "SCAND100-H4 seed17; all seven route groups sealed",
        },
        "sacson": {
            "splits": sacson_splits,
            "outer_test_protocol": "RRNWM frozen SACSoN test membership, H4-eligible intersection",
        },
        "training_preregistration": {
            "domains": ["recon", "scand", "sacson"],
            "batch_domain_purity": True,
            "selection_domain": "scand",
            "seeds": [1701, 1702, 1703],
        },
    }
    args.output_dir.mkdir(parents=True)
    lock_path = args.output_dir / "split_lock.json"
    write_json(lock_path, split_lock)
    lock_sha = sha256(lock_path)
    (args.output_dir / "split_lock.json.sha256").write_text(
        f"{lock_sha}  split_lock.json\n", encoding="utf-8"
    )
    for dataset, splits in (
        ("recon", recon_splits),
        ("scand", scand_splits),
        ("sacson", sacson_splits),
    ):
        write_json(args.output_dir / f"{dataset}_strict_splits.json", splits)
        for split in SPLITS:
            write_names(args.output_dir / dataset / split / "traj_names.txt", splits[split])
    write_names(
        args.output_dir / "scand" / "outer_test_scand100_h4" / "traj_names.txt",
        read_names(args.scand100_test_list, allow_duplicates=True),
    )
    write_names(
        args.output_dir / "sacson" / "outer_test_rrnwm_declared" / "traj_names.txt",
        read_names(args.sacson_rrnwm_test_list),
    )
    audit = {
        "status": "PASS",
        "split_lock": str(lock_path),
        "split_lock_sha256": lock_sha,
        "active_v70_parent_sha256": active_sha,
        "adapter_parent_sha256": adapter_parent_sha,
        "recon": {
            "status": "PASS",
            "unchanged_from_v70": True,
            "split_counts": {split: len(recon_splits[split]) for split in SPLITS},
        },
        "scand": scand_audit,
        "sacson": sacson_audit,
        "cross_dataset_identity_overlap": {
            f"{left}__{right}": len(
                {
                    item
                    for split in SPLITS
                    for item in split_lock[left]["splits"][split]
                }
                & {
                    item
                    for split in SPLITS
                    for item in split_lock[right]["splits"][split]
                }
            )
            for left, right in (("recon", "scand"), ("recon", "sacson"), ("scand", "sacson"))
        },
    }
    if any(audit["cross_dataset_identity_overlap"].values()):
        raise ValueError("trajectory identity collision across datasets")
    write_json(args.output_dir / "split_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
