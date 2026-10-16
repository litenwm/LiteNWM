#!/usr/bin/env python3
"""Freeze coverage, a label-free pilot manifest, and the pilot protocol.

This script deliberately reads only split identities, identity arrays from
already-produced NPZ files, image filenames/metadata, and hash-only registries.
It never opens ``traj_data.pkl`` or any candidate outcome/target array.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import identity_hash, sha256_file  # noqa: E402
from scoring.strict_inner_partitions import (  # noqa: E402
    normalize_scand_route,
    parse_scand_trajectory_id,
)


OUTPUT = ROOT / "outputs/cem_development_v1/scand_existing_train_upperbound_pilot_v1_corrected"
TRAIN_NAMES = Path("/home/rainbow/nwm/data_splits/scand/train/traj_names.txt")
TRAJECTORY_ROOT = Path("/home/rainbow/nwm/data/scand")
USED_REGISTRY = ROOT / "outputs/proposal_tail_collection_v1/used_v70_v77_identities.json"
STRICT_TEST = ROOT / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json"
INNER_MANIFESTS = tuple(
    ROOT / f"data/inner_manifests/scand/{partition}.windows.jsonl"
    for partition in ("train-fit", "scorer-cal", "cem-cal", "internal-holdout")
)
SOURCE_NPZ = (
    ROOT / "outputs/scand_domain_adaptation_v1/sources/train_fit_source.npz",
    ROOT / "outputs/scand_domain_adaptation_v1/sources/scorer_cal_source.npz",
    ROOT / "outputs/scand_domain_adaptation_v1/sources/internal_holdout_source.npz",
)
V66_LABEL_FREE = tuple(
    ROOT
    / f"outputs/cem_development_v1/dualdomain_cem_tail_v66/harvest/{split}/iteration_{iteration:02d}/source_label_free.npz"
    for split in ("scand_train_fit", "scand_scorer_cal")
    for iteration in range(3)
)
FORMAL_V70 = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
)
FORMAL_RECON_SELECTED = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/"
    "replays/recon_bank_g/proposal_aware_v70/selected_actions.npz"
)

SELECTION_SEED = 2026072201
MAX_BAGS = 15
MIN_BAGS = 12
PER_BAG_CAP = 40
TARGET_PER_BAG = 36
MIN_WINDOWS_PER_BAG = 16


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_key(*parts: object) -> str:
    return hashlib.sha256(
        "|".join((str(SELECTION_SEED), *(str(value) for value in parts))).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identity_arrays(path: Path) -> tuple[set[str], set[str]]:
    """Read identity arrays only; target/outcome arrays are never materialized."""

    trajectories: set[str] = set()
    records: set[str] = set()
    with np.load(path, allow_pickle=False) as payload:
        for key in ("trajectory", "trajectory_id", "traj_name"):
            if key in payload.files:
                trajectories.update(payload[key].astype(str).reshape(-1).tolist())
        if "record_key" in payload.files:
            records.update(payload["record_key"].astype(str).reshape(-1).tolist())
    for record in records:
        trajectory, separator, _ = record.rpartition(":")
        if separator:
            trajectories.add(trajectory)
    return trajectories, records


def parse_train_trajectory(trajectory: str) -> dict[str, str]:
    if trajectory.startswith("random_mdps_"):
        value = parse_scand_trajectory_id(trajectory)
        parts = trajectory.split("_")
        return {**value, "day": "_".join(parts[6:9])}
    parts = trajectory.split("_")
    if len(parts) == 10 and parts[0] == "Bags":
        return {
            "survey": parts[1],
            "robot": parts[2],
            "left": parts[3],
            "right": parts[4],
            "bag_id": "random_mdps_" + "_".join(parts[1:-1]),
            "segment_id": parts[-1],
            "route_group": normalize_scand_route(parts[3], parts[4]),
            "day": "_".join(parts[5:8]),
        }
    raise ValueError(f"unsupported old-train SCAND trajectory ID: {trajectory}")


def image_count(trajectory: str) -> int:
    directory = TRAJECTORY_ROOT / trajectory
    images = list(directory.glob("*.jpg"))
    indices = sorted(int(path.stem) for path in images if path.stem.isdigit())
    if indices != list(range(len(indices))):
        raise RuntimeError(f"non-contiguous image sequence: {directory}")
    return len(indices)


def available_windows(trajectory: str) -> list[int]:
    count = image_count(trajectory)
    return list(range(3, count - 4)) if count >= 8 else []


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    names = [line.strip() for line in TRAIN_NAMES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 484 or len(set(names)) != 484:
        raise RuntimeError(f"old SCAND train split contract changed: {len(names)}")

    train_bags: dict[str, list[str]] = defaultdict(list)
    train_metadata: dict[str, dict[str, str]] = {}
    for trajectory in names:
        metadata = parse_train_trajectory(trajectory)
        train_bags[metadata["bag_id"]].append(trajectory)
        train_metadata[trajectory] = metadata
    if len(train_bags) != 111:
        raise RuntimeError(f"old SCAND bag contract changed: {len(train_bags)}")

    known_trajectories: set[str] = set()
    known_records: set[str] = set()
    known_bags: set[str] = set()
    known_routes: set[str] = set()
    evidence: list[dict[str, Any]] = []
    strict = json.loads(STRICT_TEST.read_text(encoding="utf-8"))
    strict_rows = list(strict["records"])
    for row in strict_rows:
        trajectory = str(row["trajectory_id"])
        known_trajectories.add(trajectory)
        known_bags.add(str(row["bag_id"]))
        known_routes.add(str(row["route_group"]))
        known_records.add(f"{trajectory}:{int(row['curr_time'])}")
    evidence.append(
        {
            "kind": "strict_test_identity_manifest_exclusion_only",
            "path": str(STRICT_TEST),
            "sha256": sha256_file(STRICT_TEST),
            "rows": len(strict_rows),
            "outcomes_opened": False,
        }
    )
    for path in INNER_MANIFESTS:
        rows = read_jsonl(path)
        known_trajectories.update(str(row["trajectory_id"]) for row in rows)
        known_bags.update(str(row["bag_id"]) for row in rows)
        known_routes.update(str(row["route_group"]) for row in rows)
        known_records.update(f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows)
        evidence.append({"kind": "manifest", "path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
    for path in (*SOURCE_NPZ, *V66_LABEL_FREE):
        trajectories, records = identity_arrays(path)
        known_trajectories.update(trajectories)
        known_records.update(records)
        for trajectory in trajectories:
            metadata = parse_train_trajectory(trajectory)
            known_bags.add(metadata["bag_id"])
            known_routes.add(metadata["route_group"])
        evidence.append(
            {
                "kind": "identity_arrays_only_npz",
                "path": str(path),
                "sha256": sha256_file(path),
                "trajectories": len(trajectories),
                "records": len(records),
            }
        )

    registry = json.loads(USED_REGISTRY.read_text(encoding="utf-8"))
    if not registry.get("complete") or not registry.get("identity_values_are_hash_only"):
        raise RuntimeError("used v70-v77 registry is not a complete hash-only registry")
    used_bag_hashes = set(map(str, registry["bag_hashes"]))
    used_route_hashes = set(map(str, registry["route_hashes"]))
    evidence.append({"kind": "hash_only_registry", "path": str(USED_REGISTRY), "sha256": sha256_file(USED_REGISTRY)})

    bag_stats: dict[str, dict[str, Any]] = {}
    for bag, trajectories in sorted(train_bags.items()):
        metadata = train_metadata[trajectories[0]]
        windows = {trajectory: available_windows(trajectory) for trajectory in sorted(trajectories)}
        count = sum(len(value) for value in windows.values())
        exact_known = bag in known_bags or identity_hash("bag", bag) in used_bag_hashes
        route_known = metadata["route_group"] in known_routes or identity_hash("route", metadata["route_group"]) in used_route_hashes
        bag_stats[bag] = {
            "bag_id": bag,
            "route_group": metadata["route_group"],
            "survey": metadata["survey"],
            "robot": metadata["robot"],
            "trajectories": len(trajectories),
            "available_h4_windows_label_free": count,
            "exact_bag_known_proposal_tail_or_cem": exact_known,
            "route_group_previously_seen": route_known,
            "used_registry_bag_hash_overlap": identity_hash("bag", bag) in used_bag_hashes,
            "used_registry_route_hash_overlap": identity_hash("route", metadata["route_group"]) in used_route_hashes,
        }

    candidates = [
        bag for bag, record in bag_stats.items()
        if not record["exact_bag_known_proposal_tail_or_cem"]
        and record["available_h4_windows_label_free"] >= MIN_WINDOWS_PER_BAG
    ]
    by_route: dict[str, list[str]] = defaultdict(list)
    for bag in candidates:
        by_route[bag_stats[bag]["route_group"]].append(bag)
    route_first = []
    for route, bags in by_route.items():
        best = sorted(
            bags,
            key=lambda bag: (
                bag_stats[bag]["route_group_previously_seen"],
                -min(PER_BAG_CAP, bag_stats[bag]["available_h4_windows_label_free"]),
                stable_key("bag", bag),
            ),
        )[0]
        route_first.append(best)
    selected_bags = sorted(
        route_first,
        key=lambda bag: (
            bag_stats[bag]["route_group_previously_seen"],
            -min(PER_BAG_CAP, bag_stats[bag]["available_h4_windows_label_free"]),
            stable_key("route-first", bag),
        ),
    )[:MAX_BAGS]
    if len(selected_bags) < MAX_BAGS:
        remaining = [bag for bag in candidates if bag not in selected_bags]
        route_counts = Counter(bag_stats[bag]["route_group"] for bag in selected_bags)
        remaining.sort(
            key=lambda bag: (
                route_counts[bag_stats[bag]["route_group"]],
                bag_stats[bag]["route_group_previously_seen"],
                -min(PER_BAG_CAP, bag_stats[bag]["available_h4_windows_label_free"]),
                stable_key("fill", bag),
            )
        )
        selected_bags.extend(remaining[: MAX_BAGS - len(selected_bags)])
    if len(selected_bags) < MIN_BAGS:
        raise RuntimeError(f"only {len(selected_bags)} eligible uncovered bags; pilot requires {MIN_BAGS}")

    quotas = {
        bag: min(TARGET_PER_BAG, bag_stats[bag]["available_h4_windows_label_free"])
        for bag in selected_bags
    }
    target = min(600, TARGET_PER_BAG * len(selected_bags))
    deficit = target - sum(quotas.values())
    for bag in sorted(selected_bags, key=lambda value: stable_key("redistribute", value)):
        extra = min(
            deficit,
            PER_BAG_CAP - quotas[bag],
            bag_stats[bag]["available_h4_windows_label_free"] - quotas[bag],
        )
        quotas[bag] += max(extra, 0)
        deficit -= max(extra, 0)
        if deficit <= 0:
            break

    rows: list[dict[str, Any]] = []
    for bag in selected_bags:
        windows = []
        for trajectory in sorted(train_bags[bag]):
            for current in available_windows(trajectory):
                windows.append((stable_key("window", bag, trajectory, current), trajectory, current))
        for _, trajectory, current in sorted(windows)[: quotas[bag]]:
            metadata = train_metadata[trajectory]
            history = list(range(current - 3, current + 1))
            goal = current + 4
            uid = hashlib.sha256(f"pilot-v1|{trajectory}|{current}|{goal}".encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "bag_id": bag,
                    "cohort": "scand_existing_train_upperbound_pilot_v1",
                    "context_image_paths": [str(TRAJECTORY_ROOT / trajectory / f"{index}.jpg") for index in history],
                    "current_image_path": str(TRAJECTORY_ROOT / trajectory / f"{current}.jpg"),
                    "current_t": current,
                    "dataset": "scand",
                    "day": metadata["day"],
                    "goal_image_path": str(TRAJECTORY_ROOT / trajectory / f"{goal}.jpg"),
                    "goal_t": goal,
                    "history_timesteps": history,
                    "partition": "train-fit",
                    "route_group": metadata["route_group"],
                    "route_id": metadata["route_group"],
                    "selection_label_free": True,
                    "trajectory_id": trajectory,
                    "window_uid": uid,
                }
            )
    rows.sort(key=lambda row: (selected_bags.index(str(row["bag_id"])), stable_key("row-order", row["window_uid"])))
    if not 400 <= len(rows) <= 600:
        raise RuntimeError(f"pilot state target failed: {len(rows)}")
    if max(Counter(row["bag_id"] for row in rows).values()) > PER_BAG_CAP:
        raise RuntimeError("per-bag cap failed")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUTPUT / "pilot_manifest.windows.jsonl"
    coverage_path = OUTPUT / "coverage_inventory.json"
    prereg_path = OUTPUT / "preregistration.json"
    write_jsonl(manifest_path, rows)
    coverage = {
        "schema": "scand_existing_train_upperbound_coverage_v1",
        "status": "frozen_before_pilot_outcomes",
        "train_split": {"path": str(TRAIN_NAMES), "sha256": sha256_file(TRAIN_NAMES), "trajectories": len(names), "bags": len(train_bags)},
        "coverage_evidence": evidence,
        "known": {"trajectories": len(known_trajectories), "records": len(known_records), "bags": len(known_bags), "route_groups": len(known_routes)},
        "old_train_bags": list(bag_stats.values()),
        "eligible_uncovered_bags": len(candidates),
        "eligible_uncovered_route_groups": len(by_route),
        "selected": {
            "bags": selected_bags,
            "bag_count": len(selected_bags),
            "route_groups": sorted({bag_stats[bag]["route_group"] for bag in selected_bags}),
            "route_group_count": len({bag_stats[bag]["route_group"] for bag in selected_bags}),
            "states": len(rows),
            "states_per_bag": dict(sorted(Counter(row["bag_id"] for row in rows).items())),
            "exact_known_bag_overlap": 0,
            "used_registry_bag_hash_overlap": 0,
        },
        "labels_or_outcomes_opened": False,
        "traj_data_pkl_opened": False,
        "fresh_outer_or_test_opened": False,
        "reserve_accessed": False,
    }
    write_json(coverage_path, coverage)
    prereg = {
        "schema": "scand_existing_train_upperbound_pilot_preregistration_v1",
        "status": "frozen_before_candidate_generation_and_outcomes",
        "scope": "spent_old_train_pilot_only_not_independent_evidence",
        "inputs": {
            "coverage": {"path": str(coverage_path), "sha256": sha256_file(coverage_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "used_registry": {"path": str(USED_REGISTRY), "sha256": sha256_file(USED_REGISTRY)},
            "strict_test_identity_exclusion": {"path": str(STRICT_TEST), "sha256": sha256_file(STRICT_TEST)},
            "formal_v70": {"path": str(FORMAL_V70), "sha256": sha256_file(FORMAL_V70)},
            "formal_recon_selected_g": {"path": str(FORMAL_RECON_SELECTED), "sha256": sha256_file(FORMAL_RECON_SELECTED)},
        },
        "selection": {
            "seed": SELECTION_SEED,
            "bags": [MIN_BAGS, MAX_BAGS],
            "per_bag_cap": PER_BAG_CAP,
            "target_per_bag": TARGET_PER_BAG,
            "rules_in_order": [
                "exclude exact bag identity present in manifests/source NPZ/used hash registry",
                "require at least 16 H4 RGB-only windows",
                "one bag per route group before filling duplicates",
                "prefer unseen route group, then capped available-window count, then stable SHA key",
                "select windows by stable SHA key without opening traj_data.pkl",
            ],
        },
        "frozen_v70": {
            "cem_iterations": 3,
            "population": 16,
            "elite": 4,
            "proposal_scale": 0.70,
            "support_quantile": 0.999,
            "safety_thresholds_unchanged": True,
            "archive_unchanged": True,
            "recon_contract_unchanged": True,
            "sidecar_default_off_and_decision_isolated": True,
        },
        "oof": {
            "outer_folds": 5,
            "grouping": "connected component of bag_id and undirected route_group; no bag/route split",
            "inner_selection": "training-fold-only risk certification",
            "model_families": [
                "v77 route-balanced pairwise linear ensemble",
                "v77 group-DRO listwise multitask ensemble",
            ],
            "risk_profiles": "reuse frozen v77 profiles; no threshold relaxation",
        },
        "success_gate": {
            "dual_candidate_auroc_min": 0.75,
            "harm_candidate_auroc_min": 0.75,
            "must_improve_over_v77_dual_auroc_approx": 0.688,
            "must_improve_over_v77_harm_auroc_approx": 0.671,
            "risk_certified_replacements_nonzero": True,
            "replacement_harm_must_not_increase": True,
            "scand_oof_objective_relative_improvement_min": 0.05,
            "recon_objective_relative_degradation_max": 0.01,
            "recon_safety_and_validity_must_not_worsen": True,
            "threshold_relaxation_forbidden": True,
        },
        "forbidden": {
            "reserve_access": True,
            "old_test_access": True,
            "fresh_outer_use_for_tuning": True,
            "scorer_cal_use_for_tuning": True,
            "A_to_H_use_for_threshold_tuning": True,
            "v70_mutation": True,
            "automatic_full_48_bag_expansion": True,
        },
    }
    write_json(prereg_path, prereg)
    summary = {
        "coverage_sha256": sha256_file(coverage_path),
        "manifest_sha256": sha256_file(manifest_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "selected_bags": len(selected_bags),
        "selected_routes": coverage["selected"]["route_group_count"],
        "states": len(rows),
        "labels_opened": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
