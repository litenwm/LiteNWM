from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scoring.strict_inner_partitions import (
    INNER_PARTITIONS,
    allocate_route_groups,
    assert_manifest_is_reproducible,
    build_recon_h4_exclusion_manifest,
    build_recon_inner_manifests,
    build_scand_inner_manifests,
    parse_scand_trajectory_id,
    validate_recon_inner_manifests,
    validate_scand_inner_manifests,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_scand_id(
    left: str,
    right: str,
    bag: int,
    segment: int = 0,
) -> str:
    return (
        f"random_mdps_A_Spot_{left}_{right}_Mon_Jan_1_{bag}_{segment}"
    )


def make_shard(path: Path, trajectory: str, frames: int = 12) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = np.asarray(
        [f"/sealed/{trajectory}/{index}.jpg" for index in range(frames)]
    )
    positions = np.stack(
        (
            np.arange(frames, dtype=np.float64) * 0.2,
            np.zeros(frames, dtype=np.float64),
        ),
        axis=1,
    )
    np.savez_compressed(
        path,
        image_paths=image_paths,
        positions=positions,
        yaw=np.zeros(frames, dtype=np.float64),
    )
    return {
        "trajectory_id": trajectory,
        "split": "train",
        "shard_file": path.name,
        "num_frames": frames,
    }


def make_scand_fixture(root: Path) -> tuple[Path, Path]:
    latent_root = root / "scand_latents"
    shards = latent_root / "shards"
    train_ids = []
    records = []
    route_rows = []
    for index in range(8):
        trajectory = make_scand_id(f"L{index}", f"R{index}", 100 + index)
        train_ids.append(trajectory)
        records.append(make_shard(shards / f"{trajectory}.npz", trajectory))
        route_rows.append(
            {
                "route_group": f"l{index}__r{index}",
                "split": "train",
            }
        )

    forward = make_scand_id("CrossA", "CrossB", 200, 0)
    reverse = make_scand_id("CrossB", "CrossA", 201, 0)
    for trajectory in (forward, reverse):
        train_ids.append(trajectory)
        records.append(make_shard(shards / f"{trajectory}.npz", trajectory))
    route_rows.append({"route_group": "crossa__crossb", "split": "train"})

    val_id = make_scand_id("ValA", "ValB", 300)
    records.append(make_shard(shards / f"{val_id}.npz", val_id))
    records[-1]["split"] = "val"
    route_rows.append({"route_group": "vala__valb", "split": "val"})
    test_id = make_scand_id("TestA", "TestB", 400)
    route_rows.append({"route_group": "testa__testb", "split": "test"})

    split_lock_path = root / "split_lock.json"
    split_lock = {
        "policy": {"history_frames": 4, "future_frames": 4},
        "scand": {
            "splits": {
                "train": train_ids,
                "val": [val_id],
                "test": [test_id],
            },
            "route_assignment": route_rows,
        },
        "recon": {
            "splits": {"train": [], "val": [], "test": []},
            "audit": {"exact_overlap": {}, "group_overlap": {}},
        },
    }
    write_json(split_lock_path, split_lock)
    write_json(
        latent_root / "splits.json",
        {"train": train_ids, "val": [val_id], "test": []},
    )
    write_json(
        latent_root / "metadata.json",
        {
            "split_lock_dataset": "scand",
            "split_lock_sha256": sha256(split_lock_path),
            "trajectories": records,
        },
    )
    return latent_root, split_lock_path


def test_parse_scand_id_groups_reverse_routes_and_physical_bags() -> None:
    forward = parse_scand_trajectory_id(
        "random_mdps_A_Spot_AHG_Library_Mon_Nov_8_27_3"
    )
    reverse = parse_scand_trajectory_id(
        "random_mdps_A_Spot_Library_AHG_Mon_Nov_8_28_0"
    )
    sibling = parse_scand_trajectory_id(
        "random_mdps_A_Spot_AHG_Library_Mon_Nov_8_27_4"
    )
    assert forward["route_group"] == reverse["route_group"] == "ahg__library"
    assert forward["bag_id"] == sibling["bag_id"]
    assert forward["bag_id"] != reverse["bag_id"]


def test_allocate_route_groups_is_deterministic_and_nonempty() -> None:
    stats = {
        f"route_{index}": {
            "h4_windows": (index + 1) * 17,
            "trajectories": index % 3 + 1,
            "bags": index % 2 + 1,
            "route_groups": 1,
        }
        for index in range(12)
    }
    first, first_report = allocate_route_groups(stats, seed=20260719)
    second, second_report = allocate_route_groups(stats, seed=20260719)
    assert first == second
    assert first_report == second_report
    assert set(first.values()) == set(INNER_PARTITIONS)
    assert set(first) == set(stats)


def test_scand_manifests_are_four_way_group_disjoint_and_reproducible(
    tmp_path: Path,
) -> None:
    latent_root, split_lock = make_scand_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "latent_root": latent_root,
        "split_lock_path": split_lock,
        "split_seed": 20260719,
        "max_windows_per_bag": 4,
        "min_goal_displacement_m": 0.5,
    }
    report = build_scand_inner_manifests(out_root=first, **kwargs)
    build_scand_inner_manifests(out_root=second, **kwargs)
    assert report["decision"] == "STRICT_SCAND_H4_FOUR_WAY_MANIFESTS_READY"
    assert_manifest_is_reproducible(first, second)
    validation = validate_scand_inner_manifests(
        asset_root=first,
        split_lock_path=split_lock,
        out_file=first / "independent_validation.json",
    )
    assert (
        validation["decision"]
        == "STRICT_SCAND_H4_FOUR_WAY_INDEPENDENT_AUDIT_PASS"
    )
    assert validation["leakage_guard"]["test_rows"] == 0

    route_partitions: dict[str, set[str]] = {}
    bag_partitions: dict[str, set[str]] = {}
    seen_trajectories = set()
    for partition in INNER_PARTITIONS:
        rows = [
            json.loads(line)
            for line in (first / f"{partition}.windows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows
        for row in rows:
            route_partitions.setdefault(row["route_group"], set()).add(partition)
            bag_partitions.setdefault(row["bag_id"], set()).add(partition)
            seen_trajectories.add(row["trajectory_id"])
            assert row["partition"] == partition
            assert len(row["history_timesteps"]) == 4
            assert row["goal_t"] - row["current_t"] == 4
            assert "ValA" not in row["trajectory_id"]
            assert "TestA" not in row["trajectory_id"]
    assert all(len(partitions) == 1 for partitions in route_partitions.values())
    assert all(len(partitions) == 1 for partitions in bag_partitions.values())
    assert any("CrossA_CrossB" in value for value in seen_trajectories)
    assert any("CrossB_CrossA" in value for value in seen_trajectories)

    assignment = json.loads(
        (first / "partition_assignment.json").read_text(encoding="utf-8")
    )
    assert (
        assignment["trajectory_assignment"][
            make_scand_id("CrossA", "CrossB", 200)
        ]
        == assignment["trajectory_assignment"][
            make_scand_id("CrossB", "CrossA", 201)
        ]
    )
    assert assignment["label_or_model_output_used_for_assignment"] is False


def test_scand_builder_refuses_existing_output(tmp_path: Path) -> None:
    latent_root, split_lock = make_scand_fixture(tmp_path)
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_scand_inner_manifests(
            latent_root=latent_root,
            split_lock_path=split_lock,
            out_root=output,
        )


def test_recon_exclusion_manifest_counts_short_trajectories_without_test_access(
    tmp_path: Path,
) -> None:
    latent_root = tmp_path / "recon_latents"
    train = {"train_long": 8, "train_short": 7, "train_tiny": 2}
    val = {"val_long": 10, "val_short": 6}
    test_ids = ["test_sealed_a", "test_sealed_b"]
    split_lock_path = tmp_path / "recon_split_lock.json"
    split_lock = {
        "policy": {"history_frames": 4, "future_frames": 4},
        "recon": {
            "splits": {
                "train": list(train),
                "val": list(val),
                "test": test_ids,
            },
            "audit": {
                "exact_overlap": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
                "group_overlap": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            },
        },
        "scand": {
            "splits": {"train": [], "val": [], "test": []},
            "route_assignment": [],
        },
    }
    write_json(split_lock_path, split_lock)
    write_json(
        latent_root / "splits.json",
        {"train": list(train), "val": list(val), "test": []},
    )
    records = [
        {
            "trajectory_id": trajectory,
            "split": split,
            "num_frames": frames,
            "shard_file": f"{trajectory}.npz",
        }
        for split, values in (("train", train), ("val", val))
        for trajectory, frames in values.items()
    ]
    write_json(
        latent_root / "metadata.json",
        {
            "split_lock_dataset": "recon",
            "split_lock_sha256": sha256(split_lock_path),
            "trajectories": records,
        },
    )
    output = tmp_path / "recon_exclusions.json"
    report = build_recon_h4_exclusion_manifest(
        latent_root=latent_root,
        split_lock_path=split_lock_path,
        out_file=output,
    )
    assert report["decision"] == "RECON_H4_EXCLUSION_AUDIT_PASS"
    assert report["splits"]["train"]["locked_trajectories"] == 3
    assert report["splits"]["train"]["eligible_trajectories"] == 1
    assert report["splits"]["train"]["excluded_trajectories"] == 2
    assert report["splits"]["train"]["eligible_h4_windows"] == 1
    assert report["splits"]["val"]["eligible_trajectories"] == 1
    assert report["splits"]["val"]["eligible_h4_windows"] == 3
    assert report["splits"]["test"]["status"] == "SEALED_NOT_MATERIALIZED"
    assert report["splits"]["test"]["eligible_trajectories"] is None
    assert report["exclusion_count"] == 3
    assert all(row["split"] != "test" for row in report["exclusions"])


def test_recon_four_way_manifests_are_route_disjoint(
    tmp_path: Path,
) -> None:
    latent_root = tmp_path / "recon_route_latents"
    shards = latent_root / "shards"
    train_ids = []
    records = []
    for route_index in range(8):
        for repeat in range(2 if route_index == 0 else 1):
            trajectory = f"jackal_2020-01-01-12-{route_index:02d}-00_{route_index}_r{repeat:02d}"
            train_ids.append(trajectory)
            record = make_shard(
                shards / f"{trajectory}.npz",
                trajectory,
                frames=12,
            )
            record["split"] = "train"
            records.append(record)
    short_id = "jackal_2020-01-01-13-00-00_9_r00"
    short_record = make_shard(shards / f"{short_id}.npz", short_id, frames=7)
    short_record["split"] = "train"
    train_ids.append(short_id)
    records.append(short_record)
    val_id = "jackal_2020-01-02-12-00-00_0_r00"
    val_record = make_shard(shards / f"{val_id}.npz", val_id, frames=12)
    val_record["split"] = "val"
    records.append(val_record)
    test_id = "jackal_2020-01-03-12-00-00_0_r00"

    split_lock_path = tmp_path / "recon_route_split_lock.json"
    split_lock = {
        "policy": {"history_frames": 4, "future_frames": 4},
        "recon": {
            "splits": {
                "train": train_ids,
                "val": [val_id],
                "test": [test_id],
            },
            "audit": {
                "exact_overlap": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
                "group_overlap": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            },
        },
        "scand": {
            "splits": {"train": [], "val": [], "test": []},
            "route_assignment": [],
        },
    }
    write_json(split_lock_path, split_lock)
    write_json(
        latent_root / "splits.json",
        {"train": train_ids, "val": [val_id], "test": []},
    )
    write_json(
        latent_root / "metadata.json",
        {
            "split_lock_dataset": "recon",
            "split_lock_sha256": sha256(split_lock_path),
            "trajectories": records,
        },
    )
    exclusion_path = tmp_path / "recon_route_exclusions.json"
    build_recon_h4_exclusion_manifest(
        latent_root=latent_root,
        split_lock_path=split_lock_path,
        out_file=exclusion_path,
    )
    output = tmp_path / "recon_four_way"
    report = build_recon_inner_manifests(
        latent_root=latent_root,
        split_lock_path=split_lock_path,
        exclusion_manifest_path=exclusion_path,
        out_root=output,
        quota_per_route={
            "train-fit": 2,
            "scorer-cal": 1,
            "cem-cal": 1,
            "internal-holdout": 1,
        },
    )
    assert report["decision"] == "STRICT_RECON_H4_FOUR_WAY_MANIFESTS_READY"
    validation = validate_recon_inner_manifests(
        asset_root=output,
        split_lock_path=split_lock_path,
        out_file=output / "independent_validation.json",
    )
    assert (
        validation["decision"]
        == "STRICT_RECON_H4_FOUR_WAY_INDEPENDENT_AUDIT_PASS"
    )
    assert validation["leakage_guard"]["outer_val_rows"] == 0
    assert validation["leakage_guard"]["test_rows"] == 0
    routes = {}
    for partition in INNER_PARTITIONS:
        for line in (output / f"{partition}.windows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            row = json.loads(line)
            routes.setdefault(row["route_id"], set()).add(partition)
            assert row["trajectory_id"] not in {val_id, test_id}
            assert row["trajectory_id"] != short_id
    assert all(len(partitions) == 1 for partitions in routes.values())
