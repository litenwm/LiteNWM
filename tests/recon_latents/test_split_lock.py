from __future__ import annotations

import pytest

from recon_latents.split_lock import (
    assert_disjoint_splits,
    normalize_scand_route,
    recon_route_id,
    validate_parent_split_reuse,
)


def test_recon_repeat_suffix_is_grouped() -> None:
    assert recon_route_id("jackal_2019-08-02-00_r03") == "jackal_2019-08-02-00"


def test_reverse_scand_routes_share_one_group() -> None:
    assert normalize_scand_route("AHG", "Library") == normalize_scand_route(
        "Library", "AHG"
    )


def test_group_overlap_fails_closed() -> None:
    splits = {"train": ["a0"], "val": ["b0"], "test": ["a1"]}
    groups = {"a0": "route_a", "a1": "route_a", "b0": "route_b"}
    with pytest.raises(ValueError, match="group split leakage"):
        assert_disjoint_splits(splits, group_for_id=groups)


def test_disjoint_groups_pass() -> None:
    result = assert_disjoint_splits(
        {"train": ["a0"], "val": ["b0"], "test": ["c0"]},
        group_for_id={"a0": "a", "b0": "b", "c0": "c"},
    )
    assert result["group_overlap"] == {
        "train__val": 0,
        "train__test": 0,
        "val__test": 0,
    }


def test_parent_reuse_requires_unchanged_recon_and_preserved_scand(
    tmp_path,
) -> None:
    parent = {
        "recon": {"splits": {"train": ["r0"], "val": ["r1"], "test": ["r2"]}},
        "scand": {"splits": {"train": ["s0"], "val": ["s1"], "test": ["s2"]}},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(__import__("json").dumps(parent), encoding="utf-8")
    parent_sha = __import__("hashlib").sha256(parent_path.read_bytes()).hexdigest()
    active = {
        "lineage": {"parent_split_lock_sha256": parent_sha},
        "recon": parent["recon"],
        "scand": {
            "splits": {
                "train": ["s0", "s3"],
                "val": ["s1", "s4"],
                "test": ["s2", "s5"],
            }
        },
    }
    report = validate_parent_split_reuse(
        active,
        parent_split_lock=parent_path,
        artifact_split_lock_sha256=parent_sha,
        unchanged_datasets=("recon",),
        preserved_datasets=("scand",),
    )
    assert report["status"] == "PASS"


def test_parent_reuse_rejects_moved_parent_test(tmp_path) -> None:
    parent = {
        "recon": {"splits": {"train": [], "val": [], "test": []}},
        "scand": {"splits": {"train": [], "val": [], "test": ["held_out"]}},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(__import__("json").dumps(parent), encoding="utf-8")
    parent_sha = __import__("hashlib").sha256(parent_path.read_bytes()).hexdigest()
    active = {
        "lineage": {"parent_split_lock_sha256": parent_sha},
        "recon": parent["recon"],
        "scand": {
            "splits": {"train": ["held_out"], "val": [], "test": []}
        },
    }
    with pytest.raises(ValueError, match="moved or disappeared"):
        validate_parent_split_reuse(
            active,
            parent_split_lock=parent_path,
            artifact_split_lock_sha256=parent_sha,
            preserved_datasets=("scand",),
        )
