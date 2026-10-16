from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scoring.trajectory_artifact_contract_v2 import (
    LABEL_SCHEMA,
    PROPOSAL_SCHEMA,
    canonical_json,
    label_arrays,
    local_future_positions,
    proposal_arrays,
    trajectory_quality_targets,
    validate_label_artifact,
    validate_proposal_artifact,
    validate_inner_manifest,
    validate_outer_train_membership,
)


LOCK_SHA = "a" * 64


def rows() -> list[dict[str, object]]:
    return [
        {
            "window_uid": "uid0",
            "dataset": "recon",
            "partition": "train-fit",
            "trajectory_id": "traj0",
            "route_id": "route0",
            "current_t": 3,
            "goal_t": 7,
            "history_timesteps": [0, 1, 2, 3],
        },
        {
            "window_uid": "uid1",
            "dataset": "recon",
            "partition": "train-fit",
            "trajectory_id": "traj1",
            "route_id": "route1",
            "current_t": 8,
            "goal_t": 12,
            "history_timesteps": [5, 6, 7, 8],
        },
    ]


def proposal() -> dict[str, np.ndarray]:
    actions = np.zeros((2, 16, 4, 3), dtype=np.float32)
    return proposal_arrays(
        rows(),
        actions,
        dataset="recon",
        partition="train-fit",
        metadata={
            "schema": PROPOSAL_SCHEMA,
            "dataset": "recon",
            "partition": "train-fit",
            "split_lock_sha256": LOCK_SHA,
        },
    )


def test_proposal_contract_requires_exact_manifest_order() -> None:
    value = proposal()
    validate_proposal_artifact(
        value,
        rows(),
        dataset="recon",
        partition="train-fit",
        split_lock_sha256=LOCK_SHA,
    )
    value["window_uid"] = value["window_uid"][::-1]
    with pytest.raises(ValueError, match="window_uid/order"):
        validate_proposal_artifact(
            value,
            rows(),
            dataset="recon",
            partition="train-fit",
        )


def test_proposal_contract_requires_route_id() -> None:
    value = proposal()
    del value["route_id"]
    with pytest.raises(ValueError, match="route_id"):
        validate_proposal_artifact(
            value,
            rows(),
            dataset="recon",
            partition="train-fit",
        )


def test_legacy_ate_and_unclipped_rtp_targets() -> None:
    actions = np.zeros((1, 2, 4, 3), dtype=np.float32)
    actions[0, 0, :, 0] = 0.25
    ground_truth = np.asarray(
        [[(0.25, 0.0), (0.50, 0.0), (0.75, 0.0), (1.00, 0.0)]],
        dtype=np.float32,
    )
    ate, rtp = trajectory_quality_targets(actions, ground_truth)
    assert float(ate[0, 0]) == pytest.approx(0.0)
    assert float(rtp[0, 0]) == pytest.approx(1.0)
    assert float(ate[0, 1]) == pytest.approx(0.625)
    assert float(rtp[0, 1]) == pytest.approx(0.0)


def test_local_future_positions_respect_current_yaw() -> None:
    positions = np.asarray(((0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0)))
    yaw = np.full(5, np.pi / 2.0)
    local = local_future_positions(positions, yaw, current_t=0, horizon=4)
    np.testing.assert_allclose(local[:, 0], (1.0, 2.0, 3.0, 4.0), atol=1.0e-12)
    np.testing.assert_allclose(local[:, 1], 0.0, atol=1.0e-12)


def test_label_contract_binds_proposal_uid_and_split_lock() -> None:
    proposal_value = proposal()
    labels = label_arrays(
        proposal_value,
        np.zeros((2, 16), dtype=np.float32),
        np.zeros((2, 16), dtype=np.float32),
        dataset="recon",
        partition="train-fit",
        metadata={
            "schema": LABEL_SCHEMA,
            "dataset": "recon",
            "partition": "train-fit",
            "split_lock_sha256": LOCK_SHA,
        },
    )
    validate_label_artifact(
        labels,
        proposal_value,
        dataset="recon",
        partition="train-fit",
        split_lock_sha256=LOCK_SHA,
    )
    labels["window_uid"] = labels["window_uid"][::-1]
    with pytest.raises(ValueError, match="window_uid"):
        validate_label_artifact(
            labels,
            proposal_value,
            dataset="recon",
            partition="train-fit",
        )


def test_manifest_rejects_unsafe_trajectory_and_empty_route() -> None:
    value = rows()
    value[0] = dict(value[0])
    value[0]["trajectory_id"] = "../escape"
    with pytest.raises(ValueError, match="unsafe trajectory_id"):
        validate_inner_manifest(value, dataset="recon", partition="train-fit")

    value = rows()
    value[0] = dict(value[0])
    value[0]["route_id"] = ""
    with pytest.raises(ValueError, match="empty route_id"):
        validate_inner_manifest(value, dataset="recon", partition="train-fit")


def test_proposal_metadata_and_candidate_family_fail_closed() -> None:
    value = proposal()
    value["metadata"] = np.asarray(
        canonical_json(
            {
                "schema": PROPOSAL_SCHEMA,
                "dataset": "scand",
                "partition": "train-fit",
                "split_lock_sha256": LOCK_SHA,
            }
        )
    )
    with pytest.raises(ValueError, match="metadata dataset"):
        validate_proposal_artifact(
            value,
            rows(),
            dataset="recon",
            partition="train-fit",
            split_lock_sha256=LOCK_SHA,
        )

    families = np.full((2, 16), "nomad", dtype=str)
    families[0, 0] = ""
    with pytest.raises(ValueError, match="nonempty"):
        proposal_arrays(
            rows(),
            np.zeros((2, 16, 4, 3), dtype=np.float32),
            dataset="recon",
            partition="train-fit",
            metadata={},
            candidate_family=families,
        )


def test_outer_train_membership_rejects_validation_trajectory(tmp_path: Path) -> None:
    split_lock = tmp_path / "split_lock.json"
    split_lock.write_text(
        json.dumps(
            {
                "recon": {
                    "splits": {
                        "train": ["traj0"],
                        "val": ["traj1"],
                        "test": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-outer-train"):
        validate_outer_train_membership(split_lock, rows(), dataset="recon")
