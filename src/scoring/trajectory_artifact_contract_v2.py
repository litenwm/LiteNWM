"""Strict proposal and label artifacts for H4 trajectory-quality research."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from planning.nomad_action_adapter import HORIZON, integrate_local_actions


K = 16
ACTION_DIM = 3
PROPOSAL_SCHEMA = "h4_trajectory_proposal_v2"
LABEL_SCHEMA = "h4_trajectory_quality_labels_v2"
INNER_PARTITIONS = (
    "train-fit",
    "scorer-cal",
    "cem-cal",
    "internal-holdout",
)
REQUIRED_MANIFEST_FIELDS = {
    "window_uid",
    "dataset",
    "partition",
    "trajectory_id",
    "route_id",
    "current_t",
    "goal_t",
    "history_timesteps",
}
REQUIRED_PROPOSAL_ARRAYS = {
    "window_uid",
    "dataset",
    "partition",
    "trajectory_id",
    "route_id",
    "current_t",
    "goal_t",
    "history_timesteps",
    "actions_h4",
    "candidate_family",
    "metadata",
}
REQUIRED_LABEL_ARRAYS = {
    "window_uid",
    "dataset",
    "partition",
    "ate",
    "rtp",
    "metadata",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_inner_manifest(
    path: Path,
    *,
    dataset: str | None = None,
    partition: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_inner_manifest(rows, dataset=dataset, partition=partition)
    return rows


def validate_inner_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    partition: str | None = None,
) -> None:
    values = list(rows)
    if not values:
        raise ValueError("inner manifest must be nonempty")
    seen: set[str] = set()
    for index, row in enumerate(values):
        missing = REQUIRED_MANIFEST_FIELDS.difference(row)
        if missing:
            raise ValueError(f"manifest row {index} missing fields: {sorted(missing)}")
        uid = str(row["window_uid"])
        if not uid or uid in seen:
            raise ValueError(f"duplicate or empty window_uid at row {index}: {uid!r}")
        seen.add(uid)
        trajectory_id = str(row["trajectory_id"])
        if (
            not trajectory_id
            or trajectory_id in {".", ".."}
            or "/" in trajectory_id
            or "\\" in trajectory_id
        ):
            raise ValueError(
                f"manifest row {index} has unsafe trajectory_id: {trajectory_id!r}"
            )
        if not str(row["route_id"]):
            raise ValueError(f"manifest row {index} has an empty route_id")
        if dataset is not None and str(row["dataset"]) != dataset:
            raise ValueError(f"manifest dataset mismatch at row {index}")
        if partition is not None and str(row["partition"]) != partition:
            raise ValueError(f"manifest partition mismatch at row {index}")
        current_t = int(row["current_t"])
        goal_t = int(row["goal_t"])
        history = [int(value) for value in row["history_timesteps"]]
        if current_t < 3 or goal_t != current_t + HORIZON:
            raise ValueError(f"manifest row {index} violates the strict H4 window")
        if history != list(range(current_t - 3, current_t + 1)):
            raise ValueError(f"manifest row {index} history is not [t-3,t]")


def validate_outer_train_membership(
    split_lock_path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
) -> None:
    """Require every manifest trajectory to belong to the locked outer train split."""

    payload = json.loads(split_lock_path.read_text(encoding="utf-8"))
    try:
        splits = payload[dataset]["splits"]
        train_ids = set(map(str, splits["train"]))
        validation_ids = set(map(str, splits["val"]))
        test_ids = set(map(str, splits["test"]))
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"invalid split-lock schema for dataset {dataset!r}"
        ) from error
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError(f"split lock contains overlapping {dataset} assignments")
    requested = {str(row["trajectory_id"]) for row in rows}
    outside = sorted(requested.difference(train_ids))
    if outside:
        sample = ", ".join(outside[:5])
        raise ValueError(
            f"inner manifest contains {len(outside)} non-outer-train trajectories: {sample}"
        )


def proposal_arrays(
    rows: list[Mapping[str, Any]],
    actions_h4: np.ndarray,
    *,
    dataset: str,
    partition: str,
    metadata: Mapping[str, Any],
    candidate_family: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    validate_inner_manifest(rows, dataset=dataset, partition=partition)
    actions = np.asarray(actions_h4, dtype=np.float32)
    expected = (len(rows), K, HORIZON, ACTION_DIM)
    if actions.shape != expected:
        raise ValueError(f"actions_h4 must have shape {expected}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions_h4 contains NaN or Inf")
    families = (
        np.full((len(rows), K), "nomad", dtype=str)
        if candidate_family is None
        else np.asarray(candidate_family, dtype=str)
    )
    if families.shape != (len(rows), K):
        raise ValueError("candidate_family must have shape [N,16]")
    if np.any(np.char.str_len(families) == 0):
        raise ValueError("candidate_family entries must be nonempty")
    return {
        "window_uid": np.asarray([str(row["window_uid"]) for row in rows]),
        "dataset": np.asarray(dataset),
        "partition": np.asarray(partition),
        "trajectory_id": np.asarray([str(row["trajectory_id"]) for row in rows]),
        "route_id": np.asarray([str(row["route_id"]) for row in rows]),
        "current_t": np.asarray(
            [int(row["current_t"]) for row in rows], dtype=np.int64
        ),
        "goal_t": np.asarray([int(row["goal_t"]) for row in rows], dtype=np.int64),
        "history_timesteps": np.asarray(
            [row["history_timesteps"] for row in rows], dtype=np.int64
        ),
        "actions_h4": actions,
        "candidate_family": families,
        "metadata": np.asarray(canonical_json(dict(metadata))),
    }


def _metadata(array: np.ndarray) -> dict[str, Any]:
    if array.ndim != 0:
        raise ValueError("metadata must be a scalar canonical-JSON string")
    value = json.loads(str(array.item()))
    if not isinstance(value, dict):
        raise ValueError("metadata JSON must be an object")
    return value


def validate_proposal_artifact(
    proposal: Mapping[str, np.ndarray],
    rows: list[Mapping[str, Any]],
    *,
    dataset: str,
    partition: str,
    split_lock_sha256: str | None = None,
) -> dict[str, Any]:
    validate_inner_manifest(rows, dataset=dataset, partition=partition)
    missing = REQUIRED_PROPOSAL_ARRAYS.difference(proposal)
    if missing:
        raise ValueError(f"proposal missing arrays: {sorted(missing)}")
    expected_uids = np.asarray([str(row["window_uid"]) for row in rows])
    if not np.array_equal(
        np.asarray(proposal["window_uid"]).astype(str), expected_uids
    ):
        raise ValueError("proposal window_uid/order does not match inner manifest")
    if str(proposal["dataset"].item()) != dataset:
        raise ValueError("proposal dataset mismatch")
    if str(proposal["partition"].item()) != partition:
        raise ValueError("proposal partition mismatch")
    count = len(rows)
    if np.asarray(proposal["actions_h4"]).shape != (count, K, HORIZON, ACTION_DIM):
        raise ValueError("proposal actions must have shape [N,16,4,3]")
    if np.asarray(proposal["candidate_family"]).shape != (count, K):
        raise ValueError("proposal candidate_family must have shape [N,16]")
    families = np.asarray(proposal["candidate_family"]).astype(str)
    if np.any(np.char.str_len(families) == 0):
        raise ValueError("proposal candidate_family contains an empty value")
    expected_fields = {
        "trajectory_id": np.asarray([str(row["trajectory_id"]) for row in rows]),
        "route_id": np.asarray([str(row["route_id"]) for row in rows]),
        "current_t": np.asarray([int(row["current_t"]) for row in rows]),
        "goal_t": np.asarray([int(row["goal_t"]) for row in rows]),
        "history_timesteps": np.asarray(
            [row["history_timesteps"] for row in rows], dtype=np.int64
        ),
    }
    for key, expected in expected_fields.items():
        actual = np.asarray(proposal[key])
        if actual.dtype.kind in {"U", "S"}:
            actual = actual.astype(str)
        if not np.array_equal(actual, expected):
            raise ValueError(f"proposal {key} does not match inner manifest")
    if not np.isfinite(np.asarray(proposal["actions_h4"], dtype=np.float32)).all():
        raise ValueError("proposal actions contain NaN or Inf")
    metadata = _metadata(np.asarray(proposal["metadata"]))
    if metadata.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError("proposal schema mismatch")
    if metadata.get("dataset") != dataset:
        raise ValueError("proposal metadata dataset mismatch")
    if metadata.get("partition") != partition:
        raise ValueError("proposal metadata partition mismatch")
    if (
        split_lock_sha256 is not None
        and metadata.get("split_lock_sha256") != split_lock_sha256
    ):
        raise ValueError("proposal split-lock mismatch")
    return metadata


def local_future_positions(
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    current_t: int,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Return logged future XY in the robot frame at ``current_t``."""

    value = np.asarray(positions, dtype=np.float64)
    angles = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or value.shape[1] < 2 or len(value) != len(angles):
        raise ValueError("positions/yaw shapes are invalid")
    if current_t < 0 or current_t + horizon >= len(value):
        raise ValueError("ground-truth H4 window is outside the trajectory")
    if not np.isfinite(value).all() or not np.isfinite(angles).all():
        raise ValueError("ground-truth pose contains NaN or Inf")
    delta = value[current_t + 1 : current_t + horizon + 1, :2] - value[current_t, :2]
    angle = float(angles[current_t])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation_world_to_local = np.asarray(
        ((cosine, sine), (-sine, cosine)), dtype=np.float64
    )
    return delta @ rotation_world_to_local.T


def trajectory_quality_targets(
    actions_h4: np.ndarray,
    ground_truth_local_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute legacy mean H1-H4 ATE and unclipped H4 RTP in meters."""

    actions = np.asarray(actions_h4, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_local_xy, dtype=np.float64)
    if actions.ndim != 4 or actions.shape[-2:] != (HORIZON, ACTION_DIM):
        raise ValueError("actions_h4 must have shape [N,K,4,3]")
    if ground_truth.shape != (actions.shape[0], HORIZON, 2):
        raise ValueError("ground_truth_local_xy must have shape [N,4,2]")
    if not np.isfinite(actions).all() or not np.isfinite(ground_truth).all():
        raise ValueError("actions or ground truth contain NaN or Inf")
    predicted = integrate_local_actions(actions)
    error = np.linalg.norm(predicted - ground_truth[:, None], axis=-1)
    ate = error.mean(axis=-1)
    goal = ground_truth[:, -1]
    start_distance = np.linalg.norm(goal, axis=-1)
    final_distance = np.linalg.norm(predicted[:, :, -1] - goal[:, None], axis=-1)
    rtp = (start_distance[:, None] - final_distance) / np.maximum(
        start_distance[:, None], 1.0e-8
    )
    return ate.astype(np.float32), rtp.astype(np.float32)


def label_arrays(
    proposal: Mapping[str, np.ndarray],
    ate: np.ndarray,
    rtp: np.ndarray,
    *,
    dataset: str,
    partition: str,
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    count = len(proposal["window_uid"])
    ate_value = np.asarray(ate, dtype=np.float32)
    rtp_value = np.asarray(rtp, dtype=np.float32)
    if ate_value.shape != (count, K) or rtp_value.shape != (count, K):
        raise ValueError("ATE/RTP must have shape [N,16]")
    if not np.isfinite(ate_value).all() or not np.isfinite(rtp_value).all():
        raise ValueError("ATE/RTP contain NaN or Inf")
    return {
        "window_uid": np.asarray(proposal["window_uid"]).astype(str),
        "dataset": np.asarray(dataset),
        "partition": np.asarray(partition),
        "ate": ate_value,
        "rtp": rtp_value,
        "metadata": np.asarray(canonical_json(dict(metadata))),
    }


def validate_label_artifact(
    labels: Mapping[str, np.ndarray],
    proposal: Mapping[str, np.ndarray],
    *,
    dataset: str,
    partition: str,
    split_lock_sha256: str | None = None,
) -> dict[str, Any]:
    missing = REQUIRED_LABEL_ARRAYS.difference(labels)
    if missing:
        raise ValueError(f"labels missing arrays: {sorted(missing)}")
    if not np.array_equal(
        np.asarray(labels["window_uid"]).astype(str),
        np.asarray(proposal["window_uid"]).astype(str),
    ):
        raise ValueError("proposal/label window_uid mismatch")
    if str(labels["dataset"].item()) != dataset:
        raise ValueError("label dataset mismatch")
    if str(labels["partition"].item()) != partition:
        raise ValueError("label partition mismatch")
    count = len(proposal["window_uid"])
    if np.asarray(labels["ate"]).shape != (count, K):
        raise ValueError("label ATE must have shape [N,16]")
    if np.asarray(labels["rtp"]).shape != (count, K):
        raise ValueError("label RTP must have shape [N,16]")
    if not np.isfinite(np.asarray(labels["ate"], dtype=np.float32)).all():
        raise ValueError("label ATE contains NaN or Inf")
    if not np.isfinite(np.asarray(labels["rtp"], dtype=np.float32)).all():
        raise ValueError("label RTP contains NaN or Inf")
    metadata = _metadata(np.asarray(labels["metadata"]))
    if metadata.get("schema") != LABEL_SCHEMA:
        raise ValueError("label schema mismatch")
    if metadata.get("dataset") != dataset:
        raise ValueError("label metadata dataset mismatch")
    if metadata.get("partition") != partition:
        raise ValueError("label metadata partition mismatch")
    if (
        split_lock_sha256 is not None
        and metadata.get("split_lock_sha256") != split_lock_sha256
    ):
        raise ValueError("label split-lock mismatch")
    return metadata


__all__ = [
    "ACTION_DIM",
    "INNER_PARTITIONS",
    "K",
    "LABEL_SCHEMA",
    "PROPOSAL_SCHEMA",
    "canonical_json",
    "file_sha256",
    "label_arrays",
    "load_inner_manifest",
    "load_npz",
    "local_future_positions",
    "proposal_arrays",
    "trajectory_quality_targets",
    "validate_inner_manifest",
    "validate_label_artifact",
    "validate_outer_train_membership",
    "validate_proposal_artifact",
    "write_json_atomic",
    "write_npz_atomic",
]
