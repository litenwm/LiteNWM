"""Opt-in, decision-isolated logging for future SCAND proposal-tail routes.

The module deliberately has no random-number API and never returns values to
the planner.  The formal runner passes copies of already-computed arrays to the
sidecar and flushes the sidecar only after the formal selected action has been
written.  A cached dry-run mode exists solely to validate serialization on
already-spent development artifacts; it cannot reconstruct historical draws
that were rejected before population serialization.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "scand_proposal_tail_sidecar_v1"
METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")
FEATURE_WAYPOINT_UNIT_M = 0.25
COHORT_ROLES = ("spent_smoke", "development", "one_time_validation")
PHYSICAL_COORDINATE_NAMES = tuple(
    f"horizon_{step}_{axis}"
    for step in range(1, 5)
    for axis in ("dx_over_spacing", "dy_over_spacing", "dyaw_rad")
)
FIRST_REJECT_CODES = {
    "accepted": 0,
    "physical": 1,
    "support": 2,
    "ate_ucb": 3,
    "objective_ucb": 4,
    "dual_probability": 5,
    "harm_probability": 6,
    "incumbent_not_challenger": 7,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(canonical_json(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def identity_hash(kind: str, value: Any) -> str:
    return sha256_bytes(f"proposal-tail-v1|{kind}|{value}".encode("utf-8"))


def reject_forbidden_path(path: Path) -> None:
    tokens = re.split(r"[^a-z0-9]+", str(path).lower())
    if ("res" + "erve") in tokens:
        raise RuntimeError(f"refusing forbidden cohort path without opening it: {path}")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _nullable(value: Any, reason: str) -> dict[str, Any]:
    if value is None:
        return {"value": None, "unavailable_reason": reason}
    return {"value": value, "unavailable_reason": None}


def _timestamp_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    history = row.get("history_timestamps")
    if history is None and isinstance(row.get("frame_timestamps"), Mapping):
        mapping = row["frame_timestamps"]
        history = [mapping.get(str(index), mapping.get(index)) for index in row.get("history_timesteps", [])]
    current = row.get("current_timestamp")
    goal = row.get("goal_timestamp")
    return {
        "history": _nullable(history, "manifest_has_no_history_timestamps"),
        "current": _nullable(current, "manifest_has_no_current_timestamp"),
        "goal": _nullable(goal, "manifest_has_no_goal_timestamp"),
    }


def _camera_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("camera_metadata") or row.get("image_metadata")
    if not isinstance(metadata, Mapping):
        return {
            key: _nullable(None, "manifest_has_no_camera_or_image_metadata")
            for key in (
                "width",
                "height",
                "encoding",
                "jpeg_quality",
                "file_paths",
                "file_sizes",
                "file_sha256",
            )
        }
    return {
        key: _nullable(metadata.get(key), f"camera_metadata_missing_{key}")
        for key in (
            "width",
            "height",
            "encoding",
            "jpeg_quality",
            "file_paths",
            "file_sizes",
            "file_sha256",
        )
    }


def snapshot_states(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    for row in rows:
        route = row.get("route_group", row.get("route_id"))
        day = row.get("day", row.get("collection_day"))
        bag = row.get("bag_id")
        trajectory = row.get("trajectory_id")
        window = row.get("window_uid")
        record_key = f"{trajectory}:{int(row['current_t'])}"
        snapshots.append(
            {
                "record_key": record_key,
                "record_key_hash": identity_hash("record", record_key),
                "route": route,
                "route_hash": identity_hash("route", route),
                "day": _nullable(day, "manifest_has_no_collection_day"),
                "day_hash": None if day is None else identity_hash("day", day),
                "bag": bag,
                "bag_hash": identity_hash("bag", bag),
                "trajectory": trajectory,
                "trajectory_hash": identity_hash("trajectory", trajectory),
                "window": window,
                "window_hash": identity_hash("window", window),
                "history_frame_indices": list(row.get("history_timesteps", [])),
                "current_frame_index": int(row["current_t"]),
                "goal_frame_index": int(row["goal_t"]),
                "timestamps": _timestamp_snapshot(row),
                "camera": _camera_snapshot(row),
            }
        )
    return snapshots


def candidate_hashes(candidates: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float32)
    if value.ndim != 4 or value.shape[-2:] != (4, 2):
        raise ValueError("candidate waypoints must have shape [N,K,4,2]")
    result = np.empty(value.shape[:2], dtype="<U64")
    for state in range(value.shape[0]):
        for candidate in range(value.shape[1]):
            result[state, candidate] = sha256_bytes(
                np.ascontiguousarray(value[state, candidate]).tobytes()
            )
    return result


def action_structure_features(actions_h4: np.ndarray) -> np.ndarray:
    action = np.asarray(actions_h4, dtype=np.float64)
    if action.ndim != 4 or action.shape[-2:] != (4, 3):
        raise ValueError("actions_h4 must have shape [N,K,4,3]")
    xy = action[..., :2]
    speed = np.linalg.norm(xy, axis=-1)
    delta = np.diff(action, axis=-2)
    yaw = action[..., 2]
    return np.stack(
        (
            speed.sum(axis=-1),
            speed.max(axis=-1),
            np.linalg.norm(delta[..., :2], axis=-1).mean(axis=-1),
            np.abs(yaw).sum(axis=-1),
            np.abs(np.diff(yaw, axis=-1)).mean(axis=-1),
        ),
        axis=-1,
    ).astype(np.float32)


def _stable_rank(energy: np.ndarray) -> np.ndarray:
    value = np.asarray(energy)
    rank = np.empty(value.shape, dtype=np.int64)
    for state, row in enumerate(value):
        order = np.argsort(row, kind="stable")
        rank[state, order] = np.arange(len(order), dtype=np.int64)
    return rank


def scorer_components(
    prediction: Mapping[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    policy: Mapping[str, float],
) -> dict[str, np.ndarray]:
    mean = np.asarray(prediction["mean"], dtype=np.float32)
    std = np.asarray(prediction["std"], dtype=np.float32)
    ucb = mean + np.float32(policy["z"]) * std
    rank_logit = np.asarray(prediction["rank_logit"], dtype=np.float32)
    rank_scale = np.maximum(rank_logit[:, 1:].std(axis=1, keepdims=True), 1.0e-3)
    rank_z = (rank_logit - rank_logit[:, :1]) / rank_scale
    return {
        "metric_mean": mean,
        "metric_std": std,
        "metric_ucb": ucb.astype(np.float32),
        "helpful_logit": np.asarray(prediction["helpful_logit"], dtype=np.float32),
        "dual_logit": np.asarray(prediction["dual_logit"], dtype=np.float32),
        "harm_logit": np.asarray(prediction["harm_logit"], dtype=np.float32),
        "rank_logit": rank_logit,
        "rank_z": rank_z.astype(np.float32),
        "dual_probability": np.asarray(p_dual, dtype=np.float32),
        "harm_probability": np.asarray(p_harm, dtype=np.float32),
        "energy_component_ucb_ate": ucb[..., 0].astype(np.float32),
        "energy_component_ucb_rpe": (0.25 * ucb[..., 1]).astype(np.float32),
        "energy_component_ucb_endpoint": (0.25 * np.maximum(ucb[..., 2], 0.0)).astype(np.float32),
        "energy_component_ucb_objective": (0.50 * ucb[..., 3]).astype(np.float32),
        "energy_component_harm": (0.15 * p_harm).astype(np.float32),
        "energy_component_dual": (-0.05 * p_dual).astype(np.float32),
        "energy_component_rank": (-0.03 * rank_z).astype(np.float32),
    }


def safety_margins(
    components: Mapping[str, np.ndarray],
    physical: np.ndarray,
    support: np.ndarray,
    support_score: np.ndarray,
    support_threshold: float,
    policy: Mapping[str, float],
) -> dict[str, np.ndarray]:
    ucb = np.asarray(components["metric_ucb"])
    p_dual = np.asarray(components["dual_probability"])
    p_harm = np.asarray(components["harm_probability"])
    margin = {
        "physical_mask": np.asarray(physical, dtype=bool),
        "support_mask": np.asarray(support, dtype=bool),
        "support_score": np.asarray(support_score, dtype=np.float32),
        "support_margin": (support_threshold - support_score).astype(np.float32),
        "ate_ucb_margin": (-float(policy["margin_ate"]) - ucb[..., 0]).astype(np.float32),
        "objective_ucb_margin": (-float(policy["margin_objective"]) - ucb[..., 3]).astype(np.float32),
        "dual_probability_margin": (p_dual - float(policy["dual_threshold"])).astype(np.float32),
        "harm_probability_margin": (float(policy["harm_threshold"]) - p_harm).astype(np.float32),
    }
    first = np.full(physical.shape, FIRST_REJECT_CODES["accepted"], dtype=np.int8)
    ordered = (
        ("physical", ~margin["physical_mask"]),
        ("support", ~margin["support_mask"]),
        ("ate_ucb", margin["ate_ucb_margin"] <= 0),
        ("objective_ucb", margin["objective_ucb_margin"] <= 0),
        ("dual_probability", margin["dual_probability_margin"] < 0),
        ("harm_probability", margin["harm_probability_margin"] < 0),
    )
    undecided = np.ones(physical.shape, dtype=bool)
    for name, rejected in ordered:
        selected = undecided & rejected
        first[selected] = FIRST_REJECT_CODES[name]
        undecided &= ~rejected
    first[:, 0] = FIRST_REJECT_CODES["incumbent_not_challenger"]
    margin["first_reject_code"] = first
    return margin


@dataclass(frozen=True)
class SidecarBindings:
    manifest: Path
    preregistration: Path
    schema: Path
    support_checkpoint: Path
    model_result: Path
    runner_v66: Path
    runner_v70: Path
    sampler: Path


class ProposalTailSidecar:
    """Accumulate copies of live planner arrays and flush after formal output."""

    def __init__(
        self,
        output_dir: Path,
        *,
        cohort_role: str,
        cohort: str,
        rows: Sequence[Mapping[str, Any]],
        bindings: SidecarBindings,
        policy: Mapping[str, float],
        support_threshold: float,
        support_quantile: float = 0.999,
        seed_ids: Sequence[int] = (1701, 1702, 1703),
    ) -> None:
        if cohort_role not in COHORT_ROLES:
            raise ValueError(f"invalid cohort role: {cohort_role}")
        reject_forbidden_path(output_dir)
        reject_forbidden_path(bindings.manifest)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(output_dir)
        self.output_dir = output_dir
        self.cohort_role = cohort_role
        self.cohort = cohort
        self.rows = list(rows)
        self.states = snapshot_states(rows)
        self.bindings = bindings
        self.policy = dict(policy)
        self.support_threshold = float(support_threshold)
        self.support_quantile = float(support_quantile)
        self.seed_ids = tuple(int(value) for value in seed_ids)
        self._draw_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._state_events: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._pool_arrays: dict[int, dict[str, np.ndarray]] = {}
        self._active_iteration: int | None = None

    def begin_iteration(
        self,
        iteration: int,
        *,
        proposal_mean: np.ndarray,
        proposal_covariance: np.ndarray,
        incumbent: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        self._active_iteration = int(iteration)
        self._pool_arrays[iteration] = {
            "proposal_mean": np.asarray(proposal_mean).copy(),
            "proposal_covariance": np.asarray(proposal_covariance).copy(),
            "incumbent": np.asarray(incumbent).copy(),
            "reference": np.asarray(reference).copy(),
        }

    def capture_sampling_event(self, event: Mapping[str, Any]) -> None:
        if self._active_iteration is None:
            raise RuntimeError("sampling event arrived before begin_iteration")
        destination = (
            self._draw_events if event["event"] == "draw_batch" else self._state_events
        )
        copied = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in event.items()
        }
        destination[self._active_iteration].append(copied)

    def record_pool(
        self,
        source_round: int,
        *,
        candidates: np.ndarray,
        actions_h4: np.ndarray,
        support_score: np.ndarray,
        physical_guard: np.ndarray,
        support_mask: np.ndarray,
        valid: np.ndarray,
        safe: np.ndarray,
        energy: np.ndarray,
        prediction: Mapping[str, np.ndarray],
        p_dual: np.ndarray,
        p_harm: np.ndarray,
        feature_data: Mapping[str, np.ndarray],
        elite_index: np.ndarray | None = None,
        proposal_updated: np.ndarray | None = None,
        safe_count: np.ndarray | None = None,
    ) -> None:
        components = scorer_components(prediction, p_dual, p_harm, self.policy)
        margins = safety_margins(
            components,
            physical_guard,
            support_mask,
            support_score,
            self.support_threshold,
            self.policy,
        )
        arrays = self._pool_arrays.setdefault(source_round, {})
        arrays.update(
            {
                "candidates_feature_waypoint_units": np.asarray(candidates, dtype=np.float32).copy(),
                "candidate_hash": candidate_hashes(candidates),
                "actions_h4_m_rad": np.asarray(actions_h4, dtype=np.float32).copy(),
                "action_structure_features": action_structure_features(actions_h4),
                "energy": np.asarray(energy, dtype=np.float32).copy(),
                "energy_rank": _stable_rank(energy),
                "valid": np.asarray(valid, dtype=bool).copy(),
                "safe": np.asarray(safe, dtype=bool).copy(),
                **components,
                **margins,
            }
        )
        # These label-free arrays already exist in the formal feature artifact.
        # Copying them into the opt-in sidecar cannot affect planner decisions.
        for key in (
            "action_features",
            "static_features",
            "raw_ltwm_features",
            "scorer_v2_aux_features",
            "ate_by_seed",
            "rtp_by_seed",
            "support_by_seed",
            "majority_support",
        ):
            if key in feature_data:
                destination = (
                    f"ltwm_{key}"
                    if key in {"ate_by_seed", "rtp_by_seed", "support_by_seed", "majority_support"}
                    else key
                )
                arrays[destination] = np.asarray(feature_data[key]).copy()
        if elite_index is not None:
            arrays["elite_index_for_update"] = np.asarray(elite_index, dtype=np.int64).copy()
        if proposal_updated is not None:
            arrays["proposal_updated"] = np.asarray(proposal_updated, dtype=bool).copy()
        if safe_count is not None:
            arrays["safe_count"] = np.asarray(safe_count, dtype=np.int64).copy()

    def _write_draws(self, iteration: int, directory: Path) -> None:
        events = self._draw_events[iteration]
        if not events:
            raise RuntimeError(f"live sidecar missing raw draw events for iteration {iteration}")
        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for event in events:
            count = len(event["raw_draw_unclipped"])
            arrays["state_index"].append(np.full(count, event["state_index"], dtype=np.int64))
            arrays["draw_sequence"].append(np.asarray(event["draw_sequence"], dtype=np.int64))
            for key in (
                "raw_draw_unclipped",
                "draw_clipped",
                "clip_mask",
                "actions_h4",
                "normalized_action_vector",
                "physical_coordinate_margin",
                "physical_coordinate_mask",
                "physical_guard",
                "support_score",
                "support_margin",
                "support_mask",
                "accepted_mask",
                "final_candidate_index",
            ):
                arrays[key].append(np.asarray(event[key]))
        _write_npz_atomic(
            directory / "raw_draws.npz",
            **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
        )
        states = self._state_events[iteration]
        _write_json_atomic(directory / "sampling_states.json", states)

    def finalize(self, *, formal_selected_path: Path, formal_freeze_path: Path) -> Path:
        reject_forbidden_path(formal_selected_path)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        states_path = self.output_dir / "states.jsonl"
        states_path.write_text(
            "".join(canonical_json(value) + "\n" for value in self.states),
            encoding="utf-8",
        )
        for source_round, arrays in sorted(self._pool_arrays.items()):
            name = "fixed_pool" if source_round == -1 else f"iteration_{source_round:02d}"
            directory = self.output_dir / name
            directory.mkdir(parents=True, exist_ok=False)
            _write_npz_atomic(directory / "candidates_and_scores.npz", **arrays)
            if source_round >= 0:
                self._write_draws(source_round, directory)
        fixed = self._pool_arrays[-1]["candidates_feature_waypoint_units"]
        rounds = [self._pool_arrays[index]["candidates_feature_waypoint_units"][:, 1:] for index in range(3)]
        combined = np.concatenate((fixed, *rounds), axis=1).astype(np.float32)
        source_round = np.asarray([-1] * 17 + [value for value in range(3) for _ in range(16)], dtype=np.int8)
        source_candidate = np.asarray(list(range(17)) + [value for _ in range(3) for value in range(1, 17)], dtype=np.int8)
        record_key = np.asarray([value["record_key"] for value in self.states])
        candidate_table_path = self.output_dir / "candidate_table.npz"
        _write_npz_atomic(
            candidate_table_path,
            record_key=record_key,
            candidates_feature_waypoint_units=combined,
            candidate_hash=candidate_hashes(combined),
            source_round=np.broadcast_to(source_round, combined.shape[:2]),
            source_candidate_index=np.broadcast_to(source_candidate, combined.shape[:2]),
        )
        formal = np.load(formal_selected_path, allow_pickle=False)
        selected_arrays = {
            key: np.asarray(formal[key])
            for key in formal.files
            if key in {
                "record_key",
                "fixed_pool_selected_index",
                "final_cem_selected_index",
                "final_cem_energy",
                "final_cem_valid",
                "final_cem_safe",
                "archive_energy",
                "archive_source_round",
                "archive_source_candidate",
            }
        }
        if "final_cem_selected_index" in selected_arrays:
            selected_arrays["fallback_to_incumbent"] = (
                selected_arrays["final_cem_selected_index"] == 0
            )
        _write_npz_atomic(self.output_dir / "formal_selection_snapshot.npz", **selected_arrays)
        unavailable = {
            "seed_specific_rpe": "feature artifact exposes rtp_by_seed, not metric RPE by seed",
            "seed_specific_endpoint": "not emitted by the frozen three-seed LTWM feature contract",
            "seed_specific_objective": "not emitted by the frozen three-seed LTWM feature contract",
            "seed_specific_logits": "frozen ensemble_predict returns only member-averaged logits",
        }
        files = {}
        for path in sorted(self.output_dir.rglob("*")):
            if path.is_file() and path.name != "collection_manifest.json":
                files[str(path.relative_to(self.output_dir))] = sha256_file(path)
        binding_hashes = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in self.bindings.__dict__.items()
        }
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "cohort": self.cohort,
            "cohort_role": self.cohort_role,
            "evidence_role": "collection_infrastructure_only",
            "formal_behavior_influence": False,
            "raw_draws_observable": True,
            "states": len(self.states),
            "candidate_count": int(combined.shape[1]),
            "seed_count": len(self.seed_ids),
            "seed_ids": list(self.seed_ids),
            "metric_names": list(METRIC_NAMES),
            "feature_waypoint_unit_m": FEATURE_WAYPOINT_UNIT_M,
            "first_reject_codes": FIRST_REJECT_CODES,
            "identity": {
                "manifest_sha256": sha256_file(self.bindings.manifest),
                "record_key_sha256": array_sha256(record_key),
            },
            "policy": self.policy,
            "support_threshold": self.support_threshold,
            "support_quantile": self.support_quantile,
            "physical_coordinate_names": list(PHYSICAL_COORDINATE_NAMES),
            "bindings": binding_hashes,
            "formal_outputs": {
                "selected_path": str(formal_selected_path),
                "selected_sha256": sha256_file(formal_selected_path),
                "freeze_path": str(formal_freeze_path),
                "freeze_sha256": sha256_file(formal_freeze_path),
            },
            "unavailable_fields": unavailable,
            "files": files,
            "labels_opened": False,
            "reserve_accessed": False,
            "production_evidence_allowed": False,
        }
        manifest_path = self.output_dir / "collection_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        return manifest_path


__all__ = [
    "COHORT_ROLES",
    "FEATURE_WAYPOINT_UNIT_M",
    "FIRST_REJECT_CODES",
    "METRIC_NAMES",
    "PHYSICAL_COORDINATE_NAMES",
    "ProposalTailSidecar",
    "SCHEMA",
    "SidecarBindings",
    "array_sha256",
    "candidate_hashes",
    "canonical_json",
    "identity_hash",
    "reject_forbidden_path",
    "sha256_file",
    "snapshot_states",
]
