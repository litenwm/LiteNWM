#!/usr/bin/env python3
"""Compatibility loader for legacy and tri-domain scorer action support."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from planning.nomad_action_adapter import integrate_local_actions
from planning.trajectory_constraints import ActionSupportStats


DEFAULT_TAIL_SUPPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "runtime/v70_tridomain/action_support/coherent_turn_tail_v1.npz"
)
SUPPORT_RECOVERY_DISTANCE_MULTIPLIER = 1.5
SUPPORT_RECOVERY_POPULATION = 16
SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M = 0.06
SUPPORT_RECOVERY_FIRST_STEP_LATERAL_TOLERANCE_M = 0.01
SUPPORT_RECOVERY_FIRST_STEP_YAW_TOLERANCE_RAD = 0.03
REFERENCE_INTENT_MIN_MEDIAN_LATERAL_M = 0.06
REFERENCE_INTENT_MIN_AGREEMENT = 0.75
REFERENCE_INTENT_OPPOSITE_TOLERANCE_M = 0.02


def reference_intent_guard(
    reference_waypoints: np.ndarray,
    candidate_waypoints: np.ndarray,
    safe: np.ndarray,
    *,
    spacing_m: float,
    enabled: bool,
    min_median_lateral_m: float = (
        REFERENCE_INTENT_MIN_MEDIAN_LATERAL_M
    ),
    min_agreement: float = REFERENCE_INTENT_MIN_AGREEMENT,
    opposite_tolerance_m: float = (
        REFERENCE_INTENT_OPPOSITE_TOLERANCE_M
    ),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject strong CEM reversals of a confident NoMaD lateral intent.

    Frozen model scores remain untouched. The guard only masks a candidate
    when the raw NoMaD family strongly agrees on a left/right H4 endpoint and
    the challenger ends materially on the opposite side.
    """

    reference = np.asarray(reference_waypoints, dtype=np.float32)
    candidates = np.asarray(candidate_waypoints, dtype=np.float32)
    original_safe = np.asarray(safe, dtype=bool)
    if reference.ndim != 4 or reference.shape[0] != 1:
        raise ValueError(
            "reference_waypoints must have shape [1,R,H,2]"
        )
    if candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError(
            "candidate_waypoints must have shape [1,K,H,2]"
        )
    if reference.shape[-1] != 2 or candidates.shape[-1] != 2:
        raise ValueError("waypoints must contain XY coordinates")
    if original_safe.shape != candidates.shape[:2]:
        raise ValueError("safe shape does not match candidates")
    if not (
        np.isfinite(reference).all()
        and np.isfinite(candidates).all()
        and np.isfinite(float(spacing_m))
        and float(spacing_m) > 0.0
    ):
        raise ValueError("reference intent inputs must be finite")
    if not (
        np.isfinite(float(min_median_lateral_m))
        and float(min_median_lateral_m) > 0.0
        and np.isfinite(float(min_agreement))
        and 0.5 < float(min_agreement) <= 1.0
        and np.isfinite(float(opposite_tolerance_m))
        and float(opposite_tolerance_m) >= 0.0
    ):
        raise ValueError("invalid reference intent thresholds")

    reference_end_y_m = (
        reference[0, :, -1, 1] * np.float32(spacing_m)
    )
    median_end_y_m = float(np.median(reference_end_y_m))
    direction = (
        1
        if median_end_y_m > 0.0
        else (-1 if median_end_y_m < 0.0 else 0)
    )
    agreement = (
        float(
            np.mean(
                direction * reference_end_y_m
                >= float(min_median_lateral_m) * 0.5
            )
        )
        if direction
        else 0.0
    )
    active = bool(
        enabled
        and direction
        and abs(median_end_y_m) >= float(min_median_lateral_m)
        and agreement >= float(min_agreement)
    )

    compatible = np.ones_like(original_safe, dtype=bool)
    if active:
        candidate_end_y_m = (
            candidates[0, :, -1, 1] * np.float32(spacing_m)
        )
        compatible[0] = (
            direction * candidate_end_y_m
            >= -float(opposite_tolerance_m)
        )
    guarded_safe = original_safe & compatible
    audit = {
        "schema": "v70_reference_intent_guard_v1",
        "enabled": bool(enabled),
        "active": active,
        "direction": (
            "left" if direction > 0 else (
                "right" if direction < 0 else "uncertain"
            )
        ),
        "median_endpoint_y_m": median_end_y_m,
        "agreement": agreement,
        "original_safe_count": int(original_safe.sum()),
        "compatible_count": int(compatible.sum()),
        "guarded_safe_count": int(guarded_safe.sum()),
        "rejected_safe_count": int(
            (original_safe & ~compatible).sum()
        ),
    }
    return guarded_safe, audit


@lru_cache(maxsize=4)
def _load_tail_support(path_text: str) -> dict[str, np.ndarray]:
    path = Path(path_text)
    if not path.is_file():
        return {}
    required = (
        "tail_support_actions",
        "tail_normalization_lower",
        "tail_normalization_upper",
        "tail_knn_distance_max",
        "tail_min_forward_m",
        "tail_min_step_length_m",
        "tail_max_step_length_m",
        "tail_max_abs_dyaw_rad",
        "tail_max_abs_yaw_sum_rad",
        "tail_yaw_sign_tolerance_rad",
        "compatible_action_support_seeds",
        "compatible_action_support_lower",
        "compatible_action_support_upper",
        "compatible_spacing_support_sha256",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise RuntimeError(
                f"tail action support asset is missing fields: {missing}"
            )
        return {key: np.asarray(archive[key]).copy() for key in required}


def _tail_support_for(value: dict[str, Any]) -> dict[str, np.ndarray]:
    configured = os.environ.get(
        "V70_ACTION_TAIL_SUPPORT",
        str(DEFAULT_TAIL_SUPPORT_PATH),
    )
    asset = _load_tail_support(str(Path(configured).resolve()))
    if not asset:
        return {}
    seed = value.get("seed")
    seeds = asset["compatible_action_support_seeds"].reshape(-1)
    if seed is None:
        return {}
    matching = np.flatnonzero(seeds == int(seed))
    if len(matching) != 1:
        return {}
    index = int(matching[0])
    lower_key = "lower_p01" if "lower_p01" in value else "lower"
    upper_key = "upper_p99" if "upper_p99" in value else "upper"
    if lower_key not in value or upper_key not in value:
        return {}
    lower = np.asarray(value[lower_key], dtype=np.float32)
    upper = np.asarray(value[upper_key], dtype=np.float32)
    expected_lower = asset["compatible_action_support_lower"][index]
    expected_upper = asset["compatible_action_support_upper"][index]
    if not (
        lower.shape == expected_lower.shape
        and upper.shape == expected_upper.shape
        and np.allclose(lower, expected_lower, rtol=0.0, atol=1.0e-6)
        and np.allclose(upper, expected_upper, rtol=0.0, atol=1.0e-6)
    ):
        return {}
    return asset


def coherent_turn_tail_support(
    actions_h4: np.ndarray,
    *,
    spacing_support_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train-fit coherent-turn support and normalized NN distance."""

    actions = np.asarray(actions_h4, dtype=np.float32)
    if actions.ndim != 4 or actions.shape[-2:] != (4, 3):
        raise ValueError("actions_h4 must have shape [N,K,4,3]")
    configured = os.environ.get(
        "V70_ACTION_TAIL_SUPPORT",
        str(DEFAULT_TAIL_SUPPORT_PATH),
    )
    asset = _load_tail_support(str(Path(configured).resolve()))
    empty = np.zeros(actions.shape[:2], dtype=bool)
    distance = np.full(actions.shape[:2], np.inf, dtype=np.float32)
    if not asset:
        return empty, distance
    expected_sha256 = str(
        np.asarray(asset["compatible_spacing_support_sha256"]).item()
    )
    if spacing_support_sha256 != expected_sha256:
        return empty, distance

    finite = np.isfinite(actions).all(axis=(2, 3))
    safe = np.where(np.isfinite(actions), actions, 0.0)
    scale = (
        asset["tail_normalization_upper"]
        - asset["tail_normalization_lower"]
    ).reshape(1, 1, 12)
    candidate = safe.reshape(*safe.shape[:2], 1, 12) / scale
    templates = asset["tail_support_actions"].reshape(1, 1, -1, 12) / scale
    distance = np.sqrt(
        np.mean((candidate - templates) ** 2, axis=-1)
    ).min(axis=-1).astype(np.float32)

    translation = np.linalg.norm(safe[..., :2], axis=-1)
    yaw = safe[..., 2]
    tolerance = float(asset["tail_yaw_sign_tolerance_rad"].item())
    same_turn = np.all(yaw >= -tolerance, axis=-1) | np.all(
        yaw <= tolerance, axis=-1
    )
    physical = (
        finite
        & np.all(
            safe[..., 0] >= float(asset["tail_min_forward_m"].item()),
            axis=-1,
        )
        & np.all(
            translation >= float(asset["tail_min_step_length_m"].item()),
            axis=-1,
        )
        & np.all(
            translation <= float(asset["tail_max_step_length_m"].item()),
            axis=-1,
        )
        & np.all(
            np.abs(yaw) <= float(asset["tail_max_abs_dyaw_rad"].item()),
            axis=-1,
        )
        & (
            np.abs(yaw).sum(axis=-1)
            <= float(asset["tail_max_abs_yaw_sum_rad"].item())
        )
        & same_turn
    )
    support = physical & (
        distance <= float(asset["tail_knn_distance_max"].item())
    )
    return support, distance


def select_intent_preserving_tail_recovery_actions(
    reference_actions_h4: np.ndarray,
    *,
    spacing_support_sha256: str,
    population: int = SUPPORT_RECOVERY_POPULATION,
    distance_multiplier: float = SUPPORT_RECOVERY_DISTANCE_MULTIPLIER,
    intent_direction: int = 0,
    allow_partial_population: bool = False,
    minimum_unique_templates: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project NoMaD intent onto sealed train-supported H4 actions.

    This helper does not declare a candidate safe.  It only returns exact
    templates from the sealed coherent-turn training support bank when at
    least ``population`` distinct templates are close to the current NoMaD
    proposal family.  The caller must still run the unchanged physical,
    support, LTWM and selector gates.
    """

    reference = np.asarray(reference_actions_h4, dtype=np.float32)
    if reference.ndim != 3 or reference.shape[-2:] != (4, 3):
        raise ValueError(
            "reference_actions_h4 must have shape [K,4,3]"
        )
    if not np.isfinite(reference).all():
        raise ValueError("reference_actions_h4 contains NaN/Inf")
    if population <= 0:
        raise ValueError("population must be positive")
    if intent_direction not in (-1, 0, 1):
        raise ValueError("intent_direction must be -1, 0 or 1")
    if minimum_unique_templates <= 0:
        raise ValueError("minimum_unique_templates must be positive")
    if (
        not np.isfinite(distance_multiplier)
        or distance_multiplier < 1.0
    ):
        raise ValueError(
            "distance_multiplier must be finite and at least 1"
        )

    configured = os.environ.get(
        "V70_ACTION_TAIL_SUPPORT",
        str(DEFAULT_TAIL_SUPPORT_PATH),
    )
    asset = _load_tail_support(str(Path(configured).resolve()))
    empty = np.empty((0, 4, 3), dtype=np.float32)
    audit: dict[str, Any] = {
        "schema": "v70_support_recovery_audit_v1",
        "reason": "unavailable",
        "reference_count": int(reference.shape[0]),
        "template_count": 0,
        "population": int(population),
        "distance_multiplier": float(distance_multiplier),
        "allow_partial_population": bool(allow_partial_population),
        "minimum_unique_templates": int(minimum_unique_templates),
        "intent_direction": int(intent_direction),
        "tail_knn_distance_max": None,
        "max_intent_distance": None,
        "eligible_template_count": 0,
        "selected_template_indices": [],
        "selected_closest_reference_indices": [],
        "selected_intent_distances": [],
    }
    if not asset:
        audit["reason"] = "tail_support_asset_missing"
        return empty, audit

    expected_sha256 = str(
        np.asarray(asset["compatible_spacing_support_sha256"]).item()
    )
    if spacing_support_sha256 != expected_sha256:
        audit["reason"] = "spacing_support_sha256_mismatch"
        return empty, audit

    templates = np.asarray(
        asset["tail_support_actions"],
        dtype=np.float32,
    ).reshape(-1, 4, 3)
    scale = np.asarray(
        asset["tail_normalization_upper"]
        - asset["tail_normalization_lower"],
        dtype=np.float32,
    ).reshape(1, 1, 4, 3)
    if (
        templates.shape[0] < population
        or not np.isfinite(templates).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise RuntimeError("invalid coherent-turn recovery bank")

    tail_distance_max = float(
        np.asarray(asset["tail_knn_distance_max"]).item()
    )
    if not np.isfinite(tail_distance_max) or tail_distance_max <= 0.0:
        raise RuntimeError("invalid tail_knn_distance_max")
    max_intent_distance = float(
        distance_multiplier * tail_distance_max
    )
    pair_distance = np.sqrt(
        np.mean(
            (
                (
                    reference[:, None]
                    - templates[None]
                )
                / scale
            )
            ** 2,
            axis=(2, 3),
        )
    ).astype(np.float32)
    nearest_distance = pair_distance.min(axis=0)
    nearest_reference = pair_distance.argmin(axis=0)
    directional_compatible = np.ones(templates.shape[0], dtype=bool)
    if intent_direction:
        integrated = integrate_local_actions(templates[None])[0]
        endpoint_y_m = integrated[:, -1, 1]
        first_y_m = templates[:, 0, 1]
        first_yaw_rad = templates[:, 0, 2]
        directional_compatible = (
            (
                intent_direction * endpoint_y_m
                >= SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M
            )
            & (
                intent_direction * first_y_m
                >= -SUPPORT_RECOVERY_FIRST_STEP_LATERAL_TOLERANCE_M
            )
            & (
                intent_direction * first_yaw_rad
                >= -SUPPORT_RECOVERY_FIRST_STEP_YAW_TOLERANCE_RAD
            )
        )
    indices = np.arange(templates.shape[0], dtype=np.int64)
    order = np.lexsort((indices, nearest_distance))
    eligible = order[
        directional_compatible[order]
        & (nearest_distance[order] <= max_intent_distance)
    ]

    audit["directional_compatible_template_count"] = int(
        directional_compatible.sum()
    )

    audit.update(
        {
            "template_count": int(templates.shape[0]),
            "tail_knn_distance_max": tail_distance_max,
            "max_intent_distance": max_intent_distance,
            "eligible_template_count": int(eligible.size),
        }
    )
    partial_population = False
    if eligible.size < population:
        if (
            not allow_partial_population
            or eligible.size < minimum_unique_templates
        ):
            audit["reason"] = "insufficient_intent_overlap"
            return empty, audit
        # This is only enabled by the bounded, operator-supervised
        # directional-recovery branch. Repeat the nearest exact sealed
        # templates deterministically to fill the evaluator's fixed K16
        # contract; the V70/support/safety gates still decide every copy.
        selected = np.resize(eligible, population)
        partial_population = True
    else:
        selected = eligible[:population]
    selected_actions = templates[selected].copy()
    selected_support, _ = coherent_turn_tail_support(
        selected_actions[None],
        spacing_support_sha256=spacing_support_sha256,
    )
    if not bool(selected_support.all()):
        raise RuntimeError(
            "recovery bank returned an unsupported action"
        )
    audit.update(
        {
            "reason": (
                "partial_ready"
                if partial_population
                else "ready"
            ),
            "partial_population": partial_population,
            "distinct_selected_template_count": int(
                np.unique(selected).size
            ),
            "selected_template_indices": selected.tolist(),
            "selected_closest_reference_indices": (
                nearest_reference[selected].tolist()
            ),
            "selected_intent_distances": (
                nearest_distance[selected].astype(float).tolist()
            ),
        }
    )
    return selected_actions, audit


def support_stats(value: dict[str, Any]) -> ActionSupportStats:
    """Load rich legacy statistics or the latest tri-domain box bounds."""

    lower_key = "lower_p01" if "lower_p01" in value else "lower"
    upper_key = "upper_p99" if "upper_p99" in value else "upper"
    if lower_key not in value or upper_key not in value:
        raise RuntimeError(
            "action_support is missing lower/upper action bounds"
        )

    def optional_tensor(key: str) -> torch.Tensor | None:
        item = value.get(key)
        if item is None:
            return None
        return torch.as_tensor(item, dtype=torch.float32)

    def optional_float(key: str) -> float | None:
        item = value.get(key)
        return None if item is None else float(item)

    def optional_int(key: str) -> int | None:
        item = value.get(key)
        return None if item is None else int(item)

    tail = _tail_support_for(value)

    def tail_tensor(key: str) -> torch.Tensor | None:
        item = tail.get(key)
        if item is None:
            return None
        return torch.as_tensor(item, dtype=torch.float32)

    def tail_float(key: str, default: float | None = None) -> float | None:
        item = tail.get(key)
        if item is None:
            return default
        return float(np.asarray(item).item())

    stats = ActionSupportStats(
        lower=torch.as_tensor(value[lower_key], dtype=torch.float32),
        upper=torch.as_tensor(value[upper_key], dtype=torch.float32),
        mean=optional_tensor("mean"),
        covariance_inverse=optional_tensor("covariance_inverse"),
        mahalanobis_squared_max=optional_float(
            "mahalanobis_squared_p99"
        ),
        support_actions=optional_tensor("support_actions"),
        knn_k=optional_int("knn_k"),
        knn_distance_max=optional_float("knn_distance_p99"),
        roughness_max=optional_float("roughness_p99"),
        jerk_max=optional_float("jerk_p99"),
        curvature_max=optional_float("curvature_p99"),
        parent_trust_radius=optional_float("parent_trust_radius"),
        tail_support_actions=tail_tensor("tail_support_actions"),
        tail_normalization_lower=tail_tensor(
            "tail_normalization_lower"
        ),
        tail_normalization_upper=tail_tensor(
            "tail_normalization_upper"
        ),
        tail_knn_distance_max=tail_float("tail_knn_distance_max"),
        tail_min_forward_m=float(
            tail_float("tail_min_forward_m", 0.0)
        ),
        tail_min_step_length_m=float(
            tail_float("tail_min_step_length_m", 0.0)
        ),
        tail_max_step_length_m=tail_float("tail_max_step_length_m"),
        tail_max_abs_dyaw_rad=tail_float("tail_max_abs_dyaw_rad"),
        tail_max_abs_yaw_sum_rad=tail_float(
            "tail_max_abs_yaw_sum_rad"
        ),
        tail_yaw_sign_tolerance_rad=float(
            tail_float("tail_yaw_sign_tolerance_rad", 0.0)
        ),
    )
    stats.validate()
    return stats


__all__ = [
    "SUPPORT_RECOVERY_DISTANCE_MULTIPLIER",
    "SUPPORT_RECOVERY_POPULATION",
    "coherent_turn_tail_support",
    "select_intent_preserving_tail_recovery_actions",
    "support_stats",
]
