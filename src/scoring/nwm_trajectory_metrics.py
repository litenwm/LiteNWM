"""Paper-faithful planar trajectory metrics used by Navigation World Models.

The official NWM evaluator constructs an ``evo`` trajectory from the predicted
waypoint positions (without prepending the current pose) and reports:

* APE translation RMSE as ATE; and
* one-frame RPE translation RMSE as RPE.

NWM evaluates positions after dividing metric coordinates by the dataset's
average waypoint spacing.  The functions below implement the same mathematics
without adding a runtime dependency on ``evo``.
"""

from __future__ import annotations

import math

import numpy as np


NWM_OFFICIAL_COMMIT = "3f6cd8e70d6f2d1e2b9684acff510710135f0f41"
NWM_WAYPOINT_SPACING_METERS = {
    "recon": 0.25,
    "scand": 0.38,
}


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def integrate_local_actions(actions: np.ndarray) -> np.ndarray:
    """Integrate local-frame ``[dx, dy, dyaw]`` actions into planar positions.

    Args:
        actions: Array with shape ``[H, 3]``.  Each translation is expressed in
            the robot frame at the beginning of that transition.

    Returns:
        Cumulative positions with shape ``[H, 2]`` in the frame at time ``t``.
        The current pose at time ``t`` is intentionally not prepended, matching
        NWM's official ``actions_to_traj`` implementation.
    """

    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("actions must have shape [H,3]")
    if len(value) < 2:
        raise ValueError("NWM RPE requires at least two predicted waypoints")
    if not np.isfinite(value).all():
        raise ValueError("actions contain non-finite values")

    pose = np.zeros(3, dtype=np.float64)
    positions = np.empty((len(value), 2), dtype=np.float64)
    for index, (dx, dy, dyaw) in enumerate(value):
        cosine = math.cos(float(pose[2]))
        sine = math.sin(float(pose[2]))
        pose = np.asarray(
            (
                pose[0] + cosine * float(dx) - sine * float(dy),
                pose[1] + sine * float(dx) + cosine * float(dy),
                wrap_to_pi(float(pose[2]) + float(dyaw)),
            ),
            dtype=np.float64,
        )
        positions[index] = pose[:2]
    return positions


def positions_relative_to_origin(
    positions: np.ndarray,
    yaw: np.ndarray,
    current_t: int,
    horizon: int,
) -> np.ndarray:
    """Express future world positions in the planar frame at ``current_t``."""

    value = np.asarray(positions, dtype=np.float64)
    angles = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or value.shape[1] < 2:
        raise ValueError("positions must have shape [T,>=2]")
    if current_t < 0 or current_t + horizon >= len(value):
        raise IndexError("requested future horizon is outside the trajectory")
    if len(angles) != len(value):
        raise ValueError("positions/yaw length mismatch")
    if horizon < 2:
        raise ValueError("NWM RPE requires a horizon of at least two")

    origin = value[current_t, :2]
    origin_yaw = float(angles[current_t])
    cosine = math.cos(-origin_yaw)
    sine = math.sin(-origin_yaw)
    future = value[current_t + 1 : current_t + horizon + 1, :2] - origin
    result = np.empty((horizon, 2), dtype=np.float64)
    result[:, 0] = cosine * future[:, 0] - sine * future[:, 1]
    result[:, 1] = sine * future[:, 0] + cosine * future[:, 1]
    return result


def nwm_ate_rpe(
    predicted_positions_m: np.ndarray,
    reference_positions_m: np.ndarray,
    *,
    waypoint_spacing_m: float,
) -> tuple[float, float]:
    """Compute NWM ATE and translational one-frame RPE.

    This is mathematically equivalent to the official calls:

    ``evo.main_ape.ape(..., PoseRelation.translation_part, align=False,
    correct_scale=False).stats["rmse"]``

    and

    ``evo.main_rpe.rpe(..., PoseRelation.translation_part, delta=1,
    delta_unit=frames, align=False, correct_scale=False).stats["rmse"]``.
    """

    predicted = np.asarray(predicted_positions_m, dtype=np.float64)
    reference = np.asarray(reference_positions_m, dtype=np.float64)
    if predicted.shape != reference.shape:
        raise ValueError("predicted/reference trajectory shapes differ")
    if predicted.ndim != 2 or predicted.shape[1] != 2 or len(predicted) < 2:
        raise ValueError("trajectories must have matching shape [H,2], H>=2")
    if waypoint_spacing_m <= 0 or not math.isfinite(waypoint_spacing_m):
        raise ValueError("waypoint_spacing_m must be finite and positive")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("trajectory contains non-finite values")

    normalized_error = (predicted - reference) / waypoint_spacing_m
    ate = float(np.sqrt(np.mean(np.sum(np.square(normalized_error), axis=1))))

    predicted_delta = np.diff(predicted, axis=0) / waypoint_spacing_m
    reference_delta = np.diff(reference, axis=0) / waypoint_spacing_m
    relative_error = predicted_delta - reference_delta
    rpe = float(np.sqrt(np.mean(np.sum(np.square(relative_error), axis=1))))
    return ate, rpe
