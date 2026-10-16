"""Convert NoMaD cumulative waypoints to the LTWM local SE(2) action contract.

NoMaD predicts cumulative ``(x, y)`` waypoints in dataset waypoint-spacing
units.  LTWM consumes metric per-transition controls
``[dx_local, dy_local, dyaw]``.  The conversion below uses the tangent of the
predicted path as the otherwise-unobserved heading.  Integrating the returned
actions therefore reconstructs the input waypoints exactly (up to floating
point tolerance).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


HORIZON = 4
NWM_WAYPOINT_SPACING_METERS = {
    "recon": 0.25,
    "scand": 0.38,
    "sacson": 0.255,
}


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def window_seed(base_seed: int, window_uid: str) -> int:
    """Derive an order-independent Torch-compatible seed for one window."""

    if not window_uid:
        raise ValueError("window_uid must be nonempty")
    if base_seed < 0:
        raise ValueError("base_seed must be nonnegative")
    digest = hashlib.sha256(f"nomad-h4-v2|{base_seed}|{window_uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def nomad_waypoints_to_local_actions(
    waypoints: np.ndarray,
    *,
    dataset: str,
    horizon: int = HORIZON,
    yaw_epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Convert cumulative NoMaD waypoints ``[...,T,2]`` to ``[...,H,3]``.

    The waypoint path is expressed in the robot frame at the current image.
    Each segment is rotated into the frame defined by the previous path
    tangent.  A zero-length segment retains the previous heading.
    """

    value = np.asarray(waypoints, dtype=np.float32)
    if value.ndim < 2 or value.shape[-1] != 2:
        raise ValueError("waypoints must have shape [...,T,2]")
    if horizon <= 0 or value.shape[-2] < horizon:
        raise ValueError(f"waypoints must contain at least H={horizon} steps")
    if dataset not in NWM_WAYPOINT_SPACING_METERS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    if yaw_epsilon < 0 or not math.isfinite(yaw_epsilon):
        raise ValueError("yaw_epsilon must be finite and nonnegative")
    if not np.isfinite(value).all():
        raise ValueError("waypoints contain NaN or Inf")

    metric_xy = value[..., :horizon, :] * np.float32(
        NWM_WAYPOINT_SPACING_METERS[dataset]
    )
    origin = np.zeros((*metric_xy.shape[:-2], 1, 2), dtype=np.float32)
    segments = np.diff(np.concatenate((origin, metric_xy), axis=-2), axis=-2)
    flat = segments.reshape(-1, horizon, 2)
    actions = np.zeros((len(flat), horizon, 3), dtype=np.float32)

    for row_index, row in enumerate(flat):
        previous_heading = 0.0
        for step, segment in enumerate(row):
            segment_x = float(segment[0])
            segment_y = float(segment[1])
            length = math.hypot(segment_x, segment_y)
            heading = (
                previous_heading
                if length <= yaw_epsilon
                else math.atan2(segment_y, segment_x)
            )
            cosine = math.cos(previous_heading)
            sine = math.sin(previous_heading)
            dx_local = cosine * segment_x + sine * segment_y
            dy_local = -sine * segment_x + cosine * segment_y
            actions[row_index, step] = (
                dx_local,
                dy_local,
                wrap_to_pi(heading - previous_heading),
            )
            previous_heading = heading

    return actions.reshape((*segments.shape[:-2], horizon, 3))


def integrate_local_actions(actions: np.ndarray) -> np.ndarray:
    """Integrate ``[...,H,3]`` actions into cumulative local ``[...,H,2]``."""

    value = np.asarray(actions, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != 3:
        raise ValueError("actions must have shape [...,H,3]")
    if not np.isfinite(value).all():
        raise ValueError("actions contain NaN or Inf")
    flat = value.reshape(-1, value.shape[-2], 3)
    result = np.zeros((len(flat), value.shape[-2], 2), dtype=np.float64)
    for row_index, row in enumerate(flat):
        x = y = yaw = 0.0
        for step, (dx_local, dy_local, dyaw) in enumerate(row):
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            x += cosine * float(dx_local) - sine * float(dy_local)
            y += sine * float(dx_local) + cosine * float(dy_local)
            yaw = wrap_to_pi(yaw + float(dyaw))
            result[row_index, step] = (x, y)
    return result.reshape((*value.shape[:-2], value.shape[-2], 2))


__all__ = [
    "HORIZON",
    "NWM_WAYPOINT_SPACING_METERS",
    "integrate_local_actions",
    "nomad_waypoints_to_local_actions",
    "window_seed",
    "wrap_to_pi",
]
