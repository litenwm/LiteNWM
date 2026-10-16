"""Pure pose features shared by ROS and offline RGB inference."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from planning.nomad_action_adapter import nomad_waypoints_to_local_actions
from tools.evaluation.build_recon100_k17_source_v1 import action_set_features, trimmed_mean75
HORIZON = 4
SPACING_M = 0.25

@dataclass
class PoseSample:
    receive_time: float
    x: float
    y: float
    yaw: float


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def history_motion_parts(
    candidates_m: np.ndarray,
    samples: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    if candidates_m.ndim != 3 or candidates_m.shape[1:] != (HORIZON, 2):
        raise RuntimeError(
            f"expected candidates [K,{HORIZON},2], got {candidates_m.shape}"
        )

    if len(samples) != 4:
        raise RuntimeError(f"expected four poses, got {len(samples)}")

    positions = np.asarray(
        [[sample.x, sample.y] for sample in samples],
        dtype=np.float32,
    )
    yaw = np.asarray(
        [sample.yaw for sample in samples],
        dtype=np.float32,
    )

    global_velocity = np.diff(positions, axis=0)

    cosine = math.cos(float(yaw[-1]))
    sine = math.sin(float(yaw[-1]))
    rotation = np.asarray(
        [[cosine, sine], [-sine, cosine]],
        dtype=np.float32,
    )
    local_velocity = global_velocity @ rotation.T

    speed = np.linalg.norm(local_velocity, axis=1)
    yaw_delta = wrap_angle(np.diff(yaw)).astype(np.float32)

    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
    velocity_weighted = (
        local_velocity * weights[:, None]
    ).sum(axis=0)
    speed_weighted = float((speed * weights).sum())
    yaw_rate = float((yaw_delta * weights).sum())

    state = np.concatenate(
        (
            local_velocity.reshape(-1),
            speed,
            yaw_delta,
            np.asarray(
                [
                    speed.mean(),
                    speed[-1],
                    speed_weighted,
                    speed[-1] - speed[-2],
                    yaw_delta.mean(),
                    yaw_rate,
                ],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)

    if state.shape != (18,):
        raise RuntimeError(f"18D history state changed: {state.shape}")

    steps = np.arange(
        1,
        HORIZON + 1,
        dtype=np.float32,
    )[:, None]

    expected_last = steps * local_velocity[-1]
    expected_weighted = steps * velocity_weighted

    expected_twist = np.zeros((HORIZON, 2), dtype=np.float32)
    for future_step in range(HORIZON):
        heading = (future_step + 0.5) * yaw_rate
        increment = speed_weighted * np.asarray(
            [math.cos(heading), math.sin(heading)],
            dtype=np.float32,
        )
        if future_step:
            expected_twist[future_step] = expected_twist[future_step - 1]
        expected_twist[future_step] += increment

    candidate_steps = np.diff(
        np.concatenate(
            (
                np.zeros(
                    (len(candidates_m), 1, 2),
                    dtype=np.float32,
                ),
                candidates_m,
            ),
            axis=1,
        ),
        axis=1,
    )
    candidate_speed = np.linalg.norm(candidate_steps, axis=-1)

    def residual(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(
            candidates_m - expected[None],
            axis=-1,
        )
        return distance.mean(axis=1), distance[:, -1]

    last_mean, last_final = residual(expected_last)
    weighted_mean, weighted_final = residual(expected_weighted)
    twist_mean, twist_final = residual(expected_twist)

    expected_direction = velocity_weighted / max(
        float(np.linalg.norm(velocity_weighted)),
        1.0e-6,
    )
    candidate_direction = candidates_m[:, -1] / np.maximum(
        np.linalg.norm(
            candidates_m[:, -1],
            axis=1,
            keepdims=True,
        ),
        1.0e-6,
    )

    candidate_specific = np.stack(
        (
            last_mean,
            last_final,
            weighted_mean,
            weighted_final,
            twist_mean,
            twist_final,
            np.abs(candidate_speed.mean(axis=1) - speed_weighted),
            np.abs(candidate_speed[:, 0] - speed[-1]),
            np.abs(
                np.linalg.norm(candidates_m[:, -1], axis=1)
                / HORIZON
                - speed_weighted
            ),
            1.0 - candidate_direction @ expected_direction,
        ),
        axis=-1,
    ).astype(np.float32)

    if candidate_specific.shape != (len(candidates_m), 10):
        raise RuntimeError(
            f"candidate-specific feature changed: "
            f"{candidate_specific.shape}"
        )

    return state, candidate_specific


def build_history_features(
    candidates_raw: np.ndarray,
    reference_raw: np.ndarray,
    samples: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    state, candidate_specific = history_motion_parts(
        candidates_raw * SPACING_M,
        samples,
    )
    _, reference_specific = history_motion_parts(
        reference_raw * SPACING_M,
        samples,
    )

    center = np.median(reference_specific, axis=0, keepdims=True)
    scale = np.maximum(
        reference_specific.std(axis=0, keepdims=True),
        1.0e-5,
    )
    relative = np.clip(
        (candidate_specific - center) / scale,
        -8.0,
        8.0,
    )

    repeated_state = np.repeat(
        state[None],
        len(candidates_raw),
        axis=0,
    )

    result = np.concatenate(
        (
            repeated_state,
            candidate_specific,
            relative,
        ),
        axis=-1,
    ).astype(np.float32)

    if result.shape != (len(candidates_raw), 38):
        raise RuntimeError(
            f"38D history feature changed: {result.shape}"
        )

    return result, state


