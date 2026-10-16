"""Pure fail-closed safety helpers shared by the ROS bridge and tests."""

from __future__ import annotations

from collections.abc import Iterable
import math
import statistics
from typing import Any


class ClearanceRecoveryLatch:
    """Block immediately and require stable clear evidence to reopen."""

    def __init__(
        self,
        *,
        recovery_s: float,
        min_clear_observations: int,
    ) -> None:
        if not math.isfinite(recovery_s) or recovery_s <= 0.0:
            raise ValueError("recovery_s must be positive and finite")
        if min_clear_observations <= 0:
            raise ValueError("min_clear_observations must be positive")
        self.recovery_s = float(recovery_s)
        self.min_clear_observations = int(min_clear_observations)
        self.clear_observations = 0
        self.last_blocked_time: float | None = None
        self.last_observation_time: float | None = None

    def observe(self, observed_clear: bool, now: float) -> bool:
        timestamp = float(now)
        if not math.isfinite(timestamp):
            raise ValueError("observation time must be finite")
        if (
            self.last_observation_time is not None
            and timestamp < self.last_observation_time
        ):
            self.clear_observations = 0
            self.last_blocked_time = timestamp
            self.last_observation_time = timestamp
            return False
        self.last_observation_time = timestamp

        if not bool(observed_clear):
            self.clear_observations = 0
            self.last_blocked_time = timestamp
            return False

        self.clear_observations += 1
        if self.last_blocked_time is None:
            # Startup is fail-closed until one full recovery window of
            # continuously clear observations has elapsed.
            self.last_blocked_time = timestamp
        return (
            self.clear_observations >= self.min_clear_observations
            and timestamp - self.last_blocked_time >= self.recovery_s
        )


def evaluate_forward_corridor(
    points: Iterable[tuple[float, float, float]],
    *,
    min_forward_clearance_m: float,
    corridor_half_width_m: float,
    min_height_m: float,
    max_height_m: float,
    min_finite_points: int,
) -> dict[str, Any]:
    """Evaluate whether a point cloud shows a clear forward corridor."""

    if min_forward_clearance_m <= 0.0:
        raise ValueError("min_forward_clearance_m must be positive")
    if corridor_half_width_m <= 0.0:
        raise ValueError("corridor_half_width_m must be positive")
    if min_height_m >= max_height_m:
        raise ValueError("invalid obstacle height band")
    if min_finite_points <= 0:
        raise ValueError("min_finite_points must be positive")

    finite_points = 0
    corridor_points = 0
    minimum_distance_m = math.inf
    for point in points:
        if len(point) != 3:
            continue
        x, y, z = (float(value) for value in point)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        finite_points += 1
        if (
            x >= 0.0
            and abs(y) <= corridor_half_width_m
            and min_height_m <= z <= max_height_m
        ):
            corridor_points += 1
            minimum_distance_m = min(
                minimum_distance_m,
                math.hypot(x, y),
            )

    cloud_valid = finite_points >= min_finite_points
    corridor_clear = (
        cloud_valid
        and minimum_distance_m >= min_forward_clearance_m
    )
    return {
        "cloud_valid": cloud_valid,
        "corridor_clear": corridor_clear,
        "finite_points": finite_points,
        "corridor_points": corridor_points,
        "minimum_distance_m": minimum_distance_m,
    }


def evaluate_ground_aware_clearance(
    points: Iterable[tuple[float, float, float]],
    *,
    min_forward_clearance_m: float,
    corridor_half_width_m: float,
    rotation_clearance_radius_m: float,
    min_height_above_ground_m: float,
    max_height_above_ground_m: float,
    min_finite_points: int,
    rotation_ground_skimming_max_height_m: float = 0.105,
    max_isolated_rotation_ground_skimming_points: int = 3,
    ground_x_min_m: float = 0.30,
    ground_x_max_m: float = 1.50,
    ground_half_width_m: float = 0.45,
    ground_z_min_m: float = -0.60,
    ground_z_max_m: float = 0.10,
    min_ground_points: int = 30,
    ground_inlier_tolerance_m: float = 0.05,
    self_foot_x_min_m: float = 0.05,
    self_foot_x_max_m: float = 0.38,
    self_foot_abs_y_min_m: float = 0.18,
    self_foot_abs_y_max_m: float = 0.38,
    self_rear_foot_x_min_m: float = -0.42,
    self_rear_foot_x_max_m: float = -0.05,
    self_rear_foot_abs_y_min_m: float = 0.34,
    self_rear_foot_abs_y_max_m: float = 0.46,
    self_rear_foot_max_height_above_ground_m: float = 0.24,
) -> dict[str, Any]:
    """Evaluate motion clearance after estimating and removing the floor.

    The point cloud is expected in ``base_link``.  A robust near-field ground
    height is estimated first.  Points close to the floor and points inside
    measured Go2 front- and rear-foot self-return boxes are excluded before
    checking the forward corridor and the in-place rotation envelope.  The
    new rear-foot mask is height-bounded so a taller object in the same
    horizontal footprint still blocks motion. Up to three near-ground returns
    inside the rotation envelope may be treated as sparse lidar
    ground-skimming noise; a denser set or any taller return still blocks. The proven
    front-foot contract remains unchanged. Any invalid or insufficient cloud
    fails closed.
    """

    positive = {
        "min_forward_clearance_m": min_forward_clearance_m,
        "corridor_half_width_m": corridor_half_width_m,
        "rotation_clearance_radius_m": rotation_clearance_radius_m,
        "min_height_above_ground_m": min_height_above_ground_m,
        "max_height_above_ground_m": max_height_above_ground_m,
        "rotation_ground_skimming_max_height_m": (
            rotation_ground_skimming_max_height_m
        ),
        "ground_x_max_m": ground_x_max_m,
        "ground_half_width_m": ground_half_width_m,
        "ground_inlier_tolerance_m": ground_inlier_tolerance_m,
        "self_foot_x_max_m": self_foot_x_max_m,
        "self_foot_abs_y_max_m": self_foot_abs_y_max_m,
        "self_rear_foot_abs_y_min_m": self_rear_foot_abs_y_min_m,
        "self_rear_foot_abs_y_max_m": self_rear_foot_abs_y_max_m,
        "self_rear_foot_max_height_above_ground_m": (
            self_rear_foot_max_height_above_ground_m
        ),
    }
    if any(float(value) <= 0.0 for value in positive.values()):
        raise ValueError("clearance dimensions must be positive")
    if max_height_above_ground_m <= min_height_above_ground_m:
        raise ValueError("invalid obstacle height-above-ground band")
    if not (
        min_height_above_ground_m
        < rotation_ground_skimming_max_height_m
        < max_height_above_ground_m
    ):
        raise ValueError("invalid rotation ground-skimming height")
    if ground_x_max_m <= ground_x_min_m:
        raise ValueError("invalid ground-estimation x range")
    if ground_z_max_m <= ground_z_min_m:
        raise ValueError("invalid ground-estimation z range")
    if self_foot_x_max_m <= self_foot_x_min_m:
        raise ValueError("invalid self-foot x range")
    if self_foot_abs_y_max_m <= self_foot_abs_y_min_m:
        raise ValueError("invalid self-foot y range")
    if self_rear_foot_x_max_m <= self_rear_foot_x_min_m:
        raise ValueError("invalid rear self-foot x range")
    if self_rear_foot_x_max_m >= 0.0:
        raise ValueError("rear self-foot x range must remain behind base_link")
    if self_rear_foot_abs_y_max_m <= self_rear_foot_abs_y_min_m:
        raise ValueError("invalid rear self-foot y range")
    if not (
        min_height_above_ground_m
        < self_rear_foot_max_height_above_ground_m
        <= max_height_above_ground_m
    ):
        raise ValueError("invalid self-foot height limit")
    if min_finite_points <= 0 or min_ground_points <= 0:
        raise ValueError("minimum point counts must be positive")
    if (
        not isinstance(
            max_isolated_rotation_ground_skimming_points,
            int,
        )
        or max_isolated_rotation_ground_skimming_points < 0
    ):
        raise ValueError("invalid isolated rotation point allowance")

    finite: list[tuple[float, float, float]] = []
    ground_candidates: list[float] = []
    for point in points:
        if len(point) != 3:
            continue
        x, y, z = (float(value) for value in point)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        finite.append((x, y, z))
        if (
            ground_x_min_m <= x <= ground_x_max_m
            and abs(y) <= ground_half_width_m
            and ground_z_min_m <= z <= ground_z_max_m
        ):
            ground_candidates.append(z)

    cloud_valid = len(finite) >= min_finite_points
    ground_height_m = (
        float(statistics.median(ground_candidates))
        if ground_candidates
        else math.nan
    )
    ground_inliers = (
        sum(
            abs(value - ground_height_m) <= ground_inlier_tolerance_m
            for value in ground_candidates
        )
        if math.isfinite(ground_height_m)
        else 0
    )
    ground_valid = (
        cloud_valid
        and len(ground_candidates) >= min_ground_points
        and ground_inliers >= min_ground_points
    )

    minimum_forward_distance_m = math.inf
    minimum_rotation_distance_m = math.inf
    raw_minimum_rotation_distance_m = math.inf
    rotation_ground_skimming_distances: list[float] = []
    ignored_rotation_ground_skimming_points = 0
    obstacle_points = 0
    self_filtered_points = 0
    front_self_filtered_points = 0
    rear_self_filtered_points = 0
    if ground_valid:
        min_z = ground_height_m + min_height_above_ground_m
        max_z = ground_height_m + max_height_above_ground_m
        for x, y, z in finite:
            if not min_z <= z <= max_z:
                continue
            height_above_ground = z - ground_height_m
            front_self_return = (
                self_foot_x_min_m <= x <= self_foot_x_max_m
                and self_foot_abs_y_min_m
                <= abs(y)
                <= self_foot_abs_y_max_m
            )
            rear_self_return = (
                self_rear_foot_x_min_m
                <= x
                <= self_rear_foot_x_max_m
                and self_rear_foot_abs_y_min_m
                <= abs(y)
                <= self_rear_foot_abs_y_max_m
                and height_above_ground
                <= self_rear_foot_max_height_above_ground_m
            )
            if front_self_return or rear_self_return:
                self_filtered_points += 1
                front_self_filtered_points += int(front_self_return)
                rear_self_filtered_points += int(rear_self_return)
                continue
            obstacle_points += 1
            distance = math.hypot(x, y)
            raw_minimum_rotation_distance_m = min(
                raw_minimum_rotation_distance_m,
                distance,
            )
            if x >= 0.0 and abs(y) <= corridor_half_width_m:
                minimum_forward_distance_m = min(
                    minimum_forward_distance_m,
                    distance,
                )
            if (
                distance < rotation_clearance_radius_m
                and height_above_ground
                <= rotation_ground_skimming_max_height_m
            ):
                rotation_ground_skimming_distances.append(distance)
            else:
                minimum_rotation_distance_m = min(
                    minimum_rotation_distance_m,
                    distance,
                )

    rotation_ground_skimming_points = len(
        rotation_ground_skimming_distances
    )
    if (
        rotation_ground_skimming_points
        > max_isolated_rotation_ground_skimming_points
    ):
        minimum_rotation_distance_m = min(
            minimum_rotation_distance_m,
            min(rotation_ground_skimming_distances),
        )
    else:
        ignored_rotation_ground_skimming_points = (
            rotation_ground_skimming_points
        )

    forward_clear = (
        ground_valid
        and minimum_forward_distance_m >= min_forward_clearance_m
    )
    rotation_clear = (
        ground_valid
        and minimum_rotation_distance_m >= rotation_clearance_radius_m
    )
    return {
        "cloud_valid": cloud_valid,
        "ground_valid": ground_valid,
        "ground_height_m": ground_height_m,
        "ground_candidate_points": len(ground_candidates),
        "ground_inlier_points": ground_inliers,
        "finite_points": len(finite),
        "obstacle_points": obstacle_points,
        "self_filtered_points": self_filtered_points,
        "front_self_filtered_points": front_self_filtered_points,
        "rear_self_filtered_points": rear_self_filtered_points,
        "forward_clear": forward_clear,
        "rotation_clear": rotation_clear,
        "minimum_forward_distance_m": minimum_forward_distance_m,
        "minimum_rotation_distance_m": minimum_rotation_distance_m,
        "raw_minimum_rotation_distance_m": (
            raw_minimum_rotation_distance_m
        ),
        "rotation_ground_skimming_points": (
            rotation_ground_skimming_points
        ),
        "ignored_rotation_ground_skimming_points": (
            ignored_rotation_ground_skimming_points
        ),
    }


def shape_forward_steering_for_clearance(
    *,
    target_vx: float,
    target_wz: float,
    rotation_clear: bool,
    forward_steering_wz_threshold: float,
    command_epsilon: float = 1e-4,
) -> tuple[float, float, bool]:
    """Fail closed when a curved forward command cannot safely steer.

    A curved command expresses coupled translational and angular intent. If
    rotational clearance is unavailable, removing only yaw silently changes
    that intent into straight motion. Hold both components at zero instead;
    the caller can retry after the clearance recovery latch opens. Pure
    in-place rotation remains unchanged and is blocked by the rotation gate.
    """

    values = (
        target_vx,
        target_wz,
        forward_steering_wz_threshold,
        command_epsilon,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("steering-shape inputs must be finite")
    if command_epsilon <= 0.0:
        raise ValueError("command_epsilon must be positive")
    if forward_steering_wz_threshold <= command_epsilon:
        raise ValueError("forward steering threshold is too small")
    forward_steering = bool(
        target_vx > command_epsilon
        and abs(target_wz) > command_epsilon
    )
    if forward_steering and not bool(rotation_clear):
        return 0.0, 0.0, True
    return float(target_vx), float(target_wz), False


def motion_clearance_allows(
    *,
    target_vx: float,
    target_wz: float,
    forward_clear: bool,
    rotation_clear: bool,
    command_epsilon: float = 1e-4,
) -> bool:
    """Apply only the clearance gate needed by the requested motion."""

    values = (target_vx, target_wz, command_epsilon)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if command_epsilon <= 0.0:
        raise ValueError("command_epsilon must be positive")
    forward_needed = target_vx > command_epsilon
    rotation_needed = abs(target_wz) > command_epsilon
    return (
        (not forward_needed or bool(forward_clear))
        and (not rotation_needed or bool(rotation_clear))
    )
