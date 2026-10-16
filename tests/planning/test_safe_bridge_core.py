from __future__ import annotations

import math

from deploy.v70_go2_orin.safe_bridge_core import (
    evaluate_forward_corridor,
    evaluate_ground_aware_clearance,
    motion_clearance_allows,
)


def evaluate(points):
    return evaluate_forward_corridor(
        points,
        min_forward_clearance_m=0.65,
        corridor_half_width_m=0.32,
        min_height_m=-0.25,
        max_height_m=0.60,
        min_finite_points=3,
    )


def ground_points(count: int = 40):
    return [
        (
            0.30 + 0.02 * index,
            -0.20 + 0.10 * (index % 5),
            -0.21 + 0.002 * (index % 3),
        )
        for index in range(count)
    ]


def evaluate_ground_aware(points, *, min_ground_points: int = 20):
    return evaluate_ground_aware_clearance(
        points,
        min_forward_clearance_m=0.65,
        corridor_half_width_m=0.32,
        rotation_clearance_radius_m=0.48,
        min_height_above_ground_m=0.08,
        max_height_above_ground_m=0.60,
        min_finite_points=20,
        ground_x_min_m=0.30,
        ground_x_max_m=1.50,
        ground_half_width_m=0.45,
        ground_z_min_m=-0.60,
        ground_z_max_m=0.10,
        min_ground_points=min_ground_points,
        ground_inlier_tolerance_m=0.05,
    )


def test_clear_corridor_passes_with_valid_cloud() -> None:
    result = evaluate(
        [
            (1.0, 0.0, 0.1),
            (2.0, 1.0, 0.0),
            (1.5, -1.0, 0.0),
        ]
    )
    assert result["cloud_valid"]
    assert result["corridor_clear"]
    assert result["minimum_distance_m"] == 1.0


def test_close_obstacle_fails_closed() -> None:
    result = evaluate(
        [
            (0.4, 0.1, 0.0),
            (2.0, 1.0, 0.0),
            (1.5, -1.0, 0.0),
        ]
    )
    assert result["cloud_valid"]
    assert not result["corridor_clear"]
    assert result["minimum_distance_m"] < 0.65


def test_points_outside_corridor_do_not_block() -> None:
    result = evaluate(
        [
            (0.1, 0.8, 0.0),
            (0.1, -0.8, 0.0),
            (0.1, 0.0, 1.2),
        ]
    )
    assert result["cloud_valid"]
    assert result["corridor_clear"]
    assert math.isinf(result["minimum_distance_m"])


def test_sparse_or_nonfinite_cloud_fails_closed() -> None:
    result = evaluate(
        [
            (math.nan, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ]
    )
    assert not result["cloud_valid"]
    assert not result["corridor_clear"]


def test_floor_and_front_feet_do_not_block_ground_aware_clearance() -> None:
    result = evaluate_ground_aware(
        ground_points()
        + [
            (0.20, 0.27, -0.09),
            (0.20, -0.27, -0.09),
        ]
    )

    assert result["cloud_valid"]
    assert result["ground_valid"]
    assert math.isclose(
        result["ground_height_m"],
        -0.208,
        abs_tol=0.005,
    )
    assert result["self_filtered_points"] == 2
    assert result["forward_clear"]
    assert result["rotation_clear"]
    assert math.isinf(result["minimum_forward_distance_m"])


def test_real_close_obstacle_blocks_forward_and_rotation() -> None:
    result = evaluate_ground_aware(
        ground_points() + [(0.40, 0.0, 0.0)]
    )

    assert result["ground_valid"]
    assert not result["forward_clear"]
    assert not result["rotation_clear"]
    assert math.isclose(
        result["minimum_forward_distance_m"],
        0.40,
    )


def test_side_obstacle_blocks_rotation_but_not_forward() -> None:
    result = evaluate_ground_aware(
        ground_points() + [(0.20, 0.40, 0.0)]
    )

    assert result["ground_valid"]
    assert result["forward_clear"]
    assert not result["rotation_clear"]
    assert math.isinf(result["minimum_forward_distance_m"])
    assert result["minimum_rotation_distance_m"] < 0.48


def test_insufficient_ground_evidence_fails_closed() -> None:
    result = evaluate_ground_aware(
        ground_points(10)
        + [(2.0, 1.0, 0.5) for _ in range(10)],
        min_ground_points=20,
    )

    assert result["cloud_valid"]
    assert not result["ground_valid"]
    assert not result["forward_clear"]
    assert not result["rotation_clear"]


def test_motion_gate_uses_only_requested_clearance() -> None:
    assert motion_clearance_allows(
        target_vx=0.05,
        target_wz=0.0,
        forward_clear=True,
        rotation_clear=False,
    )
    assert motion_clearance_allows(
        target_vx=0.0,
        target_wz=0.1,
        forward_clear=False,
        rotation_clear=True,
    )
    assert not motion_clearance_allows(
        target_vx=0.05,
        target_wz=0.1,
        forward_clear=True,
        rotation_clear=False,
    )
    assert not motion_clearance_allows(
        target_vx=math.nan,
        target_wz=0.0,
        forward_clear=True,
        rotation_clear=True,
    )


def test_motion_gate_rejects_invalid_epsilon() -> None:
    try:
        motion_clearance_allows(
            target_vx=0.0,
            target_wz=0.0,
            forward_clear=True,
            rotation_clear=True,
            command_epsilon=0.0,
        )
    except ValueError as exc:
        assert "command_epsilon" in str(exc)
    else:
        raise AssertionError("invalid epsilon must be rejected")
