"""Pure command shaping for reliable Go2 gait entry."""

from __future__ import annotations

import math


def forward_gait_entry_for_command(
    *,
    configured_entry_vx: float,
    target_wz: float,
    max_abs_wz: float,
) -> float:
    """Enable the walking-band jump only for a near-straight drive."""

    values = (configured_entry_vx, target_wz, max_abs_wz)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gait entry selection values must be finite")
    if configured_entry_vx < 0.0 or max_abs_wz < 0.0:
        raise ValueError("gait entry selection limits must be nonnegative")
    if abs(target_wz) > max_abs_wz:
        return 0.0
    return float(configured_entry_vx)


def _approach(current: float, target: float, max_change: float) -> float:
    return current + max(-max_change, min(max_change, target - current))


def shape_forward_gait_entry(
    *,
    current_vx: float,
    target_vx: float,
    max_change: float,
    minimum_entry_vx: float,
    epsilon: float,
) -> float:
    """Skip the non-walking velocity band for an explicit drive request.

    Unitree's high-level ``Move`` controller may shift the body without
    replacing a foot contact at very small forward velocities.  Repeatedly
    ramping through that band preloads the rear legs before a real step.  If a
    planner already requests at least ``minimum_entry_vx``, enter that proven
    walking band directly.  The caller must disable the entry for turning;
    smaller scan assists retain the legacy slew rate, and a zero target is
    never held above zero.
    """

    values = (
        current_vx,
        target_vx,
        max_change,
        minimum_entry_vx,
        epsilon,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gait shaping values must be finite")
    if current_vx < 0.0 or target_vx < 0.0:
        raise ValueError("forward gait shaping does not support reverse")
    if max_change <= 0.0 or epsilon <= 0.0 or minimum_entry_vx < 0.0:
        raise ValueError("invalid gait shaping limits")

    legacy = _approach(current_vx, target_vx, max_change)
    if minimum_entry_vx <= epsilon or target_vx < minimum_entry_vx:
        return legacy
    if current_vx + epsilon < minimum_entry_vx:
        return min(target_vx, minimum_entry_vx)
    return legacy
