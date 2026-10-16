"""ROS-free authorization for one execution of each freshly selected H1."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlanAuthorization:
    allowed: bool
    reason: str


class FreshPlanGate:
    """Keep sequence provenance only; never retain or replay an action."""

    def __init__(self) -> None:
        self.last_consumed_sequence = -1

    def consume(self, *, sequence: int, interlock: bool, abstained: bool,
                action) -> PlanAuthorization:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            return PlanAuthorization(False, "invalid_sequence")
        if sequence <= self.last_consumed_sequence:
            return PlanAuthorization(False, "sequence_already_consumed_or_superseded")
        if not interlock:
            return PlanAuthorization(False, "interlock_closed")
        if abstained:
            return PlanAuthorization(False, "planner_abstained")
        try:
            values = tuple(float(x) for x in action)
        except (TypeError, ValueError):
            return PlanAuthorization(False, "invalid_action")
        if len(values) != 3 or not all(math.isfinite(x) for x in values):
            return PlanAuthorization(False, "invalid_action")
        if not any(abs(x) > 1e-8 for x in values):
            return PlanAuthorization(False, "zero_action")
        self.last_consumed_sequence = sequence
        return PlanAuthorization(True, "fresh_h1_consumed")
