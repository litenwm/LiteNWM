#!/usr/bin/env python3
"""Pure source-timestamp freshness guards for ROS sensor streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def ros_stamp_ns(message: Any) -> int:
    """Return a validated positive ROS header timestamp in nanoseconds."""

    stamp = message.header.stamp
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return 0
    value = sec * 1_000_000_000 + nanosec
    return value if value > 0 else 0


@dataclass
class SourceStampGuard:
    """Accept only strictly advancing, valid source timestamps."""

    last_stamp_ns: int | None = None
    last_progress_time: float | None = None
    accepted_count: int = 0
    duplicate_count: int = 0
    regression_count: int = 0
    invalid_count: int = 0

    def observe(self, stamp_ns: int, now: float) -> bool:
        if stamp_ns <= 0:
            self.invalid_count += 1
            return False
        if self.last_stamp_ns is not None:
            if stamp_ns == self.last_stamp_ns:
                self.duplicate_count += 1
                return False
            if stamp_ns < self.last_stamp_ns:
                self.regression_count += 1
                return False
        self.last_stamp_ns = int(stamp_ns)
        self.last_progress_time = float(now)
        self.accepted_count += 1
        return True

    def fresh(self, now: float, timeout_s: float) -> bool:
        if timeout_s <= 0.0:
            raise ValueError("freshness timeout must be positive")
        return (
            self.last_progress_time is not None
            and 0.0 <= now - self.last_progress_time <= timeout_s
        )

    def report(self) -> dict[str, int | float | None]:
        return asdict(self)
