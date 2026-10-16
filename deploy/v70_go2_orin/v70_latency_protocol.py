#!/usr/bin/env python3
"""Shared timing and reporting helpers for Go2/Orin latency benchmarks."""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import statistics
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "v70_go2_orin_latency_v1"


def synchronize_cuda(device: Any | None = None) -> None:
    """Synchronize CUDA when available without making torch a hard import."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


class StageTimer:
    """Collect non-overlapping wall-clock timings with optional CUDA fences."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.started_ns = time.perf_counter_ns()
        self.timings_ms: dict[str, float] = {}
        self.details: dict[str, Any] = {}

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        cuda: bool = False,
        device: Any | None = None,
    ) -> Iterator[None]:
        if name in self.timings_ms:
            raise ValueError(f"duplicate latency field: {name}")
        if cuda:
            synchronize_cuda(device)
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            if cuda:
                synchronize_cuda(device)
            self.timings_ms[name] = (
                time.perf_counter_ns() - started_ns
            ) / 1.0e6

    def add_ms(self, name: str, value: float) -> None:
        if name in self.timings_ms:
            raise ValueError(f"duplicate latency field: {name}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"invalid latency for {name}: {value}")
        self.timings_ms[name] = numeric

    def payload(self) -> dict[str, Any]:
        total_ms = (time.perf_counter_ns() - self.started_ns) / 1.0e6
        return {
            "schema": SCHEMA,
            "stage": self.stage,
            "clock": "time.perf_counter_ns",
            "cuda_timing": "wall clock fenced by torch.cuda.synchronize",
            "total_stage_wall_ms": total_ms,
            "timings_ms": dict(self.timings_ms),
            "details": dict(self.details),
            "host": host_metadata(),
        }


def host_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }
    try:
        import torch
    except ImportError:
        return result
    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(
            torch.cuda.current_device()
        )
        result["cuda_version"] = torch.version.cuda
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_latency(path: Path, *, expected_stage: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"latency schema mismatch: {path}")
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"latency stage mismatch: expected={expected_stage}, "
            f"actual={payload.get('stage')}"
        )
    return payload


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty latency sample")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("latency samples must be finite and non-negative")

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered),
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


__all__ = [
    "SCHEMA",
    "StageTimer",
    "host_metadata",
    "read_latency",
    "summarize_ms",
    "synchronize_cuda",
    "write_json_atomic",
]
