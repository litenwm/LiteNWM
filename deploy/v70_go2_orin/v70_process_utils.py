#!/usr/bin/env python3
"""Small process-shutdown primitives used by the V70 board runtime."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessExitReport:
    """Outcome of a bounded child-process reap."""

    was_running: bool
    terminated: bool
    killed: bool
    returncode: int | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reap_process_bounded(
    process: subprocess.Popen[bytes],
    *,
    terminate_timeout_s: float,
    kill_timeout_s: float,
    send_terminate: bool = True,
) -> ProcessExitReport:
    """Terminate and reap ``process`` without an unbounded wait.

    The caller may send SIGTERM before unblocking a pipe/socket and then call
    this function with ``send_terminate=False``.  SIGKILL is always used as
    the final bounded fallback.
    """

    if terminate_timeout_s < 0.0 or kill_timeout_s <= 0.0:
        raise ValueError("process shutdown timeouts are invalid")
    if process.poll() is not None:
        return ProcessExitReport(
            was_running=False,
            terminated=False,
            killed=False,
            returncode=process.returncode,
        )

    terminated = False
    killed = False
    error: str | None = None
    if send_terminate:
        try:
            process.terminate()
            terminated = True
        except ProcessLookupError:
            pass
        except Exception as exc:
            error = f"terminate:{type(exc).__name__}:{exc}"

    try:
        process.wait(timeout=terminate_timeout_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            killed = True
        except ProcessLookupError:
            pass
        except Exception as exc:
            suffix = f"kill:{type(exc).__name__}:{exc}"
            error = suffix if error is None else f"{error};{suffix}"
        try:
            process.wait(timeout=kill_timeout_s)
        except subprocess.TimeoutExpired as exc:
            suffix = f"unreaped_after_kill:{exc.timeout}s"
            error = suffix if error is None else f"{error};{suffix}"

    return ProcessExitReport(
        was_running=True,
        terminated=terminated or not send_terminate,
        killed=killed,
        returncode=process.poll(),
        error=error,
    )
