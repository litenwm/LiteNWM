#!/usr/bin/env python3
"""Probe the Go2 Sport lease using StopMove only; never issue Move."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("/tmp/go2_nav_stop"),
    )
    parser.add_argument(
        "--arm-file",
        type=Path,
        default=Path("/run/go2-safety/ARMED"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/v70_sport_lease_preflight.json"),
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    stop_file = args.stop_file.expanduser().resolve()
    arm_file = args.arm_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not stop_file.is_file():
        raise RuntimeError("lease preflight requires stop file present")
    if arm_file.exists():
        raise RuntimeError("lease preflight requires ARM file absent")

    report: dict[str, Any] = {
        "schema": "v70_sport_lease_preflight_v1",
        "interface": args.interface,
        "stop_present": True,
        "arm_present": False,
        "move_calls": 0,
        "lease_result": None,
        "stop_move_result": None,
        "accepted": False,
        "error": None,
    }
    started = time.monotonic()
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        ChannelFactoryInitialize(0, args.interface)
        client = SportClient(enableLease=True)
        client.SetTimeout(3.0)
        client.Init()
        lease_result = client.WaitLeaseApplied()
        report["lease_result"] = lease_result
        if lease_result not in (None, 0):
            raise RuntimeError(
                f"WaitLeaseApplied failed: code={lease_result}"
            )
        stop_result = client.StopMove()
        report["stop_move_result"] = stop_result
        if stop_result not in (None, 0):
            raise RuntimeError(
                f"StopMove lease probe failed: code={stop_result}"
            )
        report["accepted"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}:{exc}"
    finally:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch(exist_ok=True)
        report["elapsed_s"] = time.monotonic() - started
        report["stop_present_after"] = stop_file.is_file()
        report["arm_present_after"] = arm_file.exists()
        atomic_write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
