#!/usr/bin/env python3
"""Issue exactly one Go2 stand-up transition, then hold StopMove.

This is deliberately separate from the subscription-only preflight. It
refuses to start unless the navigation stop file exists and ARM is absent.
It never issues a translational or rotational Move command.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--settle-seconds", type=float, default=4.0)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1.0 <= args.settle_seconds <= 10.0:
        raise ValueError("--settle-seconds must be in [1, 10]")
    stop_file = args.stop_file.expanduser().resolve()
    arm_file = args.arm_file.expanduser().resolve()
    if not stop_file.is_file():
        raise RuntimeError("stand-up requires navigation stop file present")
    if arm_file.exists():
        raise RuntimeError("stand-up requires ARM file absent")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    client = None
    stand_code = None
    stop_code = None
    started = time.monotonic()
    try:
        ChannelFactoryInitialize(0, args.interface)
        client = SportClient(enableLease=True)
        client.SetTimeout(10.0)
        client.Init()
        client.WaitLeaseApplied()
        stand_code = int(client.StandUp())
        if stand_code != 0:
            raise RuntimeError(f"StandUp failed: code={stand_code}")
        time.sleep(args.settle_seconds)
        stop_code = int(client.StopMove())
        if stop_code != 0:
            raise RuntimeError(f"StopMove failed: code={stop_code}")
    finally:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch(exist_ok=True)
        if client is not None and stop_code is None:
            try:
                stop_code = int(client.StopMove())
            except Exception:
                pass

    print(json.dumps({
        "schema": "unitree_stand_once_v1",
        "stand_code": stand_code,
        "stop_move_code": stop_code,
        "elapsed_s": time.monotonic() - started,
        "stop_present": stop_file.is_file(),
        "arm_present": arm_file.exists(),
        "move_commands_issued": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
