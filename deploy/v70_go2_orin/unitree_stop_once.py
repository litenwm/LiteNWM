#!/usr/bin/env python3
"""Issue only Unitree StopMove as an emergency process-exit fallback."""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1 or args.attempts > 10:
        raise ValueError("--attempts must be in [1, 10]")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    ChannelFactoryInitialize(0, args.interface)
    client = SportClient(enableLease=True)
    client.SetTimeout(3.0)
    client.Init()
    client.WaitLeaseApplied()

    failures: list[str] = []
    for attempt in range(args.attempts):
        try:
            code = client.StopMove()
        except Exception as exc:
            failures.append(f"attempt_{attempt}:{type(exc).__name__}:{exc}")
        else:
            if code in (None, 0):
                print(f"StopMove accepted on attempt {attempt + 1}")
                return 0
            failures.append(f"attempt_{attempt}:return_code:{code}")
        time.sleep(0.10)
    print("StopMove failed: " + " | ".join(failures))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
