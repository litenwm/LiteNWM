#!/usr/bin/env python3
"""Inspect command, motion, and sensor streams in one already-selected SCAND bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYWORDS = (
    "joy",
    "cmd",
    "vel",
    "twist",
    "odom",
    "imu",
    "tf",
    "depth",
    "point",
    "camera_info",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        rows = [
            {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "msgcount": int(connection.msgcount),
            }
            for connection in reader.connections
            if any(keyword in connection.topic.lower() for keyword in KEYWORDS)
        ]
    result = {
        "bag": str(args.bag),
        "relevant_connections": sorted(rows, key=lambda row: row["topic"]),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
