#!/usr/bin/env python3
"""Print label-free ROS bag topic/count/time metadata as canonical JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        rows = [
            {
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "message_count": int(getattr(connection, "msgcount", 0)),
            }
            for connection in reader.connections
        ]
        result = {
            "bag": str(args.bag.resolve()),
            "start_ns": int(reader.start_time),
            "end_ns": int(reader.end_time),
            "duration_seconds": (int(reader.end_time) - int(reader.start_time)) / 1e9,
            "topics": rows,
            "message_values_opened": False,
            "reserve_content_accessed": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
