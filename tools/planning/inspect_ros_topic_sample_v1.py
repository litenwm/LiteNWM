#!/usr/bin/env python3
"""Print one ROS topic sample for an explicitly authorized development bag."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    args = parser.parse_args()
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [connection for connection in reader.connections if connection.topic == args.topic]
        if not connections:
            raise RuntimeError(f"missing topic: {args.topic}")
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            print(json.dumps({"topic": args.topic, "msgtype": connection.msgtype, "timestamp_ns": int(timestamp), "message": plain(message)}, sort_keys=True))
            return
    raise RuntimeError("topic has no messages")


if __name__ == "__main__":
    main()
