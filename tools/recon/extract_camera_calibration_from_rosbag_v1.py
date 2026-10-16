#!/usr/bin/env python3
"""Extract one physical CameraInfo calibration sidecar from a ROS bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recon_latents.camera_calibration_v1 import (  # noqa: E402
    calibration_from_ros_camera_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--max-messages", type=int, default=32)
    return parser.parse_args()


def calibration_signature(value: dict[str, object]) -> tuple[object, ...]:
    return (
        value["source_width"],
        value["source_height"],
        tuple(value["intrinsic_matrix"]),
        value["distortion_model"],
        tuple(value["distortion_coefficients"]),
        value["frame_id"],
    )


def main() -> None:
    args = parse_args()
    if args.max_messages < 1:
        raise ValueError("max-messages must be positive")
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise ImportError("install the existing rosbags dependency") from error
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader(
        [args.bag.resolve(strict=True)],
        default_typestore=typestore,
    ) as reader:
        candidates = [
            connection
            for connection in reader.connections
            if connection.msgtype.endswith("/CameraInfo")
        ]
        available = sorted(connection.topic for connection in candidates)
        if args.camera_info_topic is None:
            if len(candidates) != 1:
                raise ValueError(
                    "select --camera-info-topic; available CameraInfo topics: "
                    f"{available}"
                )
            selected = candidates[0]
        else:
            matches = [
                connection
                for connection in candidates
                if connection.topic == args.camera_info_topic
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"CameraInfo topic absent or ambiguous; available={available}"
                )
            selected = matches[0]
        rows = []
        for connection, timestamp, rawdata in reader.messages(
            connections=[selected]
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            calibration = calibration_from_ros_camera_info(
                message,
                source=f"rosbag:{args.bag.name}:{selected.topic}",
            )
            rows.append(
                {
                    **calibration.to_dict(),
                    "timestamp_ns": int(timestamp),
                }
            )
            if len(rows) >= args.max_messages:
                break
    if not rows:
        raise ValueError("selected CameraInfo topic contains no messages")
    reference = calibration_signature(rows[0])
    if any(calibration_signature(row) != reference for row in rows[1:]):
        raise ValueError("CameraInfo changes within sampled bag messages")
    output = {
        "schema_version": 1,
        "status": "PASS",
        "bag": str(args.bag),
        "camera_info_topic": selected.topic,
        "sampled_messages": len(rows),
        "calibration_constant_in_sample": True,
        "calibration": {
            key: value
            for key, value in rows[0].items()
            if key != "timestamp_ns"
        },
        "extrinsics_status": (
            "not_extracted; supply verified camera-to-base TF separately"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
