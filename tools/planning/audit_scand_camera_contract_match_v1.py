#!/usr/bin/env python3
"""Compare raw camera topics with an existing label-free processed frame."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.planning.download_prepare_new_scand_bags_v1 import crop_resize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-per-topic", type=int, default=120)
    args = parser.parse_args()
    with Image.open(args.reference) as image:
        reference = np.asarray(crop_resize(image), dtype=np.float32)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.msgtype.endswith("CompressedImage")]
        by_topic = {c.topic: [] for c in connections}
        selected = {c.topic: c for c in connections}
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            values = by_topic[connection.topic]
            if len(values) >= args.max_per_topic:
                if all(len(items) >= args.max_per_topic for items in by_topic.values()):
                    break
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            with Image.open(io.BytesIO(bytes(message.data))) as image:
                source_resolution = list(image.size)
                candidate = np.asarray(crop_resize(image), dtype=np.float32)
            mse = float(np.mean((candidate - reference) ** 2))
            values.append((mse, int(timestamp), source_resolution))
    rows = []
    for topic, values in by_topic.items():
        values.sort(key=lambda item: item[0])
        best = values[0] if values else (float("inf"), None, None)
        rows.append({
            "topic": topic,
            "samples_compared": len(values),
            "best_mse_rgb_0_255": best[0],
            "best_psnr_db": None if not np.isfinite(best[0]) else float(10 * np.log10((255.0**2) / max(best[0], 1e-12))),
            "best_timestamp_ns": best[1],
            "source_resolution": best[2],
        })
    rows.sort(key=lambda row: row["best_mse_rgb_0_255"])
    print(json.dumps({
        "schema": "scand_camera_contract_match_v1",
        "bag": str(args.bag.resolve()),
        "reference": str(args.reference.resolve()),
        "rows": rows,
        "labels_or_outcomes_opened": False,
        "reserve_content_accessed": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
