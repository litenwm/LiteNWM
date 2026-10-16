#!/usr/bin/env python3
"""Audit label-free camera topic compatibility for the frozen SCAND routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


CANONICAL_TOPICS = (
    "/image_raw/compressed",
    "/camera/rgb/image_raw/compressed",
    "/left/image_color/compressed",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    typestore = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    for selected in plan["selected"]:
        filename = selected["official_file_name"] + ".bag"
        path = args.source_dir / filename
        with AnyReader([path], default_typestore=typestore) as reader:
            images = [
                {"topic": c.topic, "message_count": int(getattr(c, "msgcount", 0)), "msgtype": c.msgtype}
                for c in reader.connections
                if c.msgtype.endswith("CompressedImage")
            ]
        names = {row["topic"] for row in images}
        canonical = next((topic for topic in CANONICAL_TOPICS if topic in names), None)
        rows.append({
            "filename": filename,
            "route_group": selected["route_group"],
            "cohort_role": selected["cohort_role"],
            "robot": selected["robot"],
            "canonical_topic": canonical,
            "canonical_available": canonical is not None,
            "compressed_image_topics": images,
        })
    result = {
        "schema": "scand_routefit_camera_topic_audit_v1",
        "status": "complete_label_free",
        "canonical_contract": list(CANONICAL_TOPICS),
        "rows": rows,
        "summary": {
            "routes": len(rows),
            "canonical_available": sum(row["canonical_available"] for row in rows),
            "canonical_missing": sum(not row["canonical_available"] for row in rows),
        },
        "labels_or_outcomes_opened": False,
        "reserve_content_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
