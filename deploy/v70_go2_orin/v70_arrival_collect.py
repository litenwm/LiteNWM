#!/usr/bin/env python3
"""Append one operator-labelled, completed V70 shadow plan to a dataset.

The script reads artifacts only. It creates no ROS node and exposes no robot
motion API. Arrival calibration samples are accepted only when the operator
explicitly confirms that the robot was standing and stationary.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


LABELS = ("far", "approach", "stop", "overshoot", "lookalike")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--confirm-standing-stationary", action="store_true")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid event at line {line_number}")
        events.append(value)
    return events


def require_finite_number(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_standing_stationary:
        raise RuntimeError(
            "refusing sample: pass --confirm-standing-stationary only "
            "after visually confirming both conditions"
        )
    run_dir = args.run_dir.expanduser().resolve()
    status_path = run_dir / "status.json"
    event_path = run_dir / "live_nav.jsonl"
    if not status_path.is_file() or not event_path.is_file():
        raise FileNotFoundError("run must contain status.json and live_nav.jsonl")
    status = read_json(status_path)
    if status.get("mode") != "shadow":
        raise RuntimeError("arrival samples must come from shadow mode")
    if status.get("fault") is not None:
        raise RuntimeError(f"shadow run faulted: {status.get('fault')}")
    if int(status.get("exit_code", -1)) != 0:
        raise RuntimeError("shadow run did not exit successfully")
    if not bool(status.get("stop_present")):
        raise RuntimeError("status does not prove stop was present")
    if bool(status.get("arm_present")):
        raise RuntimeError("status shows ARM present")
    plan = status.get("last_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("status has no completed plan")
    sequence = int(plan["sequence"])
    temporal = require_finite_number(plan, "nomad_temporal_distance")
    cosine = require_finite_number(
        plan,
        "dinov3_current_goal_cosine_distance",
    )
    goal_sha256 = str(plan["goal_sha256"])
    if len(goal_sha256) != 64:
        raise ValueError("invalid goal SHA256")

    events = read_events(event_path)
    requested = [
        event
        for event in events
        if event.get("event") == "plan_requested"
        and int(event.get("sequence", -1)) == sequence
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "plan_complete"
        and isinstance(event.get("summary"), dict)
        and int(event["summary"].get("sequence", -1)) == sequence
    ]
    if len(requested) != 1 or len(completed) != 1:
        raise RuntimeError("plan event correlation is not one-to-one")
    image_hashes = requested[0].get("image_sha256")
    if (
        not isinstance(image_hashes, list)
        or len(image_hashes) != 4
        or any(len(str(value)) != 64 for value in image_hashes)
    ):
        raise RuntimeError("plan request lacks four valid frame hashes")

    record: dict[str, Any] = {
        "schema": "v70_arrival_sample_v1",
        "collected_unix": time.time(),
        "operator_label": args.label,
        "operator_confirmed_standing_stationary": True,
        "operator_note": str(args.note),
        "run_dir": str(run_dir),
        "status_sha256": sha256_file(status_path),
        "events_sha256": sha256_file(event_path),
        "route_id": str(status.get("route_id", "")),
        "route_goal": str(status.get("route_goal", "")),
        "sequence": sequence,
        "goal_sha256": goal_sha256,
        "observation_sha256": [str(value) for value in image_hashes],
        "nomad_temporal_distance": temporal,
        "dinov3_current_goal_cosine_distance": cosine,
        "selected_first_action": plan.get("selected_first_action"),
        "abstained": bool(plan.get("abstained")),
        "goal_decision": status.get("last_goal_decision"),
        "timings_ms": plan.get("timings_ms"),
    }
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    return record


def append_unique(path: Path, record: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]
        if existing and any(
            item.get("goal_sha256") != record["goal_sha256"]
            for item in existing
        ):
            raise RuntimeError("dataset mixes different goal images")
        key = (
            record["run_dir"],
            record["sequence"],
            tuple(record["observation_sha256"]),
        )
        for item in existing:
            item_key = (
                item.get("run_dir"),
                item.get("sequence"),
                tuple(item.get("observation_sha256", [])),
            )
            if item_key == key:
                raise RuntimeError("duplicate shadow sample")
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    record = build_record(args)
    append_unique(args.dataset, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
