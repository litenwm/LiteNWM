#!/usr/bin/env python3
"""Identify SCAND joystick-to-motion dynamics with strict held-route testing."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scand_command_motion_probe_v1 import Ridge, metrics, wrap_to_pi  # noqa: E402


JOY_TOPIC = "/joystick"
ODOM_TOPIC = "/odom"
LAG_GRID_SECONDS = tuple(float(value) for value in np.arange(0.0, 0.501, 0.05))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bag", type=Path, action="append", required=True)
    parser.add_argument("--held-bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    return parser.parse_args()


def quaternion_yaw(orientation: Any) -> float:
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_bag(path: Path) -> dict[str, np.ndarray]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    joy_rows: list[tuple[float, ...]] = []
    odom_rows: list[tuple[float, ...]] = []
    with AnyReader([path], default_typestore=typestore) as reader:
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {JOY_TOPIC, ODOM_TOPIC}
        ]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == JOY_TOPIC:
                joy_rows.append((float(timestamp), *map(float, message.axes)))
            else:
                pose = message.pose.pose
                twist = message.twist.twist
                odom_rows.append(
                    (
                        float(timestamp),
                        float(pose.position.x),
                        float(pose.position.y),
                        quaternion_yaw(pose.orientation),
                        float(twist.linear.x),
                        float(twist.linear.y),
                        float(twist.angular.z),
                    )
                )
    joy = np.asarray(joy_rows, dtype=np.float64)
    odom = np.asarray(odom_rows, dtype=np.float64)
    origin_ns = min(joy[0, 0], odom[0, 0])
    joy[:, 0] = (joy[:, 0] - origin_ns) * 1.0e-9
    odom[:, 0] = (odom[:, 0] - origin_ns) * 1.0e-9
    return {"joy": joy, "odom": odom}


def held_value_at(stream: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(stream[:, 0], query, side="right") - 1
    indices = np.clip(indices, 0, len(stream) - 1)
    return stream[indices, 1:]


def transitions(raw: dict[str, np.ndarray], lag: float) -> dict[str, np.ndarray]:
    odom = raw["odom"]
    dt = odom[1:, 0] - odom[:-1, 0]
    valid = (dt >= 0.05) & (dt <= 0.75)
    previous = odom[:-1][valid]
    present = odom[1:][valid]
    dt = dt[valid]
    delta_global = present[:, 1:3] - previous[:, 1:3]
    cosine = np.cos(previous[:, 3])
    sine = np.sin(previous[:, 3])
    target = np.column_stack(
        (
            cosine * delta_global[:, 0] + sine * delta_global[:, 1],
            -sine * delta_global[:, 0] + cosine * delta_global[:, 1],
            wrap_to_pi(present[:, 3] - previous[:, 3]),
        )
    )
    query = 0.5 * (previous[:, 0] + present[:, 0]) - lag
    joystick = held_value_at(raw["joy"], query)
    velocity = previous[:, 4:7]
    previous_action = np.vstack((np.zeros((1, 3)), target[:-1]))
    state = np.column_stack((dt, velocity * dt[:, None], previous_action))
    full = np.column_stack(
        (
            state,
            joystick,
            joystick * dt[:, None],
            joystick * velocity[:, [0]],
            joystick * velocity[:, [2]],
        )
    )
    return {"target": target, "state": state, "full": full, "joystick": joystick}


def fit_score(
    train: list[dict[str, np.ndarray]],
    held: dict[str, np.ndarray],
    key: str,
    ridge: float,
) -> dict[str, Any]:
    x = np.concatenate([row[key] for row in train])
    y = np.concatenate([row["target"] for row in train])
    prediction = Ridge(ridge).fit(x, y).predict(held[key])
    return metrics(held["target"], prediction, y.mean(axis=0), y.var(axis=0))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if len(args.train_bag) < 2:
        raise ValueError("two train routes are required")
    raw_train = [read_bag(path) for path in args.train_bag]
    raw_held = read_bag(args.held_bag)

    lag_rows = []
    for lag in LAG_GRID_SECONDS:
        scores = []
        for held_index in range(len(raw_train)):
            train = [
                transitions(raw, lag)
                for index, raw in enumerate(raw_train)
                if index != held_index
            ]
            held = transitions(raw_train[held_index], lag)
            scores.append(fit_score(train, held, "full", args.ridge)["mean_nmse"])
        lag_rows.append({"lag_seconds": lag, "mean_loro_nmse": float(np.mean(scores))})
    selected_lag = min(
        lag_rows, key=lambda row: (row["mean_loro_nmse"], row["lag_seconds"])
    )["lag_seconds"]

    train = [transitions(raw, selected_lag) for raw in raw_train]
    held = transitions(raw_held, selected_lag)
    state_score = fit_score(train, held, "state", args.ridge)
    full_score = fit_score(train, held, "full", args.ridge)
    gain = (state_score["mean_nmse"] - full_score["mean_nmse"]) / max(
        state_score["mean_nmse"], 1.0e-10
    )

    train_joy = np.concatenate([row["joystick"] for row in train])
    held_joy = held["joystick"]
    result = {
        "status": "GO_JOYSTICK_TO_MOTION" if gain >= 0.05 else "NO_GO_JOYSTICK_TO_MOTION",
        "interpretation": (
            "joystick adds held-route motion information beyond current motion state"
            if gain >= 0.05
            else "joystick does not add stable held-route motion information beyond current motion state"
        ),
        "protocol": {
            "train_bags": [str(path) for path in args.train_bag],
            "held_bag": str(args.held_bag),
            "selected_lag_seconds": selected_lag,
            "lag_selected_on_train_routes_only": True,
            "all_joystick_axes_used_without_held_selection": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "sample_counts": {
            "train_transitions": int(sum(len(row["target"]) for row in train)),
            "held_transitions": int(len(held["target"])),
        },
        "joystick": {
            "dimensions": int(train_joy.shape[1]),
            "train_std": train_joy.std(axis=0).tolist(),
            "held_std": held_joy.std(axis=0).tolist(),
        },
        "lag_selection": lag_rows,
        "held_route": {
            "state_only_ridge": state_score,
            "state_plus_joystick_ridge": full_score,
            "relative_nmse_gain_from_joystick": float(gain),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
