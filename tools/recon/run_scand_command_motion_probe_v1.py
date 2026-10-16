#!/usr/bin/env python3
"""Test whether logged SCAND commands predict achieved motion across routes.

This is a deliberately small system-identification probe.  It never loads
navigation outcomes, CEM candidates, or sealed validation data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


CMD_TOPIC = "/navigation/cmd_vel"
ODOM_TOPIC = "/odom"
LAG_GRID_SECONDS = tuple(float(value) for value in np.arange(0.0, 0.501, 0.05))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bag", type=Path, action="append", required=True)
    parser.add_argument("--held-bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    return parser.parse_args()


def wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


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
    cmd_rows: list[tuple[int, float, float, float]] = []
    odom_rows: list[tuple[int, float, float, float, float, float, float]] = []
    with AnyReader([path], default_typestore=typestore) as reader:
        available = {connection.topic for connection in reader.connections}
        missing = {CMD_TOPIC, ODOM_TOPIC} - available
        if missing:
            raise ValueError(f"{path.name} is missing required topics: {sorted(missing)}")
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {CMD_TOPIC, ODOM_TOPIC}
        ]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == CMD_TOPIC:
                cmd_rows.append(
                    (
                        int(timestamp),
                        float(message.linear.x),
                        float(message.linear.y),
                        float(message.angular.z),
                    )
                )
            else:
                pose = message.pose.pose
                twist = message.twist.twist
                odom_rows.append(
                    (
                        int(timestamp),
                        float(pose.position.x),
                        float(pose.position.y),
                        quaternion_yaw(pose.orientation),
                        float(twist.linear.x),
                        float(twist.linear.y),
                        float(twist.angular.z),
                    )
                )
    if len(cmd_rows) < 2 or len(odom_rows) < 3:
        raise ValueError(f"{path.name} has too few command/odometry messages")
    cmd = np.asarray(cmd_rows, dtype=np.float64)
    odom = np.asarray(odom_rows, dtype=np.float64)
    origin_ns = min(cmd[0, 0], odom[0, 0])
    cmd[:, 0] = (cmd[:, 0] - origin_ns) * 1.0e-9
    odom[:, 0] = (odom[:, 0] - origin_ns) * 1.0e-9
    return {"cmd": cmd, "odom": odom}


def command_at(cmd: np.ndarray, query_seconds: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(cmd[:, 0], query_seconds, side="right") - 1
    indices = np.clip(indices, 0, len(cmd) - 1)
    return cmd[indices, 1:4]


def build_transitions(raw: dict[str, np.ndarray], lag_seconds: float) -> dict[str, np.ndarray]:
    cmd = raw["cmd"]
    odom = raw["odom"]
    dt = odom[1:, 0] - odom[:-1, 0]
    valid = (dt >= 0.05) & (dt <= 0.75)
    previous = odom[:-1][valid]
    present = odom[1:][valid]
    dt = dt[valid]

    delta_global = present[:, 1:3] - previous[:, 1:3]
    yaw = previous[:, 3]
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    target = np.column_stack(
        (
            cosine * delta_global[:, 0] + sine * delta_global[:, 1],
            -sine * delta_global[:, 0] + cosine * delta_global[:, 1],
            wrap_to_pi(present[:, 3] - previous[:, 3]),
        )
    )
    query = 0.5 * (previous[:, 0] + present[:, 0]) - lag_seconds
    command = command_at(cmd, query)
    state_velocity = previous[:, 4:7]
    previous_action = np.vstack((np.zeros((1, 3)), target[:-1]))

    state_features = np.column_stack(
        (
            dt,
            state_velocity * dt[:, None],
            previous_action,
        )
    )
    command_features = np.column_stack(
        (
            command * dt[:, None],
            (command - state_velocity) * dt[:, None],
            command * np.square(dt)[:, None],
        )
    )
    return {
        "target": target.astype(np.float64),
        "state": state_features.astype(np.float64),
        "full": np.column_stack((state_features, command_features)).astype(np.float64),
        "command": command.astype(np.float64),
        "dt": dt.astype(np.float64),
    }


class Ridge:
    def __init__(self, penalty: float) -> None:
        self.penalty = penalty
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.x_mean = x.mean(axis=0)
        self.x_std = np.maximum(x.std(axis=0), 1.0e-8)
        self.y_mean = y.mean(axis=0)
        self.y_std = np.maximum(y.std(axis=0), 1.0e-8)
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        design = np.column_stack((np.ones(len(x_norm)), x_norm))
        regularizer = np.eye(design.shape[1]) * self.penalty
        regularizer[0, 0] = 0.0
        self.weights = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ y_norm,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.x_mean, self.x_std, self.y_mean, self.y_std, self.weights)):
            raise RuntimeError("model is not fitted")
        x_norm = (x - self.x_mean) / self.x_std
        design = np.column_stack((np.ones(len(x_norm)), x_norm))
        return (design @ self.weights) * self.y_std + self.y_mean


def concatenate(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([row[key] for row in rows], axis=0)


def metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    train_mean: np.ndarray,
    train_variance: np.ndarray,
) -> dict[str, Any]:
    squared_error = np.square(target - prediction)
    mse = squared_error.mean(axis=0)
    nmse_per_dim = mse / np.maximum(train_variance, 1.0e-10)
    centered_energy = np.square(target - train_mean).sum(axis=0)
    error_energy = squared_error.sum(axis=0)
    r2_per_dim = 1.0 - error_energy / np.maximum(centered_energy, 1.0e-10)
    return {
        "mse_per_dim": mse.tolist(),
        "nmse_per_dim": nmse_per_dim.tolist(),
        "mean_nmse": float(nmse_per_dim.mean()),
        "r2_per_dim": r2_per_dim.tolist(),
        "mean_r2": float(r2_per_dim.mean()),
    }


def fit_and_score(
    train_rows: list[dict[str, np.ndarray]],
    evaluation: dict[str, np.ndarray],
    feature_key: str,
    penalty: float,
) -> tuple[Ridge, dict[str, Any]]:
    train_x = concatenate(train_rows, feature_key)
    train_y = concatenate(train_rows, "target")
    model = Ridge(penalty).fit(train_x, train_y)
    prediction = model.predict(evaluation[feature_key])
    return model, metrics(
        evaluation["target"],
        prediction,
        train_y.mean(axis=0),
        train_y.var(axis=0),
    )


def select_lag(
    raw_train: list[dict[str, np.ndarray]],
    penalty: float,
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for lag in LAG_GRID_SECONDS:
        route_scores = []
        for held_index in range(len(raw_train)):
            train_rows = [
                build_transitions(raw, lag)
                for index, raw in enumerate(raw_train)
                if index != held_index
            ]
            held = build_transitions(raw_train[held_index], lag)
            _, score = fit_and_score(train_rows, held, "full", penalty)
            route_scores.append(score["mean_nmse"])
        rows.append(
            {
                "lag_seconds": lag,
                "mean_loro_nmse": float(np.mean(route_scores)),
            }
        )
    selected = min(rows, key=lambda row: (row["mean_loro_nmse"], row["lag_seconds"]))
    return float(selected["lag_seconds"]), rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if len(args.train_bag) < 2:
        raise ValueError("at least two train bags are required for train-only lag selection")
    raw_train = [read_bag(path) for path in args.train_bag]
    raw_held = read_bag(args.held_bag)
    selected_lag, lag_selection = select_lag(raw_train, args.ridge)
    train_rows = [build_transitions(raw, selected_lag) for raw in raw_train]
    held = build_transitions(raw_held, selected_lag)

    train_y = concatenate(train_rows, "target")
    train_mean = train_y.mean(axis=0)
    train_variance = train_y.var(axis=0)
    _, state_score = fit_and_score(train_rows, held, "state", args.ridge)
    _, full_score = fit_and_score(train_rows, held, "full", args.ridge)

    nominal_prediction = held["command"] * held["dt"][:, None]
    zero_prediction = np.zeros_like(held["target"])
    nominal_score = metrics(held["target"], nominal_prediction, train_mean, train_variance)
    zero_score = metrics(held["target"], zero_prediction, train_mean, train_variance)
    command_gain = (
        state_score["mean_nmse"] - full_score["mean_nmse"]
    ) / max(state_score["mean_nmse"], 1.0e-10)

    result = {
        "status": (
            "GO_COMMAND_TO_MOTION"
            if command_gain >= 0.05 and full_score["mean_nmse"] < zero_score["mean_nmse"]
            else "NO_GO_COMMAND_TO_MOTION"
        ),
        "interpretation": (
            "logged command contributes held-route motion information beyond current motion state"
            if command_gain >= 0.05
            else "logged command adds too little held-route motion information for the proposed factorization"
        ),
        "protocol": {
            "train_bags": [str(path) for path in args.train_bag],
            "held_bag": str(args.held_bag),
            "lag_selected_on_train_routes_only": True,
            "selected_lag_seconds": selected_lag,
            "ridge": args.ridge,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "sample_counts": {
            "train_transitions": int(len(train_y)),
            "held_transitions": int(len(held["target"])),
        },
        "lag_selection": lag_selection,
        "held_route": {
            "zero_motion": zero_score,
            "nominal_command_integration": nominal_score,
            "state_only_ridge": state_score,
            "state_plus_command_ridge": full_score,
            "relative_nmse_gain_from_command": float(command_gain),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
