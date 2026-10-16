#!/usr/bin/env python3
"""Frozen external-bag validation of the rotation-transport LTWM.

Three previously unused Spot bags are fixed in source before execution.  The
baseline and transport checkpoints remain frozen; this runner only extracts
one bounded forward segment per bag, encodes the existing legacy320
multi-layer dual representation, and compares label-free real-future latent
prediction plus action retrieval.  It never loads navigation outcomes or
sealed CEM validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

import run_raw_resolution_paired_pilot_v1 as raw_pilot
import run_rotation_warp_ltwm_paired_pilot_v1 as common
from process_scand_bags_rosbags import (
    IMAGE_TOPICS,
    ODOM_TOPICS,
    choose_topic,
    quaternion_yaw,
)
from recon_latents.actions import derive_continuous_actions
from recon_latents.models_highres_dual_rotation_transport_v1 import (
    RotationTransportHighResMultiLayerDualLatentWM,
)
from recon_latents.models_highres_dual_rotation_warp_v1 import (
    RotationWarpHighResDualLatentWMConfig,
)
from recon_latents.models_highres_dual_v1 import (
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = (
    ROOT / "outputs/highres_dual_ltwm_v1/raw_resolution_paired_pilot_v1"
)
DEFAULT_TRANSPORT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_ltwm_paired_pilot_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/highres_dual_ltwm_v1/"
    "rotation_transport_external_bag_validation_v1"
)


@dataclass(frozen=True)
class ExternalBag:
    trajectory_id: str
    path: Path


# Fixed before any image/latent extraction for this validation.
EXTERNAL_BAGS = (
    ExternalBag(
        "external_A_Spot_Bass_Garage_Fri_Nov_26_134",
        Path("/mnt/d/VEDIOS/A_Spot_Bass_Garage_Fri_Nov_26_134.bag"),
    ),
    ExternalBag(
        "external_A_Spot_EERC_Ahg_Thu_Nov_11_80",
        Path("/mnt/d/VEDIOS/A_Spot_EERC_Ahg_Thu_Nov_11_80.bag"),
    ),
    ExternalBag(
        "external_A_Spot_Fountain_Library_Tue_Nov_16_120",
        Path("/mnt/d/VEDIOS/A_Spot_Fountain_Library_Tue_Nov_16_120.bag"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--transport", type=Path, default=DEFAULT_TRANSPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=float, default=4.0)
    parser.add_argument("--maximum-frames-per-route", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_delta(
    previous: tuple[Image.Image, np.ndarray, float, int],
    present: tuple[Image.Image, np.ndarray, float, int],
) -> float:
    delta = present[1] - previous[1]
    return float(
        delta[0] * math.cos(previous[2])
        + delta[1] * math.sin(previous[2])
    )


def read_bounded_forward_segment(
    bag_path: Path,
    *,
    sample_rate: float,
    maximum_frames: int,
) -> tuple[list[tuple[Image.Image, np.ndarray, float, int]], dict[str, Any]]:
    """Stream one deterministic early forward segment with bounded memory."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        image_topic = choose_topic(reader.connections, IMAGE_TOPICS)
        odom_topic = choose_topic(reader.connections, ODOM_TOPICS)
        selected = [
            connection
            for connection in reader.connections
            if connection.topic in {image_topic, odom_topic}
        ]
        latest_payload: bytes | None = None
        latest_position: np.ndarray | None = None
        latest_yaw: float | None = None
        next_sample_ns: int | None = None
        previous = None
        current = []
        best = []
        sampled = 0
        completed_segments = 0
        source_size = None
        for connection, timestamp, rawdata in reader.messages(
            connections=selected
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == image_topic:
                latest_payload = bytes(message.data)
            else:
                position = message.pose.pose.position
                latest_position = np.asarray(
                    (float(position.x), float(position.y)),
                    dtype=np.float32,
                )
                latest_yaw = quaternion_yaw(message.pose.pose.orientation)
            if next_sample_ns is None:
                next_sample_ns = int(timestamp)
            if (
                int(timestamp) < next_sample_ns
                or latest_payload is None
                or latest_position is None
                or latest_yaw is None
            ):
                continue
            with Image.open(io.BytesIO(latest_payload)) as decoded:
                raw_image = decoded.convert("RGB").copy()
            if source_size is None:
                source_size = [raw_image.width, raw_image.height]
            legacy, _ = raw_pilot.paired_views(raw_image)
            present = (
                legacy,
                latest_position.copy(),
                float(latest_yaw),
                int(timestamp),
            )
            sampled += 1
            next_sample_ns = int(timestamp + round(1.0e9 / sample_rate))
            if previous is not None:
                if forward_delta(previous, present) >= 1.0e-5:
                    if not current:
                        current.append(previous)
                    current.append(present)
                    if len(current) >= maximum_frames:
                        return current[:maximum_frames], {
                            "image_topic": image_topic,
                            "odom_topic": odom_topic,
                            "source_resolution": source_size,
                            "sampled_until_stop": sampled,
                            "selection": (
                                "first forward segment reaching fixed cap"
                            ),
                            "completed_forward_segments_before_selection": (
                                completed_segments
                            ),
                        }
                elif current:
                    completed_segments += 1
                    if len(current) > len(best):
                        best = current
                    current = []
            previous = present
        if len(current) > len(best):
            best = current
        common.require(
            len(best) >= 8,
            f"no usable forward segment in {bag_path}",
        )
        return best[:maximum_frames], {
            "image_topic": image_topic,
            "odom_topic": odom_topic,
            "source_resolution": source_size,
            "sampled_until_stop": sampled,
            "selection": (
                "longest completed segment because no segment reached cap"
            ),
            "completed_forward_segments_before_selection": (
                completed_segments
            ),
        }


def encode_external_bags(
    output: Path,
    *,
    sample_rate: float,
    maximum_frames: int,
    batch_size: int,
    device: torch.device,
    amp: str,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    backbone, representation, lineage = raw_pilot.load_representation(device)
    paths = {}
    rows = []
    for spec in EXTERNAL_BAGS:
        started = time.perf_counter()
        segment, audit = read_bounded_forward_segment(
            spec.path,
            sample_rate=sample_rate,
            maximum_frames=maximum_frames,
        )
        images = [value[0] for value in segment]
        positions = np.stack([value[1] for value in segment]).astype(
            np.float32
        )
        yaw = np.asarray([value[2] for value in segment], dtype=np.float32)
        timestamps = np.asarray(
            [value[3] for value in segment],
            dtype=np.int64,
        )
        actions = derive_continuous_actions(positions, yaw)
        semantic, geometry = raw_pilot.encode_view(
            images,
            backbone,
            representation,
            batch_size=batch_size,
            device=device,
            amp=amp,
        )
        shard = output / "latents" / f"{spec.trajectory_id}.npz"
        raw_pilot.write_npz_atomic(
            shard,
            semantic_latents=semantic,
            geometry_latents=geometry,
            actions_continuous=actions,
            positions=positions,
            yaw=yaw,
            timestamps_ns=timestamps,
            trajectory_id=np.asarray(spec.trajectory_id),
            split=np.asarray("external"),
        )
        paths[spec.trajectory_id] = shard
        rows.append(
            {
                "trajectory_id": spec.trajectory_id,
                "bag": str(spec.path),
                "bag_size_bytes": spec.path.stat().st_size,
                "frames": len(segment),
                "windows": len(segment) - 7,
                "latent_shard": str(shard),
                "latent_shard_sha256": sha256_file(shard),
                "runtime_seconds": time.perf_counter() - started,
                **audit,
            }
        )
        print(
            common.canonical_json(
                {
                    "stage": "encoded_external_bag",
                    "trajectory": spec.trajectory_id,
                    "frames": len(segment),
                    "windows": len(segment) - 7,
                    "runtime_seconds": rows[-1]["runtime_seconds"],
                }
            ),
            flush=True,
        )
    del backbone, representation
    torch.cuda.empty_cache()
    return paths, rows, lineage


def metric_changes(
    baseline: dict[str, Any],
    transport: dict[str, Any],
) -> dict[str, float]:
    return {
        "semantic_h4_nmse_change": (
            transport["semantic_h4_nmse"] - baseline["semantic_h4_nmse"]
        ),
        "geometry_h4_nmse_change": (
            transport["geometry_h4_nmse"] - baseline["geometry_h4_nmse"]
        ),
        "dual_h4_nmse_change": (
            transport["dual_h4_nmse"] - baseline["dual_h4_nmse"]
        ),
        "dual_h4_nmse_relative_change": (
            transport["dual_h4_nmse"] / baseline["dual_h4_nmse"] - 1.0
        ),
        "negative_gap_fraction_change": (
            transport["negative_gap_fraction_of_true_error"]
            - baseline["negative_gap_fraction_of_true_error"]
        ),
        "pairwise_action_win_rate_change": (
            transport["pairwise_action_win_rate"]
            - baseline["pairwise_action_win_rate"]
        ),
        "retrieval_top1_change": (
            transport["retrieval_top1"] - baseline["retrieval_top1"]
        ),
        "mean_rank_change": (
            transport["mean_rank"] - baseline["mean_rank"]
        ),
    }


def main() -> None:
    args = parse_args()
    parent = args.parent.resolve()
    transport_dir = args.transport.resolve()
    output = args.output.resolve()
    result_path = output / "result.json"
    common.require(
        not result_path.exists(),
        f"refusing to overwrite {result_path}",
    )
    common.require(torch.cuda.is_available(), "CUDA unavailable")
    common.require(
        args.maximum_frames_per_route >= 16,
        "maximum frames too small",
    )
    for spec in EXTERNAL_BAGS:
        common.require(spec.path.is_file(), f"missing bag: {spec.path}")
    parent_result_path = parent / "result.json"
    baseline_checkpoint_path = parent / "legacy320_pilot_final.pt"
    transport_result_path = transport_dir / "result.json"
    transport_checkpoint_path = (
        transport_dir / "rotation_transport_pilot_final.pt"
    )
    for path in (
        parent_result_path,
        baseline_checkpoint_path,
        transport_result_path,
        transport_checkpoint_path,
    ):
        common.require(path.is_file(), f"missing frozen input: {path}")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    transport_result = json.loads(
        transport_result_path.read_text(encoding="utf-8")
    )
    common.require(
        transport_result["status"] == "GO_ROTATION_TRANSPORT_LTWM",
        "transport development result did not pass",
    )
    common.require(
        sha256_file(baseline_checkpoint_path)
        == parent_result["checkpoints"]["legacy320"]["sha256"],
        "baseline checkpoint hash mismatch",
    )
    common.require(
        sha256_file(transport_checkpoint_path)
        == transport_result["checkpoint"]["sha256"],
        "transport checkpoint hash mismatch",
    )

    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.perf_counter()
    paths, bag_rows, representation_lineage = encode_external_bags(
        output,
        sample_rate=args.sample_rate,
        maximum_frames=args.maximum_frames_per_route,
        batch_size=args.encode_batch_size,
        device=device,
        amp=args.amp,
    )
    split = {trajectory_id: "external" for trajectory_id in paths}
    pooled_refs, refs_by_trajectory = raw_pilot.window_refs(
        paths,
        split,
        "external",
    )
    common.require(bool(pooled_refs), "external windows are empty")
    cache = raw_pilot.ShardCache(paths)

    baseline_payload = torch.load(
        baseline_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    baseline_model = HighResMultiLayerDualLatentWM(
        HighResDualLatentWMConfig(**baseline_payload["config"])
    ).to(device)
    baseline_model.load_state_dict(baseline_payload["model"], strict=True)
    transport_payload = torch.load(
        transport_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    transport_model = RotationTransportHighResMultiLayerDualLatentWM(
        RotationWarpHighResDualLatentWMConfig(
            **transport_payload["config"]
        )
    ).to(device)
    transport_model.load_state_dict(
        transport_payload["model"],
        strict=True,
    )

    pooled_baseline = raw_pilot.evaluate_model(
        baseline_model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    pooled_transport = raw_pilot.evaluate_model(
        transport_model,
        cache,
        pooled_refs,
        batch_size=args.eval_batch_size,
        device=device,
        amp=args.amp,
    )
    per_route = {}
    routes_dual_improved = 0
    routes_pairwise_improved = 0
    for trajectory_id in sorted(refs_by_trajectory):
        refs = [
            pooled_refs[index]
            for index in refs_by_trajectory[trajectory_id]
        ]
        baseline = raw_pilot.evaluate_model(
            baseline_model,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        transport = raw_pilot.evaluate_model(
            transport_model,
            cache,
            refs,
            batch_size=args.eval_batch_size,
            device=device,
            amp=args.amp,
        )
        changes = metric_changes(baseline, transport)
        if changes["dual_h4_nmse_change"] < 0.0:
            routes_dual_improved += 1
        if changes["pairwise_action_win_rate_change"] > 0.0:
            routes_pairwise_improved += 1
        per_route[trajectory_id] = {
            "baseline": baseline,
            "transport": transport,
            "changes": changes,
        }
    del baseline_model, transport_model
    torch.cuda.empty_cache()

    pooled_changes = metric_changes(
        pooled_baseline,
        pooled_transport,
    )
    checks = {
        "pooled_future_dual_h4_improved": (
            pooled_changes["dual_h4_nmse_change"] < 0.0
        ),
        "pooled_future_geometry_h4_not_worse": (
            pooled_changes["geometry_h4_nmse_change"] <= 0.0
        ),
        "pooled_negative_gap_fraction_improved": (
            pooled_changes["negative_gap_fraction_change"] > 0.0
        ),
        "pooled_pairwise_action_win_rate_improved": (
            pooled_changes["pairwise_action_win_rate_change"] > 0.0
        ),
        "pooled_mean_rank_improved": (
            pooled_changes["mean_rank_change"] < 0.0
        ),
        "dual_h4_improved_on_at_least_two_routes": (
            routes_dual_improved >= 2
        ),
        "pairwise_improved_on_at_least_two_routes": (
            routes_pairwise_improved >= 2
        ),
    }
    go = all(checks.values())
    result = {
        "status": (
            "GO_EXTERNAL_ROTATION_TRANSPORT"
            if go
            else "NO_GO_EXTERNAL_ROTATION_TRANSPORT"
        ),
        "decision": (
            "retain rotation transport as the leading LTWM development branch"
            if go
            else "do not promote rotation transport beyond development"
        ),
        "protocol": {
            "external_bags_fixed_in_source_before_extraction": True,
            "sample_rate_hz": args.sample_rate,
            "maximum_frames_per_route": args.maximum_frames_per_route,
            "preprocessing": (
                "raw external JPEG -> center crop 4:3 -> 320x240 JPEG95 "
                "-> 448x336"
            ),
            "model_training_or_tuning": False,
            "threshold_or_gate_tuning": False,
            "dino_frozen": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "lineage": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "baseline_checkpoint": {
                "path": str(baseline_checkpoint_path),
                "sha256": sha256_file(baseline_checkpoint_path),
            },
            "transport_checkpoint": {
                "path": str(transport_checkpoint_path),
                "sha256": sha256_file(transport_checkpoint_path),
            },
            "transport_result": {
                "path": str(transport_result_path),
                "sha256": sha256_file(transport_result_path),
            },
            "representation": representation_lineage,
            "bags": bag_rows,
        },
        "pooled": {
            "baseline": pooled_baseline,
            "transport": pooled_transport,
            "changes": pooled_changes,
        },
        "per_route": per_route,
        "route_consistency": {
            "routes": len(per_route),
            "dual_h4_improved_routes": routes_dual_improved,
            "pairwise_improved_routes": routes_pairwise_improved,
        },
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    common.write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "pooled": result["pooled"],
                "route_consistency": result["route_consistency"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
