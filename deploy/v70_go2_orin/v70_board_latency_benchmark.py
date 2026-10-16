#!/usr/bin/env python3
"""Measure the deployed V70 shadow pipeline on an Orin without moving the Go2."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEPLOY_DIR = Path(__file__).resolve().parent
ROOT = DEPLOY_DIR.parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from v70_latency_protocol import (  # noqa: E402
    host_metadata,
    read_latency,
    summarize_ms,
    write_json_atomic,
)

BENCHMARK_SCHEMA = "v70_go2_orin_board_latency_benchmark_v1"
PRIMARY_METRICS = (
    "dinov3_encode_ms",
    "nomad_candidate_generation_ms",
    "ltwm_future_inference_ms",
    "selector_fixed_pool_planning_ms",
    "cem_planning_ms",
    "selector_and_cem_planning_ms",
    "complete_planning_cycle_wall_ms",
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    latency_path: Path | None = None


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/home/unitree/v70_latency_runs") / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-image", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--interface", default="enP8p1s0")
    parser.add_argument("--frame-interval", type=float, default=0.20)
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--odom-interval", type=float, default=0.20)
    parser.add_argument("--odom-timeout", type=float, default=10.0)
    parser.add_argument("--candidates", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/project",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "runtime/v70_tridomain",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("/tmp/go2_nav_stop"),
    )
    parser.add_argument(
        "--arm-file",
        type=Path,
        default=Path("/run/go2-safety/ARMED"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print one cycle without creating files, reading sensors, or running models.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.candidates != 17:
        raise ValueError("the deployed V70 contract requires exactly 17 NoMaD samples")
    if args.frame_interval < 0.0:
        raise ValueError("--frame-interval must be non-negative")
    if args.odom_interval < 0.0:
        raise ValueError("--odom-interval must be non-negative")
    if args.odom_timeout <= 0.0:
        raise ValueError("--odom-timeout must be positive")
    if not args.goal_image.expanduser().is_file():
        raise FileNotFoundError(args.goal_image)
    if not args.python.expanduser().is_file():
        raise FileNotFoundError(args.python)


def require_zero_motion_interlock(args: argparse.Namespace) -> None:
    if not args.stop_file.is_file():
        raise RuntimeError(
            f"zero-motion stop file is missing: {args.stop_file}; "
            "the latency benchmark refuses to run"
        )
    if args.arm_file.exists():
        raise RuntimeError(
            f"motion arm file must be absent: {args.arm_file}; "
            "the latency benchmark refuses to run"
        )


def build_stages(
    args: argparse.Namespace,
    run_dir: Path,
    run_id: str,
) -> list[Stage]:
    python = str(args.python.expanduser().absolute())
    goal = str(args.goal_image.expanduser().resolve())
    runtime = args.runtime_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()

    frame_dir = run_dir / "frames"
    dino = run_dir / "dino.npz"
    nomad = run_dir / "nomad.npy"
    online = run_dir / "online.npz"
    ltwm = run_dir / "ltwm.npz"
    selector = run_dir / "selector.npz"
    cem = run_dir / "cem.npz"

    dino_latency = run_dir / "dino_latency.json"
    nomad_metadata = run_dir / "nomad_metadata.json"
    nomad_latency = run_dir / "nomad_latency.json"
    online_latency = run_dir / "online_latency.json"
    ltwm_latency = run_dir / "ltwm_latency.json"
    selector_latency = run_dir / "selector_latency.json"
    cem_latency = run_dir / "cem_latency.json"

    return [
        Stage(
            "dinov3",
            [
                python,
                str(DEPLOY_DIR / "v70_dino_live_shadow.py"),
                "--goal-image",
                goal,
                "--interface",
                args.interface,
                "--frame-interval",
                str(args.frame_interval),
                "--frame-output-dir",
                str(frame_dir),
                "--output",
                str(dino),
                "--latency-output",
                str(dino_latency),
            ],
            dino_latency,
        ),
        Stage(
            "nomad",
            [
                python,
                str(DEPLOY_DIR / "nomad_snapshot_shadow.py"),
                "--frame-dir",
                str(frame_dir),
                "--goal-image",
                goal,
                "--candidates",
                str(args.candidates),
                "--seed",
                str(args.seed),
                "--output",
                str(nomad),
                "--metadata-output",
                str(nomad_metadata),
                "--latency-output",
                str(nomad_latency),
            ],
            nomad_latency,
        ),
        Stage(
            "online_features",
            [
                python,
                str(DEPLOY_DIR / "v70_online_feature_shadow.py"),
                "--nomad-candidates",
                str(nomad),
                "--odom-topic",
                args.odom_topic,
                "--interval",
                str(args.odom_interval),
                "--timeout",
                str(args.odom_timeout),
                "--output",
                str(online),
                "--latency-output",
                str(online_latency),
            ],
            online_latency,
        ),
        Stage(
            "stamp_snapshot",
            [
                python,
                str(DEPLOY_DIR / "v70_stamp_snapshot_metadata.py"),
                "--artifact",
                str(dino),
                "--artifact",
                str(online),
                "--frame-dir",
                str(frame_dir),
                "--goal-image",
                goal,
                "--run-id",
                run_id,
            ],
        ),
        Stage(
            "ltwm",
            [
                python,
                str(DEPLOY_DIR / "v70_ltwm_tridomain_v3_shadow.py"),
                "--online-features",
                str(online),
                "--dino-features",
                str(dino),
                "--checkpoint-root",
                str(checkpoint_root),
                "--output",
                str(ltwm),
                "--latency-output",
                str(ltwm_latency),
            ],
            ltwm_latency,
        ),
        Stage(
            "selector",
            [
                python,
                str(DEPLOY_DIR / "v70_armd_tridomain_v3_shadow.py"),
                "--features",
                str(ltwm),
                "--support",
                str(runtime / "selector/support/tridomain_spacing_envelopes.npz"),
                "--v70-result",
                str(runtime / "selector/result.json"),
                "--checkpoint-dir",
                str(runtime / "selector/checkpoints"),
                "--domain",
                "recon",
                "--device",
                args.device,
                "--output",
                str(selector),
                "--latency-output",
                str(selector_latency),
            ],
            selector_latency,
        ),
        Stage(
            "cem",
            [
                python,
                str(DEPLOY_DIR / "v70_cem_tridomain_v6_shadow.py"),
                "--online-features",
                str(online),
                "--dino-features",
                str(dino),
                "--fixed-armd",
                str(selector),
                "--checkpoint-root",
                str(checkpoint_root),
                "--support",
                str(runtime / "selector/support/tridomain_spacing_envelopes.npz"),
                "--v70-result",
                str(runtime / "selector/result.json"),
                "--armd-checkpoint-dir",
                str(runtime / "selector/checkpoints"),
                "--device",
                args.device,
                "--output",
                str(cem),
                "--latency-output",
                str(cem_latency),
            ],
            cem_latency,
        ),
    ]


def run_stage(
    stage: Stage,
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[float, dict[str, Any] | None]:
    require_zero_motion_interlock(args)
    log_path = run_dir / f"{stage.name}.log"
    started_ns = time.perf_counter_ns()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(stage.command) + "\n\n")
        log.flush()
        result = subprocess.run(
            stage.command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    wall_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
    if result.returncode != 0:
        raise RuntimeError(
            f"stage {stage.name} failed with exit code {result.returncode}; "
            f"see {log_path}"
        )
    latency = (
        read_latency(stage.latency_path, expected_stage=stage.name)
        if stage.latency_path is not None
        else None
    )
    return wall_ms, latency


def timing(payload: dict[str, Any], name: str) -> float:
    values = payload.get("timings_ms")
    if not isinstance(values, dict) or name not in values:
        raise RuntimeError(
            f"latency payload for {payload.get('stage')!r} is missing {name!r}"
        )
    return float(values[name])


def detail_number(payload: dict[str, Any], name: str) -> float:
    values = payload.get("details")
    if not isinstance(values, dict) or name not in values:
        raise RuntimeError(
            f"latency payload for {payload.get('stage')!r} "
            f"is missing detail {name!r}"
        )
    return float(values[name])


def build_sample(
    *,
    sample_index: int,
    warmup: bool,
    cycle_wall_ms: float,
    process_wall_ms: dict[str, float],
    latency: dict[str, dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    dino = latency["dinov3"]
    nomad = latency["nomad"]
    online = latency["online_features"]
    ltwm = latency["ltwm"]
    selector = latency["selector"]
    cem = latency["cem"]

    selector_ms = timing(selector, "selector_fixed_pool_planning_ms")
    cem_ms = timing(cem, "cem_planning_ms")
    online_compute_ms = (
        timing(online, "candidate_pool_prepare_ms")
        + timing(online, "action_and_history_feature_build_ms")
    )
    planning_core_compute_ms = (
        timing(dino, "encode_5_images_ms")
        + timing(nomad, "candidate_generation_ms")
        + online_compute_ms
        + timing(ltwm, "ltwm_member_pipeline_ensemble_ms")
        + selector_ms
        + cem_ms
    )

    sample: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "sample_index": sample_index,
        "warmup": warmup,
        "run_dir": str(run_dir),
        "dinov3_encode_ms": timing(dino, "encode_5_images_ms"),
        "nomad_candidate_generation_ms": timing(
            nomad, "candidate_generation_ms"
        ),
        "nomad_temporal_distance": detail_number(
            nomad, "temporal_distance"
        ),
        "dinov3_current_goal_nmse": detail_number(
            dino, "current_goal_nmse"
        ),
        "dinov3_current_goal_cosine_distance": detail_number(
            dino, "current_goal_cosine_distance"
        ),
        "dinov3_current_goal_norm_ratio": detail_number(
            dino, "current_goal_norm_ratio"
        ),
        "ltwm_future_inference_ms": timing(
            ltwm, "future_inference_ensemble_ms"
        ),
        "ltwm_features_and_scorer_ms": timing(
            ltwm, "latent_features_and_scorer_ensemble_ms"
        ),
        "ltwm_member_pipeline_ms": timing(
            ltwm, "ltwm_member_pipeline_ensemble_ms"
        ),
        "selector_ensemble_predict_ms": timing(
            selector, "selector_ensemble_predict_ms"
        ),
        "selector_fixed_pool_planning_ms": selector_ms,
        "cem_planning_ms": cem_ms,
        "selector_and_cem_planning_ms": selector_ms + cem_ms,
        "planning_core_compute_ms": planning_core_compute_ms,
        "camera_capture_4_frames_ms": timing(
            dino, "camera_capture_4_frames_ms"
        ),
        "odom_collect_4_samples_ms": timing(
            online, "odom_collect_4_samples_ms"
        ),
        "complete_planning_cycle_wall_ms": cycle_wall_ms,
        "stage_process_wall_total_ms": sum(process_wall_ms.values()),
        "stage_process_wall_ms": process_wall_ms,
    }
    for stage_name, wall_ms in process_wall_ms.items():
        sample[f"{stage_name}_process_wall_ms"] = wall_ms
    return sample


def run_cycle(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    sample_index: int,
    warmup: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"v70-latency-{sample_index:04d}-{time.time_ns()}"
    stages = build_stages(args, run_dir, run_id)
    process_wall_ms: dict[str, float] = {}
    latency: dict[str, dict[str, Any]] = {}
    cycle_started_ns = time.perf_counter_ns()
    for stage in stages:
        print(f"[{sample_index:02d}] {stage.name}", flush=True)
        wall_ms, payload = run_stage(stage, args=args, run_dir=run_dir)
        process_wall_ms[stage.name] = wall_ms
        if payload is not None:
            latency[stage.name] = payload
    cycle_wall_ms = (time.perf_counter_ns() - cycle_started_ns) / 1.0e6
    sample = build_sample(
        sample_index=sample_index,
        warmup=warmup,
        cycle_wall_ms=cycle_wall_ms,
        process_wall_ms=process_wall_ms,
        latency=latency,
        run_dir=run_dir,
    )
    write_json_atomic(run_dir / "sample.json", sample)
    return sample


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [sample for sample in samples if not sample["warmup"]]
    if not measured:
        raise RuntimeError("no measured samples")
    numeric_keys = sorted(
        key
        for key, value in measured[0].items()
        if key.endswith("_ms") and isinstance(value, (int, float))
    )
    metrics = {
        key: summarize_ms([float(sample[key]) for sample in measured])
        for key in numeric_keys
    }
    return {
        "schema": BENCHMARK_SCHEMA,
        "primary_metrics": list(PRIMARY_METRICS),
        "measured_samples": len(measured),
        "warmup_samples": len(samples) - len(measured),
        "metrics_ms": metrics,
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    validate_args(args)

    preview_dir = args.output_dir.expanduser().resolve() / "preview"
    if args.print_commands:
        for stage in build_stages(args, preview_dir, "v70-latency-preview"):
            print(f"# {stage.name}\n{shlex.join(stage.command)}")
        return 0

    require_zero_motion_interlock(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": BENCHMARK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "goal_image": str(args.goal_image.expanduser().resolve()),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "candidate_count": args.candidates,
        "cem_iterations": 3,
        "cem_population": 16,
        "ltwm_members": 3,
        "selector_members": 3,
        "device": args.device,
        "zero_motion_interlock": {
            "stop_file": str(args.stop_file),
            "stop_present": True,
            "arm_file": str(args.arm_file),
            "arm_absent": True,
        },
        "host": host_metadata(),
        "measurement_notes": {
            "cuda": "model timings synchronize CUDA before reading the clock",
            "process_wall": (
                "each deployed shadow script runs as a fresh process, so wall "
                "time includes interpreter startup, imports, checkpoint loads, "
                "sensor waits, and artifact I/O"
            ),
            "motion": (
                "no bridge, ROS velocity publisher, or Unitree SportClient "
                "command is invoked"
            ),
        },
    }
    write_json_atomic(output_dir / "protocol.json", protocol)

    samples: list[dict[str, Any]] = []
    total = args.warmup + args.repeats
    for index in range(total):
        warmup = index < args.warmup
        label = "warmup" if warmup else "measure"
        run_dir = output_dir / f"{index:03d}_{label}"
        sample = run_cycle(
            args=args,
            run_dir=run_dir,
            sample_index=index,
            warmup=warmup,
        )
        samples.append(sample)
        append_jsonl(output_dir / "samples.jsonl", sample)
        print(
            f"[{index:02d}] complete cycle "
            f"{sample['complete_planning_cycle_wall_ms']:.3f} ms",
            flush=True,
        )

    summary = summarize_samples(samples)
    summary["protocol"] = protocol
    write_json_atomic(output_dir / "summary.json", summary)

    print("\n===== V70 BOARD LATENCY SUMMARY =====")
    for name in PRIMARY_METRICS:
        values = summary["metrics_ms"][name]
        print(
            f"{name}: p50={values['p50_ms']:.3f} ms, "
            f"p90={values['p90_ms']:.3f} ms, "
            f"p95={values['p95_ms']:.3f} ms"
        )
    print("saved:", output_dir)
    print("No motion command was issued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
