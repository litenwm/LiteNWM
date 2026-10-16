#!/usr/bin/env python3
"""Resume SACSoN NoMaD proposals with deterministic batched GPU forwards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.nomad_action_adapter import (  # noqa: E402
    NWM_WAYPOINT_SPACING_METERS,
    nomad_waypoints_to_local_actions,
    window_seed,
)
from planning.nomad_backend import (  # noqa: E402
    NoMaDSampler,
    add_external_paths,
    load_config,
    load_nomad_model,
)
from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    K,
    PROPOSAL_SCHEMA,
    canonical_json,
    file_sha256,
    load_inner_manifest,
    proposal_arrays,
    validate_outer_train_membership,
    validate_proposal_artifact,
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.build_nomad_h4_proposals_v2 import (  # noqa: E402
    external_source_hashes,
    git_commit,
    image_for,
    load_partial,
    save_partial,
    transform_images,
)


DATASET = "sacson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("train-fit", "scorer-cal"), required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--visualnav-root", type=Path, required=True)
    parser.add_argument("--diffusion-policy-root", type=Path, required=True)
    parser.add_argument("--nomad-config", type=Path, required=True)
    parser.add_argument("--nomad-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--window-batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1024)
    parser.add_argument("--verify-windows", type=int, default=4)
    parser.add_argument("--equivalence-atol", type=float, default=2.0e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def prepare_batch(
    rows: list[dict[str, object]],
    *,
    trajectory_root: Path,
    image_size: list[int],
    base_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    observations = []
    goals = []
    seeds = []
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        history = [int(value) for value in row["history_timesteps"]]
        observations.append(
            transform_images(
                [
                    image_for(trajectory_root, trajectory_id, timestep)
                    for timestep in history
                ],
                image_size,
            )
        )
        goals.append(
            transform_images(
                [
                    image_for(
                        trajectory_root,
                        trajectory_id,
                        int(row["goal_t"]),
                    )
                ],
                image_size,
            )
        )
        seeds.append(window_seed(base_seed, str(row["window_uid"])))
    return torch.cat(observations, dim=0), torch.cat(goals, dim=0), seeds


@torch.inference_mode()
def sample_batch(
    sampler: NoMaDSampler,
    observations: torch.Tensor,
    goals: torch.Tensor,
    *,
    candidates: int,
    seeds: list[int],
) -> np.ndarray:
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    batch_size = int(observations.shape[0])
    if observations.ndim != 4 or goals.ndim != 4:
        raise ValueError("batched observations and goals must be rank 4")
    if int(goals.shape[0]) != batch_size or len(seeds) != batch_size:
        raise ValueError("batch tensors and seed list must have equal length")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")

    observations = observations.to(sampler.device)
    goals = goals.to(sampler.device)
    goal_mask = torch.zeros(batch_size, dtype=torch.long, device=sampler.device)
    conditioning = sampler.model(
        "vision_encoder",
        obs_img=observations,
        goal_img=goals,
        input_goal_mask=goal_mask,
    ).repeat_interleave(candidates, dim=0)
    generators = [
        torch.Generator(device=sampler.device).manual_seed(seed) for seed in seeds
    ]
    trajectory_length = int(sampler.config["len_traj_pred"])
    sample = torch.cat(
        [
            torch.randn(
                (candidates, trajectory_length, 2),
                generator=generator,
                device=sampler.device,
            )
            for generator in generators
        ],
        dim=0,
    )
    sampler.scheduler.set_timesteps(int(sampler.config["num_diffusion_iters"]))
    for timestep in sampler.scheduler.timesteps:
        predicted_noise = sampler.model(
            "noise_pred_net",
            sample=sample,
            timestep=timestep,
            global_cond=conditioning,
        )
        next_samples = []
        for window_index, generator in enumerate(generators):
            start = window_index * candidates
            stop = start + candidates
            next_samples.append(
                sampler.scheduler.step(
                    model_output=predicted_noise[start:stop],
                    timestep=timestep,
                    sample=sample[start:stop],
                    generator=generator,
                ).prev_sample
            )
        sample = torch.cat(next_samples, dim=0)
    decoded = sample.reshape(-1, trajectory_length, 2)
    minimum = decoded.new_tensor((-2.5, -4.0))
    maximum = decoded.new_tensor((5.0, 4.0))
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded * (maximum - minimum) + minimum
    decoded = torch.cumsum(decoded, dim=1)
    return (
        decoded.reshape(
            batch_size,
            candidates,
            trajectory_length,
            2,
        )
        .float()
        .cpu()
        .numpy()
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_equivalence(
    sampler: NoMaDSampler,
    observations: torch.Tensor,
    goals: torch.Tensor,
    seeds: list[int],
    *,
    atol: float,
) -> dict[str, object]:
    synchronize(sampler.device)
    started = time.perf_counter()
    serial = np.stack(
        [
            sampler.sample(
                observations[index : index + 1],
                goals[index : index + 1],
                candidates=K,
                seed=seeds[index],
            )
            for index in range(len(seeds))
        ]
    )
    synchronize(sampler.device)
    serial_seconds = time.perf_counter() - started

    synchronize(sampler.device)
    started = time.perf_counter()
    batched = sample_batch(
        sampler,
        observations,
        goals,
        candidates=K,
        seeds=seeds,
    )
    synchronize(sampler.device)
    batched_seconds = time.perf_counter() - started
    absolute = np.abs(serial - batched)
    max_absolute = float(absolute.max(initial=0.0))
    mean_absolute = float(absolute.mean())
    serial_actions = nomad_waypoints_to_local_actions(
        serial, dataset=DATASET, horizon=4
    )
    batched_actions = nomad_waypoints_to_local_actions(
        batched, dataset=DATASET, horizon=4
    )
    action_absolute = np.abs(serial_actions - batched_actions)
    equivalent = bool(np.allclose(serial, batched, rtol=0.0, atol=atol))
    report = {
        "status": "PASS" if equivalent else "FAIL",
        "windows": len(seeds),
        "candidates_per_window": K,
        "serial_seconds": serial_seconds,
        "batched_seconds": batched_seconds,
        "speedup": serial_seconds / max(batched_seconds, 1.0e-12),
        "max_absolute_waypoint_difference": max_absolute,
        "mean_absolute_waypoint_difference": mean_absolute,
        "max_translation_action_difference_m": float(action_absolute[..., :2].max(initial=0.0)),
        "max_yaw_action_difference_rad": float(action_absolute[..., 2].max(initial=0.0)),
        "mean_absolute_action_difference": float(action_absolute.mean()),
        "absolute_tolerance": atol,
        "per_window_seed_preserved": True,
    }
    if not equivalent:
        raise RuntimeError(f"batched NoMaD equivalence failed: {report}")
    return report


def main() -> None:
    args = parse_args()
    if (
        args.seed < 0
        or args.window_batch_size < 1
        or args.save_every < 1
        or args.verify_windows < 1
        or args.equivalence_atol < 0
    ):
        raise ValueError("invalid seed, batch, checkpoint, verification, or tolerance")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output.exists() and not args.benchmark_only:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows = load_inner_manifest(
        args.inner_manifest,
        dataset=DATASET,
        partition=args.partition,
    )
    split_lock_sha = file_sha256(args.split_lock)
    validate_outer_train_membership(args.split_lock, rows, dataset=DATASET)
    trajectory_root = args.trajectory_root.resolve(strict=True)
    manifest_sha = file_sha256(args.inner_manifest)
    checkpoint_sha = file_sha256(args.nomad_checkpoint)
    config_sha = file_sha256(args.nomad_config)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial.npz")

    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    source_hashes = external_source_hashes(
        args.visualnav_root,
        args.diffusion_policy_root,
    )
    device = torch.device(args.device)
    model, load_report = load_nomad_model(
        args.nomad_checkpoint,
        config,
        device,
    )
    sampler = NoMaDSampler(model, config, device)
    partial_binding = {
        "schema": "h4_nomad_partial_v2",
        "dataset": DATASET,
        "partition": args.partition,
        "inner_manifest_sha256": manifest_sha,
        "split_lock_sha256": split_lock_sha,
        "nomad_checkpoint_sha256": checkpoint_sha,
        "nomad_config_sha256": config_sha,
        "visualnav_source_sha256": source_hashes,
        "base_seed": args.seed,
        "candidates": K,
        "horizon": 4,
        "action_conversion": "path_heading_se2",
    }
    actions = (
        load_partial(partial_path, rows, partial_binding)
        if partial_path.exists()
        else []
    )
    resumed_prefix_windows = len(actions)
    verify_start = min(len(actions), max(0, len(rows) - args.verify_windows))
    verify_rows = rows[verify_start : verify_start + args.verify_windows]
    observations, goals, verify_seeds = prepare_batch(
        verify_rows,
        trajectory_root=trajectory_root,
        image_size=list(config["image_size"]),
        base_seed=args.seed,
    )
    equivalence = verify_equivalence(
        sampler,
        observations,
        goals,
        verify_seeds,
        atol=args.equivalence_atol,
    )
    equivalence["verification_start_index"] = verify_start
    equivalence["window_batch_size_requested"] = args.window_batch_size
    benchmark_path = args.output.with_suffix(
        args.output.suffix + ".batch_benchmark.json"
    )
    write_json_atomic(benchmark_path, equivalence)
    print(json.dumps({"stage": "batch_equivalence", **equivalence}), flush=True)
    if args.benchmark_only:
        return

    started = time.perf_counter()
    for batch_start in range(
        len(actions),
        len(rows),
        args.window_batch_size,
    ):
        batch_rows = rows[
            batch_start : min(batch_start + args.window_batch_size, len(rows))
        ]
        observations, goals, seeds = prepare_batch(
            batch_rows,
            trajectory_root=trajectory_root,
            image_size=list(config["image_size"]),
            base_seed=args.seed,
        )
        waypoint_batch = sample_batch(
            sampler,
            observations,
            goals,
            candidates=K,
            seeds=seeds,
        )
        expected_shape = (
            len(batch_rows),
            K,
            int(config["len_traj_pred"]),
            2,
        )
        if (
            waypoint_batch.shape != expected_shape
            or not np.isfinite(waypoint_batch).all()
        ):
            raise RuntimeError(
                f"NoMaD returned invalid batched waypoints: "
                f"{waypoint_batch.shape}"
            )
        previous_count = len(actions)
        actions.extend(
            nomad_waypoints_to_local_actions(
                waypoints,
                dataset=DATASET,
                horizon=4,
            )
            for waypoints in waypoint_batch
        )
        if len(actions) // args.save_every > previous_count // args.save_every:
            save_partial(partial_path, rows, actions, partial_binding)
        print(
            json.dumps(
                {
                    "stage": "nomad_h4_proposals_batched",
                    "dataset": DATASET,
                    "partition": args.partition,
                    "completed": len(actions),
                    "total": len(rows),
                    "window_batch_size": len(batch_rows),
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    all_actions = np.stack(actions).astype(np.float32)
    metadata = {
        "schema": PROPOSAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "partition": args.partition,
        "windows": len(rows),
        "candidates": K,
        "horizon": 4,
        "split_lock_sha256": split_lock_sha,
        "inner_manifest_sha256": manifest_sha,
        "nomad_checkpoint_sha256": checkpoint_sha,
        "nomad_config_sha256": config_sha,
        "visualnav_git_commit": git_commit(args.visualnav_root),
        "visualnav_source_sha256": source_hashes,
        "load_report": load_report,
        "base_seed": args.seed,
        "per_window_seed": "sha256(nomad-h4-v2|base_seed|window_uid)",
        "proposal_family": "nomad_x16",
        "nomad_output": "cumulative_local_xy_waypoint_units",
        "action_conversion": "path_heading_se2",
        "action_units": ["meters", "meters", "radians"],
        "waypoint_spacing_m": NWM_WAYPOINT_SPACING_METERS[DATASET],
        "rgb_only": True,
        "image_preprocessing": "legacy_nomad_transform_images_no_center_crop",
        "logged_pose_opened": False,
        "labels_opened": False,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "execution": {
            "window_batch_size": args.window_batch_size,
            "partial_checkpoint_interval_windows": args.save_every,
            "resumed_prefix_windows": resumed_prefix_windows,
            "equivalence_report": equivalence,
            "numerical_result_contract": (
                "per-window RNG stream preserved; batched GPU forwards "
                "verified against serial execution"
            ),
        },
    }
    arrays = proposal_arrays(
        rows,
        all_actions,
        dataset=DATASET,
        partition=args.partition,
        metadata=metadata,
    )
    validate_proposal_artifact(
        arrays,
        rows,
        dataset=DATASET,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    write_npz_atomic(args.output, **arrays)
    metadata["output_sha256"] = file_sha256(args.output)
    write_json_atomic(Path(str(args.output) + ".schema.json"), metadata)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
