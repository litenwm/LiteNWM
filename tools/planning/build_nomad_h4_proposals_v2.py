#!/usr/bin/env python3
"""Generate strict NoMaD x16 H4 proposals for one inner-manifest partition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    transform_images,
)
from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    K,
    PROPOSAL_SCHEMA,
    INNER_PARTITIONS,
    canonical_json,
    file_sha256,
    load_inner_manifest,
    load_npz,
    proposal_arrays,
    validate_proposal_artifact,
    validate_outer_train_membership,
    write_json_atomic,
    write_npz_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("recon", "scand"), required=True)
    parser.add_argument("--partition", choices=INNER_PARTITIONS, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--inner-manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--visualnav-root", type=Path, required=True)
    parser.add_argument("--diffusion-policy-root", type=Path, required=True)
    parser.add_argument("--nomad-config", type=Path, required=True)
    parser.add_argument("--nomad-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def external_source_hashes(
    visualnav_root: Path,
    diffusion_policy_root: Path,
) -> dict[str, str]:
    paths = {
        "nomad.py": visualnav_root / "train/vint_train/models/nomad/nomad.py",
        "nomad_vint.py": visualnav_root / "train/vint_train/models/nomad/nomad_vint.py",
        "conditional_unet1d.py": diffusion_policy_root
        / "diffusion_policy/model/diffusion/conditional_unet1d.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {name: file_sha256(path) for name, path in paths.items()}


def image_for(root: Path, trajectory_id: str, timestep: int) -> Image.Image:
    path = (root / trajectory_id / f"{timestep}.jpg").resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"image path escapes trajectory root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def load_partial(
    path: Path,
    rows: list[dict[str, object]],
    binding: dict[str, object],
) -> list[np.ndarray]:
    partial = load_npz(path)
    if set(partial) != {"window_uid", "actions_h4", "metadata"}:
        raise RuntimeError("partial proposal has an unexpected schema")
    metadata_array = np.asarray(partial["metadata"])
    if metadata_array.ndim != 0:
        raise RuntimeError("partial proposal metadata must be a scalar")
    metadata = json.loads(str(metadata_array.item()))
    if metadata != binding:
        raise RuntimeError("partial proposal provenance does not match this invocation")
    uids = partial["window_uid"].astype(str)
    expected = np.asarray([str(row["window_uid"]) for row in rows[: len(uids)]])
    if not np.array_equal(uids, expected):
        raise RuntimeError("partial proposal is not an exact manifest prefix")
    actions = np.asarray(partial["actions_h4"], dtype=np.float32)
    if actions.shape != (len(uids), K, 4, 3) or not np.isfinite(actions).all():
        raise RuntimeError("partial proposal actions are invalid")
    return [value.copy() for value in actions]


def save_partial(
    path: Path,
    rows: list[dict[str, object]],
    actions: list[np.ndarray],
    binding: dict[str, object],
) -> None:
    write_npz_atomic(
        path,
        window_uid=np.asarray([str(row["window_uid"]) for row in rows[: len(actions)]]),
        actions_h4=np.stack(actions).astype(np.float32),
        metadata=np.asarray(canonical_json(binding)),
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seed < 0 or args.save_every <= 0:
        raise ValueError("seed must be nonnegative and save-every must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = load_inner_manifest(
        args.inner_manifest,
        dataset=args.dataset,
        partition=args.partition,
    )
    split_lock_sha = file_sha256(args.split_lock)
    validate_outer_train_membership(args.split_lock, rows, dataset=args.dataset)
    trajectory_root = args.trajectory_root.resolve(strict=True)
    manifest_sha = file_sha256(args.inner_manifest)
    checkpoint_sha = file_sha256(args.nomad_checkpoint)
    config_sha = file_sha256(args.nomad_config)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial.npz")
    if partial_path.exists() and not args.resume:
        raise FileExistsError(
            f"partial output exists; pass --resume after checking {partial_path}"
        )

    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    source_hashes = external_source_hashes(
        args.visualnav_root, args.diffusion_policy_root
    )
    device = torch.device(args.device)
    model, load_report = load_nomad_model(args.nomad_checkpoint, config, device)
    sampler = NoMaDSampler(model, config, device)
    partial_binding = {
        "schema": "h4_nomad_partial_v2",
        "dataset": args.dataset,
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
    started = time.perf_counter()

    for index in range(len(actions), len(rows)):
        row = rows[index]
        trajectory_id = str(row["trajectory_id"])
        history = [int(value) for value in row["history_timesteps"]]
        observation_pil = [
            image_for(trajectory_root, trajectory_id, t) for t in history
        ]
        goal_pil = image_for(trajectory_root, trajectory_id, int(row["goal_t"]))
        observation = transform_images(observation_pil, list(config["image_size"]))
        goal = transform_images([goal_pil], list(config["image_size"]))
        per_window_seed = window_seed(args.seed, str(row["window_uid"]))
        waypoints = sampler.sample(
            observation,
            goal,
            candidates=K,
            seed=per_window_seed,
        )
        expected_shape = (K, int(config["len_traj_pred"]), 2)
        if waypoints.shape != expected_shape or not np.isfinite(waypoints).all():
            raise RuntimeError(f"NoMaD returned invalid waypoints: {waypoints.shape}")
        actions.append(
            nomad_waypoints_to_local_actions(
                waypoints,
                dataset=args.dataset,
                horizon=4,
            )
        )
        if len(actions) % args.save_every == 0:
            save_partial(partial_path, rows, actions, partial_binding)
        print(
            json.dumps(
                {
                    "stage": "nomad_h4_proposals_v2",
                    "dataset": args.dataset,
                    "partition": args.partition,
                    "completed": len(actions),
                    "total": len(rows),
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    all_actions = np.stack(actions).astype(np.float32)
    metadata = {
        "schema": PROPOSAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
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
        "waypoint_spacing_m": NWM_WAYPOINT_SPACING_METERS[args.dataset],
        "rgb_only": True,
        "image_preprocessing": "legacy_nomad_transform_images_no_center_crop",
        "logged_pose_opened": False,
        "labels_opened": False,
        "outer_validation_accessed": False,
        "test_accessed": False,
    }
    arrays = proposal_arrays(
        rows,
        all_actions,
        dataset=args.dataset,
        partition=args.partition,
        metadata=metadata,
    )
    validate_proposal_artifact(
        arrays,
        rows,
        dataset=args.dataset,
        partition=args.partition,
        split_lock_sha256=split_lock_sha,
    )
    write_npz_atomic(args.output, **arrays)
    metadata["output_sha256"] = file_sha256(args.output)
    write_json_atomic(Path(str(args.output) + ".schema.json"), metadata)
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
