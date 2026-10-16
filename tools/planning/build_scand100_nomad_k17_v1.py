#!/usr/bin/env python3
"""Build the label-free strict-SCAND100 K17 pool from frozen RGB windows.

Candidate zero is the trimmed-mean75 aggregate of the same deterministic
NoMaD x16 samples stored at indices 1..16.  This stage never deserializes the
strict ``navigation_eval.pkl`` file or any trajectory pose file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json",
    )
    parser.add_argument(
        "--split-lock",
        type=Path,
        default=REPO_ROOT / "data/split_lock/scand_strict_splits.json",
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    parser.add_argument(
        "--visualnav-root", type=Path, default=Path("/home/rainbow/visualnav-transformer")
    )
    parser.add_argument(
        "--diffusion-policy-root", type=Path, default=Path("/home/rainbow/diffusion_policy")
    )
    parser.add_argument(
        "--nomad-config",
        type=Path,
        default=Path("/home/rainbow/visualnav-transformer/train/config/nomad.yaml"),
    )
    parser.add_argument(
        "--nomad-checkpoint",
        type=Path,
        default=Path(
            "/home/rainbow/visualnav-transformer/deployment/model_weights/nomad.pth"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--legacy-site-packages", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def trimmed_mean75(pool: np.ndarray) -> np.ndarray:
    center = pool.mean(axis=0)
    distance = np.mean(np.sum((pool - center[None]) ** 2, axis=2), axis=1)
    keep = max(2, int(np.ceil(0.75 * len(pool))))
    return pool[np.argsort(distance, kind="stable")[:keep]].mean(axis=0).astype(
        np.float32
    )


def trajectory_names(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result.add(str(row["trajectory_id"]))
    return result


def image_for(root: Path, trajectory_id: str, timestep: int) -> Image.Image:
    path = (root / trajectory_id / f"{timestep}.jpg").resolve(strict=True)
    path.relative_to(root)
    with Image.open(path) as image:
        return image.convert("RGB")


def preflight(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(args.manifest.read_text())
    split = json.loads(args.split_lock.read_text())
    records = list(manifest["records"])
    target = {str(row["trajectory_id"]) for row in records}
    checks: dict[str, bool] = {
        "manifest_schema_v1": manifest.get("schema_version") == 1,
        "dataset_scand": manifest.get("dataset") == "scand",
        "strict_test": manifest.get("strict_split") == "test",
        "windows_100": len(records) == 100,
        "trajectories_43": len(target) == 43,
        "exact_outer_test_set": target == set(map(str, split["test"])),
        "outer_train_overlap_zero": not target.intersection(map(str, split["train"])),
        "outer_val_overlap_zero": not target.intersection(map(str, split["val"])),
        "horizon_h4": manifest.get("horizon") == 4,
        "context_four": manifest.get("context_frames") == 4,
        "model_independent_selection": manifest.get("selection_uses_model_outputs")
        is False,
        "scand_spacing_038": NWM_WAYPOINT_SPACING_METERS["scand"] == 0.38,
    }
    overlap: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "data/inner_manifests").glob("*/*.windows.jsonl")):
        hit = sorted(target.intersection(trajectory_names(path)))
        overlap[str(path.relative_to(REPO_ROOT))] = hit
        checks[f"zero_overlap::{path.relative_to(REPO_ROOT)}"] = not hit
    for path in sorted((REPO_ROOT / "data/latent_metadata").glob("*/splits.json")):
        payload = json.loads(path.read_text())
        for partition, values in payload.items():
            if isinstance(values, list):
                hit = sorted(target.intersection(map(str, values)))
                key = f"{path.relative_to(REPO_ROOT)}::{partition}"
                overlap[key] = hit
                checks[f"zero_overlap::{key}"] = not hit
    root = args.trajectory_root.resolve(strict=True)
    image_count = 0
    missing_images: list[str] = []
    for row in records:
        times = [*map(int, row["context_times"]), int(row["goal_time"])]
        for timestep in times:
            path = root / str(row["trajectory_id"]) / f"{timestep}.jpg"
            if path.is_file():
                image_count += 1
            else:
                missing_images.append(str(path))
    checks["all_context_and_goal_images_exist"] = not missing_images
    checks["nomad_checkpoint_exists"] = args.nomad_checkpoint.is_file()
    checks["nomad_config_exists"] = args.nomad_config.is_file()
    checks["visualnav_root_exists"] = args.visualnav_root.is_dir()
    checks["diffusion_policy_root_exists"] = args.diffusion_policy_root.is_dir()
    report = {
        "schema": "scand100_three_arm_preflight_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "overlap": overlap,
        "target_windows": len(records),
        "target_trajectories": len(target),
        "rgb_files_checked": image_count,
        "missing_images": missing_images,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "split_lock": str(args.split_lock.resolve()),
        "split_lock_sha256": sha256(args.split_lock),
        "navigation_eval_deserialized": False,
        "pose_outcomes_opened": False,
        "strict_test_labels_opened": False,
        "cem": False,
    }
    write_json(args.output.parent / "preflight_audit.json", report)
    if not all(checks.values()):
        raise RuntimeError(f"strict SCAND100 preflight blocked: {checks}")
    return records, report


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".audit.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records, report = preflight(args)
    if args.legacy_site_packages is not None:
        legacy = args.legacy_site_packages.resolve(strict=True)
        if str(legacy) not in sys.path:
            sys.path.append(str(legacy))
    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    device = torch.device(args.device)
    model, load_report = load_nomad_model(args.nomad_checkpoint, config, device)
    sampler = NoMaDSampler(model, config, device)
    root = args.trajectory_root.resolve(strict=True)
    samples: list[np.ndarray] = []
    per_window_seeds: list[int] = []
    window_uids: list[str] = []
    for index, row in enumerate(records):
        trajectory_id = str(row["trajectory_id"])
        uid = f"scand:{trajectory_id}:{int(row['curr_time'])}:h4"
        seed = window_seed(args.seed, uid)
        observation = transform_images(
            [image_for(root, trajectory_id, int(t)) for t in row["context_times"]],
            list(config["image_size"]),
        )
        goal = transform_images(
            [image_for(root, trajectory_id, int(row["goal_time"]))],
            list(config["image_size"]),
        )
        waypoints = sampler.sample(observation, goal, candidates=16, seed=seed)
        if waypoints.shape != (16, int(config["len_traj_pred"]), 2):
            raise RuntimeError(f"unexpected NoMaD output shape: {waypoints.shape}")
        samples.append(waypoints[:, :4].astype(np.float32))
        per_window_seeds.append(seed)
        window_uids.append(uid)
        print(json.dumps({"stage": "nomad_x16", "completed": index + 1, "total": 100}), flush=True)
    nomad_x16 = np.stack(samples)
    incumbent = np.stack([trimmed_mean75(pool) for pool in nomad_x16])
    candidates = np.concatenate((incumbent[:, None], nomad_x16), axis=1).astype(
        np.float32
    )
    actions = nomad_waypoints_to_local_actions(candidates, dataset="scand")
    if candidates.shape != (100, 17, 4, 2) or actions.shape != (100, 17, 4, 3):
        raise RuntimeError("K17 pool shape contract failed")
    write_npz(
        args.output,
        candidates_waypoint_units=candidates,
        candidate_actions_h4=actions.astype(np.float32),
        nomad_x16_waypoint_units=nomad_x16,
        window_uid=np.asarray(window_uids),
        trajectory_id=np.asarray([str(row["trajectory_id"]) for row in records]),
        curr_time=np.asarray([int(row["curr_time"]) for row in records], dtype=np.int64),
        eval_index=np.asarray([int(row["eval_index"]) for row in records], dtype=np.int64),
        per_window_seed=np.asarray(per_window_seeds, dtype=np.int64),
    )
    audit = {
        "schema": "scand100_nomad_k17_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_pool_frozen_before_features_or_labels",
        "candidate_pool": str(args.output.resolve()),
        "candidate_pool_sha256": sha256(args.output),
        "candidate_shape": list(candidates.shape),
        "action_shape": list(actions.shape),
        "candidate_order": ["trimmed_mean75_incumbent", "deterministic_nomad_x16"],
        "incumbent_index": 0,
        "base_seed": args.seed,
        "per_window_seed": "sha256(nomad-h4-v2|base_seed|window_uid)",
        "waypoint_spacing_m": 0.38,
        "nomad_checkpoint": str(args.nomad_checkpoint.resolve()),
        "nomad_checkpoint_sha256": sha256(args.nomad_checkpoint),
        "nomad_config": str(args.nomad_config.resolve()),
        "nomad_config_sha256": sha256(args.nomad_config),
        "load_report": load_report,
        "preflight_audit": str((args.output.parent / "preflight_audit.json").resolve()),
        "preflight_audit_sha256": sha256(args.output.parent / "preflight_audit.json"),
        "manifest_sha256": report["manifest_sha256"],
        "rgb_only_generation": True,
        "navigation_eval_deserialized": False,
        "pose_outcomes_opened": False,
        "strict_test_labels_opened": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output.with_suffix(".audit.json"), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
