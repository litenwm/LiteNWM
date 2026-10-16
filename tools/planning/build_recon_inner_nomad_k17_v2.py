#!/usr/bin/env python3
"""Build canonical, resumable RECON inner K17 pools from frozen manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.nomad_action_adapter import (  # noqa: E402
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
from tools.planning.build_scand100_nomad_k17_v1 import (  # noqa: E402
    image_for,
    sha256,
    trimmed_mean75,
    write_json,
    write_npz,
)


PARTITIONS = ("train-fit", "scorer-cal", "cem-cal", "fresh-outer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spent-recon-features",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/features/"
        "strict_external100_k17_exact_features.npz",
    )
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/recon")
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
    parser.add_argument("--legacy-site-packages", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_rows(path: Path, partition: str) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("partition") != partition for row in rows):
        raise RuntimeError("manifest partition contract failed")
    for row in rows:
        history = list(map(int, row["history_timesteps"]))
        current = int(row["current_t"])
        if history != list(range(current - 3, current + 1)):
            raise RuntimeError("RECON history contract changed")
        if int(row["goal_t"]) != current + 4:
            raise RuntimeError("RECON H4 goal contract changed")
        if str(row.get("dataset")) != "recon":
            raise RuntimeError("non-RECON row in canonical manifest")
    return rows


def save_progress(
    path: Path,
    *,
    manifest_sha256: str,
    partition: str,
    seed: int,
    samples: list[np.ndarray],
    seeds: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value = (
        np.stack(samples).astype(np.float32)
        if samples
        else np.empty((0, 16, 4, 2), dtype=np.float32)
    )
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            manifest_sha256=np.asarray(manifest_sha256),
            partition=np.asarray(partition),
            seed=np.asarray(seed, dtype=np.int64),
            completed=np.asarray(len(samples), dtype=np.int64),
            nomad_x16_waypoint_units=value,
            per_window_seed=np.asarray(seeds, dtype=np.int64),
        )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    progress_path = args.output.with_suffix(".progress.npz")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.checkpoint_every < 32:
        raise ValueError("checkpoint interval is too small")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = load_rows(args.manifest, args.partition)
    manifest_hash = sha256(args.manifest)
    with np.load(args.spent_recon_features, allow_pickle=False) as payload:
        spent_records = payload["record_key"].astype(str)
    spent_trajectories = {
        str(value).rpartition(":")[0] for value in spent_records
    }
    trajectories = {str(row["trajectory_id"]) for row in rows}
    overlap = sorted(spent_trajectories.intersection(trajectories))
    if overlap:
        raise RuntimeError(
            f"canonical inner partition overlaps spent RECON100: {overlap[:5]}"
        )

    root = args.trajectory_root.resolve(strict=True)
    missing: list[str] = []
    for row in rows:
        for timestep in [*map(int, row["history_timesteps"]), int(row["goal_t"])]:
            path = root / str(row["trajectory_id"]) / f"{timestep}.jpg"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"missing {len(missing)} RECON RGB files; first={missing[0]}")

    samples: list[np.ndarray] = []
    per_window_seeds: list[int] = []
    if progress_path.exists():
        with np.load(progress_path, allow_pickle=False) as progress:
            if str(progress["manifest_sha256"].item()) != manifest_hash:
                raise RuntimeError("progress manifest hash mismatch")
            if str(progress["partition"].item()) != args.partition:
                raise RuntimeError("progress partition mismatch")
            if int(progress["seed"].item()) != args.seed:
                raise RuntimeError("progress seed mismatch")
            samples = list(progress["nomad_x16_waypoint_units"].astype(np.float32))
            per_window_seeds = list(progress["per_window_seed"].astype(np.int64))
        print(
            json.dumps(
                {
                    "stage": "resume",
                    "partition": args.partition,
                    "completed": len(samples),
                }
            ),
            flush=True,
        )

    if args.legacy_site_packages is not None:
        legacy = args.legacy_site_packages.resolve(strict=True)
        if str(legacy) not in sys.path:
            sys.path.append(str(legacy))
    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    device = torch.device(args.device)
    model, load_report = load_nomad_model(args.nomad_checkpoint, config, device)
    sampler = NoMaDSampler(model, config, device)

    for index in range(len(samples), len(rows)):
        row = rows[index]
        trajectory = str(row["trajectory_id"])
        uid = f"recon:{trajectory}:{int(row['current_t'])}:h4"
        sample_seed = window_seed(args.seed, uid)
        observation = transform_images(
            [image_for(root, trajectory, int(t)) for t in row["history_timesteps"]],
            list(config["image_size"]),
        )
        goal = transform_images(
            [image_for(root, trajectory, int(row["goal_t"]))],
            list(config["image_size"]),
        )
        waypoints = sampler.sample(
            observation, goal, candidates=16, seed=sample_seed
        )
        if waypoints.shape[0] != 16 or waypoints.shape[1] < 4 or waypoints.shape[2] != 2:
            raise RuntimeError(f"unexpected NoMaD shape: {waypoints.shape}")
        samples.append(waypoints[:, :4].astype(np.float32))
        per_window_seeds.append(sample_seed)
        completed = index + 1
        if completed % args.checkpoint_every == 0 or completed == len(rows):
            save_progress(
                progress_path,
                manifest_sha256=manifest_hash,
                partition=args.partition,
                seed=args.seed,
                samples=samples,
                seeds=per_window_seeds,
            )
            print(
                json.dumps(
                    {
                        "stage": "nomad_x16",
                        "partition": args.partition,
                        "completed": completed,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )

    nomad_x16 = np.stack(samples).astype(np.float32)
    incumbent = np.stack([trimmed_mean75(value) for value in nomad_x16])
    candidates = np.concatenate((incumbent[:, None], nomad_x16), axis=1).astype(
        np.float32
    )
    actions = nomad_waypoints_to_local_actions(candidates, dataset="recon").astype(
        np.float32
    )
    if candidates.shape != (len(rows), 17, 4, 2) or actions.shape != (
        len(rows),
        17,
        4,
        3,
    ):
        raise RuntimeError("canonical RECON K17 shape contract failed")

    write_npz(
        args.output,
        candidates_waypoint_units=candidates,
        candidate_actions_h4=actions,
        nomad_x16_waypoint_units=nomad_x16,
        window_uid=np.asarray([str(row["window_uid"]) for row in rows]),
        record_key=np.asarray(
            [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
        ),
        trajectory_id=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_id=np.asarray([str(row["route_id"]) for row in rows]),
        curr_time=np.asarray(
            [int(row["current_t"]) for row in rows], dtype=np.int64
        ),
        per_window_seed=np.asarray(per_window_seeds, dtype=np.int64),
    )
    audit = {
        "schema": "recon_inner_nomad_k17_freeze_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_pool_frozen_before_partition_labels",
        "partition": args.partition,
        "states": len(rows),
        "trajectories": len(trajectories),
        "route_groups": len({str(row["route_id"]) for row in rows}),
        "candidate_pool": str(args.output.resolve()),
        "candidate_pool_sha256": sha256(args.output),
        "candidate_shape": list(candidates.shape),
        "action_shape": list(actions.shape),
        "candidate_order": ["trimmed_mean75_incumbent", "deterministic_nomad_x16"],
        "waypoint_spacing_m": 0.25,
        "base_seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "spent_recon_features": str(args.spent_recon_features.resolve()),
        "spent_recon_features_sha256": sha256(args.spent_recon_features),
        "spent_trajectory_overlap": overlap,
        "nomad_checkpoint_sha256": sha256(args.nomad_checkpoint),
        "nomad_config_sha256": sha256(args.nomad_config),
        "load_report": load_report,
        "rgb_only_generation": True,
        "partition_outcome_status": "labels_allowed_only_in_later_source_stage",
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
