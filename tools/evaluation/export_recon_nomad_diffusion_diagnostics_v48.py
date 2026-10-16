#!/usr/bin/env python3
"""Replay frozen NoMaD K17 sampling and export outcome-free diffusion diagnostics."""

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

from planning.nomad_backend import (  # noqa: E402
    _decode_waypoints,
    add_external_paths,
    load_config,
    load_nomad_model,
    transform_images,
)
from tools.planning.build_recon_inner_nomad_k17_v2 import load_rows  # noqa: E402
from tools.planning.build_scand100_nomad_k17_v1 import (  # noqa: E402
    image_for,
    sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k17", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-root", type=Path,
        default=Path("/home/rainbow/nwm/data/recon"),
    )
    parser.add_argument(
        "--visualnav-root", type=Path,
        default=Path("/home/rainbow/visualnav-transformer"),
    )
    parser.add_argument(
        "--diffusion-policy-root", type=Path,
        default=Path("/home/rainbow/diffusion_policy"),
    )
    parser.add_argument(
        "--nomad-config", type=Path,
        default=Path("/home/rainbow/visualnav-transformer/train/config/nomad.yaml"),
    )
    parser.add_argument(
        "--nomad-checkpoint", type=Path,
        default=Path(
            "/home/rainbow/visualnav-transformer/deployment/model_weights/nomad.pth"
        ),
    )
    parser.add_argument("--legacy-site-packages", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--identity-atol", type=float, default=2e-5)
    return parser.parse_args()


def summarize_reverse(value: np.ndarray) -> np.ndarray:
    """Summarize [steps,candidates,signals] as [candidates,75]."""
    blocks = np.array_split(np.arange(value.shape[0]), 3)
    parts = []
    for block in blocks:
        local = value[block]
        parts.extend((
            local.mean(0), local.std(0),
            np.quantile(local, 0.10, axis=0),
            np.quantile(local, 0.50, axis=0),
            np.quantile(local, 0.90, axis=0),
        ))
    return np.concatenate(parts, axis=1).astype(np.float32)


@torch.inference_mode()
def replay_one(
    model: torch.nn.Module,
    scheduler: object,
    config: dict,
    observation: torch.Tensor,
    goal: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = 16
    observation = observation.to(device)
    goal = goal.to(device)
    no_mask = torch.zeros(1, dtype=torch.long, device=device)
    conditioning = model(
        "vision_encoder", obs_img=observation, goal_img=goal,
        input_goal_mask=no_mask,
    ).repeat(candidates, 1)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randn(
        (candidates, int(config["len_traj_pred"]), 2),
        generator=generator, device=device,
    )
    scheduler.set_timesteps(int(config["num_diffusion_iters"]))
    reverse = []
    for timestep in scheduler.timesteps:
        predicted_noise = model(
            "noise_pred_net", sample=sample, timestep=timestep,
            global_cond=conditioning,
        )
        previous = scheduler.step(
            model_output=predicted_noise, timestep=timestep,
            sample=sample, generator=generator,
        ).prev_sample
        reverse.append(torch.stack((
            predicted_noise.square().mean((1, 2)).sqrt(),
            predicted_noise.abs().mean((1, 2)),
            sample.square().mean((1, 2)).sqrt(),
            (previous - sample).square().mean((1, 2)).sqrt(),
            (previous.abs() >= 0.999).float().mean((1, 2)),
        ), dim=1).float().cpu().numpy())
        sample = previous
    reverse_feature = summarize_reverse(np.stack(reverse))

    probe_values = []
    timesteps = np.asarray((10, 25, 50, 75, 90), dtype=np.int64)
    for probe_index in range(2):
        probe_generator = torch.Generator(device=device).manual_seed(
            int(seed) + 104729 * (probe_index + 1)
        )
        noise = torch.randn(
            sample.shape, generator=probe_generator, device=device,
        )
        per_timestep = []
        for timestep in timesteps:
            t = torch.full(
                (candidates,), int(timestep), dtype=torch.long, device=device
            )
            noisy = scheduler.add_noise(sample, noise, t)
            predicted = model(
                "noise_pred_net", sample=noisy, timestep=t,
                global_cond=conditioning,
            )
            per_timestep.append(
                (predicted - noise).square().mean((1, 2)).float().cpu().numpy()
            )
        probe_values.append(np.stack(per_timestep, axis=1))
    probe = np.stack(probe_values, axis=2)
    probe_feature = np.concatenate((
        probe.mean(2), probe.std(2),
        probe.mean((1, 2))[:, None],
        probe.std((1, 2))[:, None],
        probe.min((1, 2))[:, None],
        probe.max((1, 2))[:, None],
        np.quantile(probe, 0.50, axis=(1, 2))[:, None],
    ), axis=1).astype(np.float32)
    feature = np.concatenate((reverse_feature, probe_feature), axis=1)
    waypoint = _decode_waypoints(sample).float().cpu().numpy()
    return waypoint, feature


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = load_rows(args.manifest, "fresh-outer")
    with np.load(args.k17, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        frozen = payload["nomad_x16_waypoint_units"].astype(np.float32)
        seeds = payload["per_window_seed"].astype(np.int64)
    manifest_record = np.asarray([
        f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows
    ])
    if not np.array_equal(record, manifest_record):
        raise RuntimeError("manifest/K17 identity mismatch")
    count = len(rows) if args.max_states is None else min(args.max_states, len(rows))
    legacy = args.legacy_site_packages.resolve(strict=True)
    if str(legacy) not in sys.path:
        sys.path.append(str(legacy))
    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    device = torch.device(args.device)
    model, load_report = load_nomad_model(args.nomad_checkpoint, config, device)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=int(config["num_diffusion_iters"]),
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        prediction_type="epsilon",
    )
    root = args.trajectory_root.resolve(strict=True)
    diagnostics, errors = [], []
    for index, row in enumerate(rows[:count]):
        trajectory = str(row["trajectory_id"])
        observation = transform_images(
            [image_for(root, trajectory, int(value))
             for value in row["history_timesteps"]],
            list(config["image_size"]),
        )
        goal = transform_images(
            [image_for(root, trajectory, int(row["goal_t"]))],
            list(config["image_size"]),
        )
        waypoint, feature = replay_one(
            model, scheduler, config, observation, goal,
            int(seeds[index]), device,
        )
        error = float(np.max(np.abs(waypoint[:, :4] - frozen[index])))
        if error > args.identity_atol:
            raise RuntimeError(
                f"NoMaD replay identity failed at {record[index]}: {error}"
            )
        if feature.shape != (16, 90) or not np.isfinite(feature).all():
            raise RuntimeError(f"diagnostic feature contract failed: {feature.shape}")
        diagnostics.append(feature)
        errors.append(error)
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(json.dumps({
                "stage": "nomad_diffusion_diagnostics_v48",
                "completed": index + 1, "total": count,
                "max_identity_error": max(errors),
            }), flush=True)
    value = np.stack(diagnostics).astype(np.float32)
    write_npz(
        args.output,
        record_key=record[:count],
        candidate_diffusion_features=value,
        per_window_seed=seeds[:count],
    )
    audit = {
        "schema": "recon_nomad_diffusion_diagnostics_v48",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "k17": str(args.k17),
        "k17_sha256": sha256(args.k17),
        "nomad_checkpoint": str(args.nomad_checkpoint),
        "nomad_checkpoint_sha256": sha256(args.nomad_checkpoint),
        "states": count,
        "candidates": 16,
        "feature_dim": value.shape[-1],
        "feature_contract": (
            "three reverse-diffusion time blocks x five statistics x five "
            "signals plus fixed-timestep two-probe conditional denoising residuals"
        ),
        "input_time": "history t-3..t, goal image t+4, candidate seed/actions",
        "online_deployable": True,
        "outcomes_accessed": False,
        "labels_in_output": False,
        "max_replay_identity_error": max(errors),
        "all_finite": bool(np.isfinite(value).all()),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "load_report": load_report,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
