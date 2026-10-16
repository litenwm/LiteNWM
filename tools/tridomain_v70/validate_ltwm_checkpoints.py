#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path("/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain")
RUN = Path("/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights")
LATENTS = {
    "recon": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/recon_dinov3_ltwm_h4"),
    "scand": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/scand_dinov3_ltwm_h4"),
    "sacson": Path("/root/autodl-tmp/dwm_datasets/processed_v70_tridomain/sacson_dinov3_ltwm_h4"),
}

sys.path.insert(0, str(REPO / "src"))
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def real_window(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    metadata = json.loads((root / "metadata.json").read_text())
    train_ids = set(json.loads((root / "splits.json").read_text())["train"])
    for trajectory in metadata["trajectories"]:
        if trajectory["trajectory_id"] not in train_ids or trajectory["num_frames"] < 8:
            continue
        with np.load(root / "shards" / trajectory["shard_file"], allow_pickle=False) as shard:
            latent = shard["patch_latents"][:8].astype(np.float32).reshape(8, 256, 384)
            actions = shard["actions_continuous"][3:7].astype(np.float32)
        history = torch.from_numpy(latent[:4]).unsqueeze(0)
        target = torch.from_numpy(latent[4:8]).unsqueeze(0)
        action = torch.from_numpy(actions).unsqueeze(0).unsqueeze(0)
        return history, target, action, str(trajectory["trajectory_id"])
    raise RuntimeError(f"no valid H4 window in {root}")


def main() -> None:
    device = torch.device("cuda")
    windows = {name: real_window(root) for name, root in LATENTS.items()}
    active_lock_sha = sha256(RUN / "splits" / "split_lock.json")
    report = {"status": "PASS", "active_split_lock_sha256": active_lock_sha, "seeds": {}}
    for seed in (1701, 1702, 1703):
        checkpoint = RUN / "ltwm" / f"seed{seed}" / "checkpoints" / "best.pt"
        summary_path = RUN / "ltwm" / f"seed{seed}" / "report" / "train_summary.json"
        summary = json.loads(summary_path.read_text())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("split_lock_sha256") != active_lock_sha:
            raise RuntimeError(f"seed {seed}: split-lock mismatch")
        config = LiteTokenFormerConfig(**payload["config"])
        if config.horizon != 4 or config.input_dim != 384:
            raise RuntimeError(f"seed {seed}: invalid config {config}")
        model = LiteTokenFormerWMV2(config)
        model.load_state_dict(payload["model"], strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 8_001_960:
            raise RuntimeError(f"seed {seed}: parameter count {parameter_count}")
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError(f"seed {seed}: non-finite parameter")
        model = model.eval().to(device)
        domains = {}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for domain, (history, target, actions, trajectory_id) in windows.items():
                output = model(history.to(device), actions.to(device))["predicted_tokens"]
                if output.shape != (1, 1, 4, 256, 384):
                    raise RuntimeError(f"seed {seed}/{domain}: invalid output {tuple(output.shape)}")
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"seed {seed}/{domain}: non-finite output")
                prediction = output[:, 0].float()
                target_gpu = target.to(device)
                nmse = (
                    (prediction - target_gpu).square().mean()
                    / target_gpu.square().mean().clamp_min(1e-8)
                )
                domains[domain] = {
                    "trajectory_id": trajectory_id,
                    "smoke_nmse": float(nmse),
                    "output_shape": list(output.shape),
                    "finite": True,
                }
        report["seeds"][str(seed)] = {
            "summary_status": summary["status"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameter_count": parameter_count,
            "strict_load": True,
            "finite_parameters": True,
            "domains": domains,
        }
        del model
        torch.cuda.empty_cache()
    output = RUN / "ltwm" / "validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
