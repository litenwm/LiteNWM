"""Run H4 LTWM + scorer on synthetic inputs or local adapted DINOv3 latents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from planning.checkpoint_io import load_checkpoint
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2
from scoring.trajectory_quality_scorer_v2 import (
    TrajectoryQualityScorerConfig, TrajectoryQualityScorerV2,
)


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(1701)
    device = torch.device(args.device)
    if args.demo:
        wm = LiteTokenFormerWMV2(LiteTokenFormerConfig(
            horizon=4, model_dim=24, heads=4, history_layers=1,
            action_layers=1, rollout_blocks=1, mlp_ratio=2,
        ))
        scorer = TrajectoryQualityScorerV2(TrajectoryQualityScorerConfig(
            hidden_dim=32, hidden_layers=1, dropout=0,
        ))
        history = torch.randn(1, 4, 256, 384)
        goal = torch.randn(1, 256, 384)
        actions = torch.randn(1, 3, 4, 3) * 0.01
        mode = "synthetic_untrained_demo"
    else:
        cfg_path = args.config.resolve()
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        def asset(key: str) -> Path:
            path = Path(cfg[key])
            path = path if path.is_absolute() else cfg_path.parent / path
            if not path.is_file():
                raise ValueError(f"Missing {key}: {path.resolve()}")
            return path

        def checked_checkpoint(key: str, hash_key: str) -> Path:
            path = asset(key)
            expected = cfg.get(hash_key)
            if expected:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"{key} SHA-256 mismatch; expected the configured release")
            return path

        # Project training artifacts include metadata beyond tensors. Use only
        # checkpoints obtained from a trusted project release.
        wm_path = checked_checkpoint("wm_checkpoint", "wm_sha256")
        scorer_path = checked_checkpoint("scorer_checkpoint", "scorer_sha256")
        wm_payload = load_checkpoint(wm_path, map_location="cpu", weights_only=False)
        scorer_payload = load_checkpoint(scorer_path, map_location="cpu", weights_only=False)
        wm = LiteTokenFormerWMV2(LiteTokenFormerConfig(**wm_payload["config"]))
        if wm.config.horizon != 4:
            raise ValueError("This entry point requires an H4 checkpoint")
        wm.load_state_dict(wm_payload["model"], strict=True)
        scorer = TrajectoryQualityScorerV2(TrajectoryQualityScorerConfig(**scorer_payload["model_config"]))
        scorer.load_state_dict(scorer_payload["model"], strict=True)
        wm_lock = wm_payload.get("split_lock_sha256")
        scorer_lock = scorer_payload.get("split_lock_sha256")
        if not wm_lock or wm_lock != scorer_lock:
            raise ValueError("World model and scorer split-lock provenance does not match")
        paired_hash = scorer_payload.get("world_model_checkpoint_sha256")
        if paired_hash and paired_hash != hashlib.sha256(wm_path.read_bytes()).hexdigest():
            raise ValueError("Scorer is bound to a different world model checkpoint")
        with np.load(asset("inputs"), allow_pickle=False) as inputs:
            history = torch.from_numpy(inputs["history_tokens"].astype(np.float32))
            goal = torch.from_numpy(inputs["goal_tokens"].astype(np.float32))
            actions = torch.from_numpy(inputs["candidate_actions"].astype(np.float32))
        mode = "checkpoint_inference"
    if history.ndim != 4 or tuple(history.shape[1:]) != (4, 256, 384):
        raise ValueError("history_tokens must have shape [B,4,256,384]")
    batch = history.shape[0]
    if tuple(goal.shape) != (batch, 256, 384):
        raise ValueError("goal_tokens must have shape [B,256,384]")
    if actions.ndim != 4 or actions.shape[0] != batch or tuple(actions.shape[2:]) != (4, 3) or actions.shape[1] < 1:
        raise ValueError("candidate_actions must have shape [B,K,4,3], K >= 1")
    if not all(torch.isfinite(x).all() for x in (history, goal, actions)):
        raise ValueError("Inputs must be finite")
    wm = wm.to(device).eval()
    scorer = scorer.to(device).eval()
    history, goal, actions = (x.to(device) for x in (history, goal, actions))
    with torch.inference_mode():
        future = wm(history, actions)["predicted_tokens"]
        prediction = scorer(
            current_latent=history[:, -1], goal_latent=goal,
            candidate_actions=actions, predicted_future=future,
        )
    if not torch.isfinite(future).all():
        raise RuntimeError("World model produced nonfinite tokens")
    return {
        "mode": mode, "device": str(device),
        "history_shape": list(history.shape), "future_shape": list(future.shape),
        "ate_lower_is_better": prediction.ate.cpu().tolist(),
        "rtp_higher_is_better": prediction.rtp.cpu().tolist(),
        "note": "Random weights and synthetic inputs; scores have no scientific meaning."
        if args.demo else "Single-member H4 scores; no CEM, ensemble, or robot actuation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="No weights or dataset needed")
    mode.add_argument("--config", type=Path, help="Local JSON for checkpoint inference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        result = run(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"Input/configuration error: {error}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
