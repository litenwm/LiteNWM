#!/usr/bin/env python3
"""Export consolidated A--T raw/spatial LTWM features for v51 screening.

The exporter never reads outcome arrays.  Frozen mode consolidates the
existing rollout caches.  Adapted mode runs only the three preregistered WM
checkpoints and deliberately does not run the old paired scorer-v2 heads.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from planning.costs import latent_trajectory_features  # noqa: E402
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402
from tools.evaluation.export_recon_spatial_rollout_features_v17 import (  # noqa: E402
    spatial_rollout_feature,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("frozen", "adapted"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=4)
    return parser.parse_args()


def bank_roots() -> dict[str, Path]:
    return {
        **{x: ROOT / "outputs/cem_development_v1/recon_target_calibration_v20" / f"bank_{x}" for x in "abcdef"},
        "g": ROOT / "outputs/cem_development_v1/recon_target_confirmation_v29",
        "h": ROOT / "outputs/cem_development_v1/recon_target_cem_confirmation_v35",
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v40" / f"bank_{x}" for x in "ijk"},
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v44" / f"bank_{x}" for x in "lmnop"},
        **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v46" / f"bank_{x}" for x in "qrst"},
    }


def feature_path(letter: str, root: Path) -> Path:
    matches = sorted((root / "features").glob("*_features.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"bank {letter} feature ambiguity: {matches}")
    return matches[0]


def load_bank(letter: str, root: Path) -> dict[str, object]:
    feature = feature_path(letter, root)
    with np.load(feature, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        value = {
            key: payload[key].copy()
            for key in (
                "candidates", "actions_h4", "action_features", "static_features",
                "raw_ltwm_features", "support_features", "majority_support",
                "record_key", "family",
            )
        }
    latent = ROOT / Path(metadata["latent_cache"])
    if not latent.is_file():
        latent = Path(metadata["latent_cache"])
    with np.load(latent, allow_pickle=False) as payload:
        history = payload["history"].copy()
        goal = payload["goal"].copy()
    score = root / "universal_scores.npz"
    with np.load(score, allow_pickle=False) as payload:
        if not np.array_equal(value["record_key"].astype(str), payload["record_key"].astype(str)):
            raise RuntimeError(f"bank {letter} universal identity mismatch")
        universal = {
            key: payload[key].copy()
            for key in ("mean", "std", "dual_probability", "harm_probability", "rank_logit")
        }
    spatial_path = root / "spatial_rollout_v17.npz"
    with np.load(spatial_path, allow_pickle=False) as payload:
        if not np.array_equal(value["record_key"].astype(str), payload["record_key"].astype(str)):
            raise RuntimeError(f"bank {letter} spatial identity mismatch")
        frozen_spatial = payload["spatial_rollout_features"].copy()
    if history.shape != (100, 4, 256, 384) or goal.shape != (100, 256, 384):
        raise RuntimeError(f"bank {letter} latent contract changed")
    return {
        "letter": letter,
        "feature_path": feature,
        "latent_path": latent,
        "score_path": score,
        "spatial_path": spatial_path,
        "history": history,
        "goal": goal,
        "universal": universal,
        "frozen_spatial": frozen_spatial,
        **value,
    }


def checkpoint_paths(directory: Path) -> list[tuple[int, Path]]:
    values = [
        (seed, directory / f"seed{seed}/checkpoints/best.pt")
        for seed in (1701, 1702, 1703)
    ]
    missing = [str(path) for _, path in values if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"adapted WM checkpoints missing: {missing}")
    return values


def adapted_features(
    banks: list[dict[str, object]], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if args.checkpoint_dir is None:
        raise ValueError("adapted mode requires --checkpoint-dir")
    device = torch.device(args.device)
    scalar_seeds, spatial_seeds, checkpoints = [], [], []
    for seed, path in checkpoint_paths(args.checkpoint_dir):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**checkpoint["config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        scalar_banks, spatial_banks = [], []
        for bank in banks:
            history_np = bank["history"]
            goal_np = bank["goal"]
            actions_np = bank["actions_h4"]
            scalar_states, spatial_states = [], []
            for start in range(0, 100, args.batch_size):
                end = min(start + args.batch_size, 100)
                history = torch.from_numpy(history_np[start:end].astype(np.float32)).to(device)
                goal = torch.from_numpy(goal_np[start:end].astype(np.float32)).to(device)
                actions = torch.from_numpy(actions_np[start:end].astype(np.float32)).to(device)
                scalar_chunks, spatial_chunks = [], []
                for candidate_start in range(0, 17, args.candidate_chunk):
                    candidate_end = min(candidate_start + args.candidate_chunk, 17)
                    chunk = actions[:, candidate_start:candidate_end]
                    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
                    with torch.inference_mode(), autocast:
                        future = model(history, chunk)["predicted_tokens"].float()
                    with torch.inference_mode():
                        scalar, _ = latent_trajectory_features(future, goal, history[:, -1])
                        spatial = spatial_rollout_feature(future, goal)
                    scalar_chunks.append(scalar.cpu())
                    spatial_chunks.append(spatial.cpu())
                scalar_states.append(torch.cat(scalar_chunks, dim=1).numpy())
                spatial_states.append(torch.cat(spatial_chunks, dim=1).numpy())
            scalar_banks.append(np.concatenate(scalar_states).astype(np.float32))
            spatial_banks.append(np.concatenate(spatial_states).astype(np.float32))
            print(json.dumps({"seed": seed, "bank": bank["letter"], "status": "complete"}), flush=True)
        scalar_seeds.append(np.concatenate(scalar_banks))
        spatial_seeds.append(np.concatenate(spatial_banks))
        checkpoints.append({"seed": seed, "path": str(path), "sha256": file_sha256(path)})
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scalar = np.stack(scalar_seeds)
    spatial = np.stack(spatial_seeds)
    raw = np.concatenate((scalar.mean(0), scalar.std(0), scalar.min(0), scalar.max(0)), axis=-1).astype(np.float32)
    spatial_summary = np.concatenate((spatial.mean(0), spatial.std(0)), axis=-1).astype(np.float32)
    return raw, spatial_summary, checkpoints


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    banks = [load_bank(letter, root) for letter, root in sorted(bank_roots().items())]
    if args.mode == "adapted":
        raw, spatial, checkpoints = adapted_features(banks, args)
    else:
        raw = np.concatenate([bank["raw_ltwm_features"] for bank in banks]).astype(np.float32)
        spatial = np.concatenate([bank["frozen_spatial"] for bank in banks]).astype(np.float32)
        checkpoints = []
    if raw.shape != (2000, 17, 112) or spatial.shape != (2000, 17, 258):
        raise RuntimeError(f"consolidated feature contract failed: {raw.shape}, {spatial.shape}")
    arrays = {
        key: np.concatenate([bank[key] for bank in banks])
        for key in (
            "candidates", "actions_h4", "action_features", "static_features",
            "support_features", "majority_support", "record_key", "family",
        )
    }
    for key in ("mean", "std", "dual_probability", "harm_probability", "rank_logit"):
        arrays[f"universal_{key}"] = np.concatenate([bank["universal"][key] for bank in banks])
    arrays["raw_ltwm_features"] = raw
    arrays["spatial_rollout_features"] = spatial
    arrays["bank"] = np.concatenate([np.full(100, str(bank["letter"])) for bank in banks])
    metadata = {
        "schema": "recon_target_adapted_rollouts_v51",
        "mode": args.mode,
        "states": 2000,
        "banks": 20,
        "outcomes_accessed": False,
        "labels_in_output": False,
        "old_scorer_v2_run_on_adapted_wm": False,
        "checkpoints": checkpoints,
        "source_features": [
            {"bank": bank["letter"], "path": str(bank["feature_path"]), "sha256": file_sha256(bank["feature_path"])}
            for bank in banks
        ],
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    write_npz(args.output, **arrays)
    metadata["output"] = str(args.output)
    metadata["output_sha256"] = file_sha256(args.output)
    write_json(args.output.with_suffix(".audit.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
