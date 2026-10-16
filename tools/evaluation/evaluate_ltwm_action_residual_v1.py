#!/usr/bin/env python3
"""Pair a warm LTWM and one challenger on locked true future latents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools/recon"),
]

from latent_window_utils import ShardAwareBatchSampler  # noqa: E402
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402
from train_litetokenformer_wm_v2_4ctx import (  # noqa: E402
    TokenSequenceStreamingDataset,
    amp_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_and_verify(entry: dict, key: str = "sha256") -> Path:
    path = (ROOT / entry["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != entry[key]:
        raise RuntimeError(f"input hash changed: {path}")
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
) -> dict[str, np.ndarray]:
    value = prediction[:, 0].float()
    target = target.float()
    current = history[:, -1].float()[:, None]
    error = (value - target).square().mean(dim=(-1, -2))
    target_energy = target.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    predicted_residual = value - current
    target_residual = target - current
    residual_error = (
        (predicted_residual - target_residual).square().mean(dim=(-1, -2))
    )
    residual_energy = target_residual.square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        value.flatten(-2),
        target.flatten(-2),
        dim=-1,
    )
    residual_cosine = 1.0 - torch.nn.functional.cosine_similarity(
        predicted_residual.flatten(-2),
        target_residual.flatten(-2),
        dim=-1,
    )
    norm_ratio = (
        torch.linalg.vector_norm(predicted_residual.flatten(-2), dim=-1)
        / torch.linalg.vector_norm(target_residual.flatten(-2), dim=-1).clamp_min(
            1.0e-8
        )
    )
    return {
        "nmse": (error / target_energy).cpu().numpy(),
        "cosine": cosine.cpu().numpy(),
        "residual_nmse": (residual_error / residual_energy).cpu().numpy(),
        "residual_cosine": residual_cosine.cpu().numpy(),
        "norm_ratio": norm_ratio.cpu().numpy(),
    }


def aggregate(rows: list[dict[str, np.ndarray]]) -> dict[str, list[float]]:
    return {
        name: np.concatenate([row[name] for row in rows], axis=0)
        .mean(axis=0)
        .astype(np.float64)
        .tolist()
        for name in rows[0]
    }


def evaluate(
    label: str,
    checkpoint_path: Path,
    latent_root: Path,
    batch_size: int,
    amp: str,
    seed: int,
) -> dict:
    set_seed(seed)
    dataset = TokenSequenceStreamingDataset(latent_root, "val", 4, 4, 8)
    sampler = ShardAwareBatchSampler(
        dataset,
        batch_size,
        drop_last=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LiteTokenFormerWMV2(LiteTokenFormerConfig(**payload["config"])).cuda()
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    rows = {"true": [], "reversed": [], "noisy": []}
    wins = []
    response = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            history = batch["history_tokens"].cuda(non_blocking=True)
            target = batch["future_tokens"].cuda(non_blocking=True)
            actions = batch["actions"].cuda(non_blocking=True)
            scale = actions.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-4)
            with amp_context(amp):
                true_output = model(history, actions)["predicted_tokens"]
                reverse_output = model(history, actions.flip(1))["predicted_tokens"]
                noisy_output = model(
                    history,
                    actions + 0.5 * scale * torch.randn_like(actions),
                )["predicted_tokens"]
            local = {
                "true": metrics(true_output, target, history),
                "reversed": metrics(reverse_output, target, history),
                "noisy": metrics(noisy_output, target, history),
            }
            for name in rows:
                rows[name].append(local[name])
            true_residual = local["true"]["residual_nmse"]
            reverse_residual = local["reversed"]["residual_nmse"]
            wins.append((reverse_residual > true_residual).astype(np.float32))
            target_residual = (
                target.float() - history[:, -1].float()[:, None]
            )
            target_energy = (
                target_residual.square().mean(dim=(-1, -2)).clamp_min(1.0e-4)
            )
            predicted_response = (
                (
                    true_output[:, 0].float() - reverse_output[:, 0].float()
                )
                .square()
                .mean(dim=(-1, -2))
                / target_energy
            )
            response.append(predicted_response.cpu().numpy())
    merged = {name: aggregate(value) for name, value in rows.items()}
    correct_action_win_rate = (
        np.concatenate(wins, axis=0).mean(axis=0).astype(np.float64).tolist()
    )
    action_response_ratio = (
        np.concatenate(response, axis=0).mean(axis=0).astype(np.float64).tolist()
    )
    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "samples": len(dataset),
        "true": merged["true"],
        "reversed": merged["reversed"],
        "noisy": merged["noisy"],
        "correct_action_residual_win_rate": correct_action_win_rate,
        "true_vs_reversed_action_response_ratio": action_response_ratio,
        "reverse_absolute_nmse_gap": (
            np.asarray(merged["reversed"]["nmse"])
            - np.asarray(merged["true"]["nmse"])
        ).tolist(),
        "reverse_residual_nmse_gap": (
            np.asarray(merged["reversed"]["residual_nmse"])
            - np.asarray(merged["true"]["residual_nmse"])
        ).tolist(),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_training":
        raise RuntimeError("training contract is not frozen")
    warm_path = resolve_and_verify(contract["inputs"]["warm_checkpoint"])
    latent_entry = contract["inputs"]["latent_root"]
    latent_root = (ROOT / latent_entry["path"]).resolve()
    if sha256_file(latent_root / "metadata.json") != latent_entry["metadata_sha256"]:
        raise RuntimeError("latent metadata hash changed")
    if sha256_file(latent_root / "splits.json") != latent_entry["splits_sha256"]:
        raise RuntimeError("latent splits hash changed")
    challenger_path = args.challenger.resolve()
    if not challenger_path.is_file():
        raise FileNotFoundError(challenger_path)
    seed = int(contract["training"]["seed"])
    warm = evaluate(
        "warm",
        warm_path,
        latent_root,
        args.batch_size,
        args.amp,
        seed,
    )
    challenger = evaluate(
        "challenger",
        challenger_path,
        latent_root,
        args.batch_size,
        args.amp,
        seed,
    )
    warm_true = warm["true"]
    challenger_true = challenger["true"]
    checks = {
        "mean_absolute_nmse": np.mean(challenger_true["nmse"])
        < np.mean(warm_true["nmse"]),
        "h4_absolute_nmse": challenger_true["nmse"][3] <= warm_true["nmse"][3],
        "mean_residual_nmse": np.mean(challenger_true["residual_nmse"])
        < np.mean(warm_true["residual_nmse"]),
        "h4_residual_nmse": challenger_true["residual_nmse"][3]
        < warm_true["residual_nmse"][3],
        "h4_correct_action_win_rate": challenger[
            "correct_action_residual_win_rate"
        ][3]
        > warm["correct_action_residual_win_rate"][3],
    }
    report = {
        "schema": "ltwm_pure_future_latent_action_residual_evaluation_v1",
        "status": "complete",
        "decision": (
            "go_world_model_checkpoint_improved"
            if all(checks.values())
            else "no_go_world_model_gate_failed"
        ),
        "go": all(checks.values()),
        "checks": checks,
        "validation_routes": contract["inputs"]["latent_root"]["validation_routes"],
        "validation_windows": warm["samples"],
        "warm": warm,
        "challenger": challenger,
        "delta_challenger_minus_warm": {
            "mean_absolute_nmse": float(
                np.mean(challenger_true["nmse"]) - np.mean(warm_true["nmse"])
            ),
            "h4_absolute_nmse": challenger_true["nmse"][3] - warm_true["nmse"][3],
            "mean_residual_nmse": float(
                np.mean(challenger_true["residual_nmse"])
                - np.mean(warm_true["residual_nmse"])
            ),
            "h4_residual_nmse": challenger_true["residual_nmse"][3]
            - warm_true["residual_nmse"][3],
            "h4_correct_action_win_rate": challenger[
                "correct_action_residual_win_rate"
            ][3]
            - warm["correct_action_residual_win_rate"][3],
        },
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "outcome_labels_used": false,
        "outer_test_used": false,
        "sealed_validation_opened": false,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
