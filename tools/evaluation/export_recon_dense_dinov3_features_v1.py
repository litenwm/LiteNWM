#!/usr/bin/env python3
"""Export DINOv3/LTWM features for the frozen RECON dense K17 pools.

The legacy dense artifacts contribute only the frozen candidate pool, the
model-independent action/past-motion features, record keys, and family IDs.
All visual and predicted-future features are recomputed with the current
DINOv3 adapter and the three paired LTWM/scorer-v2 seeds.  Labels are copied
only when ``--include-labels`` is explicitly supplied (dense-train only).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.costs import latent_trajectory_features  # noqa: E402
from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402
from planning.trajectory_constraints import (  # noqa: E402
    ActionSupportStats,
    evaluate_trajectory_constraints,
)
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


HISTORY = 4
HORIZON = 4
ACTION_DIM = 3
POOL_CHANNELS = 64
SCALAR_LATENT_DIM = 7 * HORIZON
DEFAULT_MODEL_DIR = Path(
    "/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
)
DEFAULT_TRAJECTORY_ROOT = Path("/home/rainbow/nwm/data/recon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/project",
    )
    parser.add_argument("--include-labels", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dinov3-batch-size", type=int, default=8)
    parser.add_argument("--wm-batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=17)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_source(path: Path, include_labels: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("candidates", "action_features", "record_key", "family")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise RuntimeError(f"source is missing arrays: {missing}")
        result = {key: payload[key].copy() for key in required}
        if include_labels:
            if "targets" not in payload.files:
                raise RuntimeError("--include-labels requires source targets")
            result["targets"] = payload["targets"].copy()
    candidates = result["candidates"].astype(np.float32)
    count, candidate_count = candidates.shape[:2]
    if candidates.shape != (count, candidate_count, HORIZON, 2):
        raise RuntimeError(f"invalid candidate shape: {candidates.shape}")
    if candidate_count != 17:
        raise RuntimeError("dense migration requires K17 (incumbent + NoMaD x16)")
    if result["action_features"].shape[:2] != (count, candidate_count):
        raise RuntimeError("action feature/candidate dimensions differ")
    if result["record_key"].shape != (count,) or result["family"].shape != (count,):
        raise RuntimeError("record/family dimensions differ")
    if include_labels and result["targets"].shape != (count, candidate_count, 8):
        raise RuntimeError("expected target shape [N,17,8]")
    result["candidates"] = candidates
    result["action_features"] = result["action_features"].astype(np.float32)
    result["record_key"] = result["record_key"].astype(str)
    result["family"] = result["family"].astype(str)
    return result


def parse_record_key(value: str) -> tuple[str, int]:
    trajectory, separator, timestep = str(value).rpartition(":")
    if not separator or not trajectory or not timestep.isdigit():
        raise ValueError(f"invalid record key: {value!r}")
    if trajectory in {".", ".."} or "/" in trajectory or "\\" in trajectory:
        raise ValueError(f"unsafe trajectory id: {trajectory!r}")
    return trajectory, int(timestep)


def image_path(root: Path, trajectory: str, timestep: int) -> Path:
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / trajectory / f"{timestep}.jpg").resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"image path escapes trajectory root: {path}") from error
    return path


def preprocess(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (
        array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def latent_binding(
    *,
    split: str,
    source: Path,
    source_data: dict[str, np.ndarray],
    trajectory_root: Path,
    model_dir: Path,
    adapter_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "schema": "recon_dense_targeted_dinov3_latents_v1",
        "split": split,
        "states": int(len(source_data["record_key"])),
        "candidate_count": int(source_data["candidates"].shape[1]),
        "source_sha256": file_sha256(source),
        "candidate_array_sha256": array_sha256(source_data["candidates"]),
        "record_key_sha256": array_sha256(source_data["record_key"]),
        "trajectory_root": str(trajectory_root.resolve(strict=True)),
        "dinov3_weight_sha256": file_sha256(model_dir / "model.safetensors"),
        "adapter_checkpoint_sha256": file_sha256(adapter_checkpoint),
        "preprocess": "RGB_bicubic_224_imagenet_norm",
        "history": HISTORY,
        "goal_offset": HORIZON,
        "labels_accessed": False,
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def encode_latents(
    source_data: dict[str, np.ndarray],
    *,
    binding: dict[str, Any],
    trajectory_root: Path,
    model_dir: Path,
    adapter_checkpoint: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if output.is_file():
        arrays = load_npz(output)
        actual = json.loads(str(arrays["metadata"].item()))
        if actual != binding:
            raise RuntimeError(f"latent cache provenance mismatch: {output}")
        return arrays

    adapter_payload = torch.load(
        adapter_checkpoint, map_location="cpu", weights_only=False
    )
    if adapter_payload.get("dinov3_weight_sha256") != binding["dinov3_weight_sha256"]:
        raise RuntimeError("adapter/DINOv3 weight mismatch")
    adapter = DinoV3TokenAdapter(DinoV3AdapterConfig(**adapter_payload["config"]))
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    from transformers import AutoModel

    backbone = (
        AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
        )
        .eval()
        .to(device)
        .requires_grad_(False)
    )
    adapter = adapter.eval().to(device).requires_grad_(False)

    paths: list[Path] = []
    path_index: dict[str, int] = {}
    state_keys: list[list[str]] = []
    for record_key in source_data["record_key"]:
        trajectory, current = parse_record_key(str(record_key))
        times = list(range(current - HISTORY + 1, current + 1)) + [current + HORIZON]
        keys = []
        for timestep in times:
            path = image_path(trajectory_root, trajectory, timestep)
            key = str(path)
            if key not in path_index:
                path_index[key] = len(paths)
                paths.append(path)
            keys.append(key)
        state_keys.append(keys)

    encoded_batches = []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        images = torch.stack(
            [preprocess(path) for path in paths[start : start + batch_size]]
        ).to(device)
        with torch.inference_mode():
            hidden = backbone(pixel_values=images).last_hidden_state
            adapted = adapter(extract_dinov3_patch_tokens(hidden))
        encoded_batches.append(adapted.cpu().numpy().astype(np.float16))
        print(
            canonical_json(
                {
                    "stage": "dense_targeted_dinov3_encode",
                    "split": binding["split"],
                    "completed_images": min(start + batch_size, len(paths)),
                    "total_images": len(paths),
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    encoded = np.concatenate(encoded_batches)
    history = []
    goal = []
    for keys in state_keys:
        values = encoded[[path_index[key] for key in keys]]
        history.append(values[:HISTORY].reshape(HISTORY, 256, 384))
        goal.append(values[HISTORY].reshape(256, 384))
    arrays = {
        "history": np.stack(history).astype(np.float16),
        "goal": np.stack(goal).astype(np.float16),
        "metadata": np.asarray(canonical_json(binding)),
    }
    write_npz_atomic(output, **arrays)
    del backbone, adapter, encoded, encoded_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def checkpoint_pairs(checkpoint_root: Path) -> list[dict[str, Any]]:
    result = []
    for seed in (1701, 1702, 1703):
        wm = checkpoint_root / "ltwm_expansion" / f"seed{seed}" / "best.pt"
        scorer = checkpoint_root / "scorer_v2" / f"seed{seed}" / "best.pt"
        result.append(
            {
                "seed": seed,
                "wm": wm,
                "scorer": scorer,
                "wm_sha256": file_sha256(wm),
                "scorer_sha256": file_sha256(scorer),
            }
        )
    return result


def support_stats(value: dict[str, Any]) -> ActionSupportStats:
    if "lower" in value and "upper" in value:
        return ActionSupportStats(
            lower=torch.as_tensor(value["lower"], dtype=torch.float32),
            upper=torch.as_tensor(value["upper"], dtype=torch.float32),
        )
    return ActionSupportStats(
        lower=torch.as_tensor(value["lower_p01"], dtype=torch.float32),
        upper=torch.as_tensor(value["upper_p99"], dtype=torch.float32),
        mean=torch.as_tensor(value["mean"], dtype=torch.float32),
        covariance_inverse=torch.as_tensor(
            value["covariance_inverse"], dtype=torch.float32
        ),
        mahalanobis_squared_max=float(value["mahalanobis_squared_p99"]),
        support_actions=torch.as_tensor(value["support_actions"], dtype=torch.float32),
        knn_k=int(value["knn_k"]),
        knn_distance_max=float(value["knn_distance_p99"]),
        roughness_max=float(value["roughness_p99"]),
        jerk_max=float(value["jerk_p99"]),
        curvature_max=float(value["curvature_p99"]),
        parent_trust_radius=float(value["parent_trust_radius"]),
    )


def pool_static_features(history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return candidate-independent DINOv3 route/history features [B,210]."""
    history_mean = history.mean(dim=-2)
    goal_mean = goal.mean(dim=-2)
    width = history_mean.shape[-1] // POOL_CHANNELS
    history_pool = history_mean.reshape(
        history.shape[0], HISTORY, POOL_CHANNELS, width
    ).mean(dim=-1)
    goal_pool = goal_mean.reshape(goal.shape[0], POOL_CHANNELS, width).mean(dim=-1)
    current_pool = history_pool[:, -1]
    route = torch.cat((current_pool, goal_pool, current_pool - goal_pool), dim=-1)

    goal_expanded = goal[:, None].expand(-1, HISTORY, -1, -1)
    goal_energy = goal_expanded.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    goal_nmse = (history - goal_expanded).square().mean(dim=(-1, -2)) / goal_energy
    goal_cosine = 1.0 - F.cosine_similarity(
        history.flatten(-2), goal_expanded.flatten(-2), dim=-1
    )
    norm_ratio = torch.linalg.vector_norm(history, dim=(-1, -2)) / torch.linalg.vector_norm(
        goal_expanded, dim=(-1, -2)
    ).clamp_min(1.0e-8)
    step = history[:, 1:] - history[:, :-1]
    current_energy = history[:, :-1].square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    step_energy = step.square().mean(dim=(-1, -2)) / current_energy
    goal_direction = goal[:, None] - history[:, :-1]
    direction_cosine = F.cosine_similarity(
        step.flatten(-2), goal_direction.flatten(-2), dim=-1
    )
    scalar = torch.cat(
        (goal_nmse, goal_cosine, norm_ratio, step_energy, direction_cosine), dim=-1
    )
    value = torch.cat((route, scalar), dim=-1)
    if value.shape[-1] != 210 or not torch.isfinite(value).all():
        raise RuntimeError("static DINOv3 feature contract failed")
    return value


def export_features(
    source_data: dict[str, np.ndarray],
    latents: dict[str, np.ndarray],
    *,
    pairs: list[dict[str, Any]],
    split: str,
    source: Path,
    latent_path: Path,
    output: Path,
    include_labels: bool,
    device: torch.device,
    batch_size: int,
    candidate_chunk: int,
) -> dict[str, Any]:
    actions_np = nomad_waypoints_to_local_actions(
        source_data["candidates"], dataset="recon"
    ).astype(np.float32)
    actions_cpu = torch.from_numpy(actions_np)
    state_static = []
    for start in range(0, len(actions_np), batch_size):
        end = min(start + batch_size, len(actions_np))
        history = torch.from_numpy(latents["history"][start:end].astype(np.float32))
        goal = torch.from_numpy(latents["goal"][start:end].astype(np.float32))
        state_static.append(pool_static_features(history, goal).numpy().astype(np.float32))
    state_static_array = np.concatenate(state_static)

    scalar_by_seed = []
    ate_by_seed = []
    rtp_by_seed = []
    support_by_seed = []
    started = time.perf_counter()
    for pair in pairs:
        wm_checkpoint = torch.load(pair["wm"], map_location="cpu", weights_only=False)
        scorer_checkpoint = torch.load(
            pair["scorer"], map_location="cpu", weights_only=False
        )
        if scorer_checkpoint.get("world_model_checkpoint_sha256") != pair["wm_sha256"]:
            raise RuntimeError("paired scorer/WM provenance mismatch")
        wm = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**wm_checkpoint["config"])
        ).to(device)
        wm.load_state_dict(wm_checkpoint["model"], strict=True)
        wm.eval().requires_grad_(False)
        scorer = TrajectoryQualityScorerV2(
            TrajectoryQualityScorerConfig(**scorer_checkpoint["model_config"])
        ).to(device)
        scorer.load_state_dict(scorer_checkpoint["model"], strict=True)
        scorer.eval().requires_grad_(False)

        support = evaluate_trajectory_constraints(
            actions_cpu,
            support_stats(scorer_checkpoint["action_support"]),
            parent_actions=actions_cpu,
        ).valid.numpy()
        support[:, 0] = True
        support_by_seed.append(support)
        seed_scalar = []
        seed_ate = []
        seed_rtp = []
        for start in range(0, len(actions_np), batch_size):
            end = min(start + batch_size, len(actions_np))
            history = torch.from_numpy(
                latents["history"][start:end].astype(np.float32)
            ).to(device)
            goal = torch.from_numpy(latents["goal"][start:end].astype(np.float32)).to(
                device
            )
            actions = actions_cpu[start:end].to(device)
            scalar_chunks = []
            ate_chunks = []
            rtp_chunks = []
            for candidate_start in range(0, actions.shape[1], candidate_chunk):
                candidate_end = min(candidate_start + candidate_chunk, actions.shape[1])
                chunk = actions[:, candidate_start:candidate_end]
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with torch.inference_mode(), autocast:
                    future = wm(history, chunk)["predicted_tokens"]
                future = future.float()
                with torch.inference_mode():
                    scalar, _ = latent_trajectory_features(
                        future, goal, history[:, -1]
                    )
                    prediction = scorer(
                        current_latent=history[:, -1],
                        goal_latent=goal,
                        candidate_actions=chunk,
                        predicted_future=future,
                    )
                scalar_chunks.append(scalar.cpu())
                ate_chunks.append(prediction.ate.cpu())
                rtp_chunks.append(prediction.rtp.cpu())
            seed_scalar.append(torch.cat(scalar_chunks, dim=1).numpy().astype(np.float32))
            seed_ate.append(torch.cat(ate_chunks, dim=1).numpy().astype(np.float32))
            seed_rtp.append(torch.cat(rtp_chunks, dim=1).numpy().astype(np.float32))
            print(
                canonical_json(
                    {
                        "stage": "dense_dinov3_ltwm_features",
                        "split": split,
                        "seed": pair["seed"],
                        "completed_states": end,
                        "total_states": len(actions_np),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        scalar_by_seed.append(np.concatenate(seed_scalar))
        ate_by_seed.append(np.concatenate(seed_ate))
        rtp_by_seed.append(np.concatenate(seed_rtp))
        del wm, scorer, wm_checkpoint, scorer_checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scalar_seed = np.stack(scalar_by_seed).astype(np.float32)
    ate_seed = np.stack(ate_by_seed).astype(np.float32)
    rtp_seed = np.stack(rtp_by_seed).astype(np.float32)
    support_seed = np.stack(support_by_seed)
    majority_support = support_seed.sum(axis=0) >= math.ceil(len(pairs) / 2)
    majority_support[:, 0] = True
    # Keep the independently retrained LTWM representation separate from the
    # frozen scorer-v2 predictions. Support is provenance/audit information and
    # must never silently become a learned feature.
    raw_ltwm_features = np.concatenate(
        (
            scalar_seed.mean(axis=0),
            scalar_seed.std(axis=0),
            scalar_seed.min(axis=0),
            scalar_seed.max(axis=0),
        ),
        axis=-1,
    ).astype(np.float32)
    scorer_v2_aux_features = np.concatenate(
        (
            np.moveaxis(ate_seed, 0, -1),
            np.moveaxis(rtp_seed, 0, -1),
            ate_seed.mean(axis=0)[..., None],
            ate_seed.std(axis=0)[..., None],
            rtp_seed.mean(axis=0)[..., None],
            rtp_seed.std(axis=0)[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
    support_features = support_seed.mean(axis=0)[..., None].astype(np.float32)
    if raw_ltwm_features.shape[-1] != 112:
        raise RuntimeError(f"raw LTWM dimension changed: {raw_ltwm_features.shape}")
    if scorer_v2_aux_features.shape[-1] != 10:
        raise RuntimeError(f"scorer-v2 auxiliary dimension changed: {scorer_v2_aux_features.shape}")
    if support_features.shape[-1] != 1:
        raise RuntimeError(f"support dimension changed: {support_features.shape}")

    metadata = {
        "schema": "recon_dense_dinov3_feature_artifact_v1",
        "created_at": utc_now(),
        "split": split,
        "states": int(len(actions_np)),
        "candidates": int(actions_np.shape[1]),
        "candidate_pool": "trimmed_mean75_incumbent_plus_nomad_x16",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "latent_cache": str(latent_path),
        "latent_cache_sha256": file_sha256(latent_path),
        "feature_dimensions": {
            "action": int(source_data["action_features"].shape[-1]),
            "static": int(state_static_array.shape[-1]),
            "raw_ltwm": int(raw_ltwm_features.shape[-1]),
            "scorer_v2_aux": int(scorer_v2_aux_features.shape[-1]),
            "support": int(support_features.shape[-1]),
        },
        "feature_separation": {
            "raw_ltwm_features": (
                "three-seed LTWM rollout scalar features summarized by "
                "mean/std/min/max; no scorer-v2 predictions"
            ),
            "scorer_v2_aux_features": (
                "frozen scorer-v2 ATE/RTP predictions for three seeds plus "
                "their mean/std; stacked ablation only"
            ),
            "support_features": (
                "three-seed action-support acceptance fraction; audit-only, excluded from learned features"
            ),
        },
        "labels_accessed": bool(include_labels),
        "labels_in_output": bool(include_labels),
        "old_visual_or_dwm_features_migrated": False,
        "old_action_past_features_reused": True,
        "dinov3_visual_features_recomputed": True,
        "ltwm_future_features_recomputed": True,
        "fixed_pool_only": True,
        "cem": False,
        "seeds": [
            {
                "seed": pair["seed"],
                "wm_sha256": pair["wm_sha256"],
                "scorer_sha256": pair["scorer_sha256"],
            }
            for pair in pairs
        ],
    }
    arrays: dict[str, np.ndarray] = {
        "candidates": source_data["candidates"].astype(np.float32),
        "actions_h4": actions_np,
        "action_features": source_data["action_features"].astype(np.float32),
        "static_features": np.repeat(
            state_static_array[:, None], actions_np.shape[1], axis=1
        ).astype(np.float32),
        "raw_ltwm_features": raw_ltwm_features,
        "scorer_v2_aux_features": scorer_v2_aux_features,
        "support_features": support_features,
        "ate_by_seed": ate_seed,
        "rtp_by_seed": rtp_seed,
        "support_by_seed": support_seed,
        "majority_support": majority_support,
        "record_key": source_data["record_key"],
        "family": source_data["family"],
        "metadata": np.asarray(canonical_json(metadata)),
    }
    if include_labels:
        arrays["targets"] = source_data["targets"].astype(np.float32)
    write_npz_atomic(output, **arrays)
    metadata["output"] = str(output)
    metadata["output_sha256"] = file_sha256(output)
    write_json_atomic(output.with_suffix(".audit.json"), metadata)
    return metadata


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if min(args.dinov3_batch_size, args.wm_batch_size, args.candidate_chunk) <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / f"{args.split}_features.npz"
    if feature_path.exists():
        raise FileExistsError(feature_path)
    source_data = load_source(args.source, args.include_labels)
    binding = latent_binding(
        split=args.split,
        source=args.source,
        source_data=source_data,
        trajectory_root=args.trajectory_root,
        model_dir=args.model_dir,
        adapter_checkpoint=args.adapter_checkpoint,
    )
    latent_path = args.output_dir / f"{args.split}_latents.npz"
    device = torch.device(args.device)
    latents = encode_latents(
        source_data,
        binding=binding,
        trajectory_root=args.trajectory_root,
        model_dir=args.model_dir,
        adapter_checkpoint=args.adapter_checkpoint,
        output=latent_path,
        device=device,
        batch_size=args.dinov3_batch_size,
    )
    metadata = export_features(
        source_data,
        latents,
        pairs=checkpoint_pairs(args.checkpoint_root),
        split=args.split,
        source=args.source,
        latent_path=latent_path,
        output=feature_path,
        include_labels=args.include_labels,
        device=device,
        batch_size=args.wm_batch_size,
        candidate_chunk=args.candidate_chunk,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
