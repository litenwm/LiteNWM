#!/usr/bin/env python3
"""Export compact frozen high-resolution dual-LTWM features for union65.

This is a spent-development performance probe.  It encodes only the four
history images and the already-known goal image for each frozen state, runs
the frozen action-conditioned world model on the existing union65 actions,
and writes label-free candidate-relative rollout features.  It never reads a
future image, candidate outcome, or sealed-validation artifact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
)
from recon_latents.dinov3_multilayer_dual import (  # noqa: E402
    FrozenMultiLayerDualLatentEncoder,
    MultiLayerDualLatentConfig,
    imagenet_normalize,
)
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)


LEGACY_MANIFEST = (
    ROOT
    / "outputs/cem_development_v1"
    / "scand_existing_train_upperbound_pilot_v1_corrected"
    / "pilot_manifest.windows.jsonl"
)
FRESH_MANIFEST = (
    ROOT
    / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
    / "manifest.windows.jsonl"
)
NEW_MANIFEST = (
    ROOT
    / "outputs/cem_development_v1"
    / "scand_new_routefit10_v4_development_pilot_v2_strict_identity"
    / "pilot_manifest.windows.jsonl"
)
FRESH_IMAGE_ROOT = Path("/home/rainbow/nwm/data/scand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-model", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--state-batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=13)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_key(row: dict[str, Any]) -> str:
    if "record_key" in row:
        return str(row["record_key"])
    current = row.get("curr_time", row.get("current_t"))
    return f"{row['trajectory_id']}:{current}"


def image_paths(row: dict[str, Any]) -> tuple[list[Path], Path]:
    context = row.get("context_image_paths")
    goal = row.get("goal_image_path")
    if context is None:
        trajectory_root = FRESH_IMAGE_ROOT / str(row["trajectory_id"])
        context = [
            str(trajectory_root / f"{int(timestep)}.jpg")
            for timestep in row["history_timesteps"]
        ]
    if goal is None:
        trajectory_root = FRESH_IMAGE_ROOT / str(row["trajectory_id"])
        goal = str(trajectory_root / f"{int(row['goal_t'])}.jpg")
    return [Path(value) for value in context], Path(goal)


def preprocess(
    path: Path,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (image_width, image_height),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return imagenet_normalize(tensor)


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def reconstruct_actions(data: dict[str, np.ndarray]) -> np.ndarray:
    names = data["feature_names"].astype(str).tolist()
    index = {name: position for position, name in enumerate(names)}
    delta_index = [index[f"delta_actions_h4_{value}"] for value in range(12)]
    formal_index = [index[f"formal_actions_h4_{value}"] for value in range(12)]
    feature = data["runtime_pair_feature"].astype(np.float32)
    actions = feature[..., delta_index] + feature[..., formal_index]
    actions = actions.reshape(len(feature), feature.shape[1], 4, 3)
    formal = data["formal_v70_index"].astype(np.int64)
    formal_feature = feature[np.arange(len(feature)), formal][:, formal_index]
    require(
        np.array_equal(
            actions[np.arange(len(feature)), formal].reshape(len(feature), 12),
            formal_feature,
        ),
        "formal action reconstruction changed",
    )
    require(np.isfinite(actions).all(), "actions contain NaN/Inf")
    return actions.astype(np.float32)


def branch_features(
    prediction: torch.Tensor,
    formal_prediction: torch.Tensor,
    current: torch.Tensor,
    goal: torch.Tensor,
    logvar: torch.Tensor,
    formal_logvar: torch.Tensor,
) -> list[torch.Tensor]:
    """Return eight H1-H4 candidate-minus-formal rollout channels."""

    prediction = prediction.float()
    formal_prediction = formal_prediction.float()
    current = current.float()[:, None, None]
    goal = goal.float()[:, None, None]
    formal = formal_prediction.expand_as(prediction)
    formal_lv = formal_logvar.float().expand_as(logvar)
    epsilon = 1.0e-8

    goal_energy = goal.square().mean(dim=(-1, -2)).clamp_min(epsilon)
    current_energy = current.square().mean(dim=(-1, -2)).clamp_min(epsilon)
    current_goal = (
        (current - goal).square().mean(dim=(-1, -2)) / goal_energy
    )

    def metrics(value: torch.Tensor, uncertainty: torch.Tensor) -> list[torch.Tensor]:
        goal_nmse = (
            (value - goal).square().mean(dim=(-1, -2)) / goal_energy
        )
        goal_cosine = 1.0 - F.cosine_similarity(
            value.flatten(-2),
            goal.expand_as(value).flatten(-2),
            dim=-1,
        )
        current_nmse = (
            (value - current).square().mean(dim=(-1, -2)) / current_energy
        )
        progress = current_goal - goal_nmse
        delta = value - current
        delta_energy = torch.log(
            delta.square().mean(dim=(-1, -2)).clamp_min(epsilon)
        )
        goal_direction = 1.0 - F.cosine_similarity(
            delta.flatten(-2),
            (goal - current).expand_as(delta).flatten(-2),
            dim=-1,
        )
        uncertainty_mean = uncertainty.float().mean(dim=-1)
        uncertainty_std = uncertainty.float().std(dim=-1, unbiased=False)
        return [
            goal_nmse,
            goal_cosine,
            current_nmse,
            progress,
            delta_energy,
            goal_direction,
            uncertainty_mean,
            uncertainty_std,
        ]

    candidate_metrics = metrics(prediction, logvar)
    formal_metrics = metrics(formal, formal_lv)
    return [
        candidate - formal_value
        for candidate, formal_value in zip(candidate_metrics, formal_metrics)
    ]


def pair_features(
    prediction: torch.Tensor,
    formal_prediction: torch.Tensor,
) -> list[torch.Tensor]:
    prediction = prediction.float()
    formal = formal_prediction.float().expand_as(prediction)
    energy = formal.square().mean(dim=(-1, -2)).clamp_min(1.0e-8)
    nmse = (prediction - formal).square().mean(dim=(-1, -2)) / energy
    cosine = 1.0 - F.cosine_similarity(
        prediction.flatten(-2),
        formal.flatten(-2),
        dim=-1,
    )
    return [nmse, cosine]


def feature_names() -> np.ndarray:
    names: list[str] = []
    channels = (
        "goal_nmse",
        "goal_cosine",
        "current_nmse",
        "goal_progress",
        "delta_log_energy",
        "goal_direction_cosine_distance",
        "logvar_mean",
        "logvar_std",
    )
    for branch in ("semantic", "geometry"):
        for channel in channels:
            for horizon in range(1, 5):
                names.append(f"delta_{branch}_{channel}_h{horizon}")
    for branch in ("semantic", "geometry"):
        for channel in ("candidate_formal_nmse", "candidate_formal_cosine"):
            for horizon in range(1, 5):
                names.append(f"{branch}_{channel}_h{horizon}")
    return np.asarray(names)


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    require(args.state_batch_size > 0, "state batch size must be positive")
    require(args.candidate_chunk > 0, "candidate chunk must be positive")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    table_path = args.table.resolve()
    checkpoint_path = args.checkpoint.resolve()
    require("validation" not in str(table_path).lower(), "validation table rejected")
    data = load_npz(table_path)
    require(
        data["runtime_pair_feature"].shape == (800, 65, 351),
        "frozen common table shape changed",
    )
    rows = (
        load_jsonl(LEGACY_MANIFEST)
        + load_jsonl(FRESH_MANIFEST)
        + load_jsonl(NEW_MANIFEST)
    )
    require(len(rows) == 800, "expected 800 spent-development manifest rows")
    keys = np.asarray([record_key(row) for row in rows])
    require(
        np.array_equal(keys, data["record_key"].astype(str)),
        "manifest/table record order changed",
    )
    state_images = [image_paths(row) for row in rows]
    missing = [
        str(path)
        for history, goal in state_images
        for path in (*history, goal)
        if not path.is_file()
    ]
    require(not missing, f"missing input images, first={missing[:3]}")
    actions = reconstruct_actions(data)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(checkpoint.get("outcome_gradients") is False, "LTWM outcome leakage")
    require(checkpoint.get("sealed_test_opened") is False, "LTWM sealed test opened")
    wm_config = HighResDualLatentWMConfig(**checkpoint["config"])
    world_model = HighResMultiLayerDualLatentWM(wm_config)
    world_model.load_state_dict(checkpoint["model"], strict=True)
    world_model = world_model.eval().to(device).requires_grad_(False)

    adapter_payload = torch.load(
        args.adapter_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    adapter = DinoV3TokenAdapter(
        DinoV3AdapterConfig(**adapter_payload["config"])
    )
    adapter.load_state_dict(adapter_payload["model"], strict=True)
    representation_config = MultiLayerDualLatentConfig(
        image_height=int(checkpoint["representation_contract"]["image_height"]),
        image_width=int(checkpoint["representation_contract"]["image_width"]),
        patch_size=int(checkpoint["representation_contract"]["patch_size"]),
        semantic_block=int(
            checkpoint["representation_contract"]["semantic_block"]
        ),
        geometry_blocks=tuple(
            checkpoint["representation_contract"]["geometry_blocks"]
        ),
        backbone_dim=int(
            checkpoint["representation_contract"]["backbone_dim"]
        ),
        semantic_dim=int(
            checkpoint["representation_contract"]["semantic_dim"]
        ),
        geometry_dim=int(
            checkpoint["representation_contract"]["geometry_dim"]
        ),
        geometry_projection_seed=int(
            checkpoint["representation_contract"]["geometry_projection_seed"]
        ),
    )
    representation = FrozenMultiLayerDualLatentEncoder(
        adapter,
        representation_config,
    ).eval().to(device).requires_grad_(False)
    require(
        representation.geometry_projection_sha256
        == checkpoint["representation_contract"]["geometry_projection_sha256"],
        "geometry representation contract changed",
    )

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        args.dinov3_model,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device).requires_grad_(False)

    names = feature_names()
    features = np.empty((800, 65, len(names)), dtype=np.float32)
    started = time.perf_counter()
    peak_memory = 0
    for start in range(0, 800, args.state_batch_size):
        end = min(start + args.state_batch_size, 800)
        flat_paths = [
            path
            for history, goal in state_images[start:end]
            for path in (*history, goal)
        ]
        images = torch.stack(
            [
                preprocess(
                    path,
                    image_height=representation_config.image_height,
                    image_width=representation_config.image_width,
                )
                for path in flat_paths
            ]
        ).to(device)
        with torch.inference_mode(), amp_context(device, args.amp):
            hidden_states = backbone(
                pixel_values=images,
                output_hidden_states=True,
            ).hidden_states
            encoded = representation(hidden_states)
        local_states = end - start
        semantic = encoded["semantic_tokens"].reshape(
            local_states,
            5,
            wm_config.num_tokens,
            wm_config.semantic_dim,
        )
        geometry = encoded["geometry_tokens"].reshape(
            local_states,
            5,
            wm_config.num_tokens,
            wm_config.geometry_dim,
        )
        semantic_history = semantic[:, :4]
        geometry_history = geometry[:, :4]
        semantic_goal = semantic[:, 4]
        geometry_goal = geometry[:, 4]
        local_actions = torch.from_numpy(actions[start:end]).to(device)
        local_formal = torch.from_numpy(
            data["formal_v70_index"][start:end].astype(np.int64)
        ).to(device)
        formal_actions = local_actions[
            torch.arange(local_states, device=device),
            local_formal,
        ][:, None]
        with torch.inference_mode(), amp_context(device, args.amp):
            formal_output = world_model(
                semantic_history,
                geometry_history,
                formal_actions,
            )
        for candidate_start in range(0, 65, args.candidate_chunk):
            candidate_end = min(candidate_start + args.candidate_chunk, 65)
            with torch.inference_mode(), amp_context(device, args.amp):
                output = world_model(
                    semantic_history,
                    geometry_history,
                    local_actions[:, candidate_start:candidate_end],
                )
            blocks = []
            for branch, current, goal in (
                ("semantic", semantic_history[:, -1], semantic_goal),
                ("geometry", geometry_history[:, -1], geometry_goal),
            ):
                prediction = output[f"predicted_{branch}_tokens"]
                formal_prediction = formal_output[
                    f"predicted_{branch}_tokens"
                ]
                blocks.extend(
                    branch_features(
                        prediction,
                        formal_prediction,
                        current,
                        goal,
                        output[f"{branch}_logvar"],
                        formal_output[f"{branch}_logvar"],
                    )
                )
            for branch in ("semantic", "geometry"):
                blocks.extend(
                    pair_features(
                        output[f"predicted_{branch}_tokens"],
                        formal_output[f"predicted_{branch}_tokens"],
                    )
                )
            chunk_feature = torch.cat(blocks, dim=-1)
            features[
                start:end,
                candidate_start:candidate_end,
            ] = chunk_feature.cpu().numpy().astype(np.float32)
        formal_index = data["formal_v70_index"][start:end].astype(np.int64)
        features[
            np.arange(start, end),
            formal_index,
        ] = 0.0
        del hidden_states, encoded, images, formal_output
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        print(
            canonical_json(
                {
                    "completed_states": end,
                    "states": 800,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    require(np.isfinite(features).all(), "exported features contain NaN/Inf")
    formal = data["formal_v70_index"].astype(np.int64)
    require(
        float(np.abs(features[np.arange(800), formal]).max()) == 0.0,
        "formal relative features must be exact zero",
    )
    metadata = {
        "schema": "scand_highres_dual_union65_features_v1",
        "status": "complete_label_free_spent_development",
        "states": 800,
        "candidates": 65,
        "feature_dimensions": int(features.shape[-1]),
        "feature_names": names.tolist(),
        "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "dinov3_model_sha256": sha256_file(
            args.dinov3_model / "model.safetensors"
        ),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "manifest_hashes": {
            "legacy": sha256_file(LEGACY_MANIFEST),
            "fresh": sha256_file(FRESH_MANIFEST),
            "new": sha256_file(NEW_MANIFEST),
        },
        "input_contract": "four_history_images_plus_goal_image_only",
        "future_images_read": False,
        "candidate_outcomes_read": False,
        "ltwm_trained": False,
        "sealed_validation_opened": False,
        "formal_v70_mutated": False,
        "cem_mutated": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"].astype(str),
            formal_v70_index=formal,
            highres_dual_pair_feature=features,
            feature_names=names,
            metadata=np.asarray(canonical_json(metadata)),
        )
    temporary.replace(args.output)
    print(
        canonical_json(
            {
                "status": metadata["status"],
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "feature_dimensions": int(features.shape[-1]),
                "runtime_seconds": metadata["runtime_seconds"],
                "peak_gpu_memory_bytes": peak_memory,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
