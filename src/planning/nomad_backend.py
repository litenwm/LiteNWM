"""Thin inference-only adapter for an external VisualNav NoMaD checkout."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from planning.checkpoint_io import load_checkpoint
import yaml
from PIL import Image


def add_external_paths(
    visualnav_root: Path,
    diffusion_policy_root: Path,
) -> None:
    required = (
        visualnav_root / "train",
        visualnav_root / "deployment" / "src",
        visualnav_root,
        diffusion_policy_root,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NoMaD config must be a YAML object")
    required = {
        "model_type",
        "vision_encoder",
        "encoding_size",
        "context_size",
        "len_traj_pred",
        "learn_angle",
        "image_size",
        "num_diffusion_iters",
        "mha_num_attention_heads",
        "mha_num_attention_layers",
        "mha_ff_dim_factor",
        "down_dims",
        "cond_predict_scale",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"NoMaD config missing fields: {sorted(missing)}")
    if value["model_type"] != "nomad":
        raise ValueError("only model_type=nomad is supported")
    if int(value["context_size"]) != 3:
        raise ValueError("strict H4 proposal generation requires context_size=3")
    if int(value["len_traj_pred"]) < 4:
        raise ValueError("NoMaD checkpoint predicts fewer than four steps")
    if bool(value["learn_angle"]):
        raise ValueError("this adapter expects the published XY-only NoMaD model")
    return value


def _install_warmup_unpickle_stub() -> None:
    if "warmup_scheduler.scheduler" in sys.modules:
        return
    warmup_scheduler = types.ModuleType("warmup_scheduler")
    warmup_scheduler.__path__ = []
    scheduler_module = types.ModuleType("warmup_scheduler.scheduler")

    class GradualWarmupScheduler:  # pragma: no cover - checkpoint compatibility.
        pass

    warmup_scheduler.GradualWarmupScheduler = GradualWarmupScheduler
    scheduler_module.GradualWarmupScheduler = GradualWarmupScheduler
    sys.modules["warmup_scheduler"] = warmup_scheduler
    sys.modules["warmup_scheduler.scheduler"] = scheduler_module


def load_nomad_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        from diffusion_policy.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D,
        )
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import (
            NoMaD_ViNT,
            replace_bn_with_gn,
        )
    except ImportError as error:  # pragma: no cover - external environment.
        raise RuntimeError(
            "NoMaD imports failed; install the VisualNav environment and pass "
            "the correct --visualnav-root/--diffusion-policy-root"
        ) from error

    if config["vision_encoder"] != "nomad_vint":
        raise ValueError("the pinned checkpoint requires vision_encoder=nomad_vint")
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=int(config["encoding_size"]),
        context_size=int(config["context_size"]),
        mha_num_attention_heads=int(config["mha_num_attention_heads"]),
        mha_num_attention_layers=int(config["mha_num_attention_layers"]),
        mha_ff_dim_factor=int(config["mha_ff_dim_factor"]),
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=int(config["encoding_size"]),
        down_dims=list(config["down_dims"]),
        cond_predict_scale=bool(config["cond_predict_scale"]),
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=DenseNetwork(embedding_dim=int(config["encoding_size"])),
    )
    _install_warmup_unpickle_stub()
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {
        "strict_load": True,
        "checkpoint_type": type(checkpoint).__name__,
        "state_dict_keys": len(state_dict),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def transform_images(
    images: list[Image.Image],
    image_size: list[int],
) -> torch.Tensor:
    try:
        from torchvision.transforms import functional as functional
    except ImportError as error:  # pragma: no cover - declared dependency.
        raise RuntimeError("torchvision is required for NoMaD preprocessing") from error
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    tensors = []
    for image in images:
        resized = image.resize(tuple(map(int, image_size)))
        tensor = functional.pil_to_tensor(resized).float() / 255.0
        tensors.append(((tensor - mean) / std).unsqueeze(0))
    return torch.cat(tensors, dim=1)


def _decode_waypoints(diffusion_output: torch.Tensor) -> torch.Tensor:
    deltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    minimum = deltas.new_tensor((-2.5, -4.0))
    maximum = deltas.new_tensor((5.0, 4.0))
    deltas = (deltas + 1.0) / 2.0
    deltas = deltas * (maximum - minimum) + minimum
    return torch.cumsum(deltas, dim=1)


class NoMaDSampler:
    """Deterministic-per-window diffusion sampler for the pinned NoMaD model."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError as error:  # pragma: no cover - external environment.
            raise RuntimeError(
                "diffusers is required in the NoMaD inference environment"
            ) from error
        self.model = model
        self.config = config
        self.device = device
        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(config["num_diffusion_iters"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _goal_conditioning(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
    ) -> torch.Tensor:
        observation_images = observation_images.to(self.device)
        goal_image = goal_image.to(self.device)
        no_mask = torch.zeros(1, dtype=torch.long, device=self.device)
        conditioning = self.model(
            "vision_encoder",
            obs_img=observation_images,
            goal_img=goal_image,
            input_goal_mask=no_mask,
        )
        if conditioning.ndim != 2 or conditioning.shape[0] != 1:
            raise RuntimeError(
                "unexpected NoMaD conditioning shape: "
                f"{tuple(conditioning.shape)}"
            )
        if not bool(torch.isfinite(conditioning).all()):
            raise RuntimeError("NoMaD conditioning contains NaN/Inf")
        return conditioning

    def _temporal_distance(
        self,
        conditioning: torch.Tensor,
    ) -> float:
        prediction = self.model(
            "dist_pred_net",
            obsgoal_cond=conditioning,
        ).reshape(-1)
        if prediction.numel() != 1:
            raise RuntimeError(
                "unexpected NoMaD distance shape: "
                f"{tuple(prediction.shape)}"
            )
        value = float(prediction.item())
        if not np.isfinite(value):
            raise RuntimeError("NoMaD temporal distance is NaN/Inf")
        return value

    @torch.inference_mode()
    def sample_with_distance(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
        *,
        candidates: int,
        seed: int,
    ) -> tuple[np.ndarray, float]:
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        conditioning = self._goal_conditioning(
            observation_images,
            goal_image,
        )
        temporal_distance = self._temporal_distance(conditioning)
        conditioning = conditioning.repeat(candidates, 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sample = torch.randn(
            (candidates, int(self.config["len_traj_pred"]), 2),
            generator=generator,
            device=self.device,
        )
        self.scheduler.set_timesteps(int(self.config["num_diffusion_iters"]))
        for timestep in self.scheduler.timesteps:
            predicted_noise = self.model(
                "noise_pred_net",
                sample=sample,
                timestep=timestep,
                global_cond=conditioning,
            )
            sample = self.scheduler.step(
                model_output=predicted_noise,
                timestep=timestep,
                sample=sample,
                generator=generator,
            ).prev_sample
        candidates_np = _decode_waypoints(sample).float().cpu().numpy()
        return candidates_np, temporal_distance

    @torch.inference_mode()
    def predict_distance(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
    ) -> float:
        conditioning = self._goal_conditioning(
            observation_images,
            goal_image,
        )
        return self._temporal_distance(conditioning)

    @torch.inference_mode()
    def sample(
        self,
        observation_images: torch.Tensor,
        goal_image: torch.Tensor,
        *,
        candidates: int,
        seed: int,
    ) -> np.ndarray:
        sampled, _ = self.sample_with_distance(
            observation_images,
            goal_image,
            candidates=candidates,
            seed=seed,
        )
        return sampled


__all__ = [
    "NoMaDSampler",
    "add_external_paths",
    "load_config",
    "load_nomad_model",
    "transform_images",
]
