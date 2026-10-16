"""Numerically equivalent, persistent V70 image-goal planner.

All trained models are loaded once, but every planning call preserves the
deployed V70 contracts:

* one five-image DINO batch (four observations plus the current goal);
* float16 DINO artifact quantization before LTWM, matching the shadow files;
* 17 NoMaD diffusion samples, first 16 references, trimmed-mean K17 incumbent;
* three LTWM/scorer members and three Arm-D members;
* formal RECON support, coherent-turn-tail support and frozen calibration;
* three CEM rounds with population 16 and the frozen RNG seed.

The module imports no ROS or Unitree motion API and never emits a command.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(DEPLOY_DIR)]

from planning.nomad_action_adapter import (  # noqa: E402
    integrate_local_actions,
    nomad_waypoints_to_local_actions,
)
from recon_latents.dinov3_adapter import (  # noqa: E402
    DinoV3AdapterConfig,
    DinoV3TokenAdapter,
    extract_dinov3_patch_tokens,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    trimmed_mean75,
)
from tools.planning import (  # noqa: E402
    run_dualdomain_cem_tail_v66_replay as replay,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)
from v70_action_support_compat import (  # noqa: E402
    SUPPORT_RECOVERY_FIRST_STEP_LATERAL_TOLERANCE_M,
    SUPPORT_RECOVERY_FIRST_STEP_YAW_TOLERANCE_RAD,
    SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M,
    coherent_turn_tail_support,
    reference_intent_guard,
    select_intent_preserving_tail_recovery_actions,
    support_stats,
)
from v70_armd_tridomain_v3_shadow import (  # noqa: E402
    localize_checkpoint_records,
)
import v70_cem_tridomain_v4_shadow as cem  # noqa: E402
from v70_dino_live_shadow import (  # noqa: E402
    HISTORY,
    LATENT_DIM,
    TOKEN_COUNT,
    discover_model_dir,
    load_backbone,
    preprocess_pil,
)
from v70_pose_features import PoseSample  # noqa: E402
from v70_runtime_contract import (  # noqa: E402
    require_domain_calibration,
    zero_actions_if_abstained,
)
from numpy_pickle_compat import (  # noqa: E402
    install_numpy_pickle_compat,
)
from planning.nomad_backend import (  # noqa: E402
    NoMaDSampler,
    add_external_paths,
    load_config,
    load_nomad_model,
    transform_images,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    pool_static_features,
)


DOMAIN = "recon"
DOMAIN_ID = 0
SPACING_M = 0.25
SUPPORT_QUANTILE = 0.999
NOMAD_SAMPLES = 17
REFERENCE_COUNT = 16
HORIZON = 4

ROUTE_INTENT_OPPOSITE_TOLERANCE_M = 0.02
# Keep only a small local correction opposite to the latched final-goal
# side. A full endpoint reversal is held for replanning instead of executed.
ROUTE_INTENT_LOCAL_DETOUR_MAX_M = 0.06
ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M = 0.01
ROUTE_INTENT_FIRST_STEP_HEADING_THRESHOLD_RAD = 0.03
# Strong lateral endpoints should start turning in the same direction during
# the first transition. This remains a bounded selection preference, not a
# hard feasibility rule: the learned pool may contain no early-turn sample.
ROUTE_INTENT_STRONG_ENDPOINT_M = 0.06
ROUTE_INTENT_EARLY_HEADING_MIN_RAD = 0.08
DIRECTIONAL_RESAMPLE_EXTRA_SEEDS = 2
DIRECTIONAL_MIN_SUPPORTED_CANDIDATES = 1
DIRECTIONAL_CONFLICT_MIN_AGREEMENT = 0.75


def mirror_nomad_waypoints(candidates: np.ndarray) -> np.ndarray:
    """Mirror cumulative local XY waypoints without touching weights."""

    mirrored = np.asarray(candidates, dtype=np.float32).copy()
    if mirrored.ndim != 3 or mirrored.shape[-1] != 2:
        raise ValueError("NoMaD candidates must have shape [K,H,2]")
    if not np.isfinite(mirrored).all():
        raise ValueError("NoMaD candidates contain NaN/Inf")
    mirrored[..., 1] *= np.float32(-1.0)
    return mirrored


def apply_route_intent_guard(
    candidate_waypoints: np.ndarray,
    safe: np.ndarray,
    *,
    direction: int,
    spacing_m: float,
    opposite_tolerance_m: float = ROUTE_INTENT_LOCAL_DETOUR_MAX_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep a latched final-goal side while allowing tiny local corrections.

    The model scores and candidate generation are unchanged. For an already
    latched single-goal direction, candidates whose endpoint crosses more
    than a small local-detour band are removed from the executable safe set.
    If no compatible candidate remains, the planner abstains and replans
    rather than sending the robot across the opposite side of the goal.
    """

    candidates = np.asarray(candidate_waypoints, dtype=np.float32)
    original_safe = np.asarray(safe, dtype=bool)
    if candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError(
            "candidate_waypoints must have shape [1,K,H,2]"
        )
    if original_safe.shape != candidates.shape[:2]:
        raise ValueError("safe shape does not match candidates")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    if not (
        np.isfinite(candidates).all()
        and math.isfinite(float(spacing_m))
        and float(spacing_m) > 0.0
        and math.isfinite(float(opposite_tolerance_m))
        and float(opposite_tolerance_m) >= 0.0
    ):
        raise ValueError("invalid route intent inputs")

    compatible = np.ones_like(original_safe, dtype=bool)
    endpoint_y_m = candidates[0, :, -1, 1] * np.float32(spacing_m)
    first_x_m = candidates[0, :, 0, 0] * np.float32(spacing_m)
    first_y_m = candidates[0, :, 0, 1] * np.float32(spacing_m)
    first_heading_rad = np.arctan2(first_y_m, first_x_m)
    endpoint_direction = np.sign(endpoint_y_m)
    candidate_shape_first_step_conflict = (
        (np.abs(endpoint_y_m) >= 0.06)
        & (
            endpoint_direction * first_y_m
            < -ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M
        )
    )
    route_first_step_conflict = np.zeros_like(endpoint_y_m, dtype=bool)
    early_steering_missing = np.zeros_like(endpoint_y_m, dtype=bool)
    if direction:
        # A small opposite endpoint can be a local steering correction, but
        # a full opposite-side endpoint is not executable for this final
        # image goal. First-step shape is audited and softly ranked below;
        # making it a hard rule can reject the complete learned sample pool.
        route_first_step_conflict = (
            (
                direction * first_y_m
                < -ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M
            )
            | (
                direction * first_heading_rad
                < -ROUTE_INTENT_FIRST_STEP_HEADING_THRESHOLD_RAD
            )
        )
        early_steering_missing = (
            direction * endpoint_y_m >= ROUTE_INTENT_STRONG_ENDPOINT_M
        ) & (
            direction * first_heading_rad
            < ROUTE_INTENT_EARLY_HEADING_MIN_RAD
        )
        compatible[0] = (
            (
                direction * endpoint_y_m
                >= -float(opposite_tolerance_m)
            )
        )
    guarded_safe = original_safe & compatible
    audit = {
        "schema": "v70_route_intent_guard_v1",
        "direction": (
            "right" if direction < 0 else (
                "left" if direction > 0 else "unlatched"
            )
        ),
        "endpoint_y_min_m": float(endpoint_y_m.min()),
        "endpoint_y_max_m": float(endpoint_y_m.max()),
        "original_safe_count": int(original_safe.sum()),
        "compatible_count": int(compatible.sum()),
        "guarded_safe_count": int(guarded_safe.sum()),
        "rejected_safe_count": int(
            (original_safe & ~compatible).sum()
        ),
        "soft_opposite_safe_count": int(
            (original_safe[0] & route_first_step_conflict).sum()
        ),
        "local_detour_max_m": float(opposite_tolerance_m),
        "first_step_conflict_count": int(route_first_step_conflict.sum()),
        "candidate_shape_first_step_conflict_count": int(
            candidate_shape_first_step_conflict.sum()
        ),
        "first_step_lateral_tolerance_m": float(
            ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M
        ),
        "first_step_heading_tolerance_rad": float(
            ROUTE_INTENT_FIRST_STEP_HEADING_THRESHOLD_RAD
        ),
        "early_heading_min_rad": float(
            ROUTE_INTENT_EARLY_HEADING_MIN_RAD
        ),
        "early_steering_missing_count": int(
            early_steering_missing.sum()
        ),
        "policy": "hard_global_endpoint_soft_early_steering",
    }
    return guarded_safe, audit


ROUTE_INTENT_SOFT_PENALTY_PER_M = 0.15
ROUTE_INTENT_FIRST_STEP_CONFLICT_PENALTY = 0.20
ROUTE_INTENT_EARLY_STEERING_PENALTY = 0.35
ROUTE_INTENT_FIRST_HEADING_TARGET_RAD = 0.04
ROUTE_INTENT_TWO_STEP_HEADING_TARGET_RAD = 0.12
ROUTE_INTENT_HEADING_DEFICIT_PENALTY_PER_RAD = 1.50


def route_intent_selection_energy(
    candidate_waypoints: np.ndarray,
    energy: np.ndarray,
    *,
    direction: int,
    spacing_m: float,
) -> np.ndarray:
    """Add a bounded lateral-consistency cost without changing model scores."""

    candidates = np.asarray(candidate_waypoints, dtype=np.float32)
    values = np.asarray(energy, dtype=np.float32).copy()
    if candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError("candidate_waypoints must have shape [1,K,H,2]")
    if values.shape != candidates.shape[:2]:
        raise ValueError("energy shape does not match candidates")
    if direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    endpoint_y_m = candidates[0, :, -1, 1] * np.float32(spacing_m)
    first_y_m = candidates[0, :, 0, 1] * np.float32(spacing_m)
    first_x_m = candidates[0, :, 0, 0] * np.float32(spacing_m)
    first_heading_rad = np.arctan2(first_y_m, first_x_m)
    if candidates.shape[2] >= 2:
        second_dx_m = (
            candidates[0, :, 1, 0] - candidates[0, :, 0, 0]
        ) * np.float32(spacing_m)
        second_dy_m = (
            candidates[0, :, 1, 1] - candidates[0, :, 0, 1]
        ) * np.float32(spacing_m)
        two_step_heading_rad = np.arctan2(second_dy_m, second_dx_m)
    else:
        two_step_heading_rad = first_heading_rad
    endpoint_direction = np.sign(endpoint_y_m)
    first_step_conflict = (
        (np.abs(endpoint_y_m) >= 0.06)
        & (
            endpoint_direction * first_y_m
            < -ROUTE_INTENT_FIRST_STEP_CONFLICT_THRESHOLD_M
        )
    )
    values += (
        np.float32(ROUTE_INTENT_FIRST_STEP_CONFLICT_PENALTY)
        * first_step_conflict[None, :]
    )
    if direction:
        early_steering_missing = (
            direction * endpoint_y_m >= ROUTE_INTENT_STRONG_ENDPOINT_M
        ) & (
            direction * first_heading_rad
            < ROUTE_INTENT_EARLY_HEADING_MIN_RAD
        )
        values += (
            np.float32(ROUTE_INTENT_EARLY_STEERING_PENALTY)
            * early_steering_missing[None, :]
        )
        first_heading_deficit = np.maximum(
            np.float32(0.0),
            np.float32(ROUTE_INTENT_FIRST_HEADING_TARGET_RAD)
            - np.float32(direction) * first_heading_rad,
        )
        two_step_heading_deficit = np.maximum(
            np.float32(0.0),
            np.float32(ROUTE_INTENT_TWO_STEP_HEADING_TARGET_RAD)
            - np.float32(direction) * two_step_heading_rad,
        )
        values += (
            np.float32(
                ROUTE_INTENT_HEADING_DEFICIT_PENALTY_PER_RAD
            )
            * (first_heading_deficit + two_step_heading_deficit)[
                None, :
            ]
        )
        opposite_m = np.maximum(
            np.float32(0.0),
            -np.float32(direction) * endpoint_y_m,
        )
        values += (
            np.float32(ROUTE_INTENT_SOFT_PENALTY_PER_M)
            * opposite_m[None, :]
        )
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, ...],
) -> None:
    if value.shape != expected:
        raise RuntimeError(
            f"{name}: expected {expected}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def synchronized_elapsed_ms(
    device: torch.device,
    started_ns: int,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter_ns() - started_ns) / 1.0e6


@dataclass(frozen=True)
class PersistentPlannerPaths:
    adapter_checkpoint: Path
    checkpoint_root: Path
    support: Path
    selector_result: Path
    selector_checkpoint_dir: Path
    model_dir: Path | None = None

    @classmethod
    def defaults(
        cls,
        root: Path = ROOT,
    ) -> "PersistentPlannerPaths":
        runtime = root / "runtime/v70_tridomain"
        return cls(
            adapter_checkpoint=(
                root
                / "checkpoints/project/dinov3_adapter/"
                "adapter_pilot_v1_final.pt"
            ),
            checkpoint_root=root / "checkpoints/project",
            support=(
                runtime
                / "selector/support/"
                "tridomain_spacing_envelopes.npz"
            ),
            selector_result=runtime / "selector/result.json",
            selector_checkpoint_dir=runtime / "selector/checkpoints",
        )


class PersistentV70Planner:
    """Load the sealed V70 stack once and plan repeatedly in memory."""

    def __init__(
        self,
        paths: PersistentPlannerPaths | None = None,
        *,
        device: str = "cuda",
        nomad_seed: int = cem.RNG_SEED,
        enable_support_recovery: bool = False,
        enable_reference_intent_guard: bool = False,
    ) -> None:
        install_numpy_pickle_compat()
        # V6 applies this compatibility patch before importing V4.  Apply the
        # same rich support loader for direct persistent imports.
        cem.support_stats = support_stats

        self.paths = paths or PersistentPlannerPaths.defaults()
        self.device = torch.device(
            "cuda:0" if device == "cuda" else device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.nomad_seed = int(nomad_seed)
        if self.nomad_seed < 0:
            raise ValueError("nomad_seed must be non-negative")
        self.enable_support_recovery = bool(
            enable_support_recovery
        )
        self.enable_reference_intent_guard = bool(
            enable_reference_intent_guard
        )

        self.load_timings_ms: dict[str, float] = {}
        self._load_dino()
        self._load_nomad()
        self._load_v70_ensemble()

    def _load_dino(self) -> None:
        started_ns = time.perf_counter_ns()
        adapter_path = self.paths.adapter_checkpoint.expanduser().resolve()
        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)
        model_dir = discover_model_dir(self.paths.model_dir)
        weight_path = model_dir / "model.safetensors"

        payload = torch.load(
            adapter_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_hash = payload.get("dinov3_weight_sha256")
        if expected_hash and sha256_file(weight_path) != expected_hash:
            raise RuntimeError("adapter/DINOv3 weight hash mismatch")

        adapter = DinoV3TokenAdapter(
            DinoV3AdapterConfig(**payload["config"])
        )
        adapter.load_state_dict(payload["model"], strict=True)
        self.dino_backbone = load_backbone(
            model_dir,
            self.device,
        )
        self.dino_adapter = (
            adapter.eval()
            .to(self.device)
            .requires_grad_(False)
        )
        self.dino_model_dir = model_dir
        self.dino_weight_sha256 = sha256_file(weight_path)
        self.dino_adapter_sha256 = sha256_file(adapter_path)
        self.load_timings_ms["dinov3_load_ms"] = (
            synchronized_elapsed_ms(self.device, started_ns)
        )

    @staticmethod
    def _resolve_asset(
        assets_path: Path,
        value: str,
    ) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raw = assets_path.parent / raw
        return raw.resolve()

    def _load_nomad(self) -> None:
        started_ns = time.perf_counter_ns()
        assets_path = ROOT / "configs/local_assets.json"
        assets = json.loads(
            assets_path.read_text(encoding="utf-8")
        )
        raw = assets["nomad"]
        config_path = self._resolve_asset(
            assets_path,
            raw["config"],
        )
        checkpoint_path = self._resolve_asset(
            assets_path,
            raw["checkpoint"],
        )
        visualnav_root = self._resolve_asset(
            assets_path,
            raw["visualnav_root"],
        )
        diffusion_policy_root = self._resolve_asset(
            assets_path,
            raw["diffusion_policy_root"],
        )
        for path in (
            config_path,
            checkpoint_path,
            visualnav_root,
            diffusion_policy_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        expected_checkpoint_hash = raw.get("checkpoint_sha256")
        expected_config_hash = raw.get("config_sha256")
        if (
            expected_checkpoint_hash
            and sha256_file(checkpoint_path)
            != expected_checkpoint_hash
        ):
            raise RuntimeError("NoMaD checkpoint hash mismatch")
        if (
            expected_config_hash
            and sha256_file(config_path) != expected_config_hash
        ):
            raise RuntimeError("NoMaD config hash mismatch")

        add_external_paths(
            visualnav_root=visualnav_root,
            diffusion_policy_root=diffusion_policy_root,
        )
        self.nomad_config = load_config(config_path)
        model, report = load_nomad_model(
            checkpoint_path=checkpoint_path,
            config=self.nomad_config,
            device=self.device,
        )
        self.nomad_sampler = NoMaDSampler(
            model,
            self.nomad_config,
            self.device,
        )
        self.nomad_load_report = report
        self.nomad_checkpoint_sha256 = sha256_file(checkpoint_path)
        self.load_timings_ms["nomad_load_ms"] = (
            synchronized_elapsed_ms(self.device, started_ns)
        )

    def _load_v70_ensemble(self) -> None:
        started_ns = time.perf_counter_ns()
        support_path = self.paths.support.expanduser().resolve()
        result_path = self.paths.selector_result.expanduser().resolve()
        checkpoint_root = (
            self.paths.checkpoint_root.expanduser().resolve()
        )
        selector_dir = (
            self.paths.selector_checkpoint_dir.expanduser().resolve()
        )
        for path in (
            support_path,
            result_path,
            checkpoint_root,
            selector_dir,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        self.spacing_support_sha256 = file_sha256(support_path)
        self.v70_result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        if (
            self.spacing_support_sha256
            != self.v70_result.get("support_sha256")
        ):
            raise RuntimeError("tri-domain support SHA mismatch")
        runtime = require_domain_calibration(
            self.v70_result,
            DOMAIN,
        )
        calibration = runtime["calibration"]
        self.calibration_contract = runtime[
            "calibration_contract"
        ]
        self.policy = dict(calibration["policy"])
        self.dual_temperature = float(
            calibration["dual_temperature"]
        )
        self.harm_temperature = float(
            calibration["harm_temperature"]
        )

        envelopes = replay.load_envelopes(support_path)
        self.envelope = envelopes[DOMAIN]
        self.support_threshold = replay.threshold_for_quantile(
            self.envelope,
            SUPPORT_QUANTILE,
        )
        records = localize_checkpoint_records(
            list(self.v70_result["checkpoints"]["full"]),
            selector_dir,
        )
        self.ltwm_members = cem.load_ltwm_ensemble(
            checkpoint_root,
            self.device,
        )
        self.selector_members = cem.load_selector_ensemble(
            records,
            self.device,
        )
        self.load_timings_ms["v70_ensemble_load_ms"] = (
            synchronized_elapsed_ms(self.device, started_ns)
        )

    def encode_dino(
        self,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
    ) -> dict[str, Any]:
        if len(observations) != HISTORY:
            raise ValueError(
                f"expected {HISTORY} observation images"
            )
        image_batch = torch.stack(
            [preprocess_pil(image) for image in observations]
            + [preprocess_pil(goal_image)]
        )
        if image_batch.shape != (5, 3, 224, 224):
            raise RuntimeError(
                f"DINO input shape changed: {image_batch.shape}"
            )
        image_batch = image_batch.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        with torch.inference_mode():
            hidden = self.dino_backbone(
                pixel_values=image_batch
            ).last_hidden_state
            patch_tokens = extract_dinov3_patch_tokens(hidden)
            adapted = self.dino_adapter(patch_tokens).float()
            if adapted.ndim == 4:
                if (
                    adapted.shape[1:3] == (16, 16)
                    and adapted.shape[-1] == LATENT_DIM
                ):
                    adapted = adapted.reshape(
                        adapted.shape[0],
                        TOKEN_COUNT,
                        LATENT_DIM,
                    )
                elif (
                    adapted.shape[1] == LATENT_DIM
                    and adapted.shape[2:] == (16, 16)
                ):
                    adapted = (
                        adapted.permute(0, 2, 3, 1)
                        .contiguous()
                        .reshape(
                            adapted.shape[0],
                            TOKEN_COUNT,
                            LATENT_DIM,
                        )
                    )
                else:
                    raise RuntimeError(
                        "unsupported DINO adapter grid shape: "
                        f"{tuple(adapted.shape)}"
                    )
        encode_ms = synchronized_elapsed_ms(
            self.device,
            started_ns,
        )
        if adapted.shape != (5, TOKEN_COUNT, LATENT_DIM):
            raise RuntimeError(
                f"adapted latent shape changed: {adapted.shape}"
            )

        history = adapted[:HISTORY].unsqueeze(0)
        goal = adapted[HISTORY].unsqueeze(0)
        static = pool_static_features(history, goal)
        current = history[:, -1]
        goal_energy = goal.square().mean(
            dim=(-1, -2)
        ).clamp_min(1.0e-8)
        nmse = (
            (current - goal).square().mean(dim=(-1, -2))
            / goal_energy
        )
        cosine = 1.0 - torch.cosine_similarity(
            current.flatten(1),
            goal.flatten(1),
            dim=-1,
        )
        norm_ratio = (
            torch.linalg.vector_norm(
                current,
                dim=(-1, -2),
            )
            / torch.linalg.vector_norm(
                goal,
                dim=(-1, -2),
            ).clamp_min(1.0e-8)
        )
        tensors = (history, goal, static, nmse, cosine, norm_ratio)
        if not all(
            bool(torch.isfinite(value).all()) for value in tensors
        ):
            raise RuntimeError("DINO output contains NaN/Inf")

        # Preserve the staged runtime's NPZ contract exactly: DINO latent
        # arrays are written as float16, then read back as float32 by LTWM.
        history_storage = history.cpu().numpy().astype(np.float16)
        goal_storage = goal.cpu().numpy().astype(np.float16)
        static_np = static.cpu().numpy().astype(np.float32)
        return {
            "history_latent_storage": history_storage,
            "goal_latent_storage": goal_storage,
            "history_latent": history_storage.astype(np.float32),
            "goal_latent": goal_storage.astype(np.float32),
            "static_features": static_np,
            "current_goal_nmse": float(nmse.item()),
            "current_goal_cosine_distance": float(cosine.item()),
            "current_goal_norm_ratio": float(norm_ratio.item()),
            "encode_5_images_ms": encode_ms,
        }

    def sample_nomad(
        self,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if len(observations) != HISTORY:
            raise ValueError(
                f"expected {HISTORY} observation images"
            )
        observation_tensor = transform_images(
            list(observations),
            self.nomad_config["image_size"],
        )
        goal_tensor = transform_images(
            [goal_image],
            self.nomad_config["image_size"],
        )
        actual_seed = self.nomad_seed if seed is None else int(seed)
        if actual_seed < 0:
            raise ValueError("NoMaD seed must be non-negative")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        candidates, temporal_distance = (
            self.nomad_sampler.sample_with_distance(
                observation_tensor,
                goal_tensor,
                candidates=NOMAD_SAMPLES,
                seed=actual_seed,
            )
        )
        generation_ms = synchronized_elapsed_ms(
            self.device,
            started_ns,
        )
        value = np.asarray(candidates, dtype=np.float32)
        if (
            value.ndim != 3
            or value.shape[0] != NOMAD_SAMPLES
            or value.shape[-1] != 2
        ):
            raise RuntimeError(
                f"unexpected NoMaD shape: {value.shape}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError("NoMaD candidates contain NaN/Inf")
        return {
            "candidates": value,
            "temporal_distance": float(temporal_distance),
            "candidate_generation_ms": generation_ms,
            "seed": actual_seed,
        }

    def _directional_nomad_batch_audit(
        self,
        candidates: np.ndarray,
        *,
        direction: int,
        seed: int,
    ) -> dict[str, Any]:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        reference = np.asarray(
            candidates,
            dtype=np.float32,
        )[:REFERENCE_COUNT, :HORIZON].copy()
        require_shape(
            "directional NoMaD reference",
            reference,
            (REFERENCE_COUNT, HORIZON, 2),
        )
        actions = nomad_waypoints_to_local_actions(
            reference[None],
            dataset=DOMAIN,
        ).astype(np.float32)[0]
        support, distance = coherent_turn_tail_support(
            actions[None],
            spacing_support_sha256=self.spacing_support_sha256,
        )
        support = support[0]
        distance = distance[0]
        endpoint_y_m = (
            reference[:, -1, 1] * np.float32(SPACING_M)
        )
        median_endpoint_y_m = float(np.median(endpoint_y_m))
        reference_direction = int(np.sign(median_endpoint_y_m))
        agreement = (
            float(
                np.mean(
                    reference_direction * endpoint_y_m
                    >= SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M * 0.5
                )
            )
            if reference_direction
            else 0.0
        )
        same_endpoint = (
            direction * endpoint_y_m
            >= SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M
        )
        opposite_endpoint = (
            direction * endpoint_y_m
            <= -SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M
        )
        first_step_consistent = (
            (
                direction * actions[:, 0, 1]
                >= -SUPPORT_RECOVERY_FIRST_STEP_LATERAL_TOLERANCE_M
            )
            & (
                direction * actions[:, 0, 2]
                >= -SUPPORT_RECOVERY_FIRST_STEP_YAW_TOLERANCE_RAD
            )
        )
        eligible = same_endpoint & first_step_consistent & support
        conflict = bool(
            reference_direction == -direction
            and abs(median_endpoint_y_m)
            >= SUPPORT_RECOVERY_MIN_ENDPOINT_LATERAL_M
            and agreement >= DIRECTIONAL_CONFLICT_MIN_AGREEMENT
            and not bool(same_endpoint.any())
        )
        return {
            "seed": int(seed),
            "direction": int(direction),
            "reference_direction": int(reference_direction),
            "median_endpoint_y_m": median_endpoint_y_m,
            "reference_agreement": agreement,
            "same_direction_endpoint_count": int(same_endpoint.sum()),
            "opposite_direction_endpoint_count": int(
                opposite_endpoint.sum()
            ),
            "first_step_consistent_count": int(
                first_step_consistent.sum()
            ),
            "support_valid_count": int(support.sum()),
            "eligible_count": int(eligible.sum()),
            "eligible_indices": np.flatnonzero(eligible).tolist(),
            "minimum_support_distance": float(distance.min()),
            "conflicts_with_visual": conflict,
        }

    @staticmethod
    def _branch_metrics(
        planning: dict[str, Any],
        *,
        direction: int,
    ) -> dict[str, Any]:
        evaluations: list[dict[str, Any]] = [planning["fixed"]]
        evaluations.extend(
            item["evaluation"] for item in planning["rounds"]
        )
        recovery = planning.get("support_recovery")
        if (
            isinstance(recovery, dict)
            and isinstance(recovery.get("evaluation"), dict)
        ):
            evaluations.append(recovery["evaluation"])

        evaluated_count = 0
        support_safe_count = 0
        v70_safe_count = 0
        route_safe_count = 0
        for evaluation in evaluations:
            width = int(evaluation["valid"].shape[1])
            evaluated_count += max(0, width - 1)
            support_safe_count += int(
                np.asarray(evaluation["valid"], dtype=bool)[:, 1:].sum()
            )
            v70_safe_count += int(
                np.asarray(evaluation["safe"], dtype=bool)[:, 1:].sum()
            )
            route_safe_count += int(
                np.asarray(
                    evaluation.get("planning_safe", evaluation["safe"]),
                    dtype=bool,
                )[:, 1:].sum()
            )

        selected = np.asarray(
            planning["selected_actions_h4"],
            dtype=np.float32,
        )
        endpoint_y_m = float(
            integrate_local_actions(selected[None])[0, -1, 1]
        )
        first_step_consistent = bool(
            not planning["abstained"]
            and (
                not direction
                or (
                    direction * float(selected[0, 1])
                    >= -SUPPORT_RECOVERY_FIRST_STEP_LATERAL_TOLERANCE_M
                    and direction * float(selected[0, 2])
                    >= -SUPPORT_RECOVERY_FIRST_STEP_YAW_TOLERANCE_RAD
                )
            )
        )
        endpoint_direction_consistent = bool(
            not planning["abstained"]
            and (
                not direction
                or direction * endpoint_y_m
                >= -ROUTE_INTENT_LOCAL_DETOUR_MAX_M
            )
        )
        return {
            "branch_evaluated_count": evaluated_count,
            "branch_support_safe_count": support_safe_count,
            "branch_v70_safe_count": v70_safe_count,
            "branch_route_safe_count": route_safe_count,
            "branch_selected_energy": float(planning["archive_energy"]),
            "branch_first_step_consistent": first_step_consistent,
            "branch_endpoint_direction_consistent": (
                endpoint_direction_consistent
            ),
            "branch_selected_endpoint_y_m": endpoint_y_m,
        }

    def _evaluate_pool(
        self,
        *,
        candidates: np.ndarray,
        reference: np.ndarray,
        poses: list[PoseSample],
        history: torch.Tensor,
        goal: torch.Tensor,
        state_static: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return cem.evaluate_pool(
            candidates=candidates,
            reference=reference,
            poses=poses,
            history=history,
            goal=goal,
            state_static=state_static,
            ltwm_members=self.ltwm_members,
            envelope=self.envelope,
            threshold=self.support_threshold,
            spacing_support_sha256=self.spacing_support_sha256,
            selector_members=self.selector_members,
            policy=self.policy,
            dual_temperature=self.dual_temperature,
            harm_temperature=self.harm_temperature,
            device=self.device,
        )

    def score_and_plan(
        self,
        *,
        dino: dict[str, Any],
        nomad_candidates: np.ndarray,
        poses: Sequence[PoseSample],
        route_intent_direction: int = 0,
        strict_route_intent: bool = False,
        forced_recovery_actions: np.ndarray | None = None,
        forced_recovery_audit: dict[str, Any] | None = None,
        allow_partial_support_recovery: bool = False,
    ) -> dict[str, Any]:
        if route_intent_direction not in (-1, 0, 1):
            raise ValueError("route_intent_direction must be -1, 0 or 1")
        if not isinstance(strict_route_intent, bool):
            raise ValueError("strict_route_intent must be boolean")
        if len(poses) != 4:
            raise ValueError("exactly four odometry samples are required")
        pose_list = list(poses)
        reference_raw = np.asarray(
            nomad_candidates,
            dtype=np.float32,
        )[:REFERENCE_COUNT, :HORIZON].copy()
        require_shape(
            "NoMaD reference",
            reference_raw,
            (REFERENCE_COUNT, HORIZON, 2),
        )
        incumbent_raw = trimmed_mean75(reference_raw)
        base_raw = np.concatenate(
            (incumbent_raw[None], reference_raw),
            axis=0,
        ).astype(np.float32)
        reference = reference_raw[None]
        base_candidates = base_raw[None]
        require_shape(
            "K17 candidates",
            base_candidates,
            (1, 17, 4, 2),
        )


        def apply_route_intent(
            candidates: np.ndarray,
            safe: np.ndarray,
            audit: dict[str, Any],
        ) -> np.ndarray:
            guarded, route_audit = apply_route_intent_guard(
                candidates,
                safe,
                direction=route_intent_direction,
                spacing_m=SPACING_M,
                opposite_tolerance_m=(
                    0.0
                    if strict_route_intent
                    else ROUTE_INTENT_LOCAL_DETOUR_MAX_M
                ),
            )
            audit["route_intent"] = route_audit
            return guarded

        history_np = np.asarray(
            dino["history_latent"],
            dtype=np.float32,
        )
        goal_np = np.asarray(
            dino["goal_latent"],
            dtype=np.float32,
        )
        state_static = np.asarray(
            dino["static_features"],
            dtype=np.float32,
        )
        require_shape(
            "history_latent",
            history_np,
            (1, 4, 256, 384),
        )
        require_shape(
            "goal_latent",
            goal_np,
            (1, 256, 384),
        )
        require_shape(
            "static_features",
            state_static,
            (1, 210),
        )
        history = torch.from_numpy(history_np).to(self.device)
        goal = torch.from_numpy(goal_np).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        fixed_started_ns = time.perf_counter_ns()
        fixed = self._evaluate_pool(
            candidates=base_candidates,
            reference=reference,
            poses=pose_list,
            history=history,
            goal=goal,
            state_static=state_static,
        )
        fixed_planning_safe, fixed_intent_audit = (
            reference_intent_guard(
                reference,
                base_candidates,
                fixed["safe"],
                spacing_m=SPACING_M,
                enabled=(
                    self.enable_reference_intent_guard
                                    ),
            )
        )
        fixed_planning_safe = apply_route_intent(
            base_candidates,
            fixed_planning_safe,
            fixed_intent_audit,
        )
        fixed["energy"] = route_intent_selection_energy(
            base_candidates,
            fixed["energy"],
            direction=route_intent_direction,
            spacing_m=SPACING_M,
        )
        fixed["planning_safe"] = fixed_planning_safe
        intent_audits = [fixed_intent_audit]
        fixed_pool_ms = synchronized_elapsed_ms(
            self.device,
            fixed_started_ns,
        )

        incumbent = base_candidates[:, 0].copy()
        archive: dict[str, Any] = {
            "energy": np.inf,
            "candidate": incumbent[0].copy(),
            "source_round": -2,
            "source_candidate": 0,
        }
        cem.update_archive(
            round_index=-1,
            candidates=base_candidates,
            energy=fixed["energy"],
            safe=fixed_planning_safe,
            archive=archive,
        )

        mean, covariance = replay.initial_proposal(
            reference,
            floor=cem.COVARIANCE_FLOOR,
            scale=cem.PROPOSAL_SCALE,
        )
        (
            mean,
            covariance,
            _,
            _,
            _,
        ) = replay.safe_elite_update(
            base_candidates,
            fixed["energy"],
            fixed_planning_safe,
            previous_mean=mean,
            previous_covariance=covariance,
            elite_count=cem.ELITE_COUNT,
            floor=cem.COVARIANCE_FLOOR,
            scale=cem.PROPOSAL_SCALE,
        )

        def additional_support_mask(
            actions_h4: np.ndarray,
        ) -> np.ndarray:
            support, _ = coherent_turn_tail_support(
                actions_h4,
                spacing_support_sha256=(
                    self.spacing_support_sha256
                ),
            )
            return support

        rng = np.random.default_rng(cem.RNG_SEED)
        rounds: list[dict[str, Any]] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        cem_started_ns = time.perf_counter_ns()
        for iteration in range(cem.ITERATIONS):
            candidates, sampling_audit = replay.sample_population(
                incumbent=incumbent,
                reference=reference,
                mean=mean,
                covariance=covariance,
                population=cem.POPULATION,
                rng=rng,
                support_envelope=self.envelope,
                support_threshold=self.support_threshold,
                spacing_m=SPACING_M,
                additional_support_mask=additional_support_mask,
            )
            evaluation = self._evaluate_pool(
                candidates=candidates,
                reference=reference,
                poses=pose_list,
                history=history,
                goal=goal,
                state_static=state_static,
            )
            planning_safe, intent_audit = reference_intent_guard(
                reference,
                candidates,
                evaluation["safe"],
                spacing_m=SPACING_M,
                enabled=(
                    self.enable_reference_intent_guard
                                    ),
            )
            planning_safe = apply_route_intent(
                candidates,
                planning_safe,
                intent_audit,
            )
            evaluation["energy"] = route_intent_selection_energy(
                candidates,
                evaluation["energy"],
                direction=route_intent_direction,
                spacing_m=SPACING_M,
            )
            evaluation["planning_safe"] = planning_safe
            intent_audits.append(intent_audit)
            cem.update_archive(
                round_index=iteration,
                candidates=candidates,
                energy=evaluation["energy"],
                safe=planning_safe,
                archive=archive,
            )
            (
                mean,
                covariance,
                elite_index,
                updated,
                safe_count,
            ) = replay.safe_elite_update(
                candidates,
                evaluation["energy"],
                planning_safe,
                previous_mean=mean,
                previous_covariance=covariance,
                elite_count=cem.ELITE_COUNT,
                floor=cem.COVARIANCE_FLOOR,
                scale=cem.PROPOSAL_SCALE,
            )
            rounds.append(
                {
                    "iteration": iteration,
                    "candidates": candidates,
                    "evaluation": evaluation,
                    "elite_index": elite_index,
                    "proposal_updated": updated,
                    "safe_count": safe_count,
                    "sampling_audit": sampling_audit,
                }
            )
        cem_ms = synchronized_elapsed_ms(
            self.device,
            cem_started_ns,
        )

        support_recovery: dict[str, Any] | None = None
        support_recovery_ms = 0.0
        if (
            self.enable_support_recovery
            and (
                forced_recovery_actions is not None
                or not np.isfinite(float(archive["energy"]))
            )
        ):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            recovery_started_ns = time.perf_counter_ns()
            if forced_recovery_actions is not None:
                recovery_actions = np.asarray(
                    forced_recovery_actions,
                    dtype=np.float32,
                ).copy()
                recovery_audit = dict(forced_recovery_audit or {})
                recovery_audit["reason"] = "ready"
                recovery_audit["source"] = (
                    "mirrored_reference_sealed_support"
                )
            else:
                reference_actions = nomad_waypoints_to_local_actions(
                    reference,
                    dataset=DOMAIN,
                ).astype(np.float32)[0]
                recovery_actions, recovery_audit = (
                    select_intent_preserving_tail_recovery_actions(
                        reference_actions,
                        spacing_support_sha256=(
                            self.spacing_support_sha256
                        ),
                        population=cem.POPULATION,
                        intent_direction=route_intent_direction,
                        allow_partial_population=(
                            allow_partial_support_recovery
                        ),
                    )
                )
            support_recovery = {
                **recovery_audit,
                "triggered": True,
                "evaluated_candidate_count": 0,
                "safe_count": 0,
                "selected": False,
            }
            if recovery_actions.shape == (
                cem.POPULATION,
                HORIZON,
                3,
            ):
                recovery_waypoints = (
                    integrate_local_actions(recovery_actions)
                    / SPACING_M
                ).astype(np.float32)
                recovery_candidates = np.concatenate(
                    (incumbent, recovery_waypoints),
                    axis=0,
                )[None]
                require_shape(
                    "support recovery candidates",
                    recovery_candidates,
                    (1, 17, 4, 2),
                )
                recovery_evaluation = self._evaluate_pool(
                    candidates=recovery_candidates,
                    reference=reference,
                    poses=pose_list,
                    history=history,
                    goal=goal,
                    state_static=state_static,
                )
                reference_direction = int(
                    np.sign(
                        np.median(
                            reference[0, :, -1, 1]
                            * np.float32(SPACING_M)
                        )
                    )
                )
                visual_route_overrides_reference = bool(
                    route_intent_direction
                    and reference_direction
                    and route_intent_direction != reference_direction
                )
                recovery_planning_safe, recovery_intent_audit = (
                    reference_intent_guard(
                        reference,
                        recovery_candidates,
                        recovery_evaluation["safe"],
                        spacing_m=SPACING_M,
                        enabled=(
                            self.enable_reference_intent_guard
                            and not visual_route_overrides_reference
                        ),
                    )
                )
                recovery_intent_audit[
                    "visual_route_overrides_reference"
                ] = visual_route_overrides_reference
                recovery_planning_safe = apply_route_intent(
                    recovery_candidates,
                    recovery_planning_safe,
                    recovery_intent_audit,
                )
                recovery_evaluation["energy"] = (
                    route_intent_selection_energy(
                        recovery_candidates,
                        recovery_evaluation["energy"],
                        direction=route_intent_direction,
                        spacing_m=SPACING_M,
                    )
                )
                recovery_evaluation["planning_safe"] = (
                    recovery_planning_safe
                )
                intent_audits.append(recovery_intent_audit)
                cem.update_archive(
                    round_index=cem.ITERATIONS,
                    candidates=recovery_candidates,
                    energy=recovery_evaluation["energy"],
                    safe=recovery_planning_safe,
                    archive=archive,
                )
                support_recovery.update(
                    {
                        "candidates": recovery_candidates,
                        "evaluation": recovery_evaluation,
                        "evaluated_candidate_count": int(
                            cem.POPULATION
                        ),
                        "safe_count": int(
                            recovery_planning_safe[0, 1:].sum()
                        ),
                        "selected": bool(
                            archive["source_round"]
                            == cem.ITERATIONS
                        ),
                    }
                )
            support_recovery_ms = synchronized_elapsed_ms(
                self.device,
                recovery_started_ns,
            )

        archived = np.isfinite(float(archive["energy"]))
        selected_candidate = (
            np.asarray(archive["candidate"], dtype=np.float32)
            if archived
            else incumbent[0].astype(np.float32)
        )
        diagnostic_actions = nomad_waypoints_to_local_actions(
            selected_candidate[None],
            dataset=DOMAIN,
        ).astype(np.float32)[0]
        selected_actions = zero_actions_if_abstained(
            diagnostic_actions,
            abstained=not archived,
        )
        selected_waypoints = (
            selected_candidate.copy()
            if archived
            else np.zeros_like(selected_candidate)
        )

        selection_reason = (
            "finite_safe_support_recovery"
            if archived
            and int(archive["source_round"]) == cem.ITERATIONS
            else (
                "finite_safe_archive_challenger"
                if archived
                else "no_finite_safe_challenger"
            )
        )
        intent_guard_audit = dict(fixed_intent_audit)
        intent_guard_audit.update(
            {
                "pool_count": len(intent_audits),
                "rejected_safe_count_total": sum(
                    int(item["rejected_safe_count"])
                    for item in intent_audits
                ),
                "selected_endpoint_y_m": float(
                    selected_candidate[-1, 1] * SPACING_M
                ),
            }
        )
        result: dict[str, Any] = {
            "reference_nomad_x16": reference,
            "base_candidates": base_candidates,
            "fixed": fixed,
            "rounds": rounds,
            "final_proposal_mean": mean,
            "final_proposal_covariance": covariance,
            "archive_candidate_waypoint_units": selected_candidate,
            "selected_waypoints": selected_waypoints,
            "diagnostic_selected_actions_h4": diagnostic_actions,
            "selected_actions_h4": selected_actions,
            "selected_first_action": selected_actions[0].copy(),
            "reference_intent_guard": intent_guard_audit,
            "route_intent_direction": int(route_intent_direction),
            "strict_route_intent": bool(strict_route_intent),
            "abstained": not archived,
            "selection_reason": selection_reason,
            "archive_energy": float(archive["energy"]),
            "archive_source_round": int(
                archive["source_round"]
            ),
            "archive_source_candidate": int(
                archive["source_candidate"]
            ),
            "fixed_pool_planning_ms": fixed_pool_ms,
            "cem_planning_ms": cem_ms,
        }
        if self.enable_support_recovery:
            if support_recovery is None:
                support_recovery = {
                    "schema": "v70_support_recovery_audit_v1",
                    "reason": "original_archive_already_safe",
                    "triggered": False,
                    "eligible_template_count": 0,
                    "selected_template_indices": [],
                    "selected_closest_reference_indices": [],
                    "selected_intent_distances": [],
                    "evaluated_candidate_count": 0,
                    "safe_count": 0,
                    "selected": False,
                }
            result["support_recovery"] = support_recovery
            result["support_recovery_planning_ms"] = (
                support_recovery_ms
            )
        return result

    def plan(
        self,
        *,
        observations: Sequence[Image.Image],
        goal_image: Image.Image,
        poses: Sequence[PoseSample],
        route_intent_direction: int = 0,
        strict_route_intent: bool = False,
        allow_directional_recovery: bool = False,
    ) -> dict[str, Any]:
        total_started_ns = time.perf_counter_ns()
        dino = self.encode_dino(observations, goal_image)
        nomad = self.sample_nomad(
            observations,
            goal_image,
            seed=self.nomad_seed,
        )
        selected_nomad = nomad
        attempted_batches: list[dict[str, Any]] = []
        selected_mode = "original"
        recovery_reason = "visual_recovery_not_authorized"
        mirror_probe: dict[str, Any] | None = None
        mirrored_support_actions: np.ndarray | None = None
        total_nomad_ms = float(nomad["candidate_generation_ms"])

        if allow_directional_recovery and route_intent_direction:
            first_audit = self._directional_nomad_batch_audit(
                nomad["candidates"],
                direction=route_intent_direction,
                seed=int(nomad["seed"]),
            )
            attempted_batches.append(first_audit)
            if (
                first_audit["eligible_count"]
                >= DIRECTIONAL_MIN_SUPPORTED_CANDIDATES
            ):
                recovery_reason = "original_batch_directionally_usable"
            else:
                # A reliable visual direction may agree with the NoMaD
                # endpoint while every first step/support check still
                # rejects the batch. Bounded resampling is useful in that
                # case too and remains cheaper than repeating full V70.
                all_conflict = bool(
                    first_audit["conflicts_with_visual"]
                )
                for offset in range(
                    1,
                    DIRECTIONAL_RESAMPLE_EXTRA_SEEDS + 1,
                ):
                    resampled = self.sample_nomad(
                        observations,
                        goal_image,
                        seed=self.nomad_seed + offset,
                    )
                    total_nomad_ms += float(
                        resampled["candidate_generation_ms"]
                    )
                    batch_audit = self._directional_nomad_batch_audit(
                        resampled["candidates"],
                        direction=route_intent_direction,
                        seed=int(resampled["seed"]),
                    )
                    attempted_batches.append(batch_audit)
                    if (
                        batch_audit["eligible_count"]
                        >= DIRECTIONAL_MIN_SUPPORTED_CANDIDATES
                    ):
                        selected_nomad = resampled
                        selected_mode = "resampled_nomad"
                        recovery_reason = (
                            "bounded_resample_directionally_usable"
                        )
                        all_conflict = False
                        break
                    if not batch_audit["conflicts_with_visual"]:
                        all_conflict = False

                if all_conflict and len(attempted_batches) == 3:
                    mirrored_candidates = mirror_nomad_waypoints(
                        nomad["candidates"]
                    )
                    mirrored_reference_actions = (
                        nomad_waypoints_to_local_actions(
                            mirrored_candidates[
                                None, :REFERENCE_COUNT, :HORIZON
                            ],
                            dataset=DOMAIN,
                        ).astype(np.float32)[0]
                    )
                    mirrored_support_actions, mirror_probe = (
                        select_intent_preserving_tail_recovery_actions(
                            mirrored_reference_actions,
                            spacing_support_sha256=(
                                self.spacing_support_sha256
                            ),
                            population=cem.POPULATION,
                            intent_direction=route_intent_direction,
                        )
                    )
                    if mirror_probe["reason"] == "ready":
                        selected_nomad = {
                            **nomad,
                            "candidates": mirrored_candidates,
                            "mirrored_from_seed": int(nomad["seed"]),
                        }
                        selected_mode = "mirrored_reference_recovery"
                        recovery_reason = "two_resamples_conflicted"
                    else:
                        recovery_reason = (
                            "mirrored_reference_has_no_supported_pool"
                        )
                elif selected_mode == "original":
                    recovery_reason = (
                        "bounded_resample_no_usable_batch"
                    )

        selected_nomad = dict(selected_nomad)
        selected_nomad["candidate_generation_ms"] = total_nomad_ms
        planning = self.score_and_plan(
            dino=dino,
            nomad_candidates=selected_nomad["candidates"],
            poses=poses,
            route_intent_direction=route_intent_direction,
            strict_route_intent=strict_route_intent,
            forced_recovery_actions=mirrored_support_actions,
            forced_recovery_audit=mirror_probe,
            allow_partial_support_recovery=bool(
                allow_directional_recovery
                and route_intent_direction
            ),
        )
        directional_recovery = {
            "schema": "v70_directional_recovery_v1",
            "authorized": bool(allow_directional_recovery),
            "visual_direction": int(route_intent_direction),
            "base_seed": int(self.nomad_seed),
            "attempted_seeds": [
                int(item["seed"]) for item in attempted_batches
            ],
            "attempted_batches": attempted_batches,
            "selected_mode": selected_mode,
            "selected_seed": int(selected_nomad["seed"]),
            "reason": recovery_reason,
            "mirrored": bool(
                selected_mode == "mirrored_reference_recovery"
            ),
            "mirror_probe": mirror_probe,
            **self._branch_metrics(
                planning,
                direction=route_intent_direction,
            ),
        }
        total_ms = synchronized_elapsed_ms(
            self.device,
            total_started_ns,
        )
        result = {
            **planning,
            "dino": dino,
            "nomad": selected_nomad,
            "directional_recovery": directional_recovery,
            "planning_total_ms": total_ms,
            "load_timings_ms": dict(self.load_timings_ms),
            "precision_contract": {
                "dino": "float32 five-image batch",
                "dino_ltwm_boundary": (
                    "float16 artifact-equivalent quantization then float32"
                ),
                "nomad": "original checkpoint precision",
                "ltwm_autocast": "bfloat16 exactly as deployed",
                "ltwm_members": len(self.ltwm_members),
                "selector_members": len(self.selector_members),
                "cem_iterations": cem.ITERATIONS,
                "cem_population": cem.POPULATION,
            },
        }
        return result

    @staticmethod
    def summary(result: dict[str, Any]) -> dict[str, Any]:
        action = np.asarray(
            result["selected_first_action"],
            dtype=np.float32,
        )
        actions_h4 = np.asarray(
            result["selected_actions_h4"],
            dtype=np.float32,
        )
        waypoints_h4_m = integrate_local_actions(actions_h4).astype(
            np.float32
        )
        summary = {
            "abstained": bool(result["abstained"]),
            "selection_reason": result["selection_reason"],
            "archive_energy": float(result["archive_energy"]),
            "archive_source_round": int(
                result["archive_source_round"]
            ),
            "archive_source_candidate": int(
                result["archive_source_candidate"]
            ),
            "selected_first_action": action.tolist(),
            "selected_actions_h4": actions_h4.tolist(),
            "selected_waypoints_h4_m": waypoints_h4_m.tolist(),
            "nomad_temporal_distance": float(
                result["nomad"]["temporal_distance"]
            ),
            "dinov3_current_goal_nmse": float(
                result["dino"]["current_goal_nmse"]
            ),
            "dinov3_current_goal_cosine_distance": float(
                result["dino"][
                    "current_goal_cosine_distance"
                ]
            ),
            "dinov3_current_goal_norm_ratio": float(
                result["dino"]["current_goal_norm_ratio"]
            ),
            "timings_ms": {
                "dinov3_encode": float(
                    result["dino"]["encode_5_images_ms"]
                ),
                "nomad_candidate_generation": float(
                    result["nomad"][
                        "candidate_generation_ms"
                    ]
                ),
                "fixed_pool_planning": float(
                    result["fixed_pool_planning_ms"]
                ),
                "cem_planning": float(
                    result["cem_planning_ms"]
                ),
                "persistent_total": float(
                    result["planning_total_ms"]
                ),
            },
        }
        summary["reference_intent_guard"] = dict(
            result["reference_intent_guard"]
        )
        summary["route_intent_direction"] = int(
            result.get("route_intent_direction", 0)
        )
        summary["directional_recovery"] = dict(
            result.get("directional_recovery", {})
        )
        recovery = result.get("support_recovery")
        if recovery is not None:
            summary["support_recovery"] = {
                "triggered": bool(recovery["triggered"]),
                "reason": str(recovery["reason"]),
                "eligible_template_count": int(
                    recovery["eligible_template_count"]
                ),
                "evaluated_candidate_count": int(
                    recovery["evaluated_candidate_count"]
                ),
                "safe_count": int(recovery["safe_count"]),
                "selected": bool(recovery["selected"]),
                "selected_template_indices": list(
                    recovery["selected_template_indices"]
                ),
                "selected_closest_reference_indices": list(
                    recovery[
                        "selected_closest_reference_indices"
                    ]
                ),
                "selected_intent_distances": list(
                    recovery["selected_intent_distances"]
                ),
            }
            summary["timings_ms"]["support_recovery"] = float(
                result["support_recovery_planning_ms"]
            )
        return summary


__all__ = [
    "PersistentPlannerPaths",
    "PersistentV70Planner",
    "PoseSample",
    "mirror_nomad_waypoints",
    "require_shape",
    "sha256_file",
]
