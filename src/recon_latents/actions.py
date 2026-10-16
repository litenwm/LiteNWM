"""Action utilities for RECON pose-derived transition controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_ATOMIC_LABELS = ["adjust left", "adjust right", "go forward", "turn left", "turn right"]
CONTINUOUS_ACTION_NAMES = ["dx_local", "dy_local", "dyaw"]
ACTIONCOND_V1_FEATURE_NAMES = [
    "dx_local",
    "dy_local",
    "dyaw",
    "x_cum",
    "y_cum",
    "yaw_cum",
    "step01",
    "sin_step",
    "cos_step",
]
ACTION_DIM_PRESETS = {
    "full": [0, 1, 2],
    "dx_only": [0],
    "dy_only": [1],
    "dyaw_only": [2],
    "dx_dyaw": [0, 2],
}
_TORCH_STEP_FEATURE_CACHE: dict[tuple[int, str, str], Any] = {}


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles in radians to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def derive_continuous_actions(positions: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Return [dx_local, dy_local, dyaw] for each transition t -> t + 1."""
    positions = np.asarray(positions, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape [T, 2], got {positions.shape}")
    if yaw.ndim != 1 or len(yaw) != len(positions):
        raise ValueError(f"yaw must have shape [T] matching positions, got {yaw.shape}")
    if len(positions) < 2:
        return np.zeros((0, 3), dtype=np.float32)

    delta_global = positions[1:] - positions[:-1]
    cos_yaw = np.cos(yaw[:-1])
    sin_yaw = np.sin(yaw[:-1])
    dx_local = cos_yaw * delta_global[:, 0] + sin_yaw * delta_global[:, 1]
    dy_local = -sin_yaw * delta_global[:, 0] + cos_yaw * delta_global[:, 1]
    delta_yaw = wrap_to_pi(yaw[1:] - yaw[:-1])
    return np.stack([dx_local, dy_local, delta_yaw], axis=1).astype(np.float32)


def build_actioncond_v1_features(raw_actions: Any, horizon: int = 8, validate: bool = True) -> Any:
    """Build A0-split action features from raw [dx_local, dy_local, dyaw].

    The cumulative SE(2) pose is intentionally integrated from raw,
    unnormalized actions. Callers should normalize the returned 9D features
    after this function.
    """
    if _is_torch_tensor(raw_actions):
        return _build_actioncond_v1_features_torch(raw_actions, horizon, validate)
    return _build_actioncond_v1_features_numpy(raw_actions, horizon, validate)


def _build_actioncond_v1_features_numpy(raw_actions: Any, horizon: int, validate: bool) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=np.float32)
    _validate_raw_action_shape(actions, horizon, check_finite=validate)
    batch, steps, _ = actions.shape
    dx_local = actions[:, :, 0]
    dy_local = actions[:, :, 1]
    dyaw = actions[:, :, 2]
    yaw_cum = np.cumsum(dyaw, axis=1, dtype=np.float32)
    yaw_before = np.concatenate([np.zeros((batch, 1), dtype=np.float32), yaw_cum[:, :-1]], axis=1)
    cos_yaw = np.cos(yaw_before)
    sin_yaw = np.sin(yaw_before)
    dx_global = dx_local * cos_yaw - dy_local * sin_yaw
    dy_global = dx_local * sin_yaw + dy_local * cos_yaw
    x_cum = np.cumsum(dx_global, axis=1, dtype=np.float32)
    y_cum = np.cumsum(dy_global, axis=1, dtype=np.float32)
    poses = np.stack([x_cum, y_cum, yaw_cum], axis=2).astype(np.float32, copy=False)
    step_features = _numpy_step_features(batch, steps)
    features = np.concatenate([actions, poses, step_features], axis=2).astype(np.float32, copy=False)
    _validate_actioncond_v1_features(features, check_finite=validate)
    return features


def _build_actioncond_v1_features_torch(raw_actions: Any, horizon: int, validate: bool) -> Any:
    import torch

    actions = raw_actions.to(dtype=torch.float32)
    _validate_raw_action_shape(actions, horizon, check_finite=validate)
    batch, steps, _ = actions.shape
    dx_local = actions[:, :, 0]
    dy_local = actions[:, :, 1]
    dyaw = actions[:, :, 2]
    yaw_cum = torch.cumsum(dyaw, dim=1)
    yaw_before = torch.cat([torch.zeros((batch, 1), dtype=actions.dtype, device=actions.device), yaw_cum[:, :-1]], dim=1)
    cos_yaw = torch.cos(yaw_before)
    sin_yaw = torch.sin(yaw_before)
    dx_global = dx_local * cos_yaw - dy_local * sin_yaw
    dy_global = dx_local * sin_yaw + dy_local * cos_yaw
    x_cum = torch.cumsum(dx_global, dim=1)
    y_cum = torch.cumsum(dy_global, dim=1)
    poses = torch.stack([x_cum, y_cum, yaw_cum], dim=2)
    step_features = _torch_step_features(batch, steps, actions.device, actions.dtype)
    features = torch.cat([actions, poses, step_features], dim=2)
    _validate_actioncond_v1_features(features, check_finite=validate)
    return features


def _numpy_step_features(batch: int, steps: int) -> np.ndarray:
    if steps <= 1:
        step01 = np.zeros(steps, dtype=np.float32)
    else:
        step01 = np.arange(steps, dtype=np.float32) / np.float32(steps - 1)
    phase = np.float32(2.0 * np.pi) * step01
    per_step = np.stack([step01, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
    return np.broadcast_to(per_step[None, :, :], (batch, steps, 3)).copy()


def _torch_step_features(batch: int, steps: int, device: Any, dtype: Any) -> Any:
    import torch

    key = (steps, str(device), str(dtype))
    per_step = _TORCH_STEP_FEATURE_CACHE.get(key)
    if per_step is None:
        if steps <= 1:
            step01 = torch.zeros(steps, device=device, dtype=dtype)
        else:
            step01 = torch.arange(steps, device=device, dtype=dtype) / float(steps - 1)
        phase = (2.0 * np.pi) * step01
        per_step = torch.stack([step01, torch.sin(phase), torch.cos(phase)], dim=1)[None, :, :]
        _TORCH_STEP_FEATURE_CACHE[key] = per_step
    return per_step.expand(batch, steps, 3)


def _validate_raw_action_shape(actions: Any, horizon: int, check_finite: bool = True) -> None:
    if actions.ndim != 3:
        raise ValueError(f"raw_actions must have shape [B, H, 3], got ndim={actions.ndim}")
    if actions.shape[1] != horizon:
        raise ValueError(f"ActionCond-v1 A0-split expects H={horizon}, got H={actions.shape[1]}")
    if actions.shape[2] != 3:
        raise ValueError(f"raw_actions must have action dim 3 [dx_local, dy_local, dyaw], got {actions.shape[2]}")
    if not check_finite:
        return
    if _is_torch_tensor(actions):
        import torch

        if not bool(torch.isfinite(actions).all()):
            raise ValueError("raw_actions contain NaN or Inf")
    elif not np.isfinite(actions).all():
        raise ValueError("raw_actions contain NaN or Inf")


def _validate_actioncond_v1_features(features: Any, check_finite: bool = True) -> None:
    if features.ndim != 3 or features.shape[2] != len(ACTIONCOND_V1_FEATURE_NAMES):
        raise ValueError(
            f"ActionCond-v1 features must have shape [B, H, {len(ACTIONCOND_V1_FEATURE_NAMES)}], "
            f"got {tuple(features.shape)}"
        )
    if not check_finite:
        return
    if _is_torch_tensor(features):
        import torch

        finite = bool(torch.isfinite(features).all())
    else:
        finite = bool(np.isfinite(features).all())
    if not finite:
        raise ValueError("ActionCond-v1 features contain NaN or Inf")


def one_hot_atomic_labels(labels: list[str], vocab: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Encode atomic labels as one-hot rows and return the vocabulary used."""
    if vocab is None:
        vocab = list(DEFAULT_ATOMIC_LABELS)
        for label in sorted(set(labels)):
            if label not in vocab:
                vocab.append(label)
    label_to_index = {label: i for i, label in enumerate(vocab)}
    one_hot = np.zeros((len(labels), len(vocab)), dtype=np.float32)
    for i, label in enumerate(labels):
        if label not in label_to_index:
            raise ValueError(f"unknown atomic label {label!r}; known labels={vocab}")
        one_hot[i, label_to_index[label]] = 1.0
    return one_hot, vocab


@dataclass
class ActionNormalizer:
    """Standardize continuous actions using train-split statistics."""

    mean: np.ndarray
    std: np.ndarray
    clip_low: np.ndarray | None = None
    clip_high: np.ndarray | None = None
    eps: float = 1e-6

    @classmethod
    def fit(
        cls,
        actions: np.ndarray,
        clip: bool = False,
        clip_percentiles: tuple[float, float] = (1.0, 99.0),
        eps: float = 1e-6,
    ) -> "ActionNormalizer":
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [N, A], got {actions.shape}")
        clip_low = clip_high = None
        fit_actions = actions
        if clip:
            clip_low = np.percentile(actions, clip_percentiles[0], axis=0).astype(np.float32)
            clip_high = np.percentile(actions, clip_percentiles[1], axis=0).astype(np.float32)
            fit_actions = np.clip(actions, clip_low, clip_high)
        mean = fit_actions.mean(axis=0).astype(np.float32)
        std = fit_actions.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std, clip_low=clip_low, clip_high=clip_high, eps=eps)

    def transform(self, actions: Any) -> Any:
        if _is_torch_tensor(actions):
            import torch

            mean = torch.as_tensor(self.mean, device=actions.device, dtype=actions.dtype)
            std = torch.as_tensor(self.std, device=actions.device, dtype=actions.dtype)
            x = actions
            if self.clip_low is not None and self.clip_high is not None:
                low = torch.as_tensor(self.clip_low, device=actions.device, dtype=actions.dtype)
                high = torch.as_tensor(self.clip_high, device=actions.device, dtype=actions.dtype)
                x = torch.clamp(x, low, high)
            return (x - mean) / std

        x = np.asarray(actions, dtype=np.float32)
        if self.clip_low is not None and self.clip_high is not None:
            x = np.clip(x, self.clip_low, self.clip_high)
        return (x - self.mean) / self.std

    def inverse_transform(self, actions: Any) -> Any:
        if _is_torch_tensor(actions):
            import torch

            mean = torch.as_tensor(self.mean, device=actions.device, dtype=actions.dtype)
            std = torch.as_tensor(self.std, device=actions.device, dtype=actions.dtype)
            return actions * std + mean
        return np.asarray(actions, dtype=np.float32) * self.std + self.mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_low": None if self.clip_low is None else self.clip_low.tolist(),
            "clip_high": None if self.clip_high is None else self.clip_high.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ActionNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            clip_low=None if state.get("clip_low") is None else np.asarray(state["clip_low"], dtype=np.float32),
            clip_high=None if state.get("clip_high") is None else np.asarray(state["clip_high"], dtype=np.float32),
            eps=float(state.get("eps", 1e-6)),
        )


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def action_dim_indices(preset: str) -> list[int]:
    """Return continuous-action column indices for a named ablation preset."""
    if preset not in ACTION_DIM_PRESETS:
        options = ", ".join(sorted(ACTION_DIM_PRESETS))
        raise ValueError(f"unknown action dim preset {preset!r}; expected one of: {options}")
    return list(ACTION_DIM_PRESETS[preset])


def action_dim_names(preset: str) -> list[str]:
    return [CONTINUOUS_ACTION_NAMES[index] for index in action_dim_indices(preset)]


def select_action_dims(actions: Any, preset: str) -> Any:
    """Select continuous-action dimensions from an array or tensor."""
    indices = action_dim_indices(preset)
    if _is_torch_tensor(actions):
        import torch

        index = torch.as_tensor(indices, device=actions.device, dtype=torch.long)
        return torch.index_select(actions, dim=-1, index=index)
    return np.take(np.asarray(actions), indices, axis=-1)
