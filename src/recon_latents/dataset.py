"""PyTorch dataset for offline RECON DINOv2 patch-latent shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class _Window:
    shard_path: Path
    trajectory_id: str
    history_start: int
    goal_timestep: int | None


class ReconDinoLatentDataset(Dataset):  # type: ignore[misc]
    """Load RECON latent dynamics windows.

    Disk latents are stored as NHWC float16. Returned tensors are float32.
    For a window whose history ends at timestep t:
      z_history = z[t-K+1 : t+1]
      actions = actions[t : t+H]
      z_future = z[t+1 : t+H+1]
    """

    def __init__(
        self,
        latent_root: str | Path,
        split: str = "train",
        history_len: int = 1,
        horizon: int = 1,
        stride: int = 1,
        channels_first: bool = True,
        include_goal: bool = False,
        goal_min_offset: int | None = None,
        action_mode: str = "continuous",
    ) -> None:
        if torch is None:
            raise ImportError("ReconDinoLatentDataset requires torch.")
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of: train, val, test, all")
        if history_len < 1 or horizon < 1 or stride < 1:
            raise ValueError("history_len, horizon, and stride must be >= 1")
        if action_mode not in {"continuous", "atomic", "continuous+atomic"}:
            raise ValueError("action_mode must be continuous, atomic, or continuous+atomic")

        self.latent_root = Path(latent_root)
        self.split = split
        self.history_len = history_len
        self.horizon = horizon
        self.stride = stride
        self.channels_first = channels_first
        self.include_goal = include_goal
        self.goal_min_offset = goal_min_offset
        self.action_mode = action_mode

        self.metadata = read_json(self.latent_root / "metadata.json")
        self.splits = read_json(self.latent_root / "splits.json")
        self.windows = self._build_windows()

    def _build_windows(self) -> list[_Window]:
        allowed_ids: set[str] | None = None
        if self.split != "all":
            allowed_ids = set(self.splits[self.split])

        windows: list[_Window] = []
        for traj in self.metadata["trajectories"]:
            trajectory_id = str(traj["trajectory_id"])
            if allowed_ids is not None and trajectory_id not in allowed_ids:
                continue
            num_frames = int(traj["num_frames"])
            max_start = num_frames - self.history_len - self.horizon
            if max_start < 0:
                continue
            shard_path = self.latent_root / "shards" / traj["shard_file"]
            for history_start in range(0, max_start + 1, self.stride):
                t = history_start + self.history_len - 1
                goal_timestep = None
                if self.include_goal:
                    offset = self.goal_min_offset if self.goal_min_offset is not None else self.horizon
                    goal_timestep = min(num_frames - 1, t + max(1, offset))
                windows.append(_Window(shard_path, trajectory_id, history_start, goal_timestep))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        start = window.history_start
        t = start + self.history_len - 1
        with np.load(window.shard_path, allow_pickle=False) as shard:
            z_history = shard["patch_latents"][start : t + 1]
            z_future = shard["patch_latents"][t + 1 : t + self.horizon + 1]
            actions = self._select_actions(shard, t)
            sample: dict[str, Any] = {
                "z_history": _latent_tensor(z_history, self.channels_first),
                "actions": torch.from_numpy(np.ascontiguousarray(actions.astype(np.float32, copy=False))),
                "z_future": _latent_tensor(z_future, self.channels_first),
                "trajectory_id": window.trajectory_id,
                "timestep": t,
            }
            if window.goal_timestep is not None:
                z_goal = shard["patch_latents"][window.goal_timestep]
                sample["z_goal"] = _latent_tensor(z_goal[None, ...], self.channels_first)[0]
            return sample

    def _select_actions(self, shard: Any, t: int) -> np.ndarray:
        continuous = shard["actions_continuous"][t : t + self.horizon]
        if self.action_mode == "continuous":
            return continuous
        atomic = shard["actions_atomic_onehot"][t : t + self.horizon]
        if self.action_mode == "atomic":
            return atomic
        return np.concatenate([continuous, atomic], axis=1)


def _latent_tensor(latents: np.ndarray, channels_first: bool) -> "torch.Tensor":
    arr = latents.astype(np.float32, copy=False)
    if channels_first:
        arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(arr))
