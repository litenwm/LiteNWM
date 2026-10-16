"""Streaming windows for high-resolution semantic/geometry latent shards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from recon_latents.io import read_json


@dataclass(frozen=True)
class _Trajectory:
    trajectory_id: str
    shard_path: Path
    num_frames: int


def _flatten_tokens(value: np.ndarray) -> np.ndarray:
    array = value.astype(np.float32, copy=False)
    return np.ascontiguousarray(
        array.reshape(*array.shape[:-3], array.shape[-3] * array.shape[-2], array.shape[-1])
    )


class HighResDualLatentWindowDataset(Dataset):
    def __init__(
        self,
        latent_root: Path,
        split: str,
        history_len: int = 4,
        horizon: int = 4,
        stream_cache_shards: int = 1,
    ) -> None:
        if history_len < 2 or horizon < 1:
            raise ValueError("history must be >=2 and horizon must be positive")
        if stream_cache_shards < 0:
            raise ValueError("stream cache must be non-negative")
        self.latent_root = Path(latent_root)
        self.split = split
        self.history_len = history_len
        self.horizon = horizon
        self.stream_cache_shards = stream_cache_shards
        self.metadata = read_json(self.latent_root / "metadata.json")
        self.splits = read_json(self.latent_root / "splits.json")
        representation = self.metadata["representation"]
        self.grid_height = int(representation["grid_height"])
        self.grid_width = int(representation["grid_width"])
        self.semantic_dim = int(representation["semantic_dim"])
        self.geometry_dim = int(representation["geometry_dim"])
        allowed = set(map(str, self.splits[split]))
        self.trajectories: list[_Trajectory] = []
        self.windows: list[tuple[int, int]] = []
        self._window_indices_by_traj: list[list[int]] = []
        self._cache: OrderedDict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = OrderedDict()
        for row in self.metadata["trajectories"]:
            trajectory_id = str(row["trajectory_id"])
            if trajectory_id not in allowed:
                continue
            num_frames = int(row["num_frames"])
            max_start = num_frames - history_len - horizon
            if max_start < 0:
                continue
            trajectory_index = len(self.trajectories)
            self.trajectories.append(
                _Trajectory(
                    trajectory_id,
                    self.latent_root / "shards" / str(row["shard_file"]),
                    num_frames,
                )
            )
            indices: list[int] = []
            self._window_indices_by_traj.append(indices)
            for history_start in range(max_start + 1):
                indices.append(len(self.windows))
                self.windows.append((trajectory_index, history_start))

    def __len__(self) -> int:
        return len(self.windows)

    def _load(
        self,
        trajectory_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(trajectory_index)
        if cached is not None:
            self._cache.move_to_end(trajectory_index)
            return cached
        trajectory = self.trajectories[trajectory_index]
        with np.load(trajectory.shard_path, allow_pickle=False) as shard:
            semantic = shard["semantic_latents"].copy()
            geometry = shard["geometry_latents"].copy()
            actions = shard["actions_continuous"].astype(np.float32, copy=True)
        if self.stream_cache_shards > 0:
            self._cache[trajectory_index] = (semantic, geometry, actions)
            self._cache.move_to_end(trajectory_index)
            while len(self._cache) > self.stream_cache_shards:
                self._cache.popitem(last=False)
        return semantic, geometry, actions

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_index, history_start = self.windows[index]
        trajectory = self.trajectories[trajectory_index]
        semantic, geometry, actions_all = self._load(trajectory_index)
        t = history_start + self.history_len - 1
        semantic_history = _flatten_tokens(
            semantic[history_start : t + 1]
        )
        geometry_history = _flatten_tokens(
            geometry[history_start : t + 1]
        )
        semantic_future = _flatten_tokens(
            semantic[t + 1 : t + self.horizon + 1]
        )
        geometry_future = _flatten_tokens(
            geometry[t + 1 : t + self.horizon + 1]
        )
        actions = np.ascontiguousarray(
            actions_all[t : t + self.horizon],
            dtype=np.float32,
        )
        return {
            "semantic_history": torch.from_numpy(semantic_history),
            "geometry_history": torch.from_numpy(geometry_history),
            "semantic_future": torch.from_numpy(semantic_future),
            "geometry_future": torch.from_numpy(geometry_future),
            "actions": torch.from_numpy(actions),
            "trajectory_id": trajectory.trajectory_id,
            "timestep": int(t),
        }

    def window_indices_by_shard(self) -> list[list[int]]:
        return self._window_indices_by_traj

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_class": type(self).__name__,
            "split": self.split,
            "num_shards": len(self.trajectories),
            "num_windows": len(self.windows),
            "history_len": self.history_len,
            "horizon": self.horizon,
            "grid_shape": [self.grid_height, self.grid_width],
            "semantic_dim": self.semantic_dim,
            "geometry_dim": self.geometry_dim,
        }

