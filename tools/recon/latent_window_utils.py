"""Shared latent-window helpers for RECON DINOv2 experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from recon_latents.io import read_json


def nhwc_to_chw(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(frame.astype(np.float32, copy=False), (2, 0, 1)))


class CachedLatentWindowDataset(Dataset):
    """In-memory window dataset for repeated small latent rollout experiments."""

    def __init__(self, latent_root: Path, split: str, history_len: int, horizon: int) -> None:
        self.latent_root = latent_root
        self.split = split
        self.history_len = history_len
        self.horizon = horizon
        self.metadata = read_json(latent_root / "metadata.json")
        self.splits = read_json(latent_root / "splits.json")
        allowed = set(self.splits[split])
        self.trajectories: list[dict[str, Any]] = []
        self.windows: list[tuple[int, int]] = []

        for traj in self.metadata["trajectories"]:
            if traj["trajectory_id"] not in allowed:
                continue
            with np.load(latent_root / "shards" / traj["shard_file"], allow_pickle=False) as shard:
                record = {
                    "trajectory_id": str(traj["trajectory_id"]),
                    "z": shard["patch_latents"].copy(),
                    "actions": shard["actions_continuous"].astype(np.float32, copy=True),
                }
            traj_index = len(self.trajectories)
            self.trajectories.append(record)
            max_start = len(record["z"]) - history_len - horizon
            for history_start in range(max_start + 1):
                self.windows.append((traj_index, history_start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_index, history_start = self.windows[index]
        record = self.trajectories[traj_index]
        t = history_start + self.history_len - 1
        z_t = nhwc_to_chw(record["z"][t])
        z_future = np.stack(
            [nhwc_to_chw(frame) for frame in record["z"][t + 1 : t + self.horizon + 1]],
            axis=0,
        )
        actions = record["actions"][t : t + self.horizon]
        return {
            "z_t": torch.from_numpy(z_t),
            "z_future": torch.from_numpy(np.ascontiguousarray(z_future)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "trajectory_id": record["trajectory_id"],
            "timestep": int(t),
        }

    def all_actions(self) -> np.ndarray:
        return np.concatenate([record["actions"] for record in self.trajectories], axis=0)


@dataclass(frozen=True)
class _StreamingTrajectory:
    trajectory_id: str
    shard_path: Path
    num_frames: int


class StreamingLatentWindowDataset(Dataset):
    """Memory-bounded latent-window dataset for full RECON.

    This class builds only lightweight metadata/window indices at
    initialization. Patch latents are read from the requested shard on demand,
    with an optional LRU cache bounded by shard count.
    """

    def __init__(
        self,
        latent_root: Path,
        split: str,
        history_len: int,
        horizon: int,
        stream_cache_shards: int = 8,
    ) -> None:
        if history_len < 1 or horizon < 1:
            raise ValueError("history_len and horizon must be >= 1")
        if stream_cache_shards < 0:
            raise ValueError("stream_cache_shards must be >= 0")

        self.latent_root = latent_root
        self.split = split
        self.history_len = history_len
        self.horizon = horizon
        self.stream_cache_shards = int(stream_cache_shards)
        self.metadata = read_json(latent_root / "metadata.json")
        self.splits = read_json(latent_root / "splits.json")
        allowed = set(self.splits[split])
        self.trajectories: list[_StreamingTrajectory] = []
        self.windows: list[tuple[int, int]] = []
        self._window_indices_by_traj: list[list[int]] = []
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.reset_cache_stats()

        for traj in self.metadata["trajectories"]:
            trajectory_id = str(traj["trajectory_id"])
            if trajectory_id not in allowed:
                continue
            num_frames = int(traj["num_frames"])
            max_start = num_frames - history_len - horizon
            if max_start < 0:
                continue
            traj_index = len(self.trajectories)
            self.trajectories.append(
                _StreamingTrajectory(
                    trajectory_id=trajectory_id,
                    shard_path=latent_root / "shards" / str(traj["shard_file"]),
                    num_frames=num_frames,
                )
            )
            self._window_indices_by_traj.append([])
            for history_start in range(max_start + 1):
                window_index = len(self.windows)
                self.windows.append((traj_index, history_start))
                self._window_indices_by_traj[traj_index].append(window_index)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_index, history_start = self.windows[index]
        traj = self.trajectories[traj_index]
        t = history_start + self.history_len - 1
        if self.stream_cache_shards > 0:
            z, actions_all = self._load_shard_cached(traj_index)
            z_t = nhwc_to_chw(z[t])
            z_future = np.stack(
                [nhwc_to_chw(frame) for frame in z[t + 1 : t + self.horizon + 1]],
                axis=0,
            )
            actions = actions_all[t : t + self.horizon]
        else:
            self._record_shard_load(traj_index)
            with np.load(traj.shard_path, allow_pickle=False) as shard:
                z_t = nhwc_to_chw(shard["patch_latents"][t])
                z_future = np.stack(
                    [nhwc_to_chw(frame) for frame in shard["patch_latents"][t + 1 : t + self.horizon + 1]],
                    axis=0,
                )
                actions = shard["actions_continuous"][t : t + self.horizon].astype(np.float32, copy=True)
        return {
            "z_t": torch.from_numpy(z_t),
            "z_future": torch.from_numpy(np.ascontiguousarray(z_future)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "trajectory_id": traj.trajectory_id,
            "timestep": int(t),
        }

    def _load_shard_cached(self, traj_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(traj_index)
        if cached is not None:
            self.cache_hit_count += 1
            self._cache.move_to_end(traj_index)
            return cached
        self.cache_miss_count += 1
        self._record_shard_load(traj_index)
        traj = self.trajectories[traj_index]
        with np.load(traj.shard_path, allow_pickle=False) as shard:
            z = shard["patch_latents"].copy()
            actions = shard["actions_continuous"].astype(np.float32, copy=True)
        self._cache[traj_index] = (z, actions)
        self._cache.move_to_end(traj_index)
        while len(self._cache) > self.stream_cache_shards:
            self._cache.popitem(last=False)
            self.cache_eviction_count += 1
        return z, actions

    def _record_shard_load(self, traj_index: int) -> None:
        self.shard_load_count += 1
        self.unique_shards_loaded.add(traj_index)
        try:
            self.approximate_bytes_read += self.trajectories[traj_index].shard_path.stat().st_size
        except OSError:
            pass

    def reset_cache_stats(self) -> None:
        self.shard_load_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_eviction_count = 0
        self.unique_shards_loaded: set[int] = set()
        self.approximate_bytes_read = 0

    def cache_stats(self) -> dict[str, Any]:
        total_requests = self.cache_hit_count + self.cache_miss_count
        return {
            "shard_load_count": int(self.shard_load_count),
            "cache_hit_count": int(self.cache_hit_count),
            "cache_miss_count": int(self.cache_miss_count),
            "cache_eviction_count": int(self.cache_eviction_count),
            "unique_shards_loaded": int(len(self.unique_shards_loaded)),
            "approximate_bytes_read": int(self.approximate_bytes_read),
            "cache_hit_rate": float(self.cache_hit_count / total_requests) if total_requests else None,
            "cached_shards": int(len(self._cache)),
            "stream_cache_shards": int(self.stream_cache_shards),
        }

    def window_indices_by_shard(self) -> list[list[int]]:
        return self._window_indices_by_traj

    def all_actions(self) -> np.ndarray:
        rows = []
        for traj in self.trajectories:
            with np.load(traj.shard_path, allow_pickle=False) as shard:
                rows.append(shard["actions_continuous"].astype(np.float32, copy=True))
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), dtype=np.float32)

    def iter_action_windows(self, batch_size: int = 8192) -> Iterator[np.ndarray]:
        """Yield raw action windows without reading patch latents."""
        rows: list[np.ndarray] = []
        for traj_index, group in _groupby_sorted_window_index(self.windows):
            traj = self.trajectories[traj_index]
            with np.load(traj.shard_path, allow_pickle=False) as shard:
                actions = shard["actions_continuous"].astype(np.float32, copy=False)
                for history_start in group:
                    t = history_start + self.history_len - 1
                    rows.append(np.asarray(actions[t : t + self.horizon], dtype=np.float32))
                    if len(rows) >= batch_size:
                        yield np.stack(rows, axis=0)
                        rows.clear()
        if rows:
            yield np.stack(rows, axis=0)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_class": type(self).__name__,
            "split": self.split,
            "num_shards": len(self.trajectories),
            "num_windows": len(self.windows),
            "history_len": self.history_len,
            "horizon": self.horizon,
            "stream_cache_shards": self.stream_cache_shards,
        }


class ShardAwareBatchSampler(Sampler[list[int]]):
    """Yield shard-local batches from a streaming latent-window dataset."""

    def __init__(
        self,
        dataset: StreamingLatentWindowDataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
        shard_batch_mode: str = "single",
        shard_bucket_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if shard_bucket_size < 1:
            raise ValueError("shard_bucket_size must be >= 1")
        if shard_batch_mode not in {"single", "bucket"}:
            raise ValueError("shard_batch_mode must be 'single' or 'bucket'")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shard_batch_mode = shard_batch_mode
        self.shard_bucket_size = int(shard_bucket_size)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shard_order = np.arange(len(self.dataset.window_indices_by_shard()))
        rng.shuffle(shard_order)
        bucket_size = self.shard_bucket_size if self.shard_batch_mode == "bucket" else 1
        for start in range(0, len(shard_order), bucket_size):
            bucket_indices = shard_order[start : start + bucket_size]
            bucket_windows: list[int] = []
            for shard_index in bucket_indices:
                windows = np.asarray(self.dataset.window_indices_by_shard()[int(shard_index)], dtype=np.int64)
                if windows.size == 0:
                    continue
                rng.shuffle(windows)
                bucket_windows.extend(int(index) for index in windows.tolist())
            if self.shard_batch_mode == "bucket" and len(bucket_windows) > self.batch_size:
                rng.shuffle(bucket_windows)
            for batch_start in range(0, len(bucket_windows), self.batch_size):
                batch = bucket_windows[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        total = 0
        if self.shard_batch_mode == "single":
            for windows in self.dataset.window_indices_by_shard():
                total += len(windows) // self.batch_size
                if len(windows) % self.batch_size and not self.drop_last:
                    total += 1
            return total
        bucket_size = self.shard_bucket_size
        grouped = self.dataset.window_indices_by_shard()
        for start in range(0, len(grouped), bucket_size):
            count = sum(len(windows) for windows in grouped[start : start + bucket_size])
            total += count // self.batch_size
            if count % self.batch_size and not self.drop_last:
                total += 1
        return total


def _groupby_sorted_window_index(windows: list[tuple[int, int]]) -> Iterator[tuple[int, Iterator[int]]]:
    idx = 0
    while idx < len(windows):
        traj_index = windows[idx][0]
        start = idx
        while idx < len(windows) and windows[idx][0] == traj_index:
            idx += 1
        yield traj_index, (history_start for _, history_start in islice(windows, start, idx))


def make_subset_thresholds(actions: np.ndarray) -> dict[str, float]:
    abs_dyaw = np.abs(actions[:, 2])
    dx = actions[:, 0]
    return {
        "abs_dyaw_median": float(np.median(abs_dyaw)),
        "abs_dyaw_p75": float(np.quantile(abs_dyaw, 0.75)),
        "abs_dyaw_p90": float(np.quantile(abs_dyaw, 0.90)),
        "dx_p25": float(np.quantile(dx, 0.25)),
        "dx_p75": float(np.quantile(dx, 0.75)),
    }
