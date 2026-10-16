#!/usr/bin/env python3
"""Run the frozen reranker with its one documented latent-binding correction."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


_ORIGINAL_TRANSPOSE = torch.Tensor.transpose


def _transpose_or_permute(
    value: torch.Tensor,
    *dimensions: int,
) -> torch.Tensor:
    if len(dimensions) > 2:
        return value.permute(*dimensions)
    return _ORIGINAL_TRANSPOSE(value, *dimensions)


torch.Tensor.transpose = _transpose_or_permute

from tools.evaluation import run_scand_top10_latent_reranker_v1 as runner  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_PATH = (
    ROOT / "configs/scand/cem_top10_latent_reranker_v1.binding_amendment.json"
)
_ORIGINAL_READ_JSON = runner.read_json
_ORIGINAL_WRITE_JSON = runner.write_json_atomic
_AMENDMENT = _ORIGINAL_READ_JSON(AMENDMENT_PATH)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(runner.canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve(entry: dict) -> Path:
    path = (ROOT / entry["path"]).resolve()
    runner.require(path.is_file(), f"missing binding input: {path}")
    runner.require(
        runner.sha256_file(path) == entry["sha256"],
        f"binding input hash changed: {path}",
    )
    return path


def _verify_binding(contract: dict) -> None:
    table_path = _resolve(contract["inputs"]["runtime_table"])
    evidence_path = _resolve(_AMENDMENT["binding_evidence"])
    latent_path = _resolve(_AMENDMENT["correct_input"])
    start, end = _AMENDMENT["binding_evidence"]["runtime_table_slice"]
    with np.load(table_path, allow_pickle=False) as payload:
        expected = np.asarray(payload["record_key"])[start:end]
    with np.load(evidence_path, allow_pickle=False) as payload:
        observed = np.asarray(payload["record_key"])
    runner.require(
        np.array_equal(observed, expected),
        "corrected fresh-100 record keys do not exactly bind to table states 540:640",
    )
    with np.load(latent_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        runner.require(len(payload["history"]) == len(expected), "latent state count changed")
    runner.require(
        metadata["record_key_sha256"] == _array_sha256(observed),
        "latent cache does not bind to corrected record keys",
    )


def _read_json(path: Path) -> dict:
    value = _ORIGINAL_READ_JSON(path)
    if value.get("schema") != "scand_cem_top10_latent_reranker_contract_v1":
        return value
    amended = copy.deepcopy(value)
    original = amended["inputs"]["latent_caches"][1]
    replaced = _AMENDMENT["replaced_input"]
    runner.require(
        original["path"] == replaced["path"] and original["sha256"] == replaced["sha256"],
        "base contract no longer matches binding amendment",
    )
    amended["inputs"]["latent_caches"][1] = copy.deepcopy(_AMENDMENT["correct_input"])
    _verify_binding(amended)
    return amended


def _stream_normalizer(
    rich,
    actions,
    data,
    states,
    candidates,
    route,
    batch_size,
):
    weight = runner.coarse.state_weights(route, states)
    total = 0.0
    total_sum = None
    total_square = None
    token_dimension = 0
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        token = runner.token_batch(
            rich,
            actions,
            data,
            states[start:end],
            candidates[start:end],
        ).astype(np.float64)
        token_dimension = token.shape[-1]
        local_weight = weight[start:end, None, None, None, None]
        normalizer = token.shape[1] * token.shape[2] * token.shape[3]
        local_sum = (token * local_weight).sum(axis=(0, 1, 2, 3))
        local_square = ((token * token) * local_weight).sum(axis=(0, 1, 2, 3))
        local_total = float(weight[start:end].sum() * normalizer)
        total_sum = local_sum if total_sum is None else total_sum + local_sum
        total_square = (
            local_square if total_square is None else total_square + local_square
        )
        total += local_total
    runner.require(total_sum is not None and total_square is not None, "empty normalizer")
    mean = total_sum / total
    variance = np.maximum(total_square / total - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32), token_dimension


def _write_json(path: Path, value: dict) -> None:
    if value.get("schema") == "scand_top10_latent_reranker_nested_route_oof_result_v1":
        value = copy.deepcopy(value)
        value["inputs"]["binding_amendment"] = {
            "path": str(AMENDMENT_PATH),
            "sha256": runner.sha256_file(AMENDMENT_PATH),
        }
        wrapper = Path(__file__).resolve()
        value["inputs"]["binding_corrected_runner"] = {
            "path": str(wrapper),
            "sha256": runner.sha256_file(wrapper),
        }
        value["binding_correction_applied"] = True
    _ORIGINAL_WRITE_JSON(path, value)


runner.read_json = _read_json
runner.stream_normalizer = _stream_normalizer
runner.write_json_atomic = _write_json


if __name__ == "__main__":
    runner.main()
