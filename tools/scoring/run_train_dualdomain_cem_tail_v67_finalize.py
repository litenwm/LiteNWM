#!/usr/bin/env python3
"""Finalize existing v67 members after a metadata-only loader mismatch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scoring.universal_dualdomain_raw_v2 import (
    UniversalDualDomainRawV2,
    UniversalModelConfig,
)
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as implementation
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256
from tools.scoring.train_universal_dualdomain_raw_v2 import predict_member


SEAL = (
    implementation.ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v67_checkpoint_metadata_recovery.json"
)
_write_json_atomic = implementation.write_json_atomic


def read_seal():
    return json.loads(SEAL.read_text(encoding="utf-8"))


def load_member_compatible(record, device):
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("primary_excludes_scorer_v2_predictions"):
        seal = read_seal()
        expected = seal["checkpoint_sha256"].get(path.name)
        if (
            payload.get("schema") != "dualdomain_cem_tail_v67_scand_head_member"
            or payload.get("trainable_parameters")
            != "SCAND domain-specific head only"
            or expected != file_sha256(path)
        ):
            raise RuntimeError(f"unsealed raw-only metadata recovery: {path}")
    config = UniversalModelConfig(**payload["model_config"])
    model = UniversalDualDomainRawV2(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    target_norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["target_normalization"].items()
    }
    return model, normalization, target_norm


def ensemble_predict_compatible(records, feature, domain, *, device):
    predictions = []
    for record in records:
        member = load_member_compatible(record, device)
        predictions.append(
            predict_member(member, feature, domain, device=device)
        )
        del member
        if device == "cuda":
            torch.cuda.empty_cache()
    means = np.stack([value["mean"] for value in predictions])
    variances = np.stack([value["variance"] for value in predictions])
    ensemble_mean = means.mean(axis=0)
    total_variance = (
        variances.mean(axis=0)
        + np.square(means).mean(axis=0)
        - np.square(ensemble_mean)
    )
    return {
        "mean": ensemble_mean.astype(np.float32),
        "std": np.sqrt(np.maximum(total_variance, 1.0e-8)).astype(np.float32),
        **{
            key: np.stack([value[key] for value in predictions]).mean(axis=0)
            for key in (
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            )
        },
    }


def write_json_with_recovery(path, value):
    if value.get("schema") == "dualdomain_cem_tail_v67_head_hard_negative_training_result":
        value = {
            **value,
            "checkpoint_metadata_recovery": str(SEAL),
            "checkpoint_metadata_recovery_sha256": file_sha256(SEAL),
            "checkpoint_weights_mutated_by_recovery": False,
        }
    _write_json_atomic(path, value)


def main() -> None:
    seal = read_seal()
    if seal.get("status") != "frozen_before_scorer_cal_finalization":
        raise RuntimeError("metadata recovery seal changed")
    if seal.get("weights_mutated") or seal.get("members_retrained"):
        raise RuntimeError("metadata recovery is not weight preserving")
    if seal.get("A_to_H_accessed") or seal.get("reserve_accessed"):
        raise RuntimeError("metadata recovery reports forbidden access")
    implementation.ensemble_predict = ensemble_predict_compatible
    implementation.write_json_atomic = write_json_with_recovery
    implementation.main()


if __name__ == "__main__":
    main()
