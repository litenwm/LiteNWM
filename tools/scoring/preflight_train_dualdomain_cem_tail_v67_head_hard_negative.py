#!/usr/bin/env python3
"""Preflight the sealed v67 head-only hard-negative training intervention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    DOMAINS,
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_targets,
)
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as training  # noqa: E402


OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v67_head_hard_negative/"
    "training_preflight.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prereg = training.read_json(training.PREREG)
    if prereg.get("status") != "frozen_before_v67_training":
        raise RuntimeError("v67 preregistration changed")
    base = training.read_json(training.BASE_RESULT)
    envelopes = training.implementation.load_envelopes(Path(base["support"]))
    train = training.implementation.load_domain_split("scand", "train_fit", envelopes)
    targets = build_universal_targets(train["targets_raw"])
    hard, hard_audit = training.load_scand_hard_negatives(train, targets)
    weights = training.scand_state_weights(train["group"], train["set_type"])
    per_set_weight = {
        name: float(weights[train["set_type"].astype(str) == name].sum())
        for name in training.SET_WEIGHTS
    }
    initial = base["arms"]["D"]["checkpoints"]["full"][0]
    payload = torch.load(initial["path"], map_location="cpu", weights_only=False)
    model = UniversalDualDomainRawV2(UniversalModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.heads[DOMAINS.index("scand")].parameters():
        parameter.requires_grad_(True)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(not name.startswith("heads.1.") for name in trainable_names):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names}")
    if hard_audit["total"] <= 0 or hard[:, 0].any():
        raise RuntimeError("hard-negative contract failed")
    result = {
        "schema": "dualdomain_cem_tail_v67_head_hard_negative_preflight",
        "status": "passed_before_v67_training",
        "states": int(len(train["feature"])),
        "candidate_sets": 4,
        "hard_negative_audit": hard_audit,
        "set_weight_sums": per_set_weight,
        "state_weight_sum": float(weights.sum()),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "recon_head_trainable": False,
        "shared_trunk_trainable": False,
        "preregistration_sha256": sha256(training.PREREG),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
