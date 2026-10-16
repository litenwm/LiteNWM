#!/usr/bin/env python3
"""Smoke-test the independent three-domain proposal-aware V70 model contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import feature_normalization  # noqa: E402
from tools.tridomain_v70.train_proposal_aware_cem_risk_tridomain import (  # noqa: E402
    load_initial_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    model, audit = load_initial_model(
        args.checkpoint,
        args.device,
        initialization_seed=args.seed,
    )
    if model.config.num_domains != 3:
        raise RuntimeError("model does not expose three independent domains")
    if model.domain_embedding.weight.shape != (3, 16):
        raise RuntimeError("domain embedding table is not [3,16]")
    if len(model.heads) != 3:
        raise RuntimeError("model does not have three domain-specific heads")

    feature = np.zeros((3, 2, 815), dtype=np.float32)
    domain = np.asarray((0, 1, 2), dtype=np.int64)
    normalization = feature_normalization(
        feature,
        domain,
        domain_conditioned=True,
    )
    if normalization["feature_mean"].shape != (3, 815):
        raise RuntimeError("feature normalization is not three-domain")

    model.eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(feature).to(args.device),
            torch.from_numpy(domain).to(args.device),
        )
    for key, value in output.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite output from domain-2 smoke: {key}")

    result = {
        "status": "TRIDOMAIN_MODEL_CONTRACT_OK",
        "model_domain_index": {"recon": 0, "scand": 1, "sacson": 2},
        "num_domains": model.config.num_domains,
        "domain_embedding_shape": list(model.domain_embedding.weight.shape),
        "domain_specific_heads": len(model.heads),
        "feature_normalization_shape": list(
            normalization["feature_mean"].shape
        ),
        "forward_output_shapes": {
            key: list(value.shape) for key, value in output.items()
        },
        "initialization": audit,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
