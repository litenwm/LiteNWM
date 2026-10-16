#!/usr/bin/env python3
import json
from pathlib import Path

import torch


root = Path("/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain/checkpoints/project/ltwm_expansion")
keys = (
    "epochs",
    "batches_per_epoch",
    "batch_size",
    "eval_batch_size",
    "recon_batches_per_scand",
    "latent_root",
    "scand_latent_root",
    "lr",
    "seed",
    "objective_revision",
    "warm_start_checkpoint",
)
output = {}
for seed in (1701, 1702, 1703):
    payload = torch.load(root / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    output[str(seed)] = {
        "args": {key: str(args.get(key)) for key in keys},
        "history": [
            {
                "epoch": row.get("epoch"),
                "batches": row.get("train", {}).get("batches"),
                "domain_batches": row.get("train", {}).get("domain_batches"),
                "selection_score": row.get("selection_score"),
            }
            for row in payload.get("history", [])
        ],
    }
print(json.dumps(output, indent=2))
