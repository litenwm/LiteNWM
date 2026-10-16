#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch


REPO = Path("/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain")


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def main() -> None:
    report = {}
    for seed in (1701, 1702, 1703):
        path = REPO / "checkpoints" / "project" / "ltwm_expansion" / f"seed{seed}" / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        history = payload.get("history", [])
        report[str(seed)] = {
            "checkpoint": str(path),
            "payload_keys": sorted(payload),
            "args": safe(args),
            "history_rows": len(history),
            "history": safe(history),
            "parameter_count": payload.get("parameter_count"),
            "split_lock_sha256": payload.get("split_lock_sha256"),
            "config": safe(payload.get("config", {})),
            "loss_weights": safe(payload.get("loss_weights", {})),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
