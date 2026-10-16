#!/usr/bin/env python3
"""Mechanically split an exported 123-D feature block by source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.artifacts:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        if "latent_features" not in arrays:
            required = {"raw_ltwm_features", "scorer_v2_aux_features", "support_features"}
            if required.issubset(arrays):
                print(f"already split: {path}")
                continue
            raise RuntimeError(f"{path} has neither combined nor split feature fields")
        combined = arrays.pop("latent_features")
        if combined.ndim != 3 or combined.shape[-1] != 123:
            raise RuntimeError(f"unexpected combined feature shape: {combined.shape}")
        arrays["raw_ltwm_features"] = combined[..., :112].astype(np.float32)
        arrays["scorer_v2_aux_features"] = combined[..., 112:122].astype(np.float32)
        arrays["support_features"] = combined[..., 122:123].astype(np.float32)
        metadata = json.loads(str(arrays["metadata"].item()))
        metadata["feature_dimensions"] = {
            "action": int(arrays["action_features"].shape[-1]),
            "static": int(arrays["static_features"].shape[-1]),
            "raw_ltwm": 112,
            "scorer_v2_aux": 10,
            "support": 1,
        }
        metadata["feature_separation"] = {
            "raw_ltwm_features": (
                "three-seed LTWM rollout scalar features summarized by mean/std/min/max; "
                "no scorer-v2 predictions"
            ),
            "scorer_v2_aux_features": (
                "frozen scorer-v2 ATE/RTP predictions for three seeds plus mean/std; "
                "stacked ablation only"
            ),
            "support_features": (
                "three-seed action-support acceptance fraction; audit-only, "
                "excluded from learned features"
            ),
        }
        metadata["mechanically_split_without_reencoding"] = True
        arrays["metadata"] = np.asarray(canonical_json(metadata))
        temporary = path.with_suffix(path.suffix + ".split.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
        audit_path = path.with_suffix(".audit.json")
        metadata["output"] = str(path)
        metadata["output_sha256"] = file_sha256(path)
        audit_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"split {path} sha256={metadata['output_sha256']}")


if __name__ == "__main__":
    main()
