#!/usr/bin/env python3
"""Export label-free, route-disjoint dense-meta predictions for union65.

For a SCAND route represented in the original 7,489-state train-fit split,
inference uses only the frozen member whose held-out family fold excluded that
entire route.  Routes absent from train-fit use the five-member ensemble.
This prevents exact-state and same-route leakage without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.dense_meta_v1 import family_fold, feature_by_mode, predict  # noqa: E402
from tools.evaluation import (  # noqa: E402
    export_scand_union65_frozen_densemeta_transfer_v1 as base,
)


TRAIN_FEATURES = (
    ROOT
    / "outputs/scand_domain_adaptation_v1"
    / "features/train_fit/scand_inner_train_fit_spacing038_features.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    table = base.load_npz(args.table)
    record = table["record_key"].astype(str)
    route = table["route_group"].astype(str)
    cohort = table["cohort"].astype(str)
    candidates = table["candidates_feature_waypoint_units"].astype(np.float32)
    assembled: dict[str, np.ndarray | None] = {
        key: None for key in base.FEATURE_KEYS
    }
    cohort_audit = {}
    for name, paths in base.cohort_specs():
        local, audit = base.reconstruct_cohort(paths, record, candidates)
        states = np.flatnonzero(cohort == name)
        if not np.array_equal(states, audit["table_state_index"]):
            raise RuntimeError(f"cohort/table ordering changed: {name}")
        for key in base.FEATURE_KEYS:
            if assembled[key] is None:
                assembled[key] = np.empty(
                    (len(record), candidates.shape[1], local[key].shape[-1]),
                    dtype=np.float32,
                )
            assert assembled[key] is not None
            assembled[key][states] = local[key]
        cohort_audit[name] = {
            key: value
            for key, value in audit.items()
            if key not in {"record_key", "table_state_index"}
        }
    features = {
        key: value for key, value in assembled.items() if value is not None
    }
    if len(features) != len(base.FEATURE_KEYS):
        raise RuntimeError("not all transfer features were reconstructed")
    if any(not np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("transfer features contain NaN or Inf")

    train = base.load_npz(TRAIN_FEATURES)
    train_record = train["record_key"].astype(str)
    train_family = set(train["family"].astype(str).tolist())
    transfer_family = np.asarray([f"scand::{value}" for value in route])
    seen_route = np.asarray(
        [value in train_family for value in transfer_family], dtype=bool
    )
    route_disjoint_member = np.asarray(
        [
            family_fold(value, 5) if seen else -1
            for value, seen in zip(transfer_family, seen_route)
        ],
        dtype=np.int8,
    )
    exact_overlap = np.isin(record, train_record)

    arrays: dict[str, np.ndarray] = {
        "record_key": record,
        "route_group": route,
        "formal_v70_index": table["formal_v70_index"].astype(np.int16),
        "seen_train_route": seen_route,
        "route_disjoint_member": route_disjoint_member,
        "exact_train_record_overlap": exact_overlap,
    }
    checkpoint_audit = {}
    for mode in base.MODES:
        feature = feature_by_mode(features, mode, variant="raw")
        members, member_audit = base.load_members(mode, device)
        checkpoint_audit[mode] = member_audit
        predictions_by_fold = {}
        for (model, stats), audit in zip(members, member_audit, strict=True):
            heldout = int(audit["heldout_family_fold"])
            predictions_by_fold[heldout] = predict(
                model,
                stats,
                feature,
                np.arange(len(feature)),
                device=str(device),
            )
        if set(predictions_by_fold) != set(range(5)):
            raise RuntimeError(f"{mode} checkpoint folds changed")
        for key in base.PREDICTION_KEYS:
            stacked = np.stack(
                [predictions_by_fold[fold][key] for fold in range(5)]
            )
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            for heldout in range(5):
                state = route_disjoint_member == heldout
                mean[state] = stacked[heldout, state]
                std[state] = 0.0
            arrays[f"{mode}_{key}_mean"] = mean.astype(np.float32)
            arrays[f"{mode}_{key}_std"] = std.astype(np.float32)
        del members, predictions_by_fold
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    audit = {
        "schema": "scand_union65_route_disjoint_densemeta_transfer_v1",
        "status": "complete_label_free_transfer_predictions",
        "experiment_novelty": (
            "first route-disjoint application of the frozen 7489-state K17 "
            "GroupFirst models to the four-pool union65 candidate set"
        ),
        "route_disjoint_contract": (
            "seen SCAND train routes use only the frozen member trained with "
            "that route family held out; unseen routes use all five members"
        ),
        "union65_outcomes_used": False,
        "counterfactual_action_outcomes_used": False,
        "states": int(len(record)),
        "candidates": int(candidates.shape[1]),
        "seen_train_route_states": int(seen_route.sum()),
        "unseen_train_route_states": int((~seen_route).sum()),
        "exact_train_record_overlap_states": int(exact_overlap.sum()),
        "exact_overlap_uses_excluding_member_only": True,
        "member_state_counts": {
            str(fold): int((route_disjoint_member == fold).sum())
            for fold in range(5)
        },
        "cohorts": cohort_audit,
        "checkpoints": checkpoint_audit,
        "inputs": {
            "table": {
                "path": str(args.table.resolve()),
                "sha256": sha256_file(args.table),
            },
            "train_features": {
                "path": str(TRAIN_FEATURES.resolve()),
                "sha256": sha256_file(TRAIN_FEATURES),
            },
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(jsonable(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["output"], sort_keys=True))


if __name__ == "__main__":
    main()
