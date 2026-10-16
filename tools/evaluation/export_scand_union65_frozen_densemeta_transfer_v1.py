#!/usr/bin/env python3
"""Transfer the frozen 7.5k-state K17 dense-meta ensemble to union65.

This exporter does not fit on union65 outcomes.  It reconstructs the original
candidate features for each of the four proposal pools, aligns them to the
spent union65 table by exact record/candidate identity, and applies the frozen
SCAND domain-adaptation GroupFirst checkpoints.
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

from scoring.dense_meta_v1 import GroupFirstNet, feature_by_mode, predict  # noqa: E402
from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


OLD_ROOT = (
    ROOT
    / "outputs/cem_development_v1"
    / "scand_existing_train_upperbound_pilot_v1_corrected"
)
NEW_ROOT = (
    ROOT
    / "outputs/cem_development_v1"
    / "scand_new_routefit10_v4_development_pilot_v2_strict_identity"
)
CHECKPOINT_ROOT = ROOT / "outputs/scand_domain_adaptation_v1/training_v1/checkpoints"
FEATURE_KEYS = ("action_features", "static_features", "raw_ltwm_features")
MODES = ("action", "full")
PREDICTION_KEYS = (
    "outcome",
    "p_help",
    "p_safe",
    "p_harm",
    "p_select",
    "p_rank",
    "p_challenge",
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def exact_key(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float32).tobytes()


def fresh_paths() -> list[Path]:
    spec = next(item for item in v72.COHORTS if item["name"] == "scand_fresh_outer")
    replay = v72.V70_ROOT / "replays" / spec["name"] / "proposal_aware_v70"
    freeze = v72.read_json(replay / "prelabel_freeze.json")
    return [v72.resolve_path(freeze["base_features"])] + [
        replay / f"iteration_{iteration:02d}" / "cem_features.npz"
        for iteration in range(v72.ITERATIONS)
    ]


def cohort_specs() -> list[tuple[str, list[Path]]]:
    return [
        (
            "spent_old_train_sidecar",
            [
                OLD_ROOT
                / "base_features"
                / "scand_existing_train_upperbound_pilot_v1_corrected_features.npz"
            ]
            + [
                OLD_ROOT
                / "formal_logger_on"
                / f"iteration_{iteration:02d}"
                / "cem_features.npz"
                for iteration in range(3)
            ],
        ),
        ("spent_fresh_v70_v77", fresh_paths()),
        (
            "spent_new_mainrgb_development",
            [
                NEW_ROOT
                / "base_features"
                / "scand_new_routefit10_v4_development_pilot_v2_features.npz"
            ]
            + [
                NEW_ROOT
                / "formal_v70"
                / f"iteration_{iteration:02d}"
                / "cem_features.npz"
                for iteration in range(3)
            ],
        ),
    ]


def reconstruct_cohort(
    paths: list[Path],
    table_record: np.ndarray,
    table_candidates: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pools = [load_npz(path) for path in paths]
    for path, pool in zip(paths, pools, strict=True):
        missing = sorted({"record_key", "candidates", *FEATURE_KEYS} - set(pool))
        if missing:
            raise RuntimeError(f"missing {missing}: {path}")
    record = pools[0]["record_key"].astype(str)
    if any(not np.array_equal(record, pool["record_key"].astype(str)) for pool in pools):
        raise RuntimeError("pool record orders differ")
    table_position = {name: index for index, name in enumerate(table_record.astype(str))}
    if any(name not in table_position for name in record):
        raise RuntimeError("cohort record missing from union65 table")
    target_state = np.asarray([table_position[name] for name in record], dtype=np.int64)
    target_candidates = table_candidates[target_state].astype(np.float32)
    output = {
        key: np.empty(
            (len(record), target_candidates.shape[1], pools[0][key].shape[-1]),
            dtype=np.float32,
        )
        for key in FEATURE_KEYS
    }
    reconstructed = np.empty_like(target_candidates)
    for state in range(len(record)):
        source: dict[bytes, tuple[int, int]] = {}
        for pool_index, pool in enumerate(pools):
            for candidate_index, candidate in enumerate(pool["candidates"][state]):
                source.setdefault(exact_key(candidate), (pool_index, candidate_index))
        if len(source) != target_candidates.shape[1]:
            raise RuntimeError(
                f"unique candidate count changed for {record[state]}: {len(source)}"
            )
        for candidate_index, candidate in enumerate(target_candidates[state]):
            key = exact_key(candidate)
            if key not in source:
                raise RuntimeError(f"candidate alignment failed: {record[state]}")
            pool_index, local_index = source[key]
            reconstructed[state, candidate_index] = pools[pool_index]["candidates"][
                state, local_index
            ]
            for feature_key in FEATURE_KEYS:
                output[feature_key][state, candidate_index] = pools[pool_index][
                    feature_key
                ][state, local_index]
    if not np.array_equal(reconstructed, target_candidates):
        raise RuntimeError("candidate reconstruction is not exact")
    return output, {
        "states": int(len(record)),
        "record_key": record,
        "table_state_index": target_state,
        "paths": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
    }


def load_members(mode: str, device: torch.device) -> tuple[list[tuple[Any, Any]], list[dict[str, Any]]]:
    members = []
    audit = []
    paths = sorted(CHECKPOINT_ROOT.glob(f"domain_balanced_{mode}_fold*.pt"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {mode} members, got {len(paths)}")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "scand_domain_adaptation_groupfirst_member_v1":
            raise RuntimeError(f"checkpoint schema changed: {path}")
        if payload.get("domain_regime") != "domain_balanced" or payload.get("mode") != mode:
            raise RuntimeError(f"checkpoint binding changed: {path}")
        config = payload["config"]
        model = GroupFirstNet(
            int(payload["input_dim"]),
            int(config["hidden"]),
            float(config["dropout"]),
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        members.append((model.to(device).eval(), payload["normalization"]))
        audit.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "heldout_family_fold": int(payload["heldout_family_fold"]),
            }
        )
    return members, audit


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    table = load_npz(args.table)
    record = table["record_key"].astype(str)
    candidates = table["candidates_feature_waypoint_units"].astype(np.float32)
    cohort = table["cohort"].astype(str)
    assembled = {
        key: np.empty((len(record), candidates.shape[1], 0), dtype=np.float32)
        for key in FEATURE_KEYS
    }
    dimensions: dict[str, int] = {}
    cohort_audit = {}
    for name, paths in cohort_specs():
        local, audit = reconstruct_cohort(paths, record, candidates)
        states = np.flatnonzero(cohort == name)
        if not np.array_equal(states, audit["table_state_index"]):
            raise RuntimeError(f"cohort/table ordering changed: {name}")
        for key in FEATURE_KEYS:
            if assembled[key].shape[-1] == 0:
                assembled[key] = np.empty(
                    (len(record), candidates.shape[1], local[key].shape[-1]),
                    dtype=np.float32,
                )
                dimensions[key] = int(local[key].shape[-1])
            assembled[key][states] = local[key]
        cohort_audit[name] = {
            key: value for key, value in audit.items() if key not in {"record_key", "table_state_index"}
        }
    if any(not np.isfinite(value).all() for value in assembled.values()):
        raise RuntimeError("assembled transfer features contain nonfinite values")

    arrays: dict[str, np.ndarray] = {
        "record_key": record,
        "route_group": table["route_group"].astype(str),
        "formal_v70_index": table["formal_v70_index"].astype(np.int16),
    }
    checkpoint_audit = {}
    for mode in MODES:
        feature = feature_by_mode(assembled, mode, variant="raw")
        members, checkpoint_audit[mode] = load_members(mode, device)
        member_predictions = [
            predict(
                model,
                stats,
                feature,
                np.arange(len(feature)),
                device=str(device),
            )
            for model, stats in members
        ]
        for key in PREDICTION_KEYS:
            stacked = np.stack([value[key] for value in member_predictions])
            arrays[f"{mode}_{key}_mean"] = stacked.mean(axis=0).astype(np.float32)
            arrays[f"{mode}_{key}_std"] = stacked.std(axis=0).astype(np.float32)
        del members, member_predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    audit = {
        "schema": "scand_union65_frozen_densemeta_transfer_audit_v1",
        "status": "complete_label_free_transfer_predictions",
        "experiment_novelty": (
            "first application of the frozen 7489-state K17 GroupFirst ensemble "
            "to the four-pool union65 candidate set with formal-v70 fallback"
        ),
        "union65_outcomes_used_for_training": False,
        "union65_outcomes_forwarded_to_model": False,
        "counterfactual_action_outcomes_used": False,
        "states": int(len(record)),
        "candidates": int(candidates.shape[1]),
        "feature_dimensions": dimensions,
        "modes": list(MODES),
        "cohorts": cohort_audit,
        "checkpoints": checkpoint_audit,
        "inputs": {
            "table": {
                "path": str(args.table.resolve()),
                "sha256": sha256_file(args.table),
                "identity_and_candidate_geometry_only": True,
            }
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": audit["output"]["sha256"],
                "states": audit["states"],
                "candidates": audit["candidates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
