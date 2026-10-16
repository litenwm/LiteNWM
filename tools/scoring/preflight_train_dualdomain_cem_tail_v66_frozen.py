#!/usr/bin/env python3
"""Verify every sealed v66 data and training contract before GPU fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import DOMAINS, group_fold  # noqa: E402
from tools.evaluation.export_recon_dense_dinov3_features_v1 import load_npz  # noqa: E402
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as frozen  # noqa: E402
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import load_envelopes  # noqa: E402
from tools.scoring.harvest_dualdomain_cem_tail_v66 import paths as source_paths  # noqa: E402


implementation = frozen.implementation
OUTPUT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/training_preflight.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_route(domain: str, source: dict[str, np.ndarray]) -> np.ndarray:
    route = source["route_group"].astype(str)
    if domain == "scand":
        route = np.char.add(np.char.add(route, "::bag::"), source["bag_id"].astype(str))
    return route


def audit_part(
    domain: str,
    split: str,
    envelopes: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    part = implementation.load_domain_split(domain, split, envelopes)
    source = load_npz(source_paths(domain, split)["source"])
    route = exact_route(domain, source)
    states = len(route)
    expected_group = np.tile(route, 4)
    expected_fold = np.tile(
        np.asarray([group_fold(domain, value, 5) for value in route], dtype=np.int64),
        4,
    )
    if not np.array_equal(part["group"].astype(str), expected_group):
        raise RuntimeError(f"{domain}/{split}: route identity changed")
    if not np.array_equal(part["fold_id"].astype(np.int64), expected_fold):
        raise RuntimeError(f"{domain}/{split}: OOF fold identity changed")
    if len(part["feature"]) != 4 * states:
        raise RuntimeError(f"{domain}/{split}: expected four frozen candidate sets")

    support = np.expm1(part["feature"][..., -2])
    guard = part["feature"][..., -1] > 0.5
    expected_valid = guard & (
        support <= threshold_for_quantile(envelopes[domain], 0.999)
    )
    expected_valid[:, 0] = True
    if not np.array_equal(part["valid"], expected_valid):
        raise RuntimeError(f"{domain}/{split}: universal valid contract changed")

    harvest_root = implementation.HARVEST / f"{domain}_{split}"
    harvest_result_path = harvest_root / "result.json"
    harvest_result = read_json(harvest_result_path)
    freeze_path = Path(harvest_result["preoutcome_freeze"])
    if (
        harvest_result.get("status") != "complete"
        or not harvest_result.get("labels_opened_only_after_freeze")
        or harvest_result.get("A_to_H_accessed")
        or harvest_result.get("reserve_accessed")
        or sha256(freeze_path) != harvest_result["preoutcome_freeze_sha256"]
    ):
        raise RuntimeError(f"{domain}/{split}: harvest seal failed")

    score_valid_mismatches = 0
    legacy_majority_excluded = 0
    for iteration in range(3):
        start = (iteration + 1) * states
        stop = start + states
        directory = harvest_root / f"iteration_{iteration:02d}"
        score = load_npz(directory / "scores.npz")
        raw = load_npz(directory / "features.npz")
        if not np.array_equal(part["valid"][start:stop], score["valid"].astype(bool)):
            score_valid_mismatches += 1
        legacy = np.asarray(raw["majority_support"], dtype=bool)
        legacy_majority_excluded += int((part["valid"][start:stop] & ~legacy).sum())
    if score_valid_mismatches:
        raise RuntimeError(f"{domain}/{split}: training/harvest valid mismatch")

    return part, {
        "domain": domain,
        "split": split,
        "original_states": states,
        "loaded_states": int(len(part["feature"])),
        "route_groups": int(len(np.unique(route))),
        "fold_counts": np.bincount(expected_fold[:states], minlength=5).astype(int).tolist(),
        "set_type_counts": {
            name: int(np.sum(part["set_type"].astype(str) == name))
            for name in implementation.SET_WEIGHTS
        },
        "valid_candidates": int(part["valid"].sum()),
        "legacy_majority_candidates_deliberately_retained": legacy_majority_excluded,
        "harvest_result": str(harvest_result_path),
        "harvest_result_sha256": sha256(harvest_result_path),
        "preoutcome_freeze_sha256": harvest_result["preoutcome_freeze_sha256"],
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prereg = read_json(implementation.PREREG)
    weights_seal = read_json(implementation.WEIGHT_AMENDMENT)
    correction = read_json(frozen.CORRECTION)
    if prereg.get("status") != "frozen_before_proposal_harvest":
        raise RuntimeError("preregistration changed")
    if weights_seal.get("status") != "frozen_before_any_v66_training":
        raise RuntimeError("training weight seal changed")
    if correction.get("status") != "frozen_before_any_v66_training":
        raise RuntimeError("candidate-level calibration correction changed")

    base_result = read_json(implementation.BASE_RESULT)
    envelopes = load_envelopes(Path(base_result["support"]))
    parts: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audits = []
    for split in ("train_fit", "scorer_cal"):
        for domain in DOMAINS:
            part, audit = audit_part(domain, split, envelopes)
            parts[(domain, split)] = part
            audits.append(audit)

    train_parts = [parts[(domain, "train_fit")] for domain in DOMAINS]
    train = {
        key: np.concatenate([part[key] for part in train_parts])
        for key in train_parts[0]
    }
    weights = implementation.training_weights(
        train["domain"], train["group"], train["set_type"]
    )
    weight_contract: dict[str, Any] = {"total": float(weights.sum()), "domains": {}}
    for domain_index, domain in enumerate(DOMAINS):
        domain_mask = train["domain"] == domain_index
        entry = {"total": float(weights[domain_mask].sum()), "set_types": {}}
        for name in implementation.SET_WEIGHTS:
            mask = domain_mask & (train["set_type"].astype(str) == name)
            entry["set_types"][name] = float(weights[mask].sum())
        weight_contract["domains"][domain] = entry
    expected = {
        domain: {
            "fixed": 0.25,
            "cem_iteration_0": 1.0 / 12.0,
            "cem_iteration_1": 1.0 / 12.0,
            "cem_iteration_2": 1.0 / 12.0,
        }
        for domain in DOMAINS
    }
    for domain, set_values in expected.items():
        for name, value in set_values.items():
            if not np.isclose(weight_contract["domains"][domain]["set_types"][name], value):
                raise RuntimeError(f"{domain}/{name}: frozen weight changed")

    payload = {
        "schema": "dualdomain_cem_tail_v66_training_preflight",
        "status": "passed_before_gpu_training",
        "parts": audits,
        "train_states": int(len(train["feature"])),
        "training_weights": weight_contract,
        "calibration_decision": "candidate-level safety mask then lowest conservative energy",
        "preregistration_sha256": sha256(implementation.PREREG),
        "training_weight_amendment_sha256": sha256(implementation.WEIGHT_AMENDMENT),
        "candidate_level_calibration_correction_sha256": sha256(frozen.CORRECTION),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
