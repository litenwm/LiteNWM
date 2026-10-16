#!/usr/bin/env python3
"""Fit and freeze the full-data deployment bundle for the successful v4 prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools" / "evaluation"),
]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
import run_scand_union65_safe_multitask_v1 as base  # noqa: E402
from run_scand_union65_fixed_dual_harm_v3 import (  # noqa: E402
    v3_conservative_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def resolve_entry(entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_entry(entry: dict[str, Any], label: str) -> Path:
    path = resolve_entry(entry)
    base.require(path.is_file(), f"{label} missing: {path}")
    base.require(sha256_file(path) == entry["sha256"], f"{label} hash changed")
    return path


def main() -> None:
    args = parse_args()
    deployment_contract_path = args.contract.resolve()
    output = args.output_dir.resolve()
    base.require(not output.exists(), f"refusing to overwrite: {output}")
    contract = base.read_json(deployment_contract_path)
    base.require(contract["status"] == "frozen_before_full_fit", "contract not frozen")
    source = contract["source_prototype"]
    source_contract_path = verify_entry(source["contract"], "source v4 contract")
    verify_entry(source["runner"], "source v4 runner")
    source_result_path = verify_entry(source["result"], "source v4 result")
    oof_path = verify_entry(
        source["outer_route_oof_predictions"], "source v4 OOF predictions"
    )
    verify_entry(source["outer_models"], "source v4 outer models")
    table_path = verify_entry(contract["training_table"], "training table")
    source_result = base.read_json(source_result_path)
    base.require(source_result["prototype_go"] is True, "source prototype did not pass")
    base.require(
        source_result["decision"] == source["required_decision"],
        "source decision changed",
    )
    source_contract = base.read_json(source_contract_path)
    full_fit = contract["full_fit"]
    base.require(
        int(source_contract["model"]["epochs"]) == int(full_fit["epochs"]),
        "deployment epochs differ from v4",
    )

    device = torch.device(args.device)
    base.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    data = base.load_npz(table_path)
    feature = data["runtime_pair_feature"].astype(np.float32)
    base.require(feature.shape == (800, 65, 351), f"table shape changed: {feature.shape}")
    states = np.arange(len(feature))
    routes = data["route_group"].astype(str)

    with np.load(oof_path, allow_pickle=False) as payload:
        oof_mean = np.asarray(payload["heads_mean"], dtype=np.float32)
        oof_std = np.asarray(payload["heads_std"], dtype=np.float32)
    fixed = source_contract["fixed_policy"]
    _, advantage, eligible = v3_conservative_candidates(
        oof_mean,
        oof_std,
        data,
        states,
        float(fixed["uncertainty_beta"]),
        float(fixed["harm_probability_limit"]),
        source_contract,
    )
    threshold = base.threshold_for_coverage(
        advantage, float(fixed["target_inner_oof_coverage"])
    )

    parameters: dict[str, np.ndarray] = {}
    fit_records = []
    started = time.perf_counter()
    for member, seed in enumerate(int(value) for value in full_fit["ensemble_seeds"]):
        model, mean, std, fit = base.fit_model(
            data,
            states,
            routes,
            source_contract,
            device,
            seed,
        )
        for name, tensor in model.state_dict().items():
            parameters[f"member_{member}__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"member_{member}__feature_mean"] = mean
        parameters[f"member_{member}__feature_std"] = std
        fit_records.append({"member": member, "seed": seed, **fit})
        del model

    output.mkdir(parents=True, exist_ok=False)
    bundle_path = output / "deployment_bundle.npz"
    manifest_path = output / "manifest.json"
    with bundle_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    manifest = {
        "schema": "scand_union65_fixed_dual_harm_deployment_bundle_v1",
        "status": "frozen_unvalidated_not_authorized_for_deployment",
        "source_prototype_decision": source_result["decision"],
        "source_prototype_metrics": {
            "objective_gain_m": source_result["inner_oof_safe_policy"][
                "objective_gain_m"
            ],
            "ate_gain_m": source_result["inner_oof_safe_policy"]["ate_gain_m"],
            "harm_rate": source_result["inner_oof_safe_policy"]["harm_rate"],
            "ate_p90_formal_m": source_result["inner_oof_safe_policy"][
                "formal_p90_m"
            ]["ate"],
            "ate_p90_selected_m": source_result["inner_oof_safe_policy"][
                "selected_p90_m"
            ]["ate"],
            "ate_route_cluster_lcb95_m": source_result[
                "route_cluster_bootstrap_ate_gain"
            ]["lcb95_m"],
            "ate_oracle_gain_ratio": source_result["ate_oracle_gain_ratio"],
        },
        "full_fit": {
            "states": int(len(states)),
            "routes": int(len(np.unique(routes))),
            "members": fit_records,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "frozen_policy": {
            "score": fixed["score"],
            "uncertainty_beta": float(fixed["uncertainty_beta"]),
            "harm_probability_limit": float(fixed["harm_probability_limit"]),
            "target_oof_coverage": float(fixed["target_inner_oof_coverage"]),
            "oof_advantage_threshold": float(threshold),
            "oof_eligible_states": int(eligible.sum()),
            "oof_eligible_fraction": float(eligible.mean()),
            "fallback": "formal-v70",
        },
        "artifacts": {
            "deployment_contract": {
                "path": str(deployment_contract_path),
                "sha256": sha256_file(deployment_contract_path),
            },
            "source_v4_contract": {
                "path": str(source_contract_path),
                "sha256": sha256_file(source_contract_path),
            },
            "source_v4_result": {
                "path": str(source_result_path),
                "sha256": sha256_file(source_result_path),
            },
            "source_v4_oof_predictions": {
                "path": str(oof_path),
                "sha256": sha256_file(oof_path),
            },
            "training_table": {
                "path": str(table_path),
                "sha256": sha256_file(table_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "deployment_bundle": {
                "path": str(bundle_path),
                "sha256": sha256_file(bundle_path),
            },
        },
        "validation": {
            "independent_validation_completed": False,
            "compatible_clean_holdout_found_in_repository": False,
            "deployment_authorized": False,
            "reason": (
                "existing v70 reserve outcomes are already open and the internal "
                "holdout usage audit marks its identities historically spent"
            ),
        },
    }
    base.write_json_atomic(manifest_path, manifest)
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "oof_advantage_threshold": threshold,
                "oof_eligible_fraction": float(eligible.mean()),
                "bundle": str(bundle_path),
                "bundle_sha256": sha256_file(bundle_path),
                "manifest": str(manifest_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
