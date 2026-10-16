#!/usr/bin/env python3
"""Seal the completed tri-domain V70 fair evaluation and mark cohorts spent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


DOMAINS = ("recon", "scand", "sacson")
K_VALUES = (16, 32)
METRICS = ("ate", "rte", "fde")
EXPECTED_MAPPING = {"recon": 0, "scand": 1, "sacson": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def nested_first(mapping, paths):
    for keys in paths:
        value = mapping
        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            continue
        return value
    raise KeyError(paths)


def metric_value(block, metric: str) -> float:
    candidates = {
        "ate": (("ate_nwm", "mean"), ("ate",), ("ATE",), ("metrics", "ate"), ("metrics", "ATE")),
        "rte": (("rte_nwm", "mean"), ("rte",), ("RTE",), ("metrics", "rte"), ("metrics", "RTE")),
        "fde": (("fde_nwm", "mean"), ("fde",), ("FDE",), ("metrics", "fde"), ("metrics", "FDE")),
    }
    return float(nested_first(block, candidates[metric]))


def method_block(payload, method: str):
    methods = payload.get("summaries", payload.get("methods", payload.get("results", payload)))
    require(method in methods, f"missing method {method}")
    return methods[method]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1"),
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    seal_root = run_root / "relaxed_fallback_attempts/final_cem_tridomain_v2_training_epoch39_relaxed_v1"
    fair_root = run_root / "fair_eval_final_relaxed_v1"
    require((fair_root / "COMPLETE").is_file(), "fair evaluation COMPLETE marker missing")

    result_path = seal_root / "result.json"
    seal_path = seal_root / "seal_record.json"
    prereg_path = seal_root / "fair_eval_protocol_preregistration_v1.json"
    result = load_json(result_path)
    seal = load_json(seal_path)
    prereg = load_json(prereg_path)

    require((seal_root / "RELAXED_FEASIBLE_V1").is_file(), "relaxed marker missing")
    require(result.get("seal_status") == "RELAXED_FEASIBLE_V1", "unexpected seal status")
    require(result.get("strict_all_domains_feasible") is False, "result incorrectly claims strict feasibility")
    require(result.get("relaxed_all_domains_feasible_v1") is True, "relaxed feasibility missing")
    mapping = nested_first(result, (("model_domain_index",), ("domain_mapping",), ("training", "domain_mapping")))
    require(mapping == EXPECTED_MAPPING, f"domain mapping mismatch: {mapping}")
    require(result.get("test_accessed", False) is False, "training record says test accessed")
    require(result.get("outer_val_accessed", False) is False, "outer validation was accessed")

    require(prereg.get("frozen_result_sha256") == sha256(result_path), "preregistration does not bind result SHA")
    require(prereg.get("seal_record_sha256") == sha256(seal_path), "preregistration does not bind seal SHA")
    require(prereg.get("external_test_labels_accessed_before_freeze") is False, "test labels were accessed before freeze")
    require(prereg.get("external_metrics_accessed_before_freeze") is False, "external metrics were accessed before freeze")

    checkpoint_records = []
    checkpoints = nested_first(result, (("checkpoints",), ("artifacts", "checkpoints")))
    if isinstance(checkpoints, dict) and "full" in checkpoints:
        checkpoints = checkpoints["full"]
    checkpoint_items = checkpoints.items() if isinstance(checkpoints, dict) else enumerate(checkpoints)
    for name, record in checkpoint_items:
        if isinstance(record, str):
            path, expected_sha = Path(record), None
        else:
            path = Path(record.get("path") or record.get("checkpoint"))
            expected_sha = record.get("sha256") or record.get("sha")
        require(path.is_file(), f"checkpoint missing: {path}")
        actual_sha = sha256(path)
        if expected_sha:
            require(actual_sha == expected_sha, f"checkpoint SHA mismatch: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        loaded_text = str(loaded.get("schema", loaded.get("schema_version", ""))) if isinstance(loaded, dict) else ""
        require("proposal_aware_v70_tridomain" in loaded_text, f"checkpoint schema mismatch: {path}")
        checkpoint_records.append({"name": str(name), "path": str(path), "sha256": actual_sha, "loadable": True})

    world_model_records = []
    world_models = nested_first(result, (("world_models",), ("ltwm",), ("provenance", "world_models")))
    for seed, record in world_models.items():
        path = Path(record.get("path") or record.get("checkpoint"))
        expected_sha = record.get("sha256") or record.get("sha")
        require(path.is_file(), f"LTWM missing: {path}")
        actual_sha = sha256(path)
        require(actual_sha == expected_sha, f"LTWM SHA mismatch for {seed}")
        torch.load(path, map_location="cpu", weights_only=False)
        world_model_records.append({"seed": str(seed), "path": str(path), "sha256": actual_sha, "loadable": True})

    result_text = json.dumps(result, sort_keys=True)
    for wm in world_model_records:
        require(wm["sha256"] in result_text, f"LTWM SHA not bound in result: {wm['seed']}")

    feature_cache_records = []
    for record in result.get("feature_cache_bindings", []):
        path = Path(record["path"])
        require(path.is_file(), f"feature cache missing: {path}")
        actual_sha = sha256(path)
        require(actual_sha == record["sha256"], f"feature cache SHA mismatch: {path}")
        feature_cache_records.append(
            {
                "dataset": record["dataset"],
                "split": record["split"],
                "set_type": record["set_type"],
                "path": str(path),
                "sha256": actual_sha,
            }
        )
    require(len(feature_cache_records) == 24, f"expected 24 feature-cache bindings, found {len(feature_cache_records)}")
    require({record["dataset"] for record in feature_cache_records} == set(DOMAINS), "feature caches omit a domain")

    new_paths = {
        "recon": fair_root / "recon100/metrics/results.json",
        "scand": fair_root / "scand_common100/metrics/results.json",
        "sacson": fair_root / "sacson100/metrics/results.json",
    }
    old_paths = {
        "recon": Path("/root/autodl-tmp/dwm_runs/fair_v70_20260723/recon100_primary/seed17/metrics_sealed/results.json"),
        "scand": Path("/root/autodl-tmp/dwm_runs/fair_v70_20260723/scand_common100/seed17/metrics_sealed/results.json"),
        "sacson": Path("/root/autodl-tmp/dwm_runs/v70_tridomain_fair_eval_triseed_20260728/sacson100/metrics/results.json"),
    }
    new_methods = {domain: {16: "new_v70_triseed_k16", 32: "new_v70_triseed_k32"} for domain in DOMAINS}
    old_methods = {
        "recon": {16: "A7_v70_full_k16", 32: "A8_v70_full_k32"},
        "scand": {16: "A7_v70_full_k16", 32: "A8_v70_full_k32"},
        "sacson": {16: "new_v70_triseed_k16", 32: "new_v70_triseed_k32"},
    }

    comparisons = {}
    spent = []
    ate_regressions = []
    macro_old = {k: [] for k in K_VALUES}
    macro_new = {k: [] for k in K_VALUES}
    for domain in DOMAINS:
        new_payload = load_json(new_paths[domain])
        old_payload = load_json(old_paths[domain])
        comparisons[domain] = {}
        new_text = json.dumps(new_payload, sort_keys=True)
        require("future_pose_opened" not in new_text or '"future_pose_opened": false' in new_text.lower(), f"future pose audit failed: {domain}")
        for k in K_VALUES:
            new_block = method_block(new_payload, new_methods[domain][k])
            old_block = method_block(old_payload, old_methods[domain][k])
            row = {}
            for metric in METRICS:
                old_value = metric_value(old_block, metric)
                new_value = metric_value(new_block, metric)
                delta_pct = (new_value / old_value - 1.0) * 100.0
                row[metric] = {"old": old_value, "new": new_value, "delta_percent": delta_pct}
                if metric == "ate":
                    ate_regressions.append(delta_pct)
                    macro_old[k].append(old_value)
                    macro_new[k].append(new_value)
            comparisons[domain][f"k{k}"] = row
        manifest_sha = nested_first(
            new_payload,
            (("inputs", "manifest", "sha256"), ("manifest_sha256",), ("audit", "manifest_sha256")),
        )
        old_manifest_sha = nested_first(
            old_payload,
            (("inputs", "manifest", "sha256"), ("manifest_sha256",), ("audit", "manifest_sha256")),
        )
        require(manifest_sha == old_manifest_sha, f"cohort mismatch for {domain}")
        spent.append(
            {
                "domain": domain,
                "cohort": {"recon": "RECON-100", "scand": "SCAND-scand_common100", "sacson": "SACSoN-100"}[domain],
                "manifest_sha256": manifest_sha,
                "results_path": str(new_paths[domain]),
                "results_sha256": sha256(new_paths[domain]),
                "sample_count": 100,
                "prohibited_for_future_training_or_tuning": True,
            }
        )

    macro = {}
    macro_regressions = []
    for k in K_VALUES:
        old_mean = sum(macro_old[k]) / len(macro_old[k])
        new_mean = sum(macro_new[k]) / len(macro_new[k])
        delta = (new_mean / old_mean - 1.0) * 100.0
        macro[f"k{k}"] = {"old_ate": old_mean, "new_ate": new_mean, "delta_percent": delta}
        macro_regressions.append(delta)

    adapter_trigger = max(macro_regressions) > 5.0 or max(ate_regressions) > 10.0
    require(adapter_trigger is False, "adapter retraining trigger unexpectedly met")
    timestamp = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema": "tridomain_v70_final_fair_audit_v1",
        "created_at_utc": timestamp,
        "status": "PASS",
        "seal_status": "RELAXED_FEASIBLE_V1",
        "strict_feasible": False,
        "relaxed_feasible": True,
        "domain_mapping": mapping,
        "checkpoint_records": checkpoint_records,
        "world_model_records": world_model_records,
        "feature_cache_records": feature_cache_records,
        "training_sample_counts": nested_first(
            result,
            (("train_states_by_dataset",), ("training_sample_counts",), ("sample_counts",), ("training", "sample_counts")),
        ),
        "comparisons": comparisons,
        "macro_ate": macro,
        "adapter_retrain_trigger_rule": "macro ATE regression >5% OR any domain/K ATE regression >10%",
        "adapter_retrain_triggered": False,
        "decision": "ACCEPT_CURRENT_TRIDOMAIN_V70_NO_ADAPTER_RETRAIN_TRIGGER",
        "test_leakage_audit": "PASS: frozen protocol and selection preceded future-pose label opening; cohorts are now spent",
        "spent_cohorts_path": str(fair_root / "SPENT_TEST_COHORTS.json"),
    }
    spent_payload = {
        "schema": "spent_test_cohorts_v1",
        "created_at_utc": timestamp,
        "reason": "Final tri-domain V70 external fair evaluation completed",
        "spent": True,
        "cohorts": spent,
    }
    atomic_json(fair_root / "SPENT_TEST_COHORTS.json", spent_payload)
    atomic_json(fair_root / "FINAL_AUDIT_REPORT.json", audit)
    print(json.dumps({"status": "PASS", "adapter_retrain_triggered": False, "macro_ate": macro}, indent=2))


if __name__ == "__main__":
    main()
