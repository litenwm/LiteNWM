#!/usr/bin/env python3
"""Freeze A-T bank-isolated splits for minimal existing LTWM adaptation.

Only window manifests are read. Outcome arrays and sealed reserve labels are
never opened. The source split exists only for frozen per-image encoding;
fold-specific views exclude the complete outer bank before training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/cem_development_v1/recon_target_ltwm_adaptation_v51"
PARENT = ROOT / "data/parent_split_lock/split_lock.json"
WARM_PARENT = ROOT / "data/split_lock/split_lock.json"
RESERVE = ROOT / "outputs/cem_development_v1/recon_target_calibration_v20/final_confirmation_reserve/manifest.windows.jsonl"
BANKS = {
    **{x: ROOT / "outputs/cem_development_v1/recon_target_calibration_v20" / f"bank_{x}" for x in "abcdef"},
    "g": ROOT / "outputs/cem_development_v1/recon_target_confirmation_v29",
    "h": ROOT / "outputs/cem_development_v1/recon_target_cem_confirmation_v35",
    **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v40" / f"bank_{x}" for x in "ijk"},
    **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v44" / f"bank_{x}" for x in "lmnop"},
    **{x: ROOT / "outputs/cem_development_v1/recon_target_development_v46" / f"bank_{x}" for x in "qrst"},
}
FOLDS = {
    "hard_t": {"outer": "t", "validation": "s"},
    "easy_d": {"outer": "d", "validation": "c"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def manifest(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 windows in {path}, got {len(rows)}")
    return rows


def lock(
    experiment: str,
    splits: dict[str, list[str]],
    parent_sha: str,
    warm_parent_sha: str | None,
    purpose: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": experiment,
        "lineage": {
            "parent_split_lock": str(PARENT),
            "parent_split_lock_sha256": parent_sha,
            **(
                {
                    "warm_start_parent_split_lock": str(WARM_PARENT),
                    "warm_start_parent_split_lock_sha256": warm_parent_sha,
                }
                if warm_parent_sha is not None
                else {}
            ),
            "target_development_banks": "A-T; outcomes are development-spent",
        },
        "policy": {
            "history_frames": 4,
            "future_frames": 4,
            "split_before_window_generation": True,
            "heldout_bank_images_forbidden_during_adaptation": True,
            "heldout_bank_unlabeled_distribution_forbidden_during_adaptation": True,
            "final_confirmation_reserve_sealed": True,
            "purpose": purpose,
        },
        "recon": {"splits": splits},
    }


def main() -> None:
    if set(BANKS) != set("abcdefghijklmnopqrst"):
        raise RuntimeError("A-T bank map is incomplete")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    parent_sha = sha(PARENT)
    warm_parent_sha = sha(WARM_PARENT)
    parent_test = set(map(str, parent["recon"]["splits"]["test"]))
    ids: dict[str, list[str]] = {}
    routes: dict[str, set[str]] = {}
    manifest_sha: dict[str, str] = {}
    total_states = 0
    for bank, root in sorted(BANKS.items()):
        path = root / "manifest.windows.jsonl"
        rows = manifest(path)
        total_states += len(rows)
        ids[bank] = sorted({str(row["trajectory_id"]) for row in rows})
        routes[bank] = {str(row["route_group"]) for row in rows}
        manifest_sha[bank] = sha(path)
        if len(ids[bank]) != 25 or len(routes[bank]) != 25:
            raise RuntimeError(f"bank {bank} is not 25-trajectory/25-route")
        if not set(ids[bank]) <= parent_test:
            raise RuntimeError(f"bank {bank} includes non-parent-test trajectories")

    all_ids: set[str] = set()
    all_routes: set[str] = set()
    for bank in sorted(ids):
        if all_ids & set(ids[bank]) or all_routes & routes[bank]:
            raise RuntimeError(f"bank overlap detected at {bank}")
        all_ids.update(ids[bank])
        all_routes.update(routes[bank])
    if len(all_ids) != 500 or len(all_routes) != 500:
        raise RuntimeError("A-T must have 500 disjoint trajectories/routes")

    reserve_rows = manifest(RESERVE)
    reserve_ids = {str(row["trajectory_id"]) for row in reserve_rows}
    reserve_routes = {str(row["route_group"]) for row in reserve_rows}
    if all_ids & reserve_ids or all_routes & reserve_routes:
        raise RuntimeError("A-T overlaps sealed reserve")
    if (RESERVE.parent / "source_labeled.npz").exists():
        raise RuntimeError("sealed reserve has a labeled source")

    prereg = {
        "schema": "recon_target_ltwm_adaptation_v51_preregistration",
        "status": "frozen_before_training",
        "development_evidence_only": True,
        "outer_folds": FOLDS,
        "schemes": {
            "frozen": {"train_scope": None, "role": "unchanged baseline"},
            "output_calibration": {"train_scope": "output_calibration", "learning_rate": 0.0001},
            "action_cross_attention": {"train_scope": "action_cross_attention", "learning_rate": 0.00005},
        },
        "warm_start_seeds": [1701, 1702, 1703],
        "backbone": "frozen DINOv3-L/16",
        "adapter": "frozen adapter_pilot_v1_final",
        "runtime_smoke": {
            "fold": "hard_t",
            "seed": 1701,
            "batches_per_scheme": 100,
            "selection_use": "runtime and finite check only; no outer outcomes",
        },
        "screen": {
            "folds": ["hard_t", "easy_d"],
            "seeds": [1701, 1702, 1703],
            "epochs": 1,
            "batches_per_epoch": 800,
            "eval_batches_per_epoch": 96,
            "batch_size": 2,
            "amp": "bf16",
        },
        "stage_2_safety_screen": {
            "ranker": "existing recon_target_full_listwise_v42 Ranker/train_model",
            "architecture_changes": False,
            "epochs": 180,
            "hidden_dim": 160,
            "dropout": 0.30,
            "ensemble_members": 3,
            "learning_rate": 0.0005,
            "weight_decay": 0.002,
            "hard_t": {
                "fit_banks": list("abcdefghijklmn"),
                "gate_cal_banks": list("opqrs"),
                "outer_bank": "t",
            },
            "easy_d": {
                "fit_banks": [x for x in "abcdefghijklmno" if x != "d"],
                "gate_cal_banks": list("pqrst"),
                "outer_bank": "d",
            },
            "outer_outcomes_opened_only_after_freeze": True,
            "fixed_threshold_from_gate_cal_only": True,
            "full_route_quota": False,
        },
        "safety_gates": {
            "coverage_min": 0.40,
            "dual_precision_min": 0.90,
            "wilson_lower_min": 0.80,
            "harm_rate_max": 0.10,
            "quality": "mean and p90 non-degradation",
        },
        "progression_rule": (
            "Both hard_t and easy_d must pass before full 20-bank LOBO. "
            "Full LOBO must pass before real CEM replacement and sealed confirmation."
        ),
        "labels_accessed_by_preparation": False,
    }
    prereg_sha = write_json(OUT / "preregistration.json", prereg)

    source_splits = {"train": sorted(all_ids), "val": [], "test": []}
    source_dir = OUT / "source_encoding"
    source_strict_sha = write_json(source_dir / "strict_splits.json", source_splits)
    source_lock_sha = write_json(
        source_dir / "split_lock.json",
        lock(
            "recon_target_ltwm_adaptation_v51_source_encoding",
            source_splits,
            parent_sha,
            None,
            "frozen per-trajectory encoding only; never supplied directly to training",
        ),
    )

    fold_audit: dict[str, object] = {}
    for name, spec in FOLDS.items():
        outer, val_bank = spec["outer"], spec["validation"]
        train_banks = sorted(set(BANKS) - {outer, val_bank})
        splits = {
            "train": sorted(t for bank in train_banks for t in ids[bank]),
            "val": ids[val_bank],
            "test": ids[outer],
        }
        if tuple(map(len, splits.values())) != (450, 25, 25):
            raise RuntimeError(f"invalid fold sizes for {name}")
        fold_dir = OUT / "folds" / name
        strict_sha = write_json(fold_dir / "strict_splits.json", splits)
        split_lock_sha = write_json(
            fold_dir / "split_lock.json",
            lock(
                f"recon_target_ltwm_adaptation_v51_{name}",
                splits,
                parent_sha,
                warm_parent_sha,
                "outer-bank isolated LTWM representation adaptation",
            ),
        )
        fold_audit[name] = {
            "train_banks": train_banks,
            "validation_bank": val_bank,
            "outer_bank": outer,
            "split_counts": {key: len(value) for key, value in splits.items()},
            "strict_splits_sha256": strict_sha,
            "split_lock_sha256": split_lock_sha,
            "outer_images_exposed_to_training_view": False,
            "outer_unlabeled_distribution_exposed_to_training_view": False,
        }

    audit = {
        "schema": "recon_target_ltwm_adaptation_v51_audit",
        "status": "PASS",
        "banks": 20,
        "states": total_states,
        "trajectories": len(all_ids),
        "routes": len(all_routes),
        "all_selected_ids_parent_test": True,
        "all_bank_trajectory_disjoint": True,
        "all_bank_route_disjoint": True,
        "reserve_sealed": True,
        "reserve_overlap_trajectories": 0,
        "reserve_overlap_routes": 0,
        "reserve_manifest_sha256": sha(RESERVE),
        "parent_split_lock_sha256": parent_sha,
        "warm_start_parent_split_lock_sha256": warm_parent_sha,
        "bank_manifest_sha256": manifest_sha,
        "source_strict_splits_sha256": source_strict_sha,
        "source_split_lock_sha256": source_lock_sha,
        "preregistration_sha256": prereg_sha,
        "folds": fold_audit,
        "labels_accessed": False,
    }
    audit_sha = write_json(OUT / "audit.json", audit)
    print(json.dumps({**audit, "audit_sha256": audit_sha}, indent=2))


if __name__ == "__main__":
    main()
