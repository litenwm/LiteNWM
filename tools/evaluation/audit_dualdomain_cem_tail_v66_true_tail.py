#!/usr/bin/env python3
"""Audit true CEM-tail outcomes after a sealed v66 harvest is complete.

This diagnostic is intentionally restricted to the train/calibration harvest
it is pointed at.  It refuses to open targets until the pre-outcome freeze and
the completed harvest result agree, and it never accesses A--H or reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.audit_cem_proposal_shift_v65 import (  # noqa: E402
    binary_summary,
    route_bootstrap,
    true_scores,
    within_state_rank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_vectors(
    score: dict[str, np.ndarray],
    dual: np.ndarray,
    harm: np.ndarray,
    allowed: np.ndarray,
) -> dict[str, np.ndarray]:
    allowed = np.asarray(allowed, dtype=bool).copy()
    allowed[:, 0] = False
    selected = np.where(allowed, score["energy"], np.inf).argmin(axis=1)
    exists = allowed.any(axis=1)
    row = np.flatnonzero(exists)
    column = selected[exists]
    return {
        "state": row,
        "candidate": column,
        "p_dual": score["dual_probability"][row, column],
        "p_harm": score["harm_probability"][row, column],
        "dual": dual[row, column],
        "harm": harm[row, column],
    }


def summary_selected(
    selected: dict[str, np.ndarray], route: np.ndarray, seed: int
) -> dict[str, Any]:
    result = binary_summary(
        selected["p_dual"],
        selected["p_harm"],
        selected["dual"],
        selected["harm"],
    )
    if result.get("n", 0):
        result["route_clustered_ci95"] = route_bootstrap(
            route[selected["state"]],
            selected["dual"],
            selected["harm"],
            selected["p_dual"],
            seed,
        )
    return result


def elite_vectors(
    score: dict[str, np.ndarray], dual: np.ndarray, harm: np.ndarray
) -> dict[str, np.ndarray]:
    elite = np.asarray(score["elite_index_for_update"], dtype=np.int64)
    if elite.ndim != 2:
        raise RuntimeError("elite_index_for_update must be [state, elite]")
    updated = np.asarray(score["proposal_updated"], dtype=bool)
    row = np.repeat(np.arange(len(elite), dtype=np.int64), elite.shape[1])
    column = elite.reshape(-1)
    keep = np.repeat(updated, elite.shape[1]) & (column > 0)
    row = row[keep]
    column = column[keep]
    return {
        "state": row,
        "candidate": column,
        "p_dual": score["dual_probability"][row, column],
        "p_harm": score["harm_probability"][row, column],
        "dual": dual[row, column],
        "harm": harm[row, column],
    }


def audit(harvest: Path, output: Path) -> None:
    result_path = harvest / "result.json"
    result = read_json(result_path)
    if result.get("status") != "complete" or not result.get("labels_opened_only_after_freeze"):
        raise RuntimeError("harvest is not safely finalized")
    freeze_path = Path(result["preoutcome_freeze"])
    if sha256(freeze_path) != result["preoutcome_freeze_sha256"]:
        raise RuntimeError("pre-outcome freeze hash mismatch")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "candidates_scores_and_proposals_frozen_before_training_labels":
        raise RuntimeError("pre-outcome freeze status mismatch")
    if freeze.get("A_to_H_accessed") or freeze.get("reserve_accessed"):
        raise RuntimeError("forbidden evaluation data was accessed")

    iterations = []
    pooled_selected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("p_dual", "p_harm", "dual", "harm")
    }
    for index, record in enumerate(result["iterations"]):
        score_path = Path(record["scores"])
        target_path = Path(record["targets"])
        if sha256(score_path) != record["scores_sha256"] or sha256(target_path) != record["targets_sha256"]:
            raise RuntimeError(f"iteration {index}: artifact hash mismatch")
        score = load_npz(score_path)
        target = load_npz(target_path)
        if not np.array_equal(score["record_key"].astype(str), target["record_key"].astype(str)):
            raise RuntimeError(f"iteration {index}: identity mismatch")
        true_score, dual, harm = true_scores(target["absolute_metrics"])
        valid = np.asarray(score["valid"], dtype=bool)
        safe = np.asarray(score["safe"], dtype=bool)
        challenger_valid = valid.copy()
        challenger_valid[:, 0] = False
        challenger_safe = safe.copy()
        challenger_safe[:, 0] = False
        top = selected_vectors(score, dual, harm, safe)
        elite = elite_vectors(score, dual, harm)
        for key in pooled_selected:
            pooled_selected[key].append(top[key])
        valid_row, valid_column = np.nonzero(challenger_valid)
        safe_row, safe_column = np.nonzero(challenger_safe)
        iterations.append({
            "iteration": index,
            "states": int(len(valid)),
            "valid_candidates": int(len(valid_row)),
            "safe_candidates": int(len(safe_row)),
            "safe_fraction_of_valid": float(len(safe_row) / max(len(valid_row), 1)),
            "updated_windows": int(np.asarray(score["proposal_updated"], dtype=bool).sum()),
            "safe_shortfall_windows": int((~np.asarray(score["proposal_updated"], dtype=bool)).sum()),
            "all_valid": binary_summary(
                score["dual_probability"][valid_row, valid_column],
                score["harm_probability"][valid_row, valid_column],
                dual[valid_row, valid_column],
                harm[valid_row, valid_column],
            ),
            "all_safe": binary_summary(
                score["dual_probability"][safe_row, safe_column],
                score["harm_probability"][safe_row, safe_column],
                dual[safe_row, safe_column],
                harm[safe_row, safe_column],
            ),
            "safe_top": summary_selected(top, target["route_group"].astype(str), 20260730 + index),
            "safe_elite": summary_selected(elite, target["route_group"].astype(str), 20260740 + index),
            "within_state_rank": within_state_rank(-score["energy"], true_score, valid),
        })

    pooled = {
        key: np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        for key, values in pooled_selected.items()
    }
    payload = {
        "schema": "dualdomain_cem_tail_v66_true_tail_audit",
        "status": "complete",
        "domain": result["domain"],
        "split": result["split"],
        "harvest_result": str(result_path),
        "harvest_result_sha256": sha256(result_path),
        "preoutcome_freeze_sha256": result["preoutcome_freeze_sha256"],
        "iterations": iterations,
        "pooled_safe_top": binary_summary(
            pooled["p_dual"], pooled["p_harm"], pooled["dual"], pooled["harm"]
        ),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    write_json(output, payload)
    print(json.dumps({"status": "complete", "output": str(output), "sha256": sha256(output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harvest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.harvest / "true_tail_audit.json"
    audit(args.harvest, output)


if __name__ == "__main__":
    main()
