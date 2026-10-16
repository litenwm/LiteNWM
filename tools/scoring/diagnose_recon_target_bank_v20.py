#!/usr/bin/env python3
"""Diagnose candidate availability, ranking, and gate limits for v20 banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_cem_state_gate_v6 import method_summary
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json
from tools.scoring.train_recon_target_bank_base_v20 import load_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selection_summary(candidate: np.ndarray, order: np.ndarray, bank: dict) -> dict:
    chosen = candidate[order]
    selected = np.zeros(len(candidate), dtype=np.int64)
    selected[order] = chosen + 1
    return method_summary(selected, bank["delta"], bank["absolute"])


def top40_best(
    scores: list[tuple[str, np.ndarray]], bank: dict
) -> dict:
    best = None
    valid = bank["valid"]
    for name, value in scores:
        rank = np.where(valid, value, -np.inf)
        candidate = rank.argmax(1)
        row = np.arange(len(candidate))
        confidence = rank[row, candidate]
        order = np.argsort(-confidence, kind="stable")[:40]
        metrics = selection_summary(candidate, order, bank)
        item = {"score": name, "metrics": metrics}
        key = (
            metrics["dual_precision"] or 0.0,
            -(metrics["harm_rate"] if metrics["harm_rate"] is not None else 1.0),
        )
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return best[1]


def main() -> None:
    args = parse_args()
    banks = [load_bank(args.root, value) for value in ("a", "b", "c")]
    result = json.loads(args.base_result.read_text(encoding="utf-8"))
    prediction_path = Path(result["predictions"])
    with np.load(prediction_path, allow_pickle=False) as payload:
        prediction_record = payload["record_key"].astype(str)
        p_dual = payload["p_dual"].astype(np.float32)
        p_harm = payload["p_harm"].astype(np.float32)
        p_dual_std = payload["p_dual_std"].astype(np.float32)
        p_harm_std = payload["p_harm_std"].astype(np.float32)
    offset = 0
    reports = {}
    for bank in banks:
        count = len(bank["record"])
        bank_slice = slice(offset, offset + count)
        if not np.array_equal(prediction_record[bank_slice], bank["record"]):
            raise RuntimeError("diagnostic prediction identity mismatch")
        with np.load(bank["artifacts"]["score"], allow_pickle=False) as score_payload:
            universal_dual = score_payload["dual_probability"][:, 1:].astype(np.float32)
            universal_harm = score_payload["harm_probability"][:, 1:].astype(np.float32)
            universal_rank = score_payload["rank_logit"][:, 1:].astype(np.float32)
            universal_mean = score_payload["mean"][:, 1:].astype(np.float32)
            universal_std = score_payload["std"][:, 1:].astype(np.float32)
        current_dual = p_dual[bank_slice]
        current_harm = p_harm[bank_slice]
        current_dual_std = p_dual_std[bank_slice]
        current_harm_std = p_harm_std[bank_slice]
        learned_scores = []
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            for weight in (0.5, 1.0, 2.0, 4.0):
                learned_scores.append((
                    f"pdual_minus_{weight:g}pharm_z{z:g}",
                    current_dual - z * current_dual_std
                    - weight * (current_harm + z * current_harm_std),
                ))
        universal_scores = [
            ("rank_logit", universal_rank),
            ("dual_minus_harm", universal_dual - universal_harm),
            ("dual_minus_2harm", universal_dual - 2.0 * universal_harm),
            ("negative_objective_mean", -universal_mean[..., 3]),
            (
                "objective_lcb_z1",
                -(universal_mean[..., 3] + universal_std[..., 3]),
            ),
            (
                "ate_objective_lcb_z1",
                -(universal_mean[..., 0] + universal_std[..., 0])
                -(universal_mean[..., 3] + universal_std[..., 3]),
            ),
        ]
        valid = bank["valid"]
        dual = bank["dual"].astype(bool)
        harm = bank["harm"].astype(bool)
        oracle_candidate = np.zeros(count, dtype=np.int64)
        oracle_confidence = np.full(count, -np.inf, dtype=np.float32)
        for row in range(count):
            available = np.flatnonzero(valid[row] & dual[row])
            if len(available):
                utility = -bank["delta"][row, available, 0] - bank["delta"][row, available, 3]
                chosen = available[int(np.argmax(utility))]
                oracle_candidate[row] = chosen
                oracle_confidence[row] = utility.max()
        oracle_order = np.argsort(-oracle_confidence, kind="stable")[:40]
        any_dual = (valid & dual).any(1)
        any_safe = (valid & dual & ~harm).any(1)
        reports[bank["name"]] = {
            "states_with_any_dual_candidate": int(any_dual.sum()),
            "states_with_any_dual_nonharm_candidate": int(any_safe.sum()),
            "candidate_pool_oracle_top40": selection_summary(
                oracle_candidate, oracle_order, bank
            ),
            "v20_oof_noncausal_full_bank_top40_diagnostic": top40_best(
                learned_scores, bank
            ),
            "frozen_universal_noncausal_full_bank_top40_diagnostic": top40_best(
                universal_scores, bank
            ),
            "v20_candidate_level": {
                "dual_probability_positive_mean": float(current_dual[dual].mean()),
                "dual_probability_negative_mean": float(current_dual[~dual].mean()),
                "harm_probability_positive_mean": float(current_harm[harm].mean()),
                "harm_probability_negative_mean": float(current_harm[~harm].mean()),
            },
        }
        offset += count
    report = {
        "schema": "recon_target_bank_v20_diagnostic",
        "evidence_role": "development_only_noncausal_separability_diagnostic",
        "warning": (
            "top40 rows rank all heldout bank windows and are not deployable gates; "
            "they diagnose feature/ranker separability only"
        ),
        "base_result": str(args.base_result),
        "base_result_sha256": file_sha256(args.base_result),
        "predictions_sha256": file_sha256(prediction_path),
        "banks": reports,
        "final_reserve_accessed": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
