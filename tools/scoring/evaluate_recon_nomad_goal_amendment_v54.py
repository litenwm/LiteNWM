#!/usr/bin/env python3
"""Evaluate the frozen v53 arms under the supervisory v54 trend rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.scoring.screen_recon_nomad_goal_state_gate_v53 import (  # noqa: E402
    METRICS,
    load_labels,
    quality,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


VARIANTS = (
    "distance_only",
    "condition_pca8",
    "condition_pca8_plus_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment["results_access_disclosure"]["blind_preregistration"]:
        raise RuntimeError("v54 amendment must retain its non-blind disclosure")
    with np.load(args.screen_dir / "frozen_challenger.npz") as payload:
        record = payload["record_key"].astype(str)
        bank = payload["bank"].astype(str)
        challenger = payload["challenger_index"].astype(np.int64)
        challenger_score = payload["challenger_score"].astype(np.float64)
        eligible = payload["state_eligible"].astype(bool)
    with np.load(args.screen_dir / "selected_index_top40.npz") as payload:
        frozen_selected = {
            variant: payload[f"selected_{variant}"].astype(np.int64)
            for variant in VARIANTS
        }
    _, targets, absolute = load_labels(record)
    row = np.arange(len(record))
    chosen_delta = targets[row, challenger, :4]
    dual = (chosen_delta[:, 0] < -1.0e-7) & (
        chosen_delta[:, 3] < -1.0e-7
    )
    harm = (chosen_delta[:, 0] > 1.0e-7) | (
        chosen_delta[:, 3] > 1.0e-7
    )
    incumbent = absolute[:, 0]
    challenger_absolute = absolute[row, challenger]

    baseline = np.zeros(len(record), dtype=np.int64)
    for letter in sorted(np.unique(bank)):
        local = np.flatnonzero((bank == letter) & eligible)
        order = np.argsort(-challenger_score[local], kind="stable")[:40]
        chosen = local[order]
        baseline[chosen] = challenger[chosen]

    def report(selected: np.ndarray) -> dict[str, object]:
        promote = selected != 0
        chosen = incumbent.copy()
        chosen[promote] = challenger_absolute[promote]
        per_bank = {}
        for letter in sorted(np.unique(bank)):
            local = (bank == letter) & promote
            per_bank[letter] = {
                "promotions": int(local.sum()),
                "dual": float(dual[local].mean()),
                "harm": float(harm[local].mean()),
            }
        return {
            "promotions": int(promote.sum()),
            "coverage": float(promote.mean()),
            "dual": float(dual[promote].mean()),
            "harm": float(harm[promote].mean()),
            "quality": quality(chosen),
            "per_bank": per_bank,
            "worst_bank_dual": min(v["dual"] for v in per_bank.values()),
            "worst_bank_harm": max(v["harm"] for v in per_bank.values()),
        }

    reference = report(baseline)
    arm_report = {}
    useful = []
    for variant in VARIANTS:
        value = report(frozen_selected[variant])
        quality_non_degradation = all(
            value["quality"][metric][summary]
            <= reference["quality"][metric][summary] + 1.0e-9
            for metric in METRICS
            for summary in ("mean", "p90")
        )
        dual_gain = value["dual"] - reference["dual"]
        harm_reduction = reference["harm"] - value["harm"]
        worst_dual_drop = max(
            0.0, reference["worst_bank_dual"] - value["worst_bank_dual"]
        )
        worst_harm_increase = max(
            0.0, value["worst_bank_harm"] - reference["worst_bank_harm"]
        )
        trend_pass = bool(
            (dual_gain >= 0.02 or harm_reduction >= 0.02)
            and quality_non_degradation
            and (worst_dual_drop <= 0.05 or worst_harm_increase <= 0.05)
        )
        value.update(
            {
                "pooled_dual_gain": float(dual_gain),
                "pooled_harm_reduction": float(harm_reduction),
                "quality_non_degradation": quality_non_degradation,
                "worst_bank_dual_drop": float(worst_dual_drop),
                "worst_bank_harm_increase": float(worst_harm_increase),
                "useful_trend": trend_pass,
            }
        )
        arm_report[variant] = value
        if trend_pass:
            useful.append(variant)

    result = {
        "schema": "recon_nomad_goal_v53_amendment_v54_result",
        "status": "v53_stopped_after_frozen_three_arm_amendment_evaluation",
        "development_only": True,
        "blind_preregistration": False,
        "production_safe_claim_allowed": False,
        "baseline_definition": (
            "Within each held-out bank, select the top 40 eligible frozen v47 "
            "challengers by their already-frozen challenger_score."
        ),
        "diagnostic_is_online_deployable": False,
        "amendment": str(args.amendment),
        "amendment_sha256": file_sha256(args.amendment),
        "frozen_challenger_sha256": file_sha256(
            args.screen_dir / "frozen_challenger.npz"
        ),
        "frozen_selected_index_sha256": file_sha256(
            args.screen_dir / "selected_index_top40.npz"
        ),
        "reference": reference,
        "variants": arm_report,
        "useful_trend_variants": useful,
        "v53_terminal": True,
        "new_v53_feature_or_model_allowed": False,
        "next_action": "return_to_true_iterative_candidate_level_cem",
        "reserve_accessed": False,
        "cem": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
