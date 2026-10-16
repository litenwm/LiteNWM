#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


RUN = Path("/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights")
DOMAINS = ("recon", "scand", "sacson")


def load(path: Path):
    return json.loads(path.read_text())


def domain_score(row: dict, domain: str) -> float:
    nmse = [float(value) for value in row["val"][domain]["true"]["nmse"]]
    return mean(nmse) + nmse[3]


def candidate_rows(seed: int):
    summary = load(RUN / "ltwm" / f"seed{seed}" / "report" / "train_summary.json")
    if "stages" not in summary:
        for row in summary["history"]:
            yield {
                "label": f"epoch_{row['epoch']}",
                "checkpoint": str(
                    RUN / "ltwm" / f"seed{seed}" / "checkpoints" / f"epoch_{row['epoch']:03d}.pt"
                ),
                "row": row,
            }
        return
    for stage in summary["stages"]:
        root = RUN / "ltwm" / f"seed1703_stage{stage['stage']}"
        for row in stage["history"]:
            yield {
                "label": f"stage_{stage['stage']}",
                "checkpoint": str(root / "checkpoints" / "best.pt"),
                "row": row,
            }


def main() -> None:
    seed_reports = {}
    for seed in (1701, 1702, 1703):
        candidates = []
        baseline = load(
            RUN / "ltwm" / f"baseline_seed{seed}" / "report" / "train_summary.json"
        )["history"][0]
        baseline_scores = {domain: domain_score(baseline, domain) for domain in DOMAINS}
        for candidate in candidate_rows(seed):
            row = candidate["row"]
            scores = {domain: domain_score(row, domain) for domain in DOMAINS}
            normalized = {
                domain: scores[domain] / baseline_scores[domain] for domain in DOMAINS
            }
            candidates.append(
                {
                    "label": candidate["label"],
                    "checkpoint": candidate["checkpoint"],
                    "recon_guard_pass": bool(row.get("recon_guard_pass", False)),
                    "scand_selection_score": float(row["selection_score"]),
                    "domain_scores": scores,
                    "macro_raw_score": mean(scores.values()),
                    "macro_normalized_to_old": mean(normalized.values()),
                    "h4_nmse": {
                        domain: float(row["val"][domain]["true"]["nmse"][3])
                        for domain in DOMAINS
                    },
                    "h4_norm_ratio": {
                        domain: float(row["val"][domain]["true"]["norm_ratio"][3])
                        for domain in DOMAINS
                    },
                    "reverse_gap_h4": {
                        domain: float(row["val"][domain]["reverse_nmse_gap_h4"])
                        for domain in DOMAINS
                    },
                }
            )
        eligible = [item for item in candidates if item["recon_guard_pass"]]
        by_scand = min(eligible, key=lambda item: item["scand_selection_score"])
        by_macro = min(eligible, key=lambda item: item["macro_normalized_to_old"])
        seed_reports[str(seed)] = {
            "baseline_domain_scores": baseline_scores,
            "selected_by_current_scand_rule": by_scand,
            "selected_by_normalized_tridomain_macro": by_macro,
            "selection_agrees": by_scand["checkpoint"] == by_macro["checkpoint"],
            "candidates": candidates,
        }

    train_summary = load(RUN / "ltwm" / "seed1702" / "report" / "train_summary.json")
    windows = {
        domain: int(train_summary["train_dataset"][domain]["num_windows"])
        for domain in DOMAINS
    }
    samples_per_seed = {"recon": 36_000, "scand": 18_000, "sacson": 18_000}
    coverage = {
        domain: {
            "training_windows": windows[domain],
            "scheduled_samples_per_seed": samples_per_seed[domain],
            "scheduled_samples_as_percent_of_window_count": 100.0
            * samples_per_seed[domain]
            / windows[domain],
            "note": "upper-bound exposure ratio; repeated windows can reduce unique coverage",
        }
        for domain in DOMAINS
    }
    report = {
        "status": "AUDITED",
        "data_protocol": {
            "outer_test_excluded": True,
            "domain_schedule": ["recon", "recon", "scand", "sacson"],
            "batch_size": 4,
            "updates_per_seed": 18_000,
            "samples_per_seed": samples_per_seed,
            "coverage": coverage,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "precision": "bfloat16 autocast",
            "seed1703_optimizer_continuity": False,
            "seed1703_note": "external SIGKILL forced bounded chunks; optimizer state resets between chunks",
        },
        "checkpoint_selection": seed_reports,
    }
    output = RUN / "ltwm" / "training_protocol_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
