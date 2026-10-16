#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev


RUN = Path("/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights")
DOMAINS = ("recon", "scand", "sacson")
METRICS = ("nmse", "cosine", "norm_ratio")


def load(path: Path):
    return json.loads(path.read_text())


def best_new_row(seed: int) -> tuple[dict, str, float]:
    summary = load(RUN / "ltwm" / f"seed{seed}" / "report" / "train_summary.json")
    if seed == 1703 and "stages" in summary:
        stage = next(item for item in summary["stages"] if item["stage"] == summary["best_stage"])
        row = min(
            (item for item in stage["history"] if item.get("recon_guard_pass", False)),
            key=lambda item: float(item["selection_score"]),
        )
        return row, f"stage_{summary['best_stage']}", float(row["selection_score"])
    row = min(
        (
            item
            for item in summary["history"]
            if item.get("recon_guard_pass", False)
            and math.isfinite(float(item["selection_score"]))
        ),
        key=lambda item: float(item["selection_score"]),
    )
    return row, f"epoch_{row['epoch']}", float(row["selection_score"])


def extract(metrics: dict) -> dict:
    result = {}
    for domain in DOMAINS:
        values = metrics[domain]
        result[domain] = {
            name: [float(value) for value in values["true"][name]] for name in METRICS
        }
        result[domain]["reverse_nmse_gap_h4"] = float(values["reverse_nmse_gap_h4"])
        result[domain]["noisy_nmse_gap_h4"] = float(values["noisy_nmse_gap_h4"])
    return result


def main() -> None:
    report = {
        "protocol": {
            "comparison": "old warm-start checkpoint vs new checkpoint",
            "same_split_lock": str(RUN / "splits" / "split_lock.json"),
            "evaluation_batches_per_domain": 512,
            "history_frames": 4,
            "future_frames": 4,
            "selection_rule": "lowest SCAND selection score subject to RECON guard",
        },
        "seeds": {},
    }
    for seed in (1701, 1702, 1703):
        baseline = load(
            RUN / "ltwm" / f"baseline_seed{seed}" / "report" / "train_summary.json"
        )["history"][0]
        new_row, selected_at, new_score = best_new_row(seed)
        old_metrics = extract(baseline["val"])
        new_metrics = extract(new_row["val"])
        changes = {}
        for domain in DOMAINS:
            old_h4 = old_metrics[domain]["nmse"][3]
            new_h4 = new_metrics[domain]["nmse"][3]
            changes[domain] = {
                "old_h4_nmse": old_h4,
                "new_h4_nmse": new_h4,
                "absolute_delta": new_h4 - old_h4,
                "relative_improvement_percent": 100.0 * (old_h4 - new_h4) / old_h4,
                "old_reverse_nmse_gap_h4": old_metrics[domain]["reverse_nmse_gap_h4"],
                "new_reverse_nmse_gap_h4": new_metrics[domain]["reverse_nmse_gap_h4"],
                "old_noisy_nmse_gap_h4": old_metrics[domain]["noisy_nmse_gap_h4"],
                "new_noisy_nmse_gap_h4": new_metrics[domain]["noisy_nmse_gap_h4"],
            }
        report["seeds"][str(seed)] = {
            "selected_at": selected_at,
            "old_selection_score": float(baseline["selection_score"]),
            "new_selection_score": new_score,
            "old": old_metrics,
            "new": new_metrics,
            "h4_changes": changes,
        }

    aggregate = {}
    for domain in DOMAINS:
        old_values = [report["seeds"][str(seed)]["h4_changes"][domain]["old_h4_nmse"] for seed in (1701, 1702, 1703)]
        new_values = [report["seeds"][str(seed)]["h4_changes"][domain]["new_h4_nmse"] for seed in (1701, 1702, 1703)]
        aggregate[domain] = {
            "old_h4_nmse_mean": mean(old_values),
            "old_h4_nmse_std": pstdev(old_values),
            "new_h4_nmse_mean": mean(new_values),
            "new_h4_nmse_std": pstdev(new_values),
            "relative_improvement_percent": 100.0 * (mean(old_values) - mean(new_values)) / mean(old_values),
            "per_seed_improved": sum(new < old for old, new in zip(old_values, new_values)),
            "num_seeds": 3,
        }
    report["aggregate"] = aggregate
    output = RUN / "ltwm" / "old_vs_new_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
