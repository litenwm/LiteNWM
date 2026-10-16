#!/usr/bin/env python3
"""Train and freeze the final six-bank v26 state-gate ensemble."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.evaluate_recon_frozen_universal_gate_v23 import load_score
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.screen_recon_target_state_gate_v24 import train_model
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json


@dataclass(frozen=True)
class FinalConfig:
    input_dim: int
    hidden_dim: int
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_path = args.output_dir / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    calibration = json.loads(args.calibration_result.read_text(encoding="utf-8"))
    if not calibration.get("strict") or calibration.get("future_route_quota"):
        raise RuntimeError("v26 fixed calibration was not strictly deployable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    banks = [load_bank(args.root, value) for value in "abcdef"]
    scores = [load_score(args.root, value) for value in "abcdef"]
    full = np.concatenate([bank["feature"] for bank in banks])
    group = np.concatenate([bank["route"] for bank in banks])
    challenger_values = []
    dual_values = []
    harm_values = []
    for bank, score in zip(banks, scores):
        challenger = frozen_challenger(score, bank["valid"])
        row = np.arange(100)
        challenger_values.append(challenger)
        dual_values.append(bank["dual"][row, challenger])
        harm_values.append(bank["harm"][row, challenger])
    challenger = np.concatenate(challenger_values)
    dual = np.concatenate(dual_values).astype(np.float32)
    harm = np.concatenate(harm_values).astype(np.float32)
    chosen = full[np.arange(600), challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = {}
    train_index = np.arange(600)
    for variant_number, (name, feature) in enumerate(variants.items()):
        values = []
        config = FinalConfig(
            input_dim=feature.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout
        )
        for member in range(args.ensemble_members):
            seed = args.seed + variant_number * 1000000 + member * 1009
            model, mean, scale = train_model(
                feature, dual, harm, group, train_index, args, seed
            )
            path = args.output_dir / f"{name}_member_{member}.pt"
            torch.save(
                {
                    "schema": "recon_target_state_gate_final_v26_member",
                    "variant": name,
                    "config": asdict(config),
                    "feature_mean": mean,
                    "feature_scale": scale,
                    "model": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "seed": seed,
                    "training_states": 600,
                    "training_routes": 150,
                    "route_robust_loss": "0.5 route mean + 0.5 worst-route-CVaR20",
                },
                path,
            )
            values.append({
                "path": str(path), "sha256": file_sha256(path), "seed": seed,
            })
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        checkpoints[name] = values
    report = {
        "schema": "recon_target_state_gate_final_v26_freeze",
        "status": "frozen_before_final_reserve_features_or_outcomes",
        "training_evidence": "six_spent_development_banks_A_to_F",
        "training_states": 600,
        "training_routes": 150,
        "candidate_pool": "K17_trimmed_mean75_incumbent_plus_fixed_nomad_x16",
        "challenger_contract": (
            "frozen universal p_dual - 0.5*p_harm - 0.5*(std_ate+std_objective)"
        ),
        "support_quantile": 0.999,
        "policy": calibration["policy"],
        "policy_source": str(args.calibration_result),
        "policy_source_sha256": file_sha256(args.calibration_result),
        "checkpoints": checkpoints,
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            "seed": args.seed,
        },
        "bank_sources": {
            bank["name"]: bank["sha"] for bank in banks
        },
        "online_causal": True,
        "future_route_quota": False,
        "final_reserve_features_accessed": False,
        "final_reserve_outcomes_accessed": False,
        "cem": False,
    }
    write_json(freeze_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
