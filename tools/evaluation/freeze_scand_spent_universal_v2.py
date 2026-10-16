#!/usr/bin/env python3
"""Freeze universal-v2 selections on the already-spent SCAND H4-100 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
    policy_selected,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    selected_for_policy,
    sigmoid,
)


SCAND_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
UNIVERSAL_ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=SCAND_ROOT
        / "features/scand_fresh_outer_h4_100_seed20260720_features.npz",
    )
    parser.add_argument(
        "--candidate", type=Path, default=SCAND_ROOT / "fresh_outer_k17.npz"
    )
    parser.add_argument(
        "--manifest", type=Path, default=SCAND_ROOT / "manifest.windows.jsonl"
    )
    parser.add_argument(
        "--training-result",
        type=Path,
        default=UNIVERSAL_ROOT / "training_v2/result.json",
    )
    parser.add_argument(
        "--safe-rank-result",
        type=Path,
        default=UNIVERSAL_ROOT / "safe_rank_v2/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UNIVERSAL_ROOT / "scand_spent_replay_v2/freeze",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def arm_policy(
    result: dict[str, Any], arm: str, domain: str
) -> tuple[dict[str, float], float, float]:
    value = result["arms"][arm]
    if arm == "B":
        frontier = value["calibration"]["frontier"]
        temperature = value["temperature"]["record"]
    else:
        frontier = value["calibration"]["frontier"]["domains"][domain]
        temperature = value["temperature"]["domains"][domain]
    return (
        frontier["policy"],
        float(temperature["deployment_dual_temperature"]),
        float(temperature["deployment_harm_temperature"]),
    )


def main() -> None:
    args = parse_args()
    selected_path = args.output_dir / "selected_index.npz"
    if selected_path.exists():
        raise FileExistsError(selected_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = read_manifest(args.manifest)
    if len(rows) != 100:
        raise RuntimeError("spent replay is not H4-100")
    record_key = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    data = load_features(args.features, labels_expected=False)
    if not np.array_equal(data["record_key"].astype(str), record_key):
        raise RuntimeError("feature/manifest identity mismatch")
    metadata = data["metadata"]
    if metadata.get("labels_accessed") or metadata.get("labels_in_output"):
        raise RuntimeError("selection freeze feature artifact contains labels")
    with np.load(args.candidate, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("candidate/manifest identity mismatch")

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    safe_rank = json.loads(args.safe_rank_result.read_text(encoding="utf-8"))
    if (
        training.get("cem") is not False
        or safe_rank.get("cem") is not False
        or not training.get("primary_excludes_scorer_v2_predictions")
        or not safe_rank.get("primary_excludes_scorer_v2_predictions")
    ):
        raise RuntimeError("universal replay model contract is invalid")
    envelopes = load_envelopes(Path(training["support"]))
    feature, score, guard = build_universal_features(
        data,
        spacing_m=0.38,
        support_envelope=envelopes["scand"],
    )
    threshold = threshold_for_quantile(envelopes["scand"], 0.999)
    valid = guard & (score <= threshold)
    valid[:, 0] = True
    domain = np.ones(len(feature), dtype=np.int64)

    arrays: dict[str, np.ndarray] = {"record_key": record_key}
    selection_audit: dict[str, Any] = {}
    for arm in ("B", "C", "D"):
        prediction = ensemble_predict(
            training["arms"][arm]["checkpoints"]["full"],
            feature,
            domain,
            device=args.device,
        )
        policy, dual_temperature, harm_temperature = arm_policy(
            training, arm, "scand"
        )
        selected = selected_for_policy(
            prediction,
            sigmoid(prediction["dual_logit"] / dual_temperature),
            sigmoid(prediction["harm_logit"] / harm_temperature),
            valid,
            policy,
        )
        arrays[f"universal_{arm.lower()}"] = selected
        selection_audit[f"universal_{arm.lower()}"] = {
            "promotions": int((selected != 0).sum()),
            "promotion_fraction": float((selected != 0).mean()),
            "calibration_feasible": bool(
                training["arms"][arm]["calibration"]["feasible"]
            ),
        }

    safe_prediction = ensemble_predict(
        safe_rank["checkpoints"]["full"],
        feature,
        domain,
        device=args.device,
    )
    safe_selected = policy_selected(
        safe_prediction,
        valid,
        safe_rank["calibration"]["scand"]["policy"],
    )
    arrays["safe_rank_failed_ablation"] = safe_selected
    selection_audit["safe_rank_failed_ablation"] = {
        "promotions": int((safe_selected != 0).sum()),
        "promotion_fraction": float((safe_selected != 0).mean()),
        "calibration_feasible": bool(
            safe_rank["calibration"]["scand"]["feasible"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_npz(selected_path, **arrays)
    freeze = {
        "schema": "universal_dualdomain_raw_v2_scand_spent_selection_freeze",
        "status": "selections_frozen_for_spent_development_replay",
        "evidence_role": "historically spent development; never fresh outer",
        "states": 100,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "feature_artifact": str(args.features),
        "feature_artifact_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "safe_rank_result": str(args.safe_rank_result),
        "safe_rank_result_sha256": file_sha256(args.safe_rank_result),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "support_candidate_acceptance": float(valid[:, 1:].mean()),
        "support_window_acceptance": float(valid[:, 1:].any(axis=1).mean()),
        "selection_audit": selection_audit,
        "labels_read_by_this_freeze": False,
        "historically_spent": True,
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(args.output_dir / "freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
