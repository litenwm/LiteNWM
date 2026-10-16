#!/usr/bin/env python3
"""Score an existing raw feature pool with the frozen universal-D ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    conservative_energy,
    domain_policy,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-quantile", type=float, default=0.999)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base = load_features(args.features, labels_expected=False)
    training = json.loads(
        args.training_result.read_text(encoding="utf-8")
    )
    envelopes = load_envelopes(Path(training["support"]))
    envelope = envelopes[args.domain]
    spacing_m = 0.25 if args.domain == "recon" else 0.38
    feature, support_score, physical_guard = build_universal_features(
        base,
        spacing_m=spacing_m,
        support_envelope=envelope,
    )
    valid = (
        physical_guard
        & (
            support_score
            <= threshold_for_quantile(envelope, args.support_quantile)
        )
        & np.asarray(base["majority_support"], dtype=bool)
    )
    valid[:, 0] = True
    domain_id = np.full(
        len(valid), 0 if args.domain == "recon" else 1, dtype=np.int64
    )
    prediction = ensemble_predict(
        training["arms"]["D"]["checkpoints"]["full"],
        feature,
        domain_id,
        device=args.device,
    )
    policy, dual_temperature, harm_temperature = domain_policy(
        training, args.domain
    )
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_npz(
        args.output,
        record_key=np.asarray(base["record_key"]).astype(str),
        energy=energy,
        mean=prediction["mean"].astype(np.float32),
        std=prediction["std"].astype(np.float32),
        dual_logit=prediction["dual_logit"].astype(np.float32),
        harm_logit=prediction["harm_logit"].astype(np.float32),
        dual_probability=p_dual,
        harm_probability=p_harm,
        rank_logit=prediction["rank_logit"].astype(np.float32),
        valid=valid,
    )
    audit = {
        "schema": "universal_raw_feature_pool_scores_v1",
        "domain": args.domain,
        "features": str(args.features),
        "features_sha256": file_sha256(args.features),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "support_quantile": args.support_quantile,
        "states": int(len(valid)),
        "valid_fraction": float(valid[:, 1:].mean()),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "outcomes_accessed": False,
    }
    write_json(args.output.with_suffix(".audit.json"), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
