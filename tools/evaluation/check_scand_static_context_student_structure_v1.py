#!/usr/bin/env python3
"""Label-free structure and deterministic implementation check for 561 features."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_scand_pairwise_incremental_student_v1 as core  # noqa: E402


BASE = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/static_context_student_v1"
TABLE = BASE / "spent_runtime_pair_static_context_table.npz"
TABLE_AUDIT = BASE / "spent_runtime_pair_static_context_table.audit.v3.json"
OUTPUT = BASE / "pairwise_student_v1/structure_and_determinism.audit.json"


def one_step(x: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    torch.manual_seed(seed)
    model = core.Student(x.shape[-1]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    xt = torch.from_numpy(x)
    index = torch.arange(len(x), dtype=torch.float32)
    target = torch.stack(
        (
            torch.sin(index / 17.0),
            torch.cos(index / 19.0),
            (index.remainder(2) == 0).float(),
            (index.remainder(3) == 0).float(),
            (index.remainder(5) == 0).float(),
        ),
        dim=-1,
    )
    prediction = model(xt)
    loss = F.smooth_l1_loss(prediction[:, :2], target[:, :2])
    loss += F.binary_cross_entropy_with_logits(prediction[:, 2:], target[:, 2:])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        output = model(xt).numpy()
    state = {
        name: value.detach().numpy().copy()
        for name, value in model.state_dict().items()
    }
    return state, output


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with np.load(TABLE, allow_pickle=False) as payload:
        # Deliberately do not read any outcome/GT/target arrays.
        feature = payload["runtime_pair_feature"].astype(np.float32)
        route = payload["route_group"].astype(str)
        pool = payload["physical_support_pool"].astype(bool)
        formal = payload["formal_v70_index"].astype(np.int64)
        robot = payload["robot"].astype(str)
        camera = payload["camera_contract"].astype(str)
        archive_keys = sorted(payload.files)
    if feature.shape != (800, 65, 561):
        raise RuntimeError(f"unexpected feature shape: {feature.shape}")
    if not np.isfinite(feature).all():
        raise RuntimeError("feature contains NaN/Inf")
    if not np.array_equal(
        feature[:, :, -210:],
        np.broadcast_to(feature[:, :1, -210:], (800, 65, 210)),
    ):
        raise RuntimeError("static context is not exactly candidate-independent")
    if len(np.unique(route)) != 42:
        raise RuntimeError("route count changed")
    if np.any(formal < 0) or np.any(formal >= 65):
        raise RuntimeError("formal-v70 index invalid")
    fold_a, assignment_a = core.assign_folds(route)
    fold_b, assignment_b = core.assign_folds(route)
    if not np.array_equal(fold_a, fold_b) or assignment_a != assignment_b:
        raise RuntimeError("route fold assignment is not deterministic")
    for outer in range(5):
        train = np.flatnonzero(fold_a != outer)
        test = np.flatnonzero(fold_a == outer)
        if set(route[train]) & set(route[test]):
            raise RuntimeError(f"route leakage outer={outer}")
    sample = feature[:4].reshape(-1, 561)[:256]
    state_a, pred_a = one_step(sample, 20267810)
    state_b, pred_b = one_step(sample, 20267810)
    parameter_equal = all(
        np.array_equal(state_a[name], state_b[name]) for name in state_a
    )
    prediction_equal = np.array_equal(pred_a, pred_b)
    if not parameter_equal or not prediction_equal:
        raise RuntimeError("synthetic deterministic reproduction failed")
    audit = {
        "schema": "scand_static_context_student_structure_audit_v1",
        "status": "pass_before_preregistration_and_true_label_training",
        "table_sha256": sha256_file(TABLE),
        "table_audit_sha256": sha256_file(TABLE_AUDIT),
        "states": 800,
        "candidates_per_state": 65,
        "feature_dim": 561,
        "base_feature_dim": 351,
        "appended_static_feature_dim": 210,
        "static_context_candidate_independent_exact": True,
        "routes": int(len(np.unique(route))),
        "physical_support_rows": int(pool.sum()),
        "fold_state_loads": assignment_a["fold_state_loads"],
        "route_overlap_each_outer": 0,
        "robots": sorted(np.unique(robot).tolist()),
        "camera_contracts": sorted(np.unique(camera).tolist()),
        "synthetic_determinism": {
            "true_or_outcome_labels_read": False,
            "sample_rows": len(sample),
            "seed": 20267810,
            "one_adamw_step_repeated": True,
            "state_dict_bitwise_equal": parameter_equal,
            "prediction_bitwise_equal": prediction_equal,
        },
        "archive_keys_audited_without_target_access": archive_keys,
        "adaptive_development_only": True,
        "sealed_validation_opened": False,
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(
        canonical_json(
            {
                "status": audit["status"],
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "fold_state_loads": assignment_a["fold_state_loads"],
                "deterministic": True,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "true_or_outcome_labels_read": False,
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
