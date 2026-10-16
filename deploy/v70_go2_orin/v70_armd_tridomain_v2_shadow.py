#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
)
from tools.planning import (  # noqa: E402
    run_dualdomain_cem_tail_v66_replay as replay,
)
from tools.planning import (  # noqa: E402
    run_tridomain_cem_tail_v70_runtime_v2 as v70,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)
from v70_runtime_contract import (  # noqa: E402
    DOMAIN_INDEX,
    DOMAIN_SPACING_M,
    require_domain_calibration,
    select_executable_action,
)

SUPPORT_QUANTILE = 0.999
CANDIDATES = 17
DEFAULT_RUNTIME = ROOT / "runtime/v70_tridomain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/home/unitree/v70_ltwm_tridomain_shadow.npz"),
    )
    parser.add_argument(
        "--support",
        type=Path,
        default=(
            DEFAULT_RUNTIME
            / "selector/support/tridomain_spacing_envelopes.npz"
        ),
    )
    parser.add_argument(
        "--v70-result",
        type=Path,
        default=DEFAULT_RUNTIME / "selector/result.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_RUNTIME / "selector/checkpoints",
    )
    parser.add_argument(
        "--domain",
        choices=tuple(DOMAIN_INDEX),
        default="recon",
    )
    parser.add_argument(
        "--allow-relaxed-calibration",
        action="store_true",
        help=(
            "Permit a domain only when every RELAXED_FEASIBLE_V1 seal "
            "field is present and true."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_armd_tridomain_v2_shadow.npz"),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as payload:
        return {
            key: payload[key].copy()
            for key in payload.files
        }


def require_shape(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> None:
    if value.shape != shape:
        raise RuntimeError(
            f"{name}: expected {shape}, got {value.shape}"
        )

    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def localize_checkpoint_records(
    records: list[dict],
    checkpoint_dir: Path,
) -> list[dict]:
    result: list[dict] = []

    for index, record in enumerate(records):
        expected_hash = str(record["sha256"])
        original = Path(str(record["path"]))
        local = checkpoint_dir / original.name

        if not local.is_file():
            fallbacks = (
                checkpoint_dir / f"proposal_aware_full_seed{index}.pt",
                checkpoint_dir / f"arm_D_full_seed{index}.pt",
            )
            local = next(
                (path for path in fallbacks if path.is_file()),
                local,
            )
            if not local.is_file():
                raise FileNotFoundError(
                    f"missing local Arm-D checkpoint for seed index "
                    f"{index} in {checkpoint_dir}"
                )

        actual_hash = file_sha256(local)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Arm-D checkpoint hash mismatch:\n"
                f"path={local}\n"
                f"expected={expected_hash}\n"
                f"actual={actual_hash}"
            )

        result.append({
            **record,
            "path": str(local.resolve()),
        })

    if len(result) != 3:
        raise RuntimeError(
            f"expected three Arm-D members, got {len(result)}"
        )

    return result


def main() -> None:
    args = parse_args()
    domain = args.domain
    domain_id_value = DOMAIN_INDEX[domain]
    spacing_m = DOMAIN_SPACING_M[domain]

    print("===== V70 tri-domain Arm-D fixed-pool shadow =====")
    print("No ROS publisher is created.")
    print("No SportClient command is issued.")
    print()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    source = load_npz(args.features)

    required = {
        "candidates_waypoint_units",
        "actions_h4",
        "action_features",
        "static_features",
        "raw_ltwm_features",
        "majority_support",
    }
    missing = required.difference(source)
    if missing:
        raise RuntimeError(
            f"feature artifact missing arrays: {sorted(missing)}"
        )

    candidates = source["candidates_waypoint_units"].astype(np.float32)
    actions_h4 = source["actions_h4"].astype(np.float32)
    action_features = source["action_features"].astype(np.float32)
    static_features = source["static_features"].astype(np.float32)
    raw_ltwm_features = source["raw_ltwm_features"].astype(np.float32)
    majority_support = source["majority_support"].astype(bool)

    require_shape(
        "candidates_waypoint_units",
        candidates,
        (1, CANDIDATES, 4, 2),
    )
    require_shape(
        "actions_h4",
        actions_h4,
        (1, CANDIDATES, 4, 3),
    )
    require_shape(
        "action_features",
        action_features,
        (1, CANDIDATES, 242),
    )
    require_shape(
        "static_features",
        static_features,
        (1, CANDIDATES, 210),
    )
    require_shape(
        "raw_ltwm_features",
        raw_ltwm_features,
        (1, CANDIDATES, 112),
    )
    require_shape(
        "majority_support",
        majority_support.astype(np.float32),
        (1, CANDIDATES),
    )

    data = {
        "action_features": action_features,
        "actions_h4": actions_h4,
        "static_features": static_features,
        "raw_ltwm_features": raw_ltwm_features,
    }

    if not args.support.is_file():
        raise FileNotFoundError(args.support)

    envelopes = replay.load_envelopes(args.support)
    if domain not in envelopes:
        raise RuntimeError(
            f"support envelope missing domain {domain}"
        )

    envelope = envelopes[domain]
    threshold = replay.threshold_for_quantile(
        envelope,
        SUPPORT_QUANTILE,
    )

    universal, support_score, physical_guard = (
        build_universal_features(
            data,
            spacing_m=spacing_m,
            support_envelope=envelope,
        )
    )

    require_shape(
        "universal_features",
        universal,
        (1, CANDIDATES, 815),
    )
    require_shape(
        "support_score",
        support_score,
        (1, CANDIDATES),
    )
    require_shape(
        "physical_guard",
        physical_guard.astype(np.float32),
        (1, CANDIDATES),
    )

    # 正式 fixed-pool/CEM validity contract。
    valid = (
        physical_guard
        & (support_score <= threshold)
    )
    valid[:, 0] = True

    if not args.v70_result.is_file():
        raise FileNotFoundError(args.v70_result)

    v70_result = json.loads(
        args.v70_result.read_text(encoding="utf-8")
    )

    if "checkpoints" not in v70_result:
        raise RuntimeError("v70 result missing checkpoints")
    if "calibration" not in v70_result:
        raise RuntimeError("v70 result missing calibration")

    if file_sha256(args.support) != v70_result.get("support_sha256"):
        raise RuntimeError("tri-domain support envelope SHA mismatch")
    runtime = require_domain_calibration(
        v70_result,
        domain,
        allow_relaxed=args.allow_relaxed_calibration,
    )
    calibration = runtime["calibration"]

    checkpoint_records = localize_checkpoint_records(
        list(v70_result["checkpoints"]["full"]),
        args.checkpoint_dir,
    )

    policy = dict(calibration["policy"])
    dual_temperature = float(
        calibration["dual_temperature"]
    )
    harm_temperature = float(
        calibration["harm_temperature"]
    )

    domain_id = np.asarray([domain_id_value], dtype=np.int64)

    print("universal features:", universal.shape)
    print("support threshold:", threshold)
    print(
        "physical guard:",
        int(physical_guard.sum()),
        "/",
        CANDIDATES,
    )
    print(
        "support valid:",
        int((support_score <= threshold).sum()),
        "/",
        CANDIDATES,
    )
    print(
        "formal valid:",
        int(valid.sum()),
        "/",
        CANDIDATES,
    )
    print("Arm-D members:", len(checkpoint_records))
    print("policy:", policy)
    print()

    started = time.monotonic()

    prediction = v70.compatible_ensemble_predict(
        checkpoint_records,
        universal,
        domain_id,
        device=args.device,
    )

    elapsed = time.monotonic() - started

    require_shape(
        "prediction mean",
        prediction["mean"],
        (1, CANDIDATES, 4),
    )
    require_shape(
        "prediction std",
        prediction["std"],
        (1, CANDIDATES, 4),
    )
    require_shape(
        "dual_logit",
        prediction["dual_logit"],
        (1, CANDIDATES),
    )
    require_shape(
        "harm_logit",
        prediction["harm_logit"],
        (1, CANDIDATES),
    )
    require_shape(
        "rank_logit",
        prediction["rank_logit"],
        (1, CANDIDATES),
    )

    energy, p_dual, p_harm = replay.conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )

    safe = replay.candidate_safety_mask(
        prediction,
        p_dual,
        p_harm,
        valid,
        policy,
    )

    selection = select_executable_action(
        energy,
        safe,
        actions_h4,
    )
    selected_index = int(selection["selected_index"])
    abstained = bool(selection["abstained"])
    diagnostic_waypoints = candidates[0, selected_index].copy()
    selected_waypoints = (
        np.zeros_like(diagnostic_waypoints)
        if abstained
        else diagnostic_waypoints.copy()
    )
    selected_actions = selection["selected_actions_h4"]

    z = float(policy["z"])
    ucb = prediction["mean"] + z * prediction["std"]

    print("index  valid safe      energy     p_dual     p_harm")
    for index in range(CANDIDATES):
        energy_text = (
            f"{float(energy[0, index]):+.6f}"
            if np.isfinite(energy[0, index])
            else "      inf"
        )
        print(
            f"{index:>5d}  "
            f"{int(valid[0, index]):>5d} "
            f"{int(safe[0, index]):>4d}  "
            f"{energy_text:>10s}  "
            f"{float(p_dual[0, index]):.6f}  "
            f"{float(p_harm[0, index]):.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidates_waypoint_units=candidates,
        actions_h4=actions_h4,
        universal_features=universal,
        support_score=support_score,
        physical_guard=physical_guard,
        majority_support=majority_support,
        valid=valid,
        mean=prediction["mean"],
        std=prediction["std"],
        ucb=ucb,
        dual_probability=p_dual,
        harm_probability=p_harm,
        rank_logit=prediction["rank_logit"],
        energy=energy,
        safe=safe,
        selected_index=np.asarray(
            [selected_index],
            dtype=np.int64,
        ),
        abstained=np.asarray([abstained], dtype=bool),
        selection_reason=np.asarray(selection["reason"]),
        diagnostic_selected_waypoints=diagnostic_waypoints,
        selected_waypoints=selected_waypoints,
        selected_actions_h4=selected_actions,
        selected_first_action=selection["selected_first_action"],
        domain=np.asarray(domain),
        domain_id=np.asarray(domain_id_value, dtype=np.int64),
        calibration_contract=np.asarray(
            runtime["calibration_contract"]
        ),
        support_threshold=np.asarray(
            threshold,
            dtype=np.float32,
        ),
    )

    print()
    print("===== RESULT =====")
    print("universal_features:", universal.shape)
    print("prediction mean:", prediction["mean"].shape)
    print("prediction std:", prediction["std"].shape)
    print("valid candidates:", int(valid.sum()), "/", CANDIDATES)
    print(
        "safe challengers:",
        int(safe[:, 1:].sum()),
        "/",
        CANDIDATES - 1,
    )
    print("selected index:", selected_index)
    print("decision:", "ABSTAIN_ZERO" if abstained else "SHADOW_SELECT")
    print("selection reason:", selection["reason"])
    print(
        "selection source:",
        "explicit zero-action abstention"
        if abstained
        else "safe NoMaD challenger (shadow only)",
    )
    print("selected waypoints:")
    print(
        np.array2string(
            selected_waypoints,
            precision=6,
            suppress_small=True,
        )
    )
    print("selected actions H4:")
    print(
        np.array2string(
            selected_actions,
            precision=6,
            suppress_small=True,
        )
    )
    print("selected first action:", selected_actions[0])
    print("Arm-D inference seconds:", round(elapsed, 4))
    print("saved:", args.output)
    print()
    print("V70 TRI-DOMAIN ARM-D SHADOW COMPUTE: PASS")


if __name__ == "__main__":
    main()
