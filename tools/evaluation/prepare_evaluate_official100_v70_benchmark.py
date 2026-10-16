#!/usr/bin/env python3
"""Prepare and evaluate the frozen v70/DWM-scorer-v2 official-100 benchmark.

The prepare stage is label-free.  It only binds the already-generated NoMaD
x16 candidates to the two frozen H4 official-100 indices and constructs the
canonical K17 pool (trimmed-mean incumbent plus the 16 NoMaD samples).

The evaluate stage is intentionally separate and requires an already-frozen
v70 replay.  It then opens trajectory poses, reconstructs the raw
DWM+scorer-v2 argmin, and reports matched-state metrics for NoMaD+NWM,
DWM+scorer-v2, formal v70, and an existing-union objective oracle.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.nomad_action_adapter import (  # noqa: E402
    nomad_waypoints_to_local_actions,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    metric_labels,
    trimmed_mean75,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    local_ground_truth,
)


RRNWM = Path("/home/rainbow/rrnwm")
DEFAULT_OUTPUT = ROOT / "outputs/official100_v70_dwm_scorerv2_benchmark_v1"
METRICS = ("ate", "rpe", "endpoint", "objective")

SPECS = {
    "recon": {
        "index": RRNWM
        / "data_splits/recon/server_full_nwm_unseen_h4_100/navigation_eval.pkl",
        "candidate": RRNWM
        / "eval_output/visualnav_baseline/"
        "serverfull_nwmunseen_h4_100_nomad_x16_seed0_candidates.npz",
        "candidate_json": RRNWM
        / "eval_output/visualnav_baseline/"
        "serverfull_nwmunseen_h4_100_nomad_x16_seed0_candidates.json",
        "nomad_nwm_per_sample": RRNWM
        / "eval_output/visualnav_baseline/"
        "serverfull_nwmunseen_h4_100_nwm_s_nomad_x16_d250_rep1_"
        "lpips_seed0_resumable/per_sample.csv",
        "trajectory_root": Path("/home/rainbow/nwm/data/recon"),
        "spacing_m": 0.25,
    },
    "scand": {
        "index": RRNWM
        / "data_splits/scand/official_nwm_test_h4_100_balanced_seed17/"
        "navigation_eval.pkl",
        "candidate": RRNWM
        / "eval_output/visualnav_baseline/multidataset_officialtest_h4_100_seed17/"
        "scand/nomad_x16_seed0_candidates.npz",
        "candidate_json": RRNWM
        / "eval_output/visualnav_baseline/multidataset_officialtest_h4_100_seed17/"
        "scand/nomad_x16_seed0_candidates.json",
        "nomad_nwm_per_sample": RRNWM
        / "eval_output/visualnav_baseline/multidataset_officialtest_h4_100_seed17/"
        "scand/nwm_s_nomad_x16_d250_rep1_lpips_seed0_resumable/per_sample.csv",
        "trajectory_root": Path("/home/rainbow/nwm/data/scand"),
        "spacing_m": 0.38,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--domain", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def scand_route_group(name: str) -> str:
    tokens = name.split("_")
    if len(tokens) >= 6 and tokens[0] == "random" and tokens[1] == "mdps":
        endpoints = sorted((tokens[4].lower(), tokens[5].lower()))
        return "__".join(endpoints)
    return name.rsplit("_", 1)[0]


def canonical_rows(
    domain: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        trajectory = str(record["traj_name"])
        current = int(record["curr_time"])
        if domain == "recon":
            route = re.sub(r"_r\d+$", "", trajectory)
        else:
            route = scand_route_group(trajectory)
        uid = hashlib.sha256(
            f"official100:{domain}:{trajectory}:{current}:h4".encode()
        ).hexdigest()[:24]
        rows.append(
            {
                "dataset": domain,
                "partition": "official-100-frozen-benchmark",
                "trajectory_id": trajectory,
                "route_id": route,
                "route_group": route,
                "current_t": current,
                "curr_time": current,
                "goal_t": current + 4,
                "history_timesteps": list(range(current - 3, current + 1)),
                "eval_index": index,
                "window_uid": uid,
                "selection_source": (
                    "frozen rrnwm H4 official-100 index; labels unopened"
                ),
            }
        )
    return rows


def load_and_bind_inputs(domain: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    spec = SPECS[domain]
    for key in ("index", "candidate", "candidate_json", "trajectory_root"):
        Path(spec[key]).resolve(strict=True)
    with Path(spec["index"]).open("rb") as handle:
        navigation = pickle.load(handle)
    metadata = json.loads(Path(spec["candidate_json"]).read_text(encoding="utf-8"))
    records = list(metadata["records"])
    with np.load(Path(spec["candidate"]), allow_pickle=False) as payload:
        nomad = payload["candidates"].astype(np.float32)
    expected = [
        (str(record["traj_name"]), int(record["curr_time"]), 4, 4)
        for record in records
    ]
    if navigation != expected:
        raise RuntimeError("candidate metadata differs from frozen navigation index")
    if nomad.shape != (100, 16, 4, 2) or len(records) != 100:
        raise RuntimeError(
            f"official-100 candidate contract changed: {nomad.shape}/{len(records)}"
        )
    root = Path(spec["trajectory_root"])
    missing = []
    for record in records:
        trajectory = str(record["traj_name"])
        current = int(record["curr_time"])
        trajectory_dir = root / trajectory
        if not (trajectory_dir / "traj_data.pkl").is_file():
            missing.append(str(trajectory_dir / "traj_data.pkl"))
        for timestep in (*range(current - 3, current + 1), current + 4):
            if not (trajectory_dir / f"{timestep}.jpg").is_file():
                missing.append(str(trajectory_dir / f"{timestep}.jpg"))
    if missing:
        raise RuntimeError(f"missing {len(missing)} official inputs; first={missing[0]}")
    return records, nomad


def prepare(domain: str, output_root: Path) -> None:
    directory = output_root / domain
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    records, nomad = load_and_bind_inputs(domain)
    rows = canonical_rows(domain, records)
    incumbent = np.stack([trimmed_mean75(value) for value in nomad]).astype(
        np.float32
    )
    candidates = np.concatenate((incumbent[:, None], nomad), axis=1).astype(
        np.float32
    )
    actions = nomad_waypoints_to_local_actions(candidates, dataset=domain).astype(
        np.float32
    )
    manifest = directory / "manifest.windows.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    adapted = directory / "source_records.json"
    write_json(
        adapted,
        {
            "records": [
                {
                    **record,
                    "trajectory_id": str(record["traj_name"]),
                    "curr_time": int(record["curr_time"]),
                    "eval_index": index,
                }
                for index, record in enumerate(records)
            ]
        },
    )
    candidate = directory / "k17.npz"
    write_npz(
        candidate,
        candidates_waypoint_units=candidates,
        candidate_actions_h4=actions,
        nomad_x16_waypoint_units=nomad,
        record_key=np.asarray(
            [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
        ),
        trajectory_id=np.asarray([str(row["trajectory_id"]) for row in rows]),
        route_id=np.asarray([str(row["route_id"]) for row in rows]),
        eval_index=np.arange(100, dtype=np.int64),
    )
    audit = {
        "schema": "official100_v70_dwm_scorerv2_input_freeze_v1",
        "status": "prepared_label_free_before_dwm_or_v70_selection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "states": 100,
        "horizon": 4,
        "candidate_order": ["trimmed_mean75_incumbent", "frozen_nomad_x16"],
        "candidate_shape": list(candidates.shape),
        "navigation_index": str(Path(SPECS[domain]["index"]).resolve()),
        "navigation_index_sha256": sha256(Path(SPECS[domain]["index"])),
        "upstream_candidate": str(Path(SPECS[domain]["candidate"]).resolve()),
        "upstream_candidate_sha256": sha256(Path(SPECS[domain]["candidate"])),
        "upstream_candidate_json_sha256": sha256(
            Path(SPECS[domain]["candidate_json"])
        ),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "source_records": str(adapted.resolve()),
        "source_records_sha256": sha256(adapted),
        "candidate": str(candidate.resolve()),
        "candidate_sha256": sha256(candidate),
        "trajectory_root": str(Path(SPECS[domain]["trajectory_root"]).resolve()),
        "labels_opened": False,
        "outcomes_opened": False,
        "reserve_accessed": False,
    }
    write_json(directory / "input_freeze.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


def summarize(
    selected: np.ndarray,
    candidates: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)
    row = np.arange(len(selected))
    absolute = labels["values"][row, selected, :4]
    delta = labels["targets"][row, selected, :4]
    promoted = selected != 0
    dual = promoted & (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = promoted & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    count = int(promoted.sum())
    return (
        {
            "states": int(len(selected)),
            "absolute_mean_m": {
                name: float(absolute[:, index].mean())
                for index, name in enumerate(METRICS)
            },
            "absolute_p90_m": {
                name: float(np.quantile(absolute[:, index], 0.9))
                for index, name in enumerate(METRICS)
            },
            "coverage_vs_k17_incumbent": float(promoted.mean()),
            "promotions_vs_k17_incumbent": count,
            "dual_successes": int(dual.sum()),
            "dual_precision": float(dual.sum() / count) if count else None,
            "harm_count": int(harm.sum()),
            "harm_rate": float(harm.sum() / count) if count else None,
        },
        {
            "selected_index": selected.astype(np.int64),
            "absolute_metrics_m": absolute.astype(np.float32),
            "delta_metrics_m": delta.astype(np.float32),
            "dual": dual,
            "harm": harm,
        },
    )


def load_nomad_nwm_selected(domain: str, records: list[dict[str, Any]]) -> np.ndarray:
    path = Path(SPECS[domain]["nomad_nwm_per_sample"])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 100:
        raise RuntimeError(f"NoMaD+NWM per-sample result is incomplete: {path}")
    expected = [
        (str(record["traj_name"]), int(record["curr_time"])) for record in records
    ]
    observed = [(row["traj_name"], int(row["curr_time"])) for row in rows]
    if observed != expected:
        raise RuntimeError("NoMaD+NWM state identity differs from official-100")
    selected = np.asarray([int(row["nwm_selected_index"]) for row in rows])
    if np.any((selected < 0) | (selected >= 16)):
        raise RuntimeError("NoMaD+NWM index is outside the NoMaD x16 pool")
    return selected


def evaluate(domain: str, output_root: Path) -> None:
    directory = output_root / domain
    freeze_path = directory / "input_freeze.json"
    replay_dir = directory / "v70_replay"
    selected_path = replay_dir / "selected_actions.npz"
    feature_path = directory / "features" / f"official100_{domain}_features.npz"
    result_path = directory / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for key in ("manifest", "candidate"):
        path = Path(freeze[key])
        if sha256(path) != freeze[f"{key}_sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")
    records, nomad = load_and_bind_inputs(domain)
    rows = [
        json.loads(line)
        for line in Path(freeze["manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ground_truth = np.stack(
        [
            local_ground_truth(
                Path(SPECS[domain]["trajectory_root"])
                / str(row["trajectory_id"])
                / "traj_data.pkl",
                int(row["current_t"]),
            )
            for row in rows
        ]
    ).astype(np.float32)
    with np.load(Path(freeze["candidate"]), allow_pickle=False) as payload:
        k17_native = payload["candidates_waypoint_units"].astype(np.float32)
        expected_record = payload["record_key"].astype(str)
    bridge = np.float32(1.0 if domain == "recon" else 0.38 / 0.25)
    k17 = k17_native * bridge
    with np.load(feature_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected_record):
            raise RuntimeError("base feature identity mismatch")
        if not np.array_equal(payload["candidates"].astype(np.float32), k17):
            raise RuntimeError("base feature candidate mismatch")
        aux = payload["scorer_v2_aux_features"].astype(np.float32)
    # Frozen raw scorer-v2 deployment proxy: 0.4*ATE mean + 0.6*RTP mean.
    dwm_energy = np.float32(0.4) * aux[:, :, 6] + np.float32(0.6) * aux[:, :, 8]
    dwm_selected = np.argmin(dwm_energy, axis=1).astype(np.int64)

    with np.load(selected_path, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected_record):
            raise RuntimeError("v70 replay identity mismatch")
        v70_candidates = payload["final_cem_feature_waypoint_units"].astype(
            np.float32
        )
        v70_selected = payload["final_cem_selected_index"].astype(np.int64)

    all_candidates = [k17]
    for iteration in range(3):
        with np.load(
            replay_dir / f"iteration_{iteration:02d}/population.npz",
            allow_pickle=False,
        ) as payload:
            population = payload["candidates_feature_waypoint_units"].astype(
                np.float32
            )
        all_candidates.append(population[:, 1:])
    union = np.concatenate(all_candidates, axis=1)
    union_metrics = metric_labels(union, ground_truth, 0.25, 0.4, 0.6)["values"]
    oracle_selected = np.argmin(union_metrics[:, :, 3], axis=1).astype(np.int64)

    nomad_nwm_raw = load_nomad_nwm_selected(domain, records)
    nomad_nwm_candidates = nomad * bridge
    # Prepend K17 incumbent so summaries share the same reference and indices.
    nomad_nwm_pool = np.concatenate((k17[:, :1], nomad_nwm_candidates), axis=1)
    nomad_nwm_selected = nomad_nwm_raw + 1

    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "nomad_plus_nwm_rep1_lpips": (nomad_nwm_pool, nomad_nwm_selected),
        "dwm_plus_scorer_v2_raw_argmin_k17": (k17, dwm_selected),
        "formal_v70_three_round_cem": (v70_candidates, v70_selected),
        "existing_union_objective_oracle": (union, oracle_selected),
    }
    summaries: dict[str, Any] = {}
    output_arrays: dict[str, np.ndarray] = {
        "record_key": expected_record,
        "ground_truth_local_m": ground_truth,
    }
    for name, (candidates, selected) in methods.items():
        summary, arrays = summarize(selected, candidates, ground_truth)
        summaries[name] = summary
        for key, value in arrays.items():
            output_arrays[f"{name}__{key}"] = value
    selected_metrics = directory / "selected_metrics.npz"
    write_npz(selected_metrics, **output_arrays)
    result = {
        "schema": "official100_v70_dwm_scorerv2_matched_result_v1",
        "status": "completed_frozen_same_state_h4_official100_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "states": 100,
        "horizon": 4,
        "objective": "0.4*ATE+0.6*endpoint",
        "methods": summaries,
        "input_freeze": str(freeze_path.resolve()),
        "input_freeze_sha256": sha256(freeze_path),
        "base_features": str(feature_path.resolve()),
        "base_features_sha256": sha256(feature_path),
        "v70_selected_actions": str(selected_path.resolve()),
        "v70_selected_actions_sha256": sha256(selected_path),
        "selected_metrics": str(selected_metrics.resolve()),
        "selected_metrics_sha256": sha256(selected_metrics),
        "thresholds_or_models_tuned_on_official100": False,
        "labels_opened_only_after_v70_selection_freeze": True,
        "reserve_accessed": False,
        "independent_generalization_claim": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare(args.domain, args.output_root)
    else:
        evaluate(args.domain, args.output_root)


if __name__ == "__main__":
    main()
