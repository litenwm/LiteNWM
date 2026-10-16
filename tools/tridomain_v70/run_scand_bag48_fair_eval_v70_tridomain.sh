#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

RUN="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
OLD_REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70
OLD_RUN=/root/autodl-tmp/dwm_runs/fair_v70_20260723/scand_bag48/seed17
NWM_SITE=/root/autodl-tmp/dwm_datasets/envs/nwm-runtime/site-packages
export PYTHONPATH="${NWM_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${REPO}/runtime/python"
FINAL="${RUN}/relaxed_fallback_attempts/final_cem_tridomain_v2_training_epoch39_relaxed_v1/result.json"
SEAL_MARKER="${RUN}/relaxed_fallback_attempts/final_cem_tridomain_v2_training_epoch39_relaxed_v1/RELAXED_FEASIBLE_V1"
PROTOCOL_PREREG="${RUN}/relaxed_fallback_attempts/final_cem_tridomain_v2_training_epoch39_relaxed_v1/fair_eval_protocol_preregistration_v1.json"
CHECKPOINT_ROOT="${RUN}/feature_checkpoint_root"
OUT="${RUN}/fair_eval_scand_bag48_relaxed_v1"
MANIFEST="${REPO}/data_splits/scand/fair_common_20260723/scand_common_bag_holdout_h4_48_seed20260723/manifest.json"
TRAJECTORY=/root/autodl-tmp/dwm_datasets/raw/scand_320
POOL_ROOT="${OLD_RUN}/01_nomad_pool"
MODEL_DIR=/root/autodl-tmp/dwm_datasets/deps/dinov3-vitl16-pretrain-lvd1689m
ADAPTER="${REPO}/checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt"
EXPECTED_MANIFEST_SHA=f5b2204b4b584391a92d264ec3a0b9435d7b20aee10bc39ebb9bbe03bcfdd056

test -f "${FINAL}"
test -f "${SEAL_MARKER}"
test -f "${PROTOCOL_PREREG}"
test -d "${CHECKPOINT_ROOT}"
test -f "${MANIFEST}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_MANIFEST_SHA}"
test -f "${POOL_ROOT}/v70_k16_candidates.npz"
test -f "${POOL_ROOT}/v70_k32_batch0_candidates.npz"
test -f "${POOL_ROOT}/v70_k32_batch1_candidates.npz"
test -f "${OLD_RUN}/metrics_sealed/results.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test ! -e "${OUT}"
mkdir -p "${OUT}/logs"
cd "${REPO}"

run_arm () {
  local arm=$1
  local candidate=$2
  local arm_root="${OUT}/scand_bag48/${arm}"
  local split="scand_bag48_${arm}_new_v70"
  local source="${arm_root}/source/dense_source.npz"
  local features="${arm_root}/features/${split}_features.npz"
  local latents="${arm_root}/features/${split}_latents.npz"

  "${PY}" tools/tridomain_v70/build_fair_dense_source_tridomain.py \
    --dataset scand \
    --manifest "${MANIFEST}" \
    --candidate "${candidate}" \
    --trajectory-root "${TRAJECTORY}" \
    --output "${source}" \
    > "${OUT}/logs/scand_bag48_${arm}_source.log" 2>&1

  "${PY}" tools/evaluation/export_recon_dense_dinov3_features_v1.py \
    --split "${split}" \
    --source "${source}" \
    --output-dir "${arm_root}/features" \
    --trajectory-root "${TRAJECTORY}" \
    --model-dir "${MODEL_DIR}" \
    --adapter-checkpoint "${ADAPTER}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --device cuda \
    --dinov3-batch-size 8 \
    --wm-batch-size 16 \
    > "${OUT}/logs/scand_bag48_${arm}_features.log" 2>&1

  TRIDOMAIN_V70_RESULT="${FINAL}" \
  TRIDOMAIN_V70_PROTOCOL_PREREG="${PROTOCOL_PREREG}" \
  TRIDOMAIN_V70_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  TRIDOMAIN_RUNTIME_DATASET=scand \
  "${PY}" tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py \
    --domain scand \
    --model-role proposal_aware_v66 \
    --cohort "${split}" \
    --manifest "${MANIFEST}" \
    --candidate "${candidate}" \
    --base-features "${features}" \
    --latent-cache "${latents}" \
    --trajectory-root "${TRAJECTORY}" \
    --output-dir "${arm_root}/v70" \
    --base-result "${FINAL}" \
    --v66-result "${FINAL}" \
    --device cuda \
    --wm-batch-size 16 \
    > "${OUT}/logs/scand_bag48_${arm}_v70.log" 2>&1
}

run_arm k16 "${POOL_ROOT}/v70_k16_candidates.npz"
run_arm k32_batch0 "${POOL_ROOT}/v70_k32_batch0_candidates.npz"
run_arm k32_batch1 "${POOL_ROOT}/v70_k32_batch1_candidates.npz"

"${PY}" "${OLD_REPO}/experiment_prep/fair_v70_benchmark_20260723/merge_v70_k32_batches.py" \
  --batch0 "${OUT}/scand_bag48/k32_batch0/v70/selected_actions.npz" \
  --batch1 "${OUT}/scand_bag48/k32_batch1/v70/selected_actions.npz" \
  --output "${OUT}/scand_bag48/k32_merged.npz" \
  > "${OUT}/logs/scand_bag48_k32_merge.log" 2>&1

"${PY}" tools/tridomain_v70/freeze_fair_v70_k16_k32_selection.py \
  --dataset scand \
  --cohort-id scand_bag48 \
  --manifest "${MANIFEST}" \
  --k16-selected "${OUT}/scand_bag48/k16/v70/selected_actions.npz" \
  --k32-merged "${OUT}/scand_bag48/k32_merged.npz" \
  --output-dir "${OUT}/scand_bag48/selection" \
  > "${OUT}/logs/scand_bag48_freeze_selection.log" 2>&1

"${PY}" tools/tridomain_v70/evaluate_new_v70_fair_pipeline.py metrics \
  --dataset scand \
  --cohort-id scand_bag48 \
  --manifest "${MANIFEST}" \
  --trajectory-root "${TRAJECTORY}" \
  --selection-dir "${OUT}/scand_bag48/selection" \
  --output-dir "${OUT}/scand_bag48/metrics" \
  --bootstrap 10000 \
  --bootstrap-seed 20260729 \
  > "${OUT}/logs/scand_bag48_metrics.log" 2>&1

"${PY}" - "${OUT}" "${OLD_RUN}/metrics_sealed/results.json" <<'PY'
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
old_path = Path(sys.argv[2])
new_path = out / "scand_bag48/metrics/results.json"
old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))

if new["status"] != "complete_after_frozen_selection":
    raise RuntimeError("new metrics are not sealed after frozen selection")
if new["states"] != 48 or old["states"] != 48:
    raise RuntimeError("SCAND bag48 state count changed")
old_manifest = old["inputs"]["manifest"]["sha256"]
if new["manifest_sha256"] != old_manifest:
    raise RuntimeError("SCAND bag48 manifest differs from the old fair benchmark")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

comparison = {}
for k, old_name, new_name in (
    (16, "A7_v70_full_k16", "new_v70_triseed_k16"),
    (32, "A8_v70_full_k32", "new_v70_triseed_k32"),
):
    old_summary = old["summaries"][old_name]
    new_summary = new["summaries"][new_name]
    row = {}
    for metric in ("ate", "rte", "fde"):
        key = metric + "_nwm"
        old_value = float(old_summary[key]["mean"])
        new_value = float(new_summary[key]["mean"])
        row[metric.upper()] = {
            "old": old_value,
            "new": new_value,
            "delta_percent": (new_value / old_value - 1.0) * 100.0,
        }
    row["SR@1"] = float(new_summary["success_fde_1step"]["rate"])
    row["SR@2"] = float(new_summary["success_fde_2step"]["rate"])
    comparison[f"K{k}"] = row

payload = {
    "schema": "scand_bag48_new_tridomain_v70_comparison_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS",
    "cohort_id": "scand_bag48",
    "states": 48,
    "manifest_sha256": new["manifest_sha256"],
    "new_results_path": str(new_path),
    "new_results_sha256": digest(new_path),
    "old_results_path": str(old_path),
    "old_results_sha256": digest(old_path),
    "comparison": comparison,
    "selection_frozen_before_future_poses": True,
    "test_labels_used_by_selection": False,
}
(out / "comparison_to_old_scand_bag48.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
spent = {
    "schema": "spent_test_cohort_v1",
    "created_at_utc": payload["created_at_utc"],
    "cohort_id": "scand_bag48",
    "states": 48,
    "manifest_sha256": new["manifest_sha256"],
    "spent": True,
    "prohibited_for_future_training_or_tuning": True,
}
(out / "SPENT_TEST_COHORT.json").write_text(
    json.dumps(spent, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

date -u +%FT%TZ > "${OUT}/COMPLETE"
