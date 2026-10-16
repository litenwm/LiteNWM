#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

RUN="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
OLD_REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70
NWM_SITE=/root/autodl-tmp/dwm_datasets/envs/nwm-runtime/site-packages
export PYTHONPATH="${NWM_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${REPO}/runtime/python"
FINAL="${TRIDOMAIN_FINAL_RESULT:-${RUN}/final_cem_tridomain_v2_training/result.json}"
CHECKPOINT_ROOT="${RUN}/feature_checkpoint_root"
OUT="${TRIDOMAIN_FAIR_OUT:-${RUN}/fair_eval_final}"
SEAL_MARKER="${TRIDOMAIN_SEAL_MARKER:-${RUN}/FINAL_CEM_TRAINING_COMPLETE}"
PROTOCOL_PREREG="${TRIDOMAIN_PROTOCOL_PREREG:-${FINAL%/*}/fair_eval_protocol_preregistration_v1.json}"
MODEL_DIR=/root/autodl-tmp/dwm_datasets/deps/dinov3-vitl16-pretrain-lvd1689m
ADAPTER="${REPO}/checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt"
RESUME_FROZEN_SELECTIONS="${TRIDOMAIN_RESUME_FROZEN_SELECTIONS:-NO}"

test -f "${SEAL_MARKER}"
test -f "${PROTOCOL_PREREG}"
test -f "${FINAL}"
test -d "${CHECKPOINT_ROOT}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -e "${OUT}" ]]; then
  test "${RESUME_FROZEN_SELECTIONS}" = YES
else
  test "${RESUME_FROZEN_SELECTIONS}" = NO
fi
mkdir -p "${OUT}/logs"
cd "${REPO}"


run_cohort () {
  local dataset=$1
  local cohort=$2
  local manifest=$3
  local trajectory=$4
  local pool_root=$5
  local runtime_domain=$dataset
  local replay_domain=$dataset
  local replay_manifest=$manifest
  local base_result=$FINAL
  if [[ "$dataset" == sacson ]]; then
    replay_domain=scand
    replay_manifest="${OUT}/${cohort}/sacson_manifest_alias.json"
  fi
  mkdir -p "${OUT}/${cohort}"

  local selection_audit="${OUT}/${cohort}/selection/selection_audit.json"
  if [[ -f "${selection_audit}" ]]; then
    test "${RESUME_FROZEN_SELECTIONS}" = YES
    "${PY}" - \
      "${selection_audit}" \
      "${OUT}/${cohort}/selection/selected_actions.npz" \
      "${manifest}" \
      "${dataset}" \
      "${cohort}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

audit_path = Path(sys.argv[1]).resolve(strict=True)
selection = Path(sys.argv[2]).resolve(strict=True)
manifest = Path(sys.argv[3]).resolve(strict=True)
dataset, cohort = sys.argv[4:6]
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("schema") != "proposal_aware_v70_tridomain_fair_selection_v1":
    raise RuntimeError("unsupported frozen-selection audit")
if audit.get("status") != "actions_frozen_before_future_poses":
    raise RuntimeError("selection is not frozen before future poses")
if audit.get("future_pose_opened") or audit.get("test_labels_used_by_selection"):
    raise RuntimeError("frozen selection reports test-label access")
if audit.get("dataset") != dataset or audit.get("cohort_id") != cohort:
    raise RuntimeError("frozen selection cohort identity changed")
if audit.get("manifest_sha256") != sha256(manifest):
    raise RuntimeError("frozen selection manifest SHA changed")
    if audit.get("selection_sha256") != sha256(selection):
        raise RuntimeError("frozen selected-actions SHA changed")
PY
    local completed_metrics="${OUT}/${cohort}/metrics/results.json"
    if [[ -f "${completed_metrics}" ]]; then
      "${PY}" - \
        "${completed_metrics}" \
        "${OUT}/${cohort}/selection/selected_actions.npz" \
        "${manifest}" \
        "${dataset}" \
        "${cohort}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selection = Path(sys.argv[2]).resolve(strict=True)
manifest = Path(sys.argv[3]).resolve(strict=True)
dataset, cohort = sys.argv[4:6]
if result.get("schema") != "new_v70_triseed_fair_h4_metrics_v1":
    raise RuntimeError("unsupported completed-metrics schema")
if result.get("status") != "complete_after_frozen_selection":
    raise RuntimeError("metrics are not complete after frozen selection")
if result.get("dataset") != dataset or result.get("cohort_id") != cohort:
    raise RuntimeError("completed metrics cohort identity changed")
if result.get("manifest_sha256") != sha256(manifest):
    raise RuntimeError("completed metrics manifest SHA changed")
if result.get("selection_sha256") != sha256(selection):
    raise RuntimeError("completed metrics selection SHA changed")
PY
      return 0
    fi
    test ! -e "${OUT}/${cohort}/metrics"
  else
  for arm in k16 k32_batch0 k32_batch1; do
    local candidate
    case "$arm" in
      k16) candidate="${pool_root}/v70_k16_candidates.npz" ;;
      k32_batch0) candidate="${pool_root}/v70_k32_batch0_candidates.npz" ;;
      k32_batch1) candidate="${pool_root}/v70_k32_batch1_candidates.npz" ;;
    esac
    local arm_root="${OUT}/${cohort}/${arm}"
    local source="${arm_root}/source/dense_source.npz"
    local alias_args=()
    local replay_candidate=$candidate
    if [[ "$dataset" == sacson ]]; then
      replay_candidate="${arm_root}/source/candidate_alias_scand.npz"
      alias_args=(
        --alias-manifest-output "${replay_manifest}"
        --alias-candidate-output "${replay_candidate}"
      )
    fi
    "${PY}" tools/tridomain_v70/build_fair_dense_source_tridomain.py \
      --dataset "${dataset}" \
      --manifest "${manifest}" \
      --candidate "${candidate}" \
      --trajectory-root "${trajectory}" \
      --output "${source}" \
      "${alias_args[@]}" \
      > "${OUT}/logs/${cohort}_${arm}_source.log" 2>&1

    local split="${cohort}_${arm}_new_v70"
    local features="${arm_root}/features/${split}_features.npz"
    local latents="${arm_root}/features/${split}_latents.npz"
    "${PY}" tools/evaluation/export_recon_dense_dinov3_features_v1.py \
      --split "${split}" \
      --source "${source}" \
      --output-dir "${arm_root}/features" \
      --trajectory-root "${trajectory}" \
      --model-dir "${MODEL_DIR}" \
      --adapter-checkpoint "${ADAPTER}" \
      --checkpoint-root "${CHECKPOINT_ROOT}" \
      --device cuda \
      --dinov3-batch-size 8 \
      --wm-batch-size 16 \
      > "${OUT}/logs/${cohort}_${arm}_features.log" 2>&1

    TRIDOMAIN_V70_RESULT="${FINAL}" \
    TRIDOMAIN_V70_PROTOCOL_PREREG="${PROTOCOL_PREREG}" \
    TRIDOMAIN_V70_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    TRIDOMAIN_RUNTIME_DATASET="${runtime_domain}" \
    "${PY}" tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py \
      --domain "${replay_domain}" \
      --model-role proposal_aware_v66 \
      --cohort "${cohort}_${arm}_new_v70" \
      --manifest "${replay_manifest}" \
      --candidate "${replay_candidate}" \
      --base-features "${features}" \
      --latent-cache "${latents}" \
      --trajectory-root "${trajectory}" \
      --output-dir "${arm_root}/v70" \
      --base-result "${base_result}" \
      --v66-result "${FINAL}" \
      --device cuda \
      --wm-batch-size 16 \
      > "${OUT}/logs/${cohort}_${arm}_v70.log" 2>&1
  done

  "${PY}" \
    "${OLD_REPO}/experiment_prep/fair_v70_benchmark_20260723/merge_v70_k32_batches.py" \
    --batch0 "${OUT}/${cohort}/k32_batch0/v70/selected_actions.npz" \
    --batch1 "${OUT}/${cohort}/k32_batch1/v70/selected_actions.npz" \
    --output "${OUT}/${cohort}/k32_merged.npz" \
    > "${OUT}/logs/${cohort}_k32_merge.log" 2>&1

  "${PY}" tools/tridomain_v70/freeze_fair_v70_k16_k32_selection.py \
    --dataset "${dataset}" \
    --cohort-id "${cohort}" \
    --manifest "${manifest}" \
    --k16-selected "${OUT}/${cohort}/k16/v70/selected_actions.npz" \
    --k32-merged "${OUT}/${cohort}/k32_merged.npz" \
    --output-dir "${OUT}/${cohort}/selection" \
    > "${OUT}/logs/${cohort}_freeze_selection.log" 2>&1
  fi

  "${PY}" tools/tridomain_v70/evaluate_new_v70_fair_pipeline.py metrics \
    --dataset "${dataset}" \
    --cohort-id "${cohort}" \
    --manifest "${manifest}" \
    --trajectory-root "${trajectory}" \
    --selection-dir "${OUT}/${cohort}/selection" \
    --output-dir "${OUT}/${cohort}/metrics" \
    --bootstrap 10000 \
    --bootstrap-seed 20260728 \
    > "${OUT}/logs/${cohort}_metrics.log" 2>&1
}

run_cohort \
  recon \
  recon100 \
  "${REPO}/data_splits/recon/server_full_nwm_unseen_h4_100/manifest.json" \
  /root/autodl-tmp/dwm_datasets/processed/recon_h4_100_nomad_seed17 \
  /root/autodl-tmp/dwm_runs/fair_v70_20260723/recon100_primary/seed17/01_nomad_pool

run_cohort \
  scand \
  scand_common100 \
  "${REPO}/data_splits/scand/fair_common_20260723/scand_common_exact_holdout_h4_100_seed20260723/manifest.json" \
  /root/autodl-tmp/dwm_datasets/raw/scand_320 \
  /root/autodl-tmp/dwm_runs/fair_v70_20260723/scand_common100/seed17/01_nomad_pool

run_cohort \
  sacson \
  sacson100 \
  /root/autodl-tmp/dwm_datasets/evaluation/sacson/strict_test_h4_100_seed17/manifest.json \
  /root/autodl-tmp/dwm_datasets/processed/sacson_processed \
  /root/autodl-tmp/dwm_runs/v70_tridomain_fair_eval_triseed_20260728/sacson100/01_nomad_pool

"${PY}" - "${OUT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
compact = {}
for cohort in ("recon100", "scand_common100", "sacson100"):
    path = root / cohort / "metrics/results.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    compact[cohort] = {}
    for method, summary in value["summaries"].items():
        compact[cohort][method] = {
            "ATE": summary["ate_nwm"]["mean"],
            "RTE": summary["rte_nwm"]["mean"],
            "FDE": summary["fde_nwm"]["mean"],
            "SR@1": summary["success_fde_1step"]["rate"],
            "SR@2": summary["success_fde_2step"]["rate"],
        }
(root / "compact_results.json").write_text(
    json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
PY

date -u +%FT%TZ > "${OUT}/COMPLETE"
