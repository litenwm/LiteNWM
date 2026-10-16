#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

RUN="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
PY=/root/DWM/.venv/bin/python
STRICT="${RUN}/final_cem_tridomain_v2_training"
STAGE1="${RUN}/stage1_tridomain_v2_training"
MANIFEST="${RUN}/final_cem_tridomain_v2_data_manifest.json"
RELAXED_ROOT="${RUN}/relaxed_fallback_attempts"
RELAXED="${RELAXED_ROOT}/final_cem_tridomain_v2_training_epoch39_relaxed_v1"
LOG="${RUN}/relaxed_fallback_v1_finalize.log"
FAIR_OUT="${RUN}/fair_eval_final_relaxed_v1"

test -f "${STRICT}/CALIBRATION_INFEASIBLE.json"
test -f "${STRICT}/restart_state.pt"
test -f "${STAGE1}/result.json"
test -f "${MANIFEST}"
test ! -e "${RELAXED}"
test ! -e "${RELAXED}_stream_cache"
test ! -e "${FAIR_OUT}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "${RELAXED_ROOT}"
cd "${REPO}"

"${PY}" tools/tridomain_v70/train_proposal_aware_cem_risk_tridomain.py \
  --data-manifest "${MANIFEST}" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed0.pt" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed1.pt" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed2.pt" \
  --output-dir "${RELAXED}" \
  --epochs 40 \
  --batch-states 4096 \
  --learning-rate 1.0e-4 \
  --weight-decay 1.0e-4 \
  --gradient-clip-norm 5.0 \
  --patience 5 \
  --hard-negative-multiplier 6.0 \
  --sacson-risk-hard-negative-multiplier 12.0 \
  --resume-state "${STRICT}/restart_state.pt" \
  --relaxed-finalize-v1 \
  --support-quantile 0.999 \
  --result-status frozen_after_v69_spent_failure_before_v70_spent_replay \
  --device cuda \
  > "${LOG}" 2>&1

"${PY}" - "${STRICT}" "${RELAXED}" "${MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

strict = Path(sys.argv[1]).resolve()
relaxed = Path(sys.argv[2]).resolve()
manifest = Path(sys.argv[3]).resolve()
result_path = relaxed / "result.json"
result = json.loads(result_path.read_text(encoding="utf-8"))
if result["seal_status"] != "RELAXED_FEASIBLE_V1":
    raise RuntimeError("relaxed seal status mismatch")
if result["strict_all_domains_feasible"]:
    raise RuntimeError("relaxed result incorrectly claims strict feasibility")
if not result["relaxed_all_domains_feasible_v1"]:
    raise RuntimeError("fixed relaxed contract is not feasible")
if result["model_domain_index"] != {"recon": 0, "scand": 1, "sacson": 2}:
    raise RuntimeError("three-domain mapping changed")
if result["test_accessed"] or result["outer_val_accessed"]:
    raise RuntimeError("external labels were accessed before sealing")
if result["best_epoch"] != 39:
    raise RuntimeError("unexpected preregistered best epoch")
for record in result["world_models"].values():
    if sha256(Path(record["path"])) != record["sha256"]:
        raise RuntimeError("LTWM checkpoint hash changed")
for record in result["feature_cache_bindings"]:
    if sha256(Path(record["path"])) != record["sha256"]:
        raise RuntimeError("feature cache hash changed")
for record in result["checkpoints"]["full"]:
    checkpoint = Path(record["path"])
    if sha256(checkpoint) != record["sha256"]:
        raise RuntimeError("selector checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "proposal_aware_v70_tridomain_member_v1":
        raise RuntimeError("selector checkpoint cannot be validated")
strict_failure = strict / "CALIBRATION_INFEASIBLE.json"
restart_state = strict / "restart_state.pt"
seal = {
    "status": "RELAXED_FEASIBLE_V1",
    "strict_status": "CALIBRATION_INFEASIBLE_AFTER_40_EPOCHS",
    "strict_contract": result["calibration_contract"],
    "relaxed_contract": result["relaxed_calibration_contract_v1"],
    "calibration": result["calibration"],
    "best_epoch": result["best_epoch"],
    "model_domain_index": result["model_domain_index"],
    "train_states": result["train_states"],
    "train_states_by_dataset": result["train_states_by_dataset"],
    "world_models": result["world_models"],
    "feature_cache_bindings": result["feature_cache_bindings"],
    "checkpoints": result["checkpoints"],
    "result": str(result_path),
    "result_sha256": sha256(result_path),
    "manifest": str(manifest),
    "manifest_sha256": sha256(manifest),
    "strict_failure": str(strict_failure),
    "strict_failure_sha256": sha256(strict_failure),
    "source_restart_state": str(restart_state),
    "source_restart_state_sha256": sha256(restart_state),
    "test_accessed": False,
}
(relaxed / "seal_record.json").write_text(
    json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(relaxed / "RELAXED_FEASIBLE_V1").write_text(
    "fixed preregistered fallback; external tests not accessed\n",
    encoding="utf-8",
)
PY

sha256sum \
  "${MANIFEST}" \
  "${STRICT}/CALIBRATION_INFEASIBLE.json" \
  "${STRICT}/restart_state.pt" \
  "${RELAXED}/result.json" \
  "${RELAXED}/seal_record.json" \
  "${RELAXED}/checkpoints/proposal_aware_full_seed0.pt" \
  "${RELAXED}/checkpoints/proposal_aware_full_seed1.pt" \
  "${RELAXED}/checkpoints/proposal_aware_full_seed2.pt" \
  > "${RELAXED}/SHA256SUMS"

TRIDOMAIN_FINAL_RESULT="${RELAXED}/result.json" \
TRIDOMAIN_FAIR_OUT="${FAIR_OUT}" \
TRIDOMAIN_SEAL_MARKER="${RELAXED}/RELAXED_FEASIBLE_V1" \
  bash tools/tridomain_v70/run_fair_eval_final_v70_tridomain.sh \
    "${RUN}" "${REPO}"
