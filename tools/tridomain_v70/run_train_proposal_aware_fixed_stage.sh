#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO_ROOT="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
OLD_REPO_ROOT="${3:-/root/autodl-tmp/dinov3-nomad-ltwm-v70}"
PYTHON="${PYTHON:-/root/DWM/.venv/bin/python}"

FIXED_ROOT="${RUN_ROOT}/fixed"
MANIFEST="${RUN_ROOT}/stage1_tridomain_v2_data_manifest.json"
OUTPUT="${RUN_ROOT}/stage1_tridomain_v2_training"
LOG="${RUN_ROOT}/stage1_tridomain_v2_training.log"

for dataset in recon scand sacson; do
  test -f "${FIXED_ROOT}/${dataset}/FIXED_POOL_COMPLETE.json"
done
test ! -e "${OUTPUT}"

"${PYTHON}" - "${FIXED_ROOT}" "${MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

fixed = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
spacing = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}

completed = {
    dataset: json.loads(
        (fixed / dataset / 'FIXED_POOL_COMPLETE.json').read_text(encoding='utf-8')
    )
    for dataset in spacing
}
world_models = completed['recon']['world_models']
for dataset, value in completed.items():
    if value['world_models'] != world_models:
        raise RuntimeError(f'LTWM bindings differ for {dataset}')
for seed, record in world_models.items():
    if file_sha256(Path(record['path'])) != record['sha256']:
        raise RuntimeError(f'LTWM hash changed for seed {seed}')

def entry(dataset: str, partition: str) -> dict:
    root = fixed / dataset
    return {
        "dataset": dataset,
        "features": str(root / partition / "base_features.npz"),
        'features_sha256': file_sha256(root / partition / 'base_features.npz'),
        "targets": str(root / partition / "targets.npz"),
        'targets_sha256': file_sha256(root / partition / 'targets.npz'),
        "support": str(root / "support_envelope.npz"),
        'support_sha256': file_sha256(root / 'support_envelope.npz'),
        "spacing_m": spacing[dataset],
        "set_type": "fixed_nomad_k16",
    }

value = {
    "schema": "proposal_aware_v70_tridomain_data_manifest_v1",
    'model_domain_index': {'recon': 0, 'scand': 1, 'sacson': 2},
    'world_models': world_models,
    "stage": "fixed_pool_initialization",
    "train": [entry(name, "train-fit") for name in spacing],
    "calibration": [entry(name, "scorer-cal") for name in spacing],
    "external_test_accessed": False,
    "outer_val_accessed": False,
    "test_cohorts": [],
}
output.write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

cd "${REPO_ROOT}"
nvidia-smi \
  --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader

"${PYTHON}" tools/tridomain_v70/train_proposal_aware_cem_risk_tridomain.py \
  --data-manifest "${MANIFEST}" \
  --initial-checkpoint "${OLD_REPO_ROOT}/outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/training/checkpoints/arm_D_full_seed0.pt" \
  --initial-checkpoint "${OLD_REPO_ROOT}/outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/training/checkpoints/arm_D_full_seed1.pt" \
  --initial-checkpoint "${OLD_REPO_ROOT}/outputs/cem_development_v1/dualdomain_cem_tail_v69_scand_route_crossfit/training/checkpoints/arm_D_full_seed2.pt" \
  --output-dir "${OUTPUT}" \
  --epochs 20 \
  --batch-states 4096 \
  --learning-rate 1.0e-4 \
  --weight-decay 1.0e-4 \
  --gradient-clip-norm 5.0 \
  --patience 5 \
  --hard-negative-multiplier 6.0 \
  --support-quantile 0.999 \
  --result-status trained_and_frozen_before_A_to_H_replay \
  --device cuda \
  2>&1 | tee "${LOG}"

test -f "${OUTPUT}/result.json"
sha256sum \
  "${MANIFEST}" \
  "${OUTPUT}/result.json" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed0.pt" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed1.pt" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed2.pt" \
  > "${RUN_ROOT}/stage1_tridomain_v2_training.sha256"
date -u +%FT%TZ > "${RUN_ROOT}/STAGE1_TRIDOMAIN_V2_COMPLETE"
