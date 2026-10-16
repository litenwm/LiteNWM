#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

RUN="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
PY=/root/DWM/.venv/bin/python
STAGE1="${RUN}/stage1_tridomain_v2_training"
HARVEST="${RUN}/cem_harvest"
MANIFEST="${RUN}/final_cem_tridomain_v2_data_manifest.json"
OUTPUT="${RUN}/final_cem_tridomain_v2_training"
LOG="${RUN}/final_cem_tridomain_v2_training.log"

test -f "${RUN}/CEM_HARVEST_COMPLETE.json"
test -f "${STAGE1}/result.json"
test ! -e "${OUTPUT}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

"${PY}" - "${RUN}" "${MANIFEST}" <<'PY'
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

run = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
spacing = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}
support = run / "stage1_tridomain_v2_training/support/tridomain_spacing_envelopes.npz"

stage1_result = json.loads(
    (run / 'stage1_tridomain_v2_training/result.json').read_text(encoding='utf-8')
)
world_models = stage1_result['world_models']
for seed, record in world_models.items():
    if file_sha256(Path(record['path'])) != record['sha256']:
        raise RuntimeError(f'LTWM hash changed for seed {seed}')

def fixed(dataset: str, partition: str) -> dict:
    root = run / "fixed" / dataset / partition
    return {
        "dataset": dataset,
        "features": str(root / "base_features.npz"),
        'features_sha256': file_sha256(root / 'base_features.npz'),
        "targets": str(root / "targets.npz"),
        'targets_sha256': file_sha256(root / 'targets.npz'),
        "support": str(support),
        'support_sha256': file_sha256(support),
        "spacing_m": spacing[dataset],
        "set_type": "fixed_nomad_k16",
    }

def cem(dataset: str, partition: str, iteration: int, train: bool) -> dict:
    root = run / "cem_harvest" / dataset / partition / f"iteration_{iteration:02d}"
    value = {
        "dataset": dataset,
        "features": str(root / "base_features.npz"),
        'features_sha256': file_sha256(root / 'base_features.npz'),
        "targets": str(root / "targets.npz"),
        'targets_sha256': file_sha256(root / 'targets.npz'),
        "support": str(support),
        'support_sha256': file_sha256(support),
        "spacing_m": spacing[dataset],
        "set_type": f"cem_iteration_{iteration:02d}",
    }
    if train:
        value["hard_negative"] = str(root / "hard_negatives.npz")
        value['hard_negative_sha256'] = file_sha256(root / 'hard_negatives.npz')
    return value

train = []
calibration = []
for dataset in spacing:
    train.append(fixed(dataset, "train-fit"))
    calibration.append(fixed(dataset, "scorer-cal"))
    for iteration in range(3):
        train.append(cem(dataset, "train-fit", iteration, True))
        calibration.append(cem(dataset, "scorer-cal", iteration, False))

value = {
    "schema": "proposal_aware_v70_tridomain_data_manifest_v1",
    'model_domain_index': {'recon': 0, 'scand': 1, 'sacson': 2},
    'world_models': world_models,
    "stage": "fixed_plus_three_round_cem_hard_negative",
    "train": train,
    "calibration": calibration,
    "hard_negative_definition": "stage1 predicted safe AND true proposal-aware harm",
    "external_test_accessed": False,
    "outer_val_accessed": False,
    "test_cohorts": [],
}
output.write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

cd "${REPO}"
nvidia-smi \
  --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader

"${PY}" tools/tridomain_v70/train_proposal_aware_cem_risk_tridomain.py \
  --data-manifest "${MANIFEST}" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed0.pt" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed1.pt" \
  --initial-checkpoint "${STAGE1}/checkpoints/proposal_aware_full_seed2.pt" \
  --output-dir "${OUTPUT}" \
  --epochs 40 \
  --batch-states 4096 \
  --learning-rate 1.0e-4 \
  --weight-decay 1.0e-4 \
  --gradient-clip-norm 5.0 \
  --patience 5 \
  --hard-negative-multiplier 6.0 \
  --sacson-risk-hard-negative-multiplier 12.0 \
  --support-quantile 0.999 \
  --result-status frozen_after_v69_spent_failure_before_v70_spent_replay \
  --device cuda \
  2>&1 | tee "${LOG}"

test -f "${OUTPUT}/result.json"
sha256sum \
  "${MANIFEST}" \
  "${OUTPUT}/result.json" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed0.pt" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed1.pt" \
  "${OUTPUT}/checkpoints/proposal_aware_full_seed2.pt" \
  > "${RUN}/final_cem_training.sha256"
date -u +%FT%TZ > "${RUN}/FINAL_CEM_TRAINING_COMPLETE"
