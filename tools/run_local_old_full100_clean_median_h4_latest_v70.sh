#!/usr/bin/env bash
set -euo pipefail

repo=/home/rainbow/1/dinov3-ltwm-scorer-v2-mainline
python_bin=/home/rainbow/miniconda3/envs/nwm/bin/python
experiment="${repo}/outputs/local_old_full100_clean_median_h4_nwm_s_v70_v1"
manifest="${experiment}/manifest_h4.json"
manifest_jsonl="${experiment}/manifest_h4.jsonl"
trajectory_root=/home/rainbow/nwm/data/scand
runtime="${experiment}/v70_runtime"
pool="${experiment}/nomad_pool"
out="${experiment}/latest_v70"
visualnav=/home/rainbow/visualnav-transformer
nomad_artifact="${repo}/artifacts/v70_go2_orin_20260723/dinov3_ltwm_v70_go2_orin_20260723/external/nomad"
diffusion_policy=/home/rainbow/diffusion_policy
dinov3="${repo}/artifacts/v70_go2_orin_20260723/dinov3_ltwm_v70_go2_orin_20260723/external/dinov3-vitl16-pretrain-lvd1689m"
adapter="${repo}/server_exports/v70_tridomain_latest_20260729/weights/adapter/adapter_pilot_v1_final.pt"
pool_builder="${repo}/artifacts/sacson100_h4_a1_a6_portable/_server_reference/fair_v70_benchmark_20260723/build_nomad_k32_batched_route_v1.py"
merge_script="${repo}/artifacts/sacson100_h4_a1_a6_portable/_server_reference/fair_v70_benchmark_20260723/merge_v70_k32_batches.py"

if [[ -e "${out}" ]]; then
  echo "Refusing to overwrite existing V70 output" >&2
  exit 2
fi

cd "${repo}"
if [[ ! -e "${runtime}" ]]; then
  "${python_bin}" tools/prepare_local_tridomain_v70_runtime.py \
    2>&1 | tee "${experiment}/v70_prepare_runtime.log"
fi

if [[ ! -e "${pool}/AUDIT.json" ]]; then
FAIR_V70_REPO_ROOT="${repo}" \
PYTHONPATH="${repo}:${repo}/src:${PYTHONPATH:-}:${experiment}/nomad_purepy_deps" \
"${python_bin}" "${pool_builder}" \
  --dataset scand \
  --manifest "${manifest}" \
  --trajectory-root "${trajectory_root}" \
  --visualnav-root "${visualnav}" \
  --diffusion-policy-root "${diffusion_policy}" \
  --nomad-config "${nomad_artifact}/nomad.yaml" \
  --nomad-checkpoint "${nomad_artifact}/nomad.pth" \
  --output-dir "${pool}" \
  --seed 17 \
  --checkpoint-every 1 \
  --device cuda \
  2>&1 | tee "${experiment}/v70_nomad_pool.log"
fi

mkdir -p "${out}/logs"
for arm in k16 k32_batch0 k32_batch1; do
  case "${arm}" in
    k16) candidate="${pool}/v70_k16_candidates.npz" ;;
    k32_batch0) candidate="${pool}/v70_k32_batch0_candidates.npz" ;;
    k32_batch1) candidate="${pool}/v70_k32_batch1_candidates.npz" ;;
  esac
  arm_root="${out}/${arm}"
  split="old_full100_clean_median_${arm}_new_v70"

  "${python_bin}" tools/tridomain_v70/build_fair_dense_source_tridomain.py \
    --dataset scand \
    --manifest "${manifest}" \
    --candidate "${candidate}" \
    --trajectory-root "${trajectory_root}" \
    --output "${arm_root}/source/dense_source.npz" \
    > "${out}/logs/${arm}_source.log" 2>&1

  "${python_bin}" tools/evaluation/export_recon_dense_dinov3_features_v1.py \
    --split "${split}" \
    --source "${arm_root}/source/dense_source.npz" \
    --output-dir "${arm_root}/features" \
    --trajectory-root "${trajectory_root}" \
    --model-dir "${dinov3}" \
    --adapter-checkpoint "${adapter}" \
    --checkpoint-root "${runtime}/feature_checkpoint_root" \
    --device cuda \
    --dinov3-batch-size 1 \
    --wm-batch-size 4 \
    > "${out}/logs/${arm}_features.log" 2>&1

  TRIDOMAIN_V70_RESULT="${runtime}/result.json" \
  TRIDOMAIN_V70_PROTOCOL_PREREG="${runtime}/fair_eval_protocol_preregistration_v1.json" \
  TRIDOMAIN_V70_CHECKPOINT_ROOT="${runtime}/feature_checkpoint_root" \
  TRIDOMAIN_RUNTIME_DATASET=scand \
  "${python_bin}" tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py \
    --domain scand \
    --model-role proposal_aware_v66 \
    --cohort "${split}" \
    --manifest "${manifest_jsonl}" \
    --candidate "${candidate}" \
    --base-features "${arm_root}/features/${split}_features.npz" \
    --latent-cache "${arm_root}/features/${split}_latents.npz" \
    --trajectory-root "${trajectory_root}" \
    --output-dir "${arm_root}/v70" \
    --base-result "${runtime}/result.json" \
    --v66-result "${runtime}/result.json" \
    --device cuda \
    --wm-batch-size 4 \
    > "${out}/logs/${arm}_v70.log" 2>&1
done

"${python_bin}" "${merge_script}" \
  --batch0 "${out}/k32_batch0/v70/selected_actions.npz" \
  --batch1 "${out}/k32_batch1/v70/selected_actions.npz" \
  --output "${out}/k32_merged.npz" \
  > "${out}/logs/k32_merge.log" 2>&1

"${python_bin}" tools/tridomain_v70/freeze_fair_v70_k16_k32_selection.py \
  --dataset scand \
  --cohort-id old_full100_clean_median_h4 \
  --manifest "${manifest}" \
  --k16-selected "${out}/k16/v70/selected_actions.npz" \
  --k32-merged "${out}/k32_merged.npz" \
  --output-dir "${out}/selection" \
  > "${out}/logs/freeze_selection.log" 2>&1

"${python_bin}" tools/tridomain_v70/evaluate_new_v70_fair_pipeline.py metrics \
  --dataset scand \
  --cohort-id old_full100_clean_median_h4 \
  --manifest "${manifest}" \
  --trajectory-root "${trajectory_root}" \
  --selection-dir "${out}/selection" \
  --output-dir "${out}/metrics" \
  --bootstrap 10000 \
  --bootstrap-seed 20260728 \
  > "${out}/logs/metrics.log" 2>&1

touch "${out}/COMPLETE"
