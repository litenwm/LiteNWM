#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/rainbow/1/dinov3-ltwm-scorer-v2-mainline
nwm_root=/home/rainbow/rrnwm
python_bin=/home/rainbow/miniconda3/envs/nwm/bin/python
config_path="${repo_root}/configs/scand/nwm_cdit_s_scand_local320_h4_eval.yaml"
experiment_root="${repo_root}/outputs/local_scand_common100_representative_nwm_s_full100_alignment_v1"
output_dir="${experiment_root}/after_full100_aligned_h4_n100_d150_rep1_seed0_attempt2_chunk10"
index_path="${experiment_root}/navigation_eval_h4_single.pkl"

if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite ${output_dir}" >&2
  exit 2
fi

cd "${nwm_root}"
PYTHONPATH="${nwm_root}:${PYTHONPATH:-}" "${python_bin}" \
  "${repo_root}/tools/planning_eval_local_chunked.py" \
  --exp "${config_path}" \
  --ckp 0100000 \
  --datasets scand \
  --navigation_eval_index "${index_path}" \
  --trajectory_eval_horizon 4 \
  --eval_data_folder /home/rainbow/nwm/data/scand \
  --num_workers 0 \
  --batch_size 1 \
  --rollout_stride 1 \
  --topk 5 \
  --opt_steps 1 \
  --score_mode lpips \
  --planner_variant standard \
  --candidate_batch_size 10 \
  --output_dir "${output_dir}" \
  --diffusion_steps 150 \
  --seed 0 \
  --num_samples 100 \
  --num_repeat_eval 1 \
  2>&1 | tee "${experiment_root}/after_attempt2.log"

touch "${experiment_root}/COMPLETE"
