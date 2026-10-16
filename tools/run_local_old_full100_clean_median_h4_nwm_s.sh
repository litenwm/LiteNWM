#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/rainbow/1/dinov3-ltwm-scorer-v2-mainline
nwm_root=/home/rainbow/rrnwm
python_bin=/home/rainbow/miniconda3/envs/nwm/bin/python
config_path="${repo_root}/configs/scand/nwm_cdit_s_scand_local320_h4_eval.yaml"
experiment_root="${repo_root}/outputs/local_old_full100_clean_median_h4_nwm_s_v70_v1"
output_dir="${experiment_root}/nwm_s_h4_n120_d250_rep3_seed17"
index_path="${experiment_root}/navigation_eval_h4_single.pkl"
trajectory_id=random_mdps_A_Jackal_AHG_Library_Thu_Oct_28_1_0

if [[ -e "${experiment_root}" ]]; then
  echo "Refusing to overwrite ${experiment_root}" >&2
  exit 2
fi
mkdir -p "${experiment_root}"

"${python_bin}" -c \
  'import pickle,sys; pickle.dump([(sys.argv[2], int(sys.argv[3]), 4, 4)], open(sys.argv[1], "wb"))' \
  "${index_path}" "${trajectory_id}" 115

cd "${nwm_root}"
"${python_bin}" planning_eval.py \
  --exp "${config_path}" \
  --ckp 0100000 \
  --datasets scand \
  --output_dir "${output_dir}" \
  --navigation_eval_index "${index_path}" \
  --trajectory_eval_horizon 4 \
  --eval_data_folder /home/rainbow/nwm/data/scand \
  --num_workers 0 \
  --batch_size 1 \
  --diffusion_steps 250 \
  --seed 17 \
  --num_samples 120 \
  --rollout_stride 1 \
  --topk 5 \
  --opt_steps 1 \
  --num_repeat_eval 3 \
  --score_mode lpips \
  --planner_variant standard \
  --candidate_batch_size 10 \
  2>&1 | tee "${experiment_root}/nwm_s.log"

touch "${experiment_root}/NWM_S_COMPLETE"
