#!/usr/bin/env bash
# Source from the extracted tri-domain bundle root.

set -e

V70_TRIDOMAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V70_TRIDOMAIN_VENV="${V70_TRIDOMAIN_VENV:-/home/unitree/venvs/v70-tridomain-orin}"

if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [ -f /home/unitree/jp6_legacy_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /home/unitree/jp6_legacy_ws/install/setup.bash
fi
if [ -f "${V70_TRIDOMAIN_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${V70_TRIDOMAIN_VENV}/bin/activate"
fi

export V70_TRIDOMAIN_ROOT
export V70_TRIDOMAIN_VENV
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GO2_NAV_STOP_FILE="${GO2_NAV_STOP_FILE:-/tmp/go2_nav_stop}"
export GO2_MOTION_ARM_FILE="${GO2_MOTION_ARM_FILE:-/run/go2-safety/ARMED}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${V70_TRIDOMAIN_ROOT}:${V70_TRIDOMAIN_ROOT}/src:${V70_TRIDOMAIN_ROOT}/external/visualnav-transformer/train:${V70_TRIDOMAIN_ROOT}/external/diffusion_policy${PYTHONPATH:+:${PYTHONPATH}}"

echo "V70_TRIDOMAIN_ROOT=${V70_TRIDOMAIN_ROOT}"
echo "V70_TRIDOMAIN_VENV=${V70_TRIDOMAIN_VENV}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "GO2_NAV_STOP_FILE=${GO2_NAV_STOP_FILE}"
echo "GO2_MOTION_ARM_FILE=${GO2_MOTION_ARM_FILE}"
