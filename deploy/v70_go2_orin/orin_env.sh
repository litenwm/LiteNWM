#!/usr/bin/env bash
set -e
# ROS Humble/colcon overlays reference optional, unset tracing variables.
# Preserve the caller's nounset setting after loading those scripts.
V70_RESTORE_NOUNSET=0
if [[ "$-" == *u* ]]; then V70_RESTORE_NOUNSET=1; set +u; fi
V70_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_SETUP="${V70_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f "${ROS_SETUP}" ]]; then source "${ROS_SETUP}"; fi
if [[ -n "${V70_SENSOR_SETUP:-}" ]]; then source "${V70_SENSOR_SETUP}"; fi
export PYTHONPATH="${V70_ROOT}:${V70_ROOT}/src:${V70_ROOT}/deploy/v70_go2_orin${PYTHONPATH:+:${PYTHONPATH}}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "${V70_RESTORE_NOUNSET}" == "1" ]]; then set -u; fi
unset V70_RESTORE_NOUNSET
