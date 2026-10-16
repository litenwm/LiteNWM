#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-shadow}"
ROUTE_MANIFEST="${2:-${HOME}/litenwm_routes/goal_single.json}"
V70_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V70_PYTHON="${V70_PYTHON:-$(command -v python3)}"
V70_INTERFACE="${V70_INTERFACE:-enP8p1s0}"
V70_ODOM_TOPIC="${V70_ODOM_TOPIC:-/utlidar/robot_odom}"
V70_CLOUD_TOPIC="${V70_CLOUD_TOPIC:-/utlidar/cloud_base}"
STOP_FILE="${GO2_NAV_STOP_FILE:-/tmp/go2_nav_stop}"
ARM_FILE="${GO2_MOTION_ARM_FILE:-/run/go2-safety/ARMED}"
READY_FILE="/tmp/v70_nav_ready"
FAULT_FILE="/tmp/v70_motion_executor_fault"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${V70_OUTPUT_DIR:-${HOME}/litenwm_runs/${RUN_STAMP}}"
SHUTDOWN_GRACE_STEPS="${V70_SHUTDOWN_GRACE_STEPS:-300}"
OPERATOR_STOP_ONLY="${V70_OPERATOR_STOP_ONLY:-0}"
OPERATOR_STOP_ACK="${V70_OPERATOR_STOP_ACK:-}"
SCAN_ONLY="${V70_SCAN_ONLY:-0}"
AGGRESSIVE_NAV="${V70_AGGRESSIVE_NAV:-1}"
ADAPTIVE_CEM="${V70_ADAPTIVE_CEM:-1}"
FIXED_POOL_FAST_PATH="${V70_FIXED_POOL_FAST_PATH:-0}"
DINO_FEATURE_CACHE="${V70_DINO_FEATURE_CACHE:-0}"
SETTLE_PLAN_PREFETCH="${V70_SETTLE_PLAN_PREFETCH:-0}"
ROLLING_PLAN_PREFETCH="${V70_ROLLING_PLAN_PREFETCH:-0}"
ASYNC_PLAN_IMAGES="${V70_ASYNC_PLAN_IMAGES:-0}"
PLANNER_ENTRYPOINT="${V70_PLANNER_ENTRYPOINT:-deploy/v70_go2_orin/v70_live_nav_fast.py}"
SENSOR_PREFLIGHT_DURATION_S="${V70_SENSOR_PREFLIGHT_DURATION_S:-10.0}"

# Board-calibrated Go2 live actuation profile.  The V70 target action and all
# model/planner constraints remain unchanged; only SportClient velocity mapping
# is lifted above the physical locomotion deadband observed on this robot.
GO2_LIVE_MAX_VX="${V70_LIVE_MAX_VX:-0.30}"
GO2_LIVE_MAX_WZ="0.35"
GO2_LIVE_MAX_AX="${V70_LIVE_MAX_AX:-0.75}"
GO2_LIVE_MAX_AWZ="1.00"
GO2_LIVE_GAIT_ENTRY_VX="${V70_LIVE_GAIT_ENTRY_VX:-0.0}"
EXECUTOR_NEUTRAL_HOLD_S="${V70_EXECUTOR_NEUTRAL_HOLD_S:-0.0}"
EXECUTOR_PLANNED_NEUTRAL_MOVE="${V70_PLANNED_NEUTRAL_MOVE:-0}"

if [[ "${MODE}" != "shadow" && "${MODE}" != "live" ]]; then
    echo "usage: $0 [shadow|live] ROUTE_MANIFEST" >&2
    exit 2
fi
if [[ "${OPERATOR_STOP_ONLY}" != "0" && \
      "${OPERATOR_STOP_ONLY}" != "1" ]]; then
    echo "V70_OPERATOR_STOP_ONLY must be 0 or 1" >&2
    exit 2
fi
if [[ "${SCAN_ONLY}" != "0" && "${SCAN_ONLY}" != "1" ]]; then
    echo "V70_SCAN_ONLY must be 0 or 1" >&2
    exit 2
fi
if [[ "${AGGRESSIVE_NAV}" != "0" && "${AGGRESSIVE_NAV}" != "1" ]]; then
    echo "V70_AGGRESSIVE_NAV must be 0 or 1" >&2
    exit 2
fi
if [[ "${ADAPTIVE_CEM}" != "0" && "${ADAPTIVE_CEM}" != "1" ]]; then
    echo "V70_ADAPTIVE_CEM must be 0 or 1" >&2
    exit 2
fi
if [[ "${FIXED_POOL_FAST_PATH}" != "0" && \
      "${FIXED_POOL_FAST_PATH}" != "1" ]]; then
    echo "V70_FIXED_POOL_FAST_PATH must be 0 or 1" >&2
    exit 2
fi
if [[ "${DINO_FEATURE_CACHE}" != "0" && \
      "${DINO_FEATURE_CACHE}" != "1" ]]; then
    echo "V70_DINO_FEATURE_CACHE must be 0 or 1" >&2
    exit 2
fi
if [[ "${SETTLE_PLAN_PREFETCH}" != "0" && \
      "${SETTLE_PLAN_PREFETCH}" != "1" ]]; then
    echo "V70_SETTLE_PLAN_PREFETCH must be 0 or 1" >&2
    exit 2
fi
if [[ "${ROLLING_PLAN_PREFETCH}" != "0" && \
      "${ROLLING_PLAN_PREFETCH}" != "1" ]]; then
    echo "V70_ROLLING_PLAN_PREFETCH must be 0 or 1" >&2
    exit 2
fi
if [[ "${ASYNC_PLAN_IMAGES}" != "0" && \
      "${ASYNC_PLAN_IMAGES}" != "1" ]]; then
    echo "V70_ASYNC_PLAN_IMAGES must be 0 or 1" >&2
    exit 2
fi
if [[ "${EXECUTOR_PLANNED_NEUTRAL_MOVE}" != "0" && \
      "${EXECUTOR_PLANNED_NEUTRAL_MOVE}" != "1" ]]; then
    echo "V70_PLANNED_NEUTRAL_MOVE must be 0 or 1" >&2
    exit 2
fi
if [[ "${EXECUTOR_PLANNED_NEUTRAL_MOVE}" == "1" ]]; then
    EXECUTOR_PLANNED_NEUTRAL_MOVE_ROS=true
else
    EXECUTOR_PLANNED_NEUTRAL_MOVE_ROS=false
fi
if [[ "${MODE}" != "live" && "${SCAN_ONLY}" == "1" ]]; then
    echo "scan-only is valid only in live mode" >&2
    exit 2
fi
if [[ "${SCAN_ONLY}" == "1" && \
      "${OPERATOR_STOP_ONLY}" == "1" ]]; then
    echo "scan-only cannot be combined with operator-stop-only" >&2
    exit 2
fi
if [[ "${MODE}" != "live" && "${OPERATOR_STOP_ONLY}" == "1" ]]; then
    echo "operator-stop-only is valid only in live mode" >&2
    exit 2
fi
if [[ "${MODE}" == "live" && "${OPERATOR_STOP_ONLY}" != "1" && \
      ( "${SETTLE_PLAN_PREFETCH}" == "1" || "${ROLLING_PLAN_PREFETCH}" == "1" ) ]]; then
    echo "prefetch requires operator-stop-only mode; disable it for calibrated automatic arrival" >&2
    exit 2
fi
if [[ "${OPERATOR_STOP_ONLY}" == "1" && \
      "${OPERATOR_STOP_ACK}" != "I_WILL_STOP_THE_ROBOT_MANUALLY" ]]; then
    echo "operator-stop-only requires "\
"V70_OPERATOR_STOP_ACK=I_WILL_STOP_THE_ROBOT_MANUALLY" >&2
    exit 3
fi
if [[ ! -x "${V70_PYTHON}" ]]; then
    echo "missing V70 Python: ${V70_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${ROUTE_MANIFEST}" ]]; then
    echo "missing route manifest: ${ROUTE_MANIFEST}" >&2
    exit 2
fi
if [[ "${PLANNER_ENTRYPOINT}" = /* || \
      "${PLANNER_ENTRYPOINT}" == *".."* || \
      ! -f "${V70_ROOT}/${PLANNER_ENTRYPOINT}" ]]; then
    echo "invalid/missing V70_PLANNER_ENTRYPOINT: ${PLANNER_ENTRYPOINT}" >&2
    exit 2
fi
if ! [[ "${SHUTDOWN_GRACE_STEPS}" =~ ^[0-9]+$ ]] || \
        ((SHUTDOWN_GRACE_STEPS < 1 || SHUTDOWN_GRACE_STEPS > 600)); then
    echo "V70_SHUTDOWN_GRACE_STEPS must be an integer in [1, 600]" >&2
    exit 2
fi
for required_command in setsid timeout; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "missing required command: ${required_command}" >&2
        exit 2
    fi
done
if [[ ! -f "${STOP_FILE}" ]]; then
    echo "refusing startup: stop file must already exist" >&2
    exit 3
fi
if [[ -e "${ARM_FILE}" ]]; then
    echo "refusing startup: arm file must be absent" >&2
    exit 3
fi
if [[ -e "${READY_FILE}" ]]; then
    echo "refusing startup: stale ready file ${READY_FILE}" >&2
    exit 3
fi
if [[ -e "${FAULT_FILE}" ]]; then
    echo "refusing startup: inspect/remove latched fault ${FAULT_FILE}" >&2
    exit 3
fi
if pgrep -f "v70_live_nav_fast.py|v70_live_nav.py|baseline_live_nav.py|safe_bridge_v5.py|motion_executor_v2.py" >/dev/null; then
    echo "refusing startup: a V70 navigation process is already running" >&2
    exit 3
fi

if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
    echo "refusing startup: output path is not a directory: ${OUTPUT_DIR}" >&2
    exit 3
fi
if [[ -d "${OUTPUT_DIR}" ]] && \
        [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing startup: output directory is not empty: ${OUTPUT_DIR}" >&2
    echo "choose a new V70_OUTPUT_DIR; existing run data is preserved" >&2
    exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cd "${V70_ROOT}"

# ROS Humble setup scripts legitimately reference unset tracing variables.
# Disable nounset only while sourcing the sealed runtime environment.
set +u
# shellcheck disable=SC1091
if [[ -n "${V70_VENV:-}" ]]; then source "${V70_VENV}/bin/activate"; fi
# shellcheck disable=SC1091
source deploy/v70_go2_orin/orin_env.sh
# shellcheck disable=SC1091
source deploy/v70_go2_orin/v70_shutdown_helpers.bash
set -u

if [[ "${MODE}" == "live" ]]; then
    if [[ "${V70_OPERATOR_POSTURE:-}" != "standing" ]]; then
        echo "refusing live startup: export V70_OPERATOR_POSTURE=standing" >&2
        exit 3
    fi
    echo "Running zero-motion standing sensor preflight..."
    timeout --signal=TERM --kill-after=2s 15s \
        "${V70_PYTHON}" \
        deploy/v70_go2_orin/v70_sensor_preflight.py \
        --interface "${V70_INTERFACE}" \
        --expected-ip "${V70_EXPECTED_IP:-192.168.123.18}" \
        --odom-topic "${V70_ODOM_TOPIC}" \
        --cloud-topic "${V70_CLOUD_TOPIC}" \
        --posture standing \
        --duration-s "${SENSOR_PREFLIGHT_DURATION_S}" \
        --stop-file "${STOP_FILE}" \
        --arm-file "${ARM_FILE}" \
        --output "${OUTPUT_DIR}/standing_preflight.json" \
        >"${OUTPUT_DIR}/standing_preflight.log" 2>&1
    echo "Probing Sport lease with StopMove only..."
    timeout --signal=TERM --kill-after=2s 12s \
        "${V70_PYTHON}" \
        deploy/v70_go2_orin/v70_sport_lease_preflight.py \
        --interface "${V70_INTERFACE}" \
        --stop-file "${STOP_FILE}" \
        --arm-file "${ARM_FILE}" \
        --output "${OUTPUT_DIR}/sport_lease_preflight.json" \
        >"${OUTPUT_DIR}/sport_lease_preflight.log" 2>&1
fi

DRY_RUN="true"
LIVE_FLAG=()
SHADOW_ONCE_FLAG=()
if [[ "${MODE}" == "live" ]]; then
    DRY_RUN="false"
    # Operator-stop-only navigation uses endpoint history and bounded forward
    # continuation; it must never enter the active left/right scan state.
    # Diagnostic live modes retain the explicit scan workflow below.
    LIVE_FLAG=(
        --live
        --max-consecutive-plan-holds \
            "${V70_MAX_CONSECUTIVE_PLAN_HOLDS:-5}"
        --cem-iterations "${V70_CEM_ITERATIONS:-3}"
        --live-max-vx "${GO2_LIVE_MAX_VX}"
        --retry-delay "${V70_RETRY_DELAY:-0.10}"
    )
    if [[ "${ADAPTIVE_CEM}" == "1" ]]; then
        LIVE_FLAG+=(
            --adaptive-cem
            --adaptive-cem-min-safe
            "${V70_ADAPTIVE_CEM_MIN_SAFE:-2}"
        )
    fi
    if [[ "${FIXED_POOL_FAST_PATH}" == "1" ]]; then
        LIVE_FLAG+=(--fixed-pool-fast-path)
    fi
    if [[ "${DINO_FEATURE_CACHE}" == "1" ]]; then
        LIVE_FLAG+=(--dino-feature-cache)
    fi
    if [[ "${SETTLE_PLAN_PREFETCH}" == "1" ]]; then
        LIVE_FLAG+=(--settle-plan-prefetch)
    fi
    if [[ "${ROLLING_PLAN_PREFETCH}" == "1" ]]; then
        LIVE_FLAG+=(--rolling-plan-prefetch)
    fi
    if [[ "${ASYNC_PLAN_IMAGES}" == "1" ]]; then
        LIVE_FLAG+=(--save-plan-images --async-plan-images)
    fi
    if [[ "${AGGRESSIVE_NAV}" == "1" ]]; then
        LIVE_FLAG+=(--aggressive-navigation)
    fi
    if [[ "${SCAN_ONLY}" == "1" ]]; then
        LIVE_FLAG+=(--scan-only)
    elif [[ "${OPERATOR_STOP_ONLY}" == "1" ]]; then
        LIVE_FLAG+=(--operator-stop-only --save-plan-images)
    fi
elif [[ "${V70_SHADOW_ONCE:-0}" == "1" ]]; then
    SHADOW_ONCE_FLAG=(--shadow-once)
fi

PIDS=()
SAFE_PID=""
EXECUTOR_PID=""
PLANNER_PID=""
SHUTTING_DOWN=0
shutdown_all() {
    if [[ "${SHUTTING_DOWN}" == "1" ]]; then
        return
    fi
    SHUTTING_DOWN=1
    touch "${STOP_FILE}"
    if ! rm -f "${ARM_FILE}"; then
        echo "arm_clear_failed=true" >>"${OUTPUT_DIR}/shutdown.log"
    fi
    {
        echo "shutdown_start=$(date -Ins)"
        echo "planner_pgid=${PLANNER_PID:-}"
        echo "safe_bridge_pgid=${SAFE_PID:-}"
        echo "executor_pgid=${EXECUTOR_PID:-}"
    } >>"${OUTPUT_DIR}/shutdown.log"

    # Give a healthy executor one control-timeout window to observe the stop
    # file and issue StopMove. If the executor itself died in live mode, use
    # an independent one-shot Unitree client as the final zero-command path.
    if [[ "${MODE}" == "live" ]]; then
        if [[ -n "${EXECUTOR_PID}" ]] && kill -0 "${EXECUTOR_PID}" 2>/dev/null; then
            sleep 0.40
        else
            timeout --signal=TERM --kill-after=2s 8s \
                "${V70_PYTHON}" \
                deploy/v70_go2_orin/unitree_stop_once.py \
                --interface "${V70_INTERFACE}" \
                >>"${OUTPUT_DIR}/emergency_stop.log" 2>&1 || true
        fi
    fi

    for pid in "${PLANNER_PID}" "${SAFE_PID}"; do
        if [[ -n "${pid}" ]]; then
            v70_signal_group "${pid}" TERM
        fi
    done
    sleep 0.10
    if [[ -n "${EXECUTOR_PID}" ]]; then
        v70_signal_group "${EXECUTOR_PID}" TERM
    fi
    if ! v70_wait_groups_bounded \
            "${SHUTDOWN_GRACE_STEPS}" "${PIDS[@]}"; then
        echo "shutdown_escalation=SIGKILL" \
            >>"${OUTPUT_DIR}/shutdown.log"
        for pid in "${PIDS[@]}"; do
            v70_signal_group "${pid}" KILL
        done
        v70_wait_groups_bounded 20 "${PIDS[@]}" || true
    fi
    if ! v70_reap_children "${PIDS[@]}"; then
        echo "shutdown_residual_process_group=true" \
            >>"${OUTPUT_DIR}/shutdown.log"
    fi
    rm -f "${READY_FILE}"
    echo "shutdown_done=$(date -Ins)" >>"${OUTPUT_DIR}/shutdown.log"
}

on_hangup() {
    shutdown_all
    exit 129
}

on_interrupt() {
    shutdown_all
    exit 130
}

on_terminate() {
    shutdown_all
    exit 143
}

trap shutdown_all EXIT
trap on_hangup HUP
trap on_interrupt INT
trap on_terminate TERM

setsid "${V70_PYTHON}" deploy/v70_go2_orin/safe_bridge_v5.py \
    --ros-args \
    -p odom_topic:="${V70_ODOM_TOPIC}" \
    -p dry_run:="${DRY_RUN}" \
    -p cloud_topic:="${V70_CLOUD_TOPIC}" \
    -p stop_file:="${STOP_FILE}" \
    -p arm_file:="${ARM_FILE}" \
    -p max_vx:="${GO2_LIVE_MAX_VX}" \
    -p max_wz:="${GO2_LIVE_MAX_WZ}" \
    -p max_ax:="${GO2_LIVE_MAX_AX}" \
    -p max_awz:="${GO2_LIVE_MAX_AWZ}" \
    -p min_forward_gait_entry_vx:="${GO2_LIVE_GAIT_ENTRY_VX}" \
    >"${OUTPUT_DIR}/safe_bridge.log" 2>&1 &
PIDS+=("$!")
SAFE_PID="${PIDS[-1]}"

# Keep rclpy off CycloneDDS in this process: Unitree Python SportClient also
# embeds CycloneDDS, and initializing both bindings in one process is invalid.
setsid env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    "${V70_PYTHON}" deploy/v70_go2_orin/motion_executor_v2.py \
    --ros-args \
    -p interface:="${V70_INTERFACE}" \
    -p dry_run:="${DRY_RUN}" \
    -p stop_file:="${STOP_FILE}" \
    -p arm_file:="${ARM_FILE}" \
    -p fault_file:="${FAULT_FILE}" \
    -p max_vx:="${GO2_LIVE_MAX_VX}" \
    -p max_wz:="${GO2_LIVE_MAX_WZ}" \
    -p neutral_hold_s:="${EXECUTOR_NEUTRAL_HOLD_S}" \
    -p planned_neutral_move:="${EXECUTOR_PLANNED_NEUTRAL_MOVE_ROS}" \
    >"${OUTPUT_DIR}/motion_executor.log" 2>&1 &
PIDS+=("$!")
EXECUTOR_PID="${PIDS[-1]}"

setsid "${V70_PYTHON}" "${PLANNER_ENTRYPOINT}" \
    --interface "${V70_INTERFACE}" \
    --odom-topic "${V70_ODOM_TOPIC}" \
    --route-manifest "${ROUTE_MANIFEST}" \
    --stop-file "${STOP_FILE}" \
    --arm-file "${ARM_FILE}" \
    --ready-file "${READY_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${LIVE_FLAG[@]}" \
    "${SHADOW_ONCE_FLAG[@]}" \
    >"${OUTPUT_DIR}/planner.log" 2>&1 &
PIDS+=("$!")
PLANNER_PID="${PIDS[-1]}"

echo "V70 ${MODE} closed loop started."
echo "output: ${OUTPUT_DIR}"
echo "stop file remains present; arm file remains absent."
echo "ready marker (after models and sensors stabilize): ${READY_FILE}"
if [[ "${MODE}" == "live" ]]; then
    echo "The launcher does not arm or remove the stop file."
    if [[ "${SCAN_ONLY}" == "1" ]]; then
        echo "SCAN-ONLY: four frames, automatic STOP, no planning or drive."
    elif [[ "${OPERATOR_STOP_ONLY}" == "1" ]]; then
        echo "OPERATOR-STOP-ONLY: auto arrival disabled; manual STOP required."
    fi
fi

set +e
wait -n "${PIDS[@]}"
STATUS=$?
set -e
touch "${STOP_FILE}"
echo "A V70 process exited with status ${STATUS}; stop file asserted." >&2
exit "${STATUS}"
