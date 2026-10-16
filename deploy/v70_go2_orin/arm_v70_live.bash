#!/usr/bin/env bash
set -euo pipefail

ACK="${1:-}"
STOP_FILE="${GO2_NAV_STOP_FILE:-/tmp/go2_nav_stop}"
ARM_FILE="${GO2_MOTION_ARM_FILE:-/run/go2-safety/ARMED}"
READY_FILE="/tmp/v70_nav_ready"
FAULT_FILE="/tmp/v70_motion_executor_fault"

if [[ "${ACK}" != "I_UNDERSTAND_THIS_WILL_MOVE_THE_ROBOT" ]]; then
    echo "usage: $0 I_UNDERSTAND_THIS_WILL_MOVE_THE_ROBOT" >&2
    exit 2
fi
if [[ ! -f "${READY_FILE}" ]]; then
    echo "planner is not ready: ${READY_FILE} missing" >&2
    exit 3
fi
read -r READY_SCHEMA READY_MODE READY_PID <"${READY_FILE}" || true
if [[ "${READY_SCHEMA:-}" != "v70_live_ready_v1" || "${READY_MODE:-}" != "live" ]]; then
    echo "ready marker is not from a live-mode V70 planner" >&2
    exit 3
fi
if [[ ! "${READY_PID:-}" =~ ^[0-9]+$ ]] || ! kill -0 "${READY_PID}" 2>/dev/null; then
    echo "ready marker planner PID is not alive" >&2
    exit 3
fi
if [[ ! -f "${STOP_FILE}" ]]; then
    echo "stop file is already absent; refusing ambiguous arm" >&2
    exit 3
fi
if [[ -e "${ARM_FILE}" ]]; then
    echo "arm file already exists; refusing ambiguous arm" >&2
    exit 3
fi
if [[ -e "${FAULT_FILE}" ]]; then
    echo "executor fault is latched: ${FAULT_FILE}" >&2
    exit 3
fi
for pattern in \
    "v70_live_nav(_fast)?\.py" \
    "safe_bridge_v5.py" \
    "motion_executor_v2.py"
do
    if ! pgrep -f "${pattern}" >/dev/null; then
        echo "required process missing: ${pattern}" >&2
        exit 3
    fi
done

# Safe order: arm is created while stop is still asserted.  Only after the
# arm file exists is the stop interlock removed.
touch "${ARM_FILE}"
sleep 0.20
if [[ ! -f "${READY_FILE}" || -e "${FAULT_FILE}" ]]; then
    touch "${STOP_FILE}"
    rm -f "${ARM_FILE}"
    echo "readiness changed while arming; restored stopped state" >&2
    exit 4
fi
for pattern in "v70_live_nav(_fast)?\.py" "safe_bridge_v5.py" "motion_executor_v2.py"; do
    pgrep -f "${pattern}" >/dev/null || {
        touch "${STOP_FILE}"; rm -f "${ARM_FILE}"; exit 4;
    }
done
if ! rm -f "${STOP_FILE}"; then
    touch "${STOP_FILE}"
    rm -f "${ARM_FILE}"
    echo "failed to remove stop file; restored disarmed state" >&2
    exit 4
fi
echo "V70 live loop armed. Keep the hardware emergency stop available."
