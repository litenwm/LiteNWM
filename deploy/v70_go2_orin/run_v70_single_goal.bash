#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-shadow}"
MANIFEST="${2:-${HOME}/litenwm_routes/goal_single.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${MODE}" != "shadow" && "${MODE}" != "live" ]]; then
    echo "usage: $0 [shadow|live] SINGLE_GOAL_MANIFEST" >&2
    exit 2
fi

# v70_live_nav.py rejects manifests containing zero or multiple goals. This
# wrapper is the only supported V4 user entry point; long-range routing stays
# disabled until it is separately designed and validated.
# Keep the precision scan main-method-only. Baseline launchers call the shared
# run_v70_closed_loop.bash directly and therefore retain their own behavior.
export V70_SCAN_PATTERN="${V70_SCAN_PATTERN:-symmetric_compact}"
export V70_SCAN_FAULT_PASSIVE_FALLBACK="${V70_SCAN_FAULT_PASSIVE_FALLBACK:-1}"
export V70_ADAPTIVE_ACTION_HISTORY="${V70_ADAPTIVE_ACTION_HISTORY:-1}"
# Keep the adaptive CEM1/CEM3 planner.  The stability correction below is
# intentionally execution-only and does not change candidate selection.
export V70_ADAPTIVE_CEM="${V70_ADAPTIVE_CEM:-1}"
export V70_STABLE_CONTINUOUS_EXECUTION="${V70_STABLE_CONTINUOUS_EXECUTION:-1}"
# OpenTrackVLA-style receding-horizon handoff: plan during the final verified
# drive segment, then preserve the planned world endpoint with an SE(2) rebase.
# The shared baseline launcher keeps this disabled by default.
export V70_ROLLING_PLAN_PREFETCH="${V70_ROLLING_PLAN_PREFETCH:-1}"
export V70_RETRY_DELAY="${V70_RETRY_DELAY:-0.20}"
# Main-method-only OpenTrack-style gait continuity: a fresh intentional zero
# uses Move(0,0,0); operator STOP, disarm, stale input and faults still use
# StopMove. Baseline launchers do not set this switch.
export V70_EXECUTOR_NEUTRAL_HOLD_S="${V70_EXECUTOR_NEUTRAL_HOLD_S:-0.0}"
export V70_PLANNED_NEUTRAL_MOVE="${V70_PLANNED_NEUTRAL_MOVE:-1}"
# Keep adaptive CEM selection unchanged while restoring the board-calibrated
# velocity and acceleration envelope used by the shared controller.
export V70_LIVE_MAX_VX="${V70_LIVE_MAX_VX:-0.30}"
export V70_LIVE_MAX_AX="${V70_LIVE_MAX_AX:-0.75}"
export V70_LIVE_GAIT_ENTRY_VX="${V70_LIVE_GAIT_ENTRY_VX:-0.0}"
# A pure low-speed yaw command made the body lean while all four feet remained
# planted.  The proven run used a short 0.20 m/s arc, capped to 3 cm, so every
# scan turn crosses the Go2's practical stepping threshold without drifting.
export V70_SCAN_TURN_ARC_VX="${V70_SCAN_TURN_ARC_VX:-0.0}"
export V70_SCAN_TURN_ARC_MAX_DISTANCE_M="${V70_SCAN_TURN_ARC_MAX_DISTANCE_M:-0.0}"

exec bash "${SCRIPT_DIR}/run_v70_closed_loop.bash" \
    "${MODE}" \
    "${MANIFEST}"
