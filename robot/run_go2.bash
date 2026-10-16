#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-shadow}"
MANIFEST="${2:-}"
if [[ "$MODE" != "shadow" && "$MODE" != "live" ]] || [[ -z "$MANIFEST" ]]; then
    echo "usage: bash robot/run_go2.bash [shadow|live] /absolute/goal_manifest.json" >&2
    exit 2
fi
export V70_PYTHON="${V70_PYTHON:-$(command -v python3)}"
export V70_FIXED_POOL_FAST_PATH="${V70_FIXED_POOL_FAST_PATH:-1}"
export V70_DINO_FEATURE_CACHE="${V70_DINO_FEATURE_CACHE:-1}"
export V70_NOMAD_CONTEXT_REUSE="${V70_NOMAD_CONTEXT_REUSE:-1}"
export V70_ASYNC_PLAN_IMAGES="${V70_ASYNC_PLAN_IMAGES:-1}"
export V70_ADAPTIVE_CEM="${V70_ADAPTIVE_CEM:-1}"
export V70_CEM_ITERATIONS="${V70_CEM_ITERATIONS:-1}"
export V70_CEM_BREAK_MIN_SAFE="${V70_CEM_BREAK_MIN_SAFE:-1}"
export V70_ADAPTIVE_CEM_MIN_SAFE="${V70_ADAPTIVE_CEM_MIN_SAFE:-2}"
export V70_AGGRESSIVE_NAV="${V70_AGGRESSIVE_NAV:-0}"
export V70_ROLLING_PLAN_PREFETCH="${V70_ROLLING_PLAN_PREFETCH:-0}"
export V70_SETTLE_PLAN_PREFETCH="${V70_SETTLE_PLAN_PREFETCH:-${V70_OPERATOR_STOP_ONLY:-0}}"
export V70_LTWM_TRT="${V70_LTWM_TRT:-0}"
export V70_DINO_TF32="${V70_DINO_TF32:-0}"
export V70_PLANNED_NEUTRAL_MOVE="${V70_PLANNED_NEUTRAL_MOVE:-0}"
export V70_EXECUTOR_NEUTRAL_HOLD_S="${V70_EXECUTOR_NEUTRAL_HOLD_S:-0.0}"
export V70_SHADOW_ONCE="${V70_SHADOW_ONCE:-1}"
cd "$ROOT"
# Load ROS/driver overlay before checking Python module availability.
source "$ROOT/deploy/v70_go2_orin/orin_env.sh"
"${V70_PYTHON}" robot/preflight.py --assets --hardware --route "$MANIFEST" --mode "$MODE"
exec bash deploy/v70_go2_orin/run_v70_single_goal.bash "$MODE" "$MANIFEST"
