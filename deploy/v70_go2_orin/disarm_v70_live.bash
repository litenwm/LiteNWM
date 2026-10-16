#!/usr/bin/env bash
set -euo pipefail

STOP_FILE="${GO2_NAV_STOP_FILE:-/tmp/go2_nav_stop}"
ARM_FILE="${GO2_MOTION_ARM_FILE:-/run/go2-safety/ARMED}"

# Safe order: assert stop first, then remove authorization.
touch "${STOP_FILE}"
sleep 0.40
rm -f "${ARM_FILE}"
echo "V70 navigation disarmed; stop file is asserted."
