#!/usr/bin/env bash
# Bounded process-group shutdown helpers for the V70 launcher.

v70_group_alive() {
    local leader_pid="$1"
    # Test the process group, not only its original leader.  The camera
    # worker can briefly outlive the planner leader during abnormal exit.
    kill -0 -- "-${leader_pid}" 2>/dev/null
}

v70_signal_group() {
    local leader_pid="$1"
    local signal_name="$2"
    if v70_group_alive "${leader_pid}"; then
        kill "-${signal_name}" -- "-${leader_pid}" 2>/dev/null || \
            kill "-${signal_name}" "${leader_pid}" 2>/dev/null || true
    fi
}

v70_wait_groups_bounded() {
    local grace_steps="$1"
    shift
    local step pid alive
    for ((step = 0; step < grace_steps; step += 1)); do
        alive=0
        for pid in "$@"; do
            if [[ -n "${pid}" ]] && v70_group_alive "${pid}"; then
                alive=1
                break
            fi
        done
        if [[ "${alive}" == "0" ]]; then
            return 0
        fi
        sleep 0.10
    done
    return 1
}

v70_reap_children() {
    local pid state residual
    residual=0
    for pid in "$@"; do
        if [[ -n "${pid}" ]]; then
            state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
            if [[ -z "${state}" || "${state}" == Z* ]]; then
                wait "${pid}" 2>/dev/null || true
            else
                # Never turn the final reap into another unbounded wait.
                residual=1
            fi
        fi
    done
    return "${residual}"
}
