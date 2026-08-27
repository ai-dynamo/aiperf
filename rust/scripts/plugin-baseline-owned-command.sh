#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared process/singleton contract for baseline capture and refresh scripts.
# The caller sets failure_ledger and may override AIPERF_PLUGIN_BASELINE_LOCK.

owned_pid=
owned_stdin_file=
baseline_lock_acquired=0

acquire_baseline_lock() {
    baseline_lock=${AIPERF_PLUGIN_BASELINE_LOCK:-${CARGO_TARGET_DIR%/}/native-plugin-baseline/.transition-lock}
    if ! mkdir "$baseline_lock" 2>/dev/null; then
        echo "another native-plugin baseline transition owns $baseline_lock" >&2
        return 73
    fi
    baseline_lock_acquired=1
}

record_baseline_failure() {
    failure_label=$1
    failure_reason=$2
    if ! mkdir -p "$(dirname "$failure_ledger")" || \
        ! printf '%s\t%s\n' "$failure_label" "$failure_reason" >>"$failure_ledger"
    then
        printf '%s\t%s\t%s\n' "$failure_label" "$failure_reason" \
            "could not persist capture failure ledger $failure_ledger" >&2
    fi
    return 0
}

terminate_owned_group() {
    pid=$1
    label=$2
    [ -n "$pid" ] || return 0
    kill -TERM "-$pid" 2>/dev/null || true
    grace=${AIPERF_PLUGIN_TERM_GRACE_SECONDS:-${AIPERF_CAPTURE_TERM_GRACE_SECONDS:-10}}
    deadline=$(($(date +%s) + grace))
    while kill -0 "-$pid" 2>/dev/null; do
        [ "$(date +%s)" -lt "$deadline" ] || break
        sleep 0.1
    done
    if kill -0 "-$pid" 2>/dev/null; then
        kill -KILL "-$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    deadline=$(($(date +%s) + grace))
    while kill -0 "-$pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "$label process group $pid survived TERM/KILL teardown" >&2
            return 74
        fi
        sleep 0.1
    done
}

run_owned_from() {
    timeout_seconds=$1
    label=$2
    stdin_path=$3
    shift 3
    [ -z "$owned_pid" ] || {
        echo "nested owned command is forbidden: $label" >&2
        return 70
    }
    setsid "$@" <"$stdin_path" &
    owned_pid=$!
    deadline=$(($(date +%s) + timeout_seconds))
    timed_out=0
    while kill -0 "$owned_pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            timed_out=1
            break
        fi
        sleep 0.1
    done
    if [ "$timed_out" -eq 1 ]; then
        record_baseline_failure "$label" "timeout after ${timeout_seconds}s"
        terminate_owned_group "$owned_pid" "$label" || return 74
        owned_pid=
        return 124
    fi
    if wait "$owned_pid"; then owned_command_status=0; else owned_command_status=$?; fi
    if kill -0 "-$owned_pid" 2>/dev/null; then
        record_baseline_failure "$label" "leader exited while descendant survived"
        terminate_owned_group "$owned_pid" "$label" || return 74
        owned_pid=
        if [ "$owned_command_status" -ne 0 ]; then
            record_baseline_failure "$label" "exit status $owned_command_status"
            return "$owned_command_status"
        fi
        return 74
    fi
    owned_pid=
    if [ "$owned_command_status" -ne 0 ]; then
        record_baseline_failure "$label" "exit status $owned_command_status"
    fi
    return "$owned_command_status"
}

run_owned() {
    timeout_seconds=$1
    label=$2
    shift 2
    run_owned_from "$timeout_seconds" "$label" /dev/null "$@"
}

run_owned_with_stdin() {
    timeout_seconds=$1
    label=$2
    shift 2
    [ -z "$owned_stdin_file" ] || {
        echo "owned stdin spool is already active: $label" >&2
        return 70
    }
    owned_stdin_file=$(mktemp "$(dirname "$failure_ledger")/.capture-stdin.XXXXXX")
    chmod 0600 "$owned_stdin_file"
    cat >"$owned_stdin_file"
    if run_owned_from "$timeout_seconds" "$label" "$owned_stdin_file" "$@"; then
        owned_stdin_status=0
    else
        owned_stdin_status=$?
    fi
    rm -f -- "$owned_stdin_file"
    owned_stdin_file=
    return "$owned_stdin_status"
}

release_baseline_lock() {
    [ "$baseline_lock_acquired" -eq 1 ] || return 0
    rmdir "$baseline_lock" 2>/dev/null || return 74
    baseline_lock_acquired=0
}

require_free_bytes() {
    capacity_path=$1
    required_bytes=$2
    available_bytes=$(df -Pk "$capacity_path" | awk 'NR == 2 {printf "%.0f\n", $4 * 1024}')
    case "$available_bytes" in
        ''|*[!0-9]*) echo "could not determine free-space capacity for $capacity_path" >&2; return 66 ;;
    esac
    [ "$available_bytes" -ge "$required_bytes" ] || {
        echo "$capacity_path requires $required_bytes free bytes, found $available_bytes" >&2
        return 66
    }
    printf '%s\n' "$available_bytes"
}
