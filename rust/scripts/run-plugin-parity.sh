#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: run-plugin-parity.sh TARGET_PATH -- COMMAND [ARG ...]" >&2
    exit 2
fi

target_path=$1
shift 2

if [ -z "$target_path" ]; then
    echo "TARGET_PATH must be explicit and non-empty" >&2
    exit 2
fi

mkdir -p "$target_path"
for argument in "$@"; do
    case "$argument" in
        --target-dir|--target-dir=*)
            echo "COMMAND must not override the explicit TARGET_PATH" >&2
            exit 2
            ;;
    esac
done
environment_receipt=$target_path.plugin-parity-build-environment.txt
{
    echo "CARGO_INCREMENTAL=1"
    for variable in RUSTC_WRAPPER SCCACHE_DIR SCCACHE_CACHE_SIZE SCCACHE_IDLE_TIMEOUT CARGO_BUILD_JOBS RUSTFLAGS; do
        value=$(printenv "$variable" 2>/dev/null || :)
        printf '%s=%s\n' "$variable" "$value"
    done
    printf 'COMMAND='
    printf '%s\037' "$@"
    printf '\n'
} >"$environment_receipt"

CARGO_INCREMENTAL=1 CARGO_TARGET_DIR=$target_path "$@"
