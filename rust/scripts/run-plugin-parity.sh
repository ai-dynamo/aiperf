#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
    echo "usage: run-plugin-parity.sh TARGET_PATH RECEIPT_PATH -- COMMAND [ARG ...]" >&2
    exit 2
fi

target_path=$1
receipt_path=$2
shift 3

if [ -z "$target_path" ] || [ -z "$receipt_path" ]; then
    echo "TARGET_PATH and RECEIPT_PATH must be explicit and non-empty" >&2
    exit 2
fi

mkdir -p "$target_path"
target_absolute=$(realpath -m -- "$target_path")
receipt_absolute=$(realpath -m -- "$receipt_path")
case "$receipt_absolute" in
    "$target_absolute"|"$target_absolute"/*)
        echo "RECEIPT_PATH must be outside TARGET_PATH" >&2
        exit 2
        ;;
esac
for argument in "$@"; do
    case "$argument" in
        --target-dir|--target-dir=*)
            echo "COMMAND must not override the explicit TARGET_PATH" >&2
            exit 2
            ;;
    esac
done

hex_value() {
    LC_ALL=C od -An -v -tx1 | tr -d ' \n'
}

write_environment_entry() {
    variable=$1
    eval "variable_state=\${$variable+x}"
    if [ "$variable_state" = x ]; then
        eval "variable_value=\${$variable}"
        variable_hex=$(printf '%s' "$variable_value" | hex_value)
        printf '{"name":"%s","state":"set","value_hex":"%s"}' "$variable" "$variable_hex"
    else
        printf '{"name":"%s","state":"unset","value_hex":null}' "$variable"
    fi
}

receipt_parent=$(dirname -- "$receipt_absolute")
mkdir -p "$receipt_parent"
{
    target_hex=$(printf '%s' "$target_path" | hex_value)
    printf '{"applied_environment":[{"name":"CARGO_INCREMENTAL","value_hex":"31"},{"name":"CARGO_TARGET_DIR","value_hex":"%s"}],"argv_hex":[' "$target_hex"
    separator=
    for argument in "$@"; do
        argument_hex=$(printf '%s' "$argument" | hex_value)
        printf '%s"%s"' "$separator" "$argument_hex"
        separator=,
    done
    printf '],"environment":['
    separator=
    for variable in CARGO_BUILD_JOBS CC CXX PATH RUSTC_WRAPPER RUSTFLAGS SCCACHE_CACHE_SIZE SCCACHE_DIR SCCACHE_IDLE_TIMEOUT; do
        printf '%s' "$separator"
        write_environment_entry "$variable"
        separator=,
    done
    printf '],"schema_version":1}\n'
} >"$receipt_absolute"

CARGO_INCREMENTAL=1 CARGO_TARGET_DIR=$target_path "$@"
