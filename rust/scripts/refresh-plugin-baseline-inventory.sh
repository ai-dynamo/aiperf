#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

usage() {
    echo "usage: $0 pre-capture|post-capture|postpublication GENERATION [CAPTURE_ROOT]" >&2
    exit 64
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
mode=$1
generation=$2
capture_root=${3:-${AIPERF_PLUGIN_CAPTURE_ROOT:-}}
case "$generation" in
    ''|*[!a-z0-9]*|review1|review1b|review1c|review1d|review1e|review1f|review1g|review1h)
        echo "invalid baseline generation: $generation" >&2
        exit 65
        ;;
esac
case "$mode" in
    pre-capture) [ -z "$capture_root" ] || usage ;;
    post-capture|postpublication) [ -n "$capture_root" ] || usage ;;
    *) usage ;;
esac

repository=$(git rev-parse --show-toplevel)
inventory=$repository/rust/benchmarks/plugin-parity.yaml
topology=$repository/artifacts/native-plugin-baseline/package-topology.json
baseline=caa3ff6fcf20ffe36a7704abe16274bedadbb9fb
refresh_parent=${AIPERF_PLUGIN_REFRESH_ROOT:-${CARGO_TARGET_DIR:?}/native-plugin-baseline}
case "$refresh_parent" in
    /*) ;;
    *) echo "refresh parent must be an absolute existing directory: $refresh_parent" >&2; exit 65 ;;
esac
[ -d "$refresh_parent" ] || {
    echo "refresh parent must be an absolute existing directory: $refresh_parent" >&2
    exit 66
}
refresh_parent=$(CDPATH= cd -- "$refresh_parent" && pwd -P)
[ "$refresh_parent" != / ] || {
    echo "refresh parent must not be the filesystem root" >&2
    exit 65
}
ownership_helper=$repository/rust/scripts/plugin-baseline-owned-command.sh
. "$ownership_helper"
failure_ledger=$refresh_parent/task1-$generation-refresh-failures.txt
acquire_baseline_lock
trap 'status=$?; trap - EXIT; release_baseline_lock || status=74; exit "$status"' EXIT
refresh_root=$(mktemp -d "$refresh_parent/task1-$generation-refresh.XXXXXX")
refresh_marker=$refresh_root/.aiperf-plugin-refresh-owned
: >"$refresh_marker"
cleanup() {
    status=$?
    trap - EXIT HUP INT TERM
    if [ -n "$owned_pid" ]; then
        terminate_owned_group "$owned_pid" refresh-active-child || status=74
        owned_pid=
    fi
    if [ -n "$owned_stdin_file" ]; then
        rm -f -- "$owned_stdin_file" || status=74
        owned_stdin_file=
    fi
    case "$refresh_root" in
        "$refresh_parent"/task1-"$generation"-refresh.*)
            if [ -f "$refresh_marker" ]; then
                run_owned 600 refresh-cleanup rm -rf -- "$refresh_root" || status=74
            else
                echo "refusing cleanup without owned refresh marker: $refresh_root" >&2
            fi
            ;;
        *) echo "refusing cleanup of unexpected refresh root: $refresh_root" >&2 ;;
    esac
    release_baseline_lock || status=74
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

refresh_required_free_bytes=${AIPERF_PLUGIN_REFRESH_REQUIRED_FREE_BYTES:-21474836480}
refresh_available_free_bytes=$(require_free_bytes "$refresh_parent" "$refresh_required_free_bytes")
refresh_tmpdir=${AIPERF_PLUGIN_REFRESH_TMPDIR:-$refresh_root/tmp}
case "$refresh_tmpdir" in
    "$refresh_parent"/*) ;;
    *) echo "refresh TMPDIR must be owned by refresh parent: $refresh_tmpdir" >&2; exit 65 ;;
esac
mkdir -p "$refresh_tmpdir"
TMPDIR=$refresh_tmpdir
export TMPDIR

if [ "$mode" = pre-capture ]; then
    receipts=$refresh_root/receipts
    base=$refresh_root/base
    effective=$refresh_root/effective
    mkdir -p "$receipts" "$base" "$effective"
    run_owned 300 refresh-git-archive git -C "$repository" archive --format=tar \
        "$baseline" >"$receipts/source-tree.tar"
    run_owned 300 refresh-extract-base tar -xf "$receipts/source-tree.tar" -C "$base"
    run_owned 300 refresh-extract-effective tar -xf "$receipts/source-tree.tar" -C "$effective"
    cp "$base/rust/Cargo.lock" "$receipts/baseline-Cargo.lock"
    cp "$repository/rust/benchmarks/plugin-baseline-measurement-source-projection.txt" \
        "$receipts/measurement-source-projection.txt"
    cp "$repository/rust/benchmarks/plugin-baseline-invalidations.tsv" \
        "$receipts/invalidations.tsv"
    cp "$repository/rust/benchmarks/exporter-observable-policy.json" \
        "$receipts/exporter-observable-policy.json"
    cp "$repository/rust/benchmarks/exporter-static-calibration-corpus.json" \
        "$receipts/exporter-static-calibration-corpus.json"
    while IFS= read -r projected_path; do
        case "$projected_path" in
            ''|/*|../*|*/../*) echo "unsafe projection path: $projected_path" >&2; exit 65 ;;
        esac
        [ -f "$repository/$projected_path" ] || {
            echo "missing projection source: $projected_path" >&2
            exit 66
        }
        mkdir -p "$effective/$(dirname "$projected_path")"
        cp "$repository/$projected_path" "$effective/$projected_path"
    done <"$receipts/measurement-source-projection.txt"
    run_owned 300 refresh-normalize-directories find "$effective" -type d -exec chmod 0755 {} +
    run_owned 300 refresh-normalize-executables find "$effective" -type f -perm /0111 -exec chmod 0755 {} +
    run_owned 300 refresh-normalize-files find "$effective" -type f ! -perm /0111 -exec chmod 0644 {} +
    run_owned 300 refresh-projection-archive tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -C "$effective" -cf "$receipts/measurement-source-projection.tar" \
        -T "$receipts/measurement-source-projection.txt"
    run_owned 300 refresh-effective-archive tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -C "$effective" -cf "$receipts/effective-source-tree.tar" .
    cp "$effective/rust/Cargo.lock" "$receipts/Cargo.lock"
    cp "$effective/rust/scripts/capture-plugin-baseline.sh" \
        "$receipts/capture-plugin-baseline.sh"
    cp "$effective/rust/scripts/plugin-baseline-owned-command.sh" \
        "$receipts/plugin-baseline-owned-command.sh"
    source_root=$effective
else
    receipts=$capture_root/evidence/identity
    [ -d "$receipts" ] || {
        echo "missing capture identity root: $receipts" >&2
        exit 66
    }
    source_root=$refresh_root/effective
    mkdir -p "$source_root"
    run_owned 300 refresh-extract-captured-source tar -xf \
        "$receipts/effective-source-tree.tar" -C "$source_root"
fi

candidate_inventory=$source_root/rust/benchmarks/plugin-parity.yaml
candidate_compact=$source_root/artifacts/native-plugin-baseline
candidate_topology=$candidate_compact/package-topology.json
mkdir -p "$(dirname "$candidate_inventory")" "$candidate_compact"
cp "$inventory" "$candidate_inventory"
for compact_name in README.md allocation-probe.json evidence-manifest.json bundle-locator.json package-topology.json; do
    compact_source=$repository/artifacts/native-plugin-baseline/$compact_name
    if [ -f "$compact_source" ]; then
        cp "$compact_source" "$candidate_compact/$compact_name"
    fi
done

tool_target=$refresh_root/tool-target
run_owned 1800 refresh-tool-build sh -c \
    'cd "$1"; cargo build --locked -p aiperf-bench-tools --bin evidence_digest --target-dir "$2"' \
    sh "$source_root/rust" "$tool_target"
digest_tool=$tool_target/debug/evidence_digest
[ -x "$digest_tool" ] || {
    echo "refresh evidence tool was not built" >&2
    exit 70
}

run_owned 600 refresh-cargo-metadata sh -c \
    'cd "$1"; cargo metadata --locked --format-version 1' sh "$source_root/rust" \
    >"$refresh_root/cargo-metadata.raw.json"
run_owned 600 refresh-cargo-tree-workspace sh -c \
    'cd "$1"; cargo tree --locked --workspace --edges normal,build --prefix depth' \
    sh "$source_root/rust" \
    >"$refresh_root/cargo-tree-workspace.raw.txt"
run_owned 600 refresh-cargo-tree-cli sh -c \
    'cd "$1"; cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth' \
    sh "$source_root/rust" \
    >"$refresh_root/cargo-tree-cli.raw.txt"
run_owned 300 refresh-normalize-metadata "$digest_tool" normalize-cargo-receipt metadata "$source_root" \
    "$refresh_root/cargo-metadata.raw.json" "$refresh_root/cargo-metadata.json"
run_owned 300 refresh-normalize-workspace-tree "$digest_tool" normalize-cargo-receipt tree "$source_root" \
    "$refresh_root/cargo-tree-workspace.raw.txt" \
    "$refresh_root/cargo-tree-workspace.txt"
run_owned 300 refresh-normalize-cli-tree "$digest_tool" normalize-cargo-receipt tree "$source_root" \
    "$refresh_root/cargo-tree-cli.raw.txt" "$refresh_root/cargo-tree-cli.txt"
rustc_identity="$(rustc --version);LLVM $(rustc -vV | sed -n 's/^LLVM version: //p')"
run_owned 300 refresh-topology "$digest_tool" topology "$generation" "$baseline" "$rustc_identity" \
    "$(rustc -vV | sed -n 's/^host: //p')" release "$source_root/rust/Cargo.lock" \
    "$refresh_root/cargo-metadata.json" \
    "$refresh_root/cargo-tree-workspace.txt" \
    "$refresh_root/cargo-tree-cli.txt" >"$refresh_root/generated-topology.json"
if [ "$mode" = pre-capture ]; then
    cp "$refresh_root/cargo-metadata.json" "$receipts/cargo-metadata.json"
    cp "$refresh_root/cargo-tree-workspace.txt" "$receipts/cargo-tree-workspace.txt"
    cp "$refresh_root/cargo-tree-cli.txt" "$receipts/cargo-tree-cli.txt"
    cp "$refresh_root/generated-topology.json" "$receipts/package-topology.json"
else
    cmp "$refresh_root/cargo-metadata.json" "$receipts/cargo-metadata.json"
    cmp "$refresh_root/cargo-tree-workspace.txt" "$receipts/cargo-tree-workspace.txt"
    cmp "$refresh_root/cargo-tree-cli.txt" "$receipts/cargo-tree-cli.txt"
    cmp "$refresh_root/generated-topology.json" "$receipts/package-topology.json"
fi

if [ "$mode" = pre-capture ]; then
    run_owned 300 refresh-contract-first "$digest_tool" refresh-contract \
        "$mode" "$generation" "$candidate_inventory" "$receipts" "$candidate_topology"
else
    run_owned 300 refresh-contract-first "$digest_tool" refresh-contract \
        "$mode" "$generation" "$candidate_inventory" "$capture_root" "$candidate_topology"
fi
cp "$candidate_inventory" "$refresh_root/first.yaml"
if [ "$mode" = pre-capture ]; then
    run_owned 300 refresh-contract-second "$digest_tool" refresh-contract \
        "$mode" "$generation" "$candidate_inventory" "$receipts" "$candidate_topology"
else
    run_owned 300 refresh-contract-second "$digest_tool" refresh-contract \
        "$mode" "$generation" "$candidate_inventory" "$capture_root" "$candidate_topology"
    cmp "$candidate_topology" "$receipts/package-topology.json"
fi
cmp "$refresh_root/first.yaml" "$candidate_inventory"
run_owned 1800 refresh-inventory-validator sh -c \
    'cd "$1"; AIPERF_PLUGIN_BASELINE_VALIDATION_ROOT="$2" cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory baseline_inventory_is_complete_and_self_authenticating -- --exact --test-threads=1' \
    sh "$repository/rust" "$source_root"
run_owned 1800 refresh-topology-validator sh -c \
    'cd "$1"; AIPERF_PLUGIN_BASELINE_VALIDATION_ROOT="$2" cargo test -p aiperf-e2e-tests --test plugin_baseline_inventory package_topology_dependency_claims_match_cargo_metadata -- --exact --test-threads=1' \
    sh "$repository/rust" "$source_root"
run_owned 300 refresh-publish "$digest_tool" publish-baseline \
    "$mode" "$source_root" "$repository"
cmp "$candidate_inventory" "$inventory"
cmp "$candidate_topology" "$topology"
if [ "$mode" != pre-capture ]; then
    for compact_name in allocation-probe.json evidence-manifest.json bundle-locator.json; do
        cmp "$candidate_compact/$compact_name" \
            "$repository/artifacts/native-plugin-baseline/$compact_name"
    done
fi
printf 'baseline inventory %s fixed point: %s\n' "$mode" "$generation"
