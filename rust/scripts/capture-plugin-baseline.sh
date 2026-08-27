#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

usage() {
    echo "usage: $0 BASELINE_SOURCE OUTPUT_ROOT TARGET_ROOT" >&2
    exit 64
}

require_evidence_output_root() {
    validated_output_root=$1
    while [ "${validated_output_root%/}" != "$validated_output_root" ]; do
        validated_output_root=${validated_output_root%/}
    done
    [ "${validated_output_root##*/}" = evidence ] || {
        echo "OUTPUT_ROOT basename must be \`evidence\`: $1" >&2
        return 65
    }
}

case "${1:-}" in
    --output-root-validation-self-test)
        [ "$#" -eq 2 ] || usage
        require_evidence_output_root "$2"
        exit 0
        ;;
    --ownership-self-test|--post-seal-failure-self-test|--stdin-self-test|\
        --stdin-timeout-self-test|--stdin-signal-self-test|--bundle-verification-self-test)
        ;;
    *)
        [ "$#" -eq 3 ] || usage
        require_evidence_output_root "$2"
        ;;
esac

harness_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$harness_root/rust/scripts/plugin-baseline-owned-command.sh"
baseline_revision=caa3ff6fcf20ffe36a7704abe16274bedadbb9fb
mock_pid=
owned_pid=
owned_stdin_file=
selftest_pidfile=
selftest_mode=0

if [ "${1:-}" = "--ownership-self-test" ]; then
    [ "$#" -eq 3 ] || usage
    selftest_mode=1
    capture_lock=$2
    selftest_pidfile=$3
    output_root=$(dirname "$capture_lock")
    failure_ledger=$output_root/capture-failures.txt
elif [ "${1:-}" = "--post-seal-failure-self-test" ]; then
    [ "$#" -eq 5 ] || usage
    selftest_mode=2
    capture_lock=$2
    selftest_pidfile=$3
    output_root=$4
    failure_ledger=$5
elif [ "${1:-}" = "--stdin-self-test" ]; then
    [ "$#" -eq 3 ] || usage
    selftest_mode=3
    capture_lock=$2
    selftest_output=$3
    output_root=$(dirname "$capture_lock")
    failure_ledger=$output_root/capture-failures.txt
elif [ "${1:-}" = "--stdin-timeout-self-test" ]; then
    [ "$#" -eq 2 ] || usage
    selftest_mode=4
    capture_lock=$2
    output_root=$(dirname "$capture_lock")
    failure_ledger=$output_root/capture-failures.txt
elif [ "${1:-}" = "--stdin-signal-self-test" ]; then
    [ "$#" -eq 3 ] || usage
    selftest_mode=5
    capture_lock=$2
    selftest_pidfile=$3
    output_root=$(dirname "$capture_lock")
    failure_ledger=$output_root/capture-failures.txt
elif [ "${1:-}" = "--bundle-verification-self-test" ]; then
    [ "$#" -eq 7 ] || usage
    selftest_mode=6
    capture_lock=$2
    digest_tool=$3
    selftest_bundle=$4
    selftest_manifest=$5
    selftest_extraction_root=$6
    selftest_verification_receipt=$7
    output_root=$(dirname "$capture_lock")
    failure_ledger=$output_root/capture-failures.txt
else
    [ "$#" -eq 3 ] || usage
    baseline_source=$1
    output_root=$2
    target_root=$3
    capture_lock=${AIPERF_PLUGIN_BASELINE_LOCK:-${CARGO_TARGET_DIR%/}/native-plugin-baseline/.transition-lock}
    failure_ledger=$(dirname "$output_root")/capture-failures.txt
fi

if [ "$selftest_mode" -eq 0 ] && [ "${AIPERF_CAPTURE_SANITIZED:-}" != 1 ]; then
    controlled_tmpdir=${AIPERF_PLUGIN_CAPTURE_TMPDIR:-$target_root/tmp}
    set -- env -i \
        "AIPERF_CAPTURE_SANITIZED=1" \
        "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS" \
        "CARGO_HOME=$CARGO_HOME" \
        "CARGO_INCREMENTAL=$CARGO_INCREMENTAL" \
        "CARGO_TARGET_DIR=$CARGO_TARGET_DIR" \
        "HOME=$HOME" \
        "LANG=C" \
        "LC_ALL=C" \
        "NO_PROXY=127.0.0.1,localhost" \
        "PATH=$PATH" \
        "RUSTUP_HOME=$RUSTUP_HOME" \
        "RUST_VERSION=$RUST_VERSION" \
        "AIPERF_PLUGIN_BASELINE_LOCK=$capture_lock" \
        "AIPERF_PLUGIN_CAPTURE_TMPDIR=$controlled_tmpdir" \
        "TMPDIR=$controlled_tmpdir"
    [ "${RUSTC_WRAPPER+x}" != x ] || set -- "$@" "RUSTC_WRAPPER=$RUSTC_WRAPPER"
    [ "${RUSTFLAGS+x}" != x ] || set -- "$@" "RUSTFLAGS=$RUSTFLAGS"
    [ "${CARGO_ENCODED_RUSTFLAGS+x}" != x ] || \
        set -- "$@" "CARGO_ENCODED_RUSTFLAGS=$CARGO_ENCODED_RUSTFLAGS"
    [ "${SCCACHE_DIR+x}" != x ] || set -- "$@" "SCCACHE_DIR=$SCCACHE_DIR"
    [ "${SCCACHE_ENDPOINT+x}" != x ] || set -- "$@" "SCCACHE_ENDPOINT=$SCCACHE_ENDPOINT"
    [ "${SCCACHE_CACHE_SIZE+x}" != x ] || \
        set -- "$@" "SCCACHE_CACHE_SIZE=$SCCACHE_CACHE_SIZE"
    exec "$@" sh "$harness_root/rust/scripts/capture-plugin-baseline.sh" \
        "$baseline_source" "$output_root" "$target_root"
fi

export AIPERF_PLUGIN_BASELINE_LOCK="$capture_lock"
acquire_baseline_lock

cleanup() {
    original_status=$?
    trap - EXIT
    teardown_status=0
    if [ -n "$owned_pid" ]; then
        terminate_owned_group "$owned_pid" active-child || teardown_status=74
    fi
    if [ -n "$mock_pid" ]; then
        terminate_owned_group "$mock_pid" mock-server || teardown_status=74
    fi
    if [ -n "$owned_stdin_file" ]; then
        rm -f -- "$owned_stdin_file" || teardown_status=74
        owned_stdin_file=
    fi
    if [ "$teardown_status" -eq 0 ]; then
        release_baseline_lock || teardown_status=74
    fi
    if [ "$original_status" -ne 0 ]; then
        exit "$original_status"
    fi
    exit "$teardown_status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

require_output() {
    output_contract_kind=$1
    output_contract_label=$2
    output_contract_path=$3
    case "$output_contract_kind" in
        file) test -f "$output_contract_path" ;;
        nonempty) test -s "$output_contract_path" ;;
        directory) test -d "$output_contract_path" ;;
        executable) test -x "$output_contract_path" ;;
        *) echo "unknown output contract kind: $output_contract_kind" >&2; return 70 ;;
    esac && return 0
    record_baseline_failure "output-contract-$output_contract_label" \
        "$output_contract_kind output is absent: $output_contract_path"
    echo "$output_contract_label did not produce required $output_contract_kind output $output_contract_path" >&2
    return 66
}

verify_completed_bundle() {
    completed_tool=$1
    completed_generation=$2
    completed_bundle=$3
    completed_manifest=$4
    completed_extraction_root=$5
    completed_verification_receipt=$6
    run_owned 1800 bundle-verify-and-receipt "$completed_tool" verify-staged-bundle \
        "$completed_generation" "$completed_bundle" "$completed_manifest" \
        "$completed_extraction_root" \
        "$completed_verification_receipt"
    require_output nonempty bundle-verification "$completed_verification_receipt"
}

if [ "$selftest_mode" -eq 6 ]; then
    verify_completed_bundle "$digest_tool" review1i "$selftest_bundle" \
        "$selftest_manifest" "$selftest_extraction_root" \
        "$selftest_verification_receipt"
    exit 0
fi

if [ "$selftest_mode" -eq 1 ]; then
    export AIPERF_CAPTURE_TERM_GRACE_SECONDS=1
    if run_owned 5 adversarial-child sh -c \
        'trap "" TERM; (trap "" TERM; while :; do sleep 1; done) & child=$!; echo "$child" >"$1"; exit 0' \
        sh "$selftest_pidfile"; then
        selftest_status=0
    else
        selftest_status=$?
    fi
    [ "$selftest_status" -eq 74 ] || exit 75
    exit 0
fi

if [ "$selftest_mode" -eq 2 ]; then
    export AIPERF_CAPTURE_TERM_GRACE_SECONDS=1
    mkdir -p "$output_root"
    printf '%s\n' sealed >"$output_root/marker.txt"
    chmod -R a-w "$output_root"
    run_owned 5 post-seal-helper sh -c \
        'trap "" TERM; (trap "" TERM; while :; do sleep 1; done) & child=$!; echo "$child" >"$1"; exit 42' \
        sh "$selftest_pidfile"
fi

if [ "$selftest_mode" -eq 3 ]; then
    run_owned_with_stdin 5 stdin-empty sh -c \
        'if IFS= read -r value; then exit 78; fi; printf "empty\n" >"$1"' sh \
        "$selftest_output.empty" <<'EOF'
EOF
    run_owned_with_stdin 5 stdin-contract sh -c \
        'IFS= read -r value; printf "%s\n" "$value" >"$1"' sh \
        "$selftest_output" <<'EOF'
owned-stdin-preserved
EOF
    run_owned_with_stdin 5 stdin-multiline sh -c \
        'cat >"$1"' sh "$selftest_output.multiline" <<'EOF'
first-line
second-line
EOF
    require_output nonempty stdin-empty "$selftest_output.empty"
    require_output nonempty stdin-contract "$selftest_output"
    require_output nonempty stdin-multiline "$selftest_output.multiline"
    grep -qx empty "$selftest_output.empty"
    grep -qx owned-stdin-preserved "$selftest_output"
    printf 'first-line\nsecond-line\n' | cmp -s - "$selftest_output.multiline"
    if find "$(dirname "$failure_ledger")" -maxdepth 1 -name '.capture-stdin.*' \
        -print -quit | grep -q .
    then
        echo "stdin spool file survived owned-command teardown" >&2
        exit 76
    fi
    exit 0
fi

if [ "$selftest_mode" -eq 4 ]; then
    export AIPERF_CAPTURE_TERM_GRACE_SECONDS=1
    run_owned_with_stdin 1 stdin-timeout sh -c \
        'IFS= read -r value; [ "$value" = timeout-input ]; trap "" TERM; while :; do sleep 1; done' <<'EOF'
timeout-input
EOF
fi

if [ "$selftest_mode" -eq 5 ]; then
    export AIPERF_CAPTURE_TERM_GRACE_SECONDS=1
    run_owned_with_stdin 60 stdin-signal sh -c \
        'IFS= read -r value; [ "$value" = signal-input ]; echo "$$" >"$1"; trap "" TERM; while :; do sleep 1; done' \
        sh "$selftest_pidfile" <<'EOF'
signal-input
EOF
fi

[ "$(git -C "$baseline_source" rev-parse HEAD)" = "$baseline_revision" ] || {
    echo "baseline source is not $baseline_revision" >&2
    exit 65
}
[ ! -e "$output_root" ] || {
    echo "output root already exists: $output_root" >&2
    exit 65
}
[ ! -e "$target_root" ] || {
    echo "target root already exists: $target_root" >&2
    exit 65
}
case "$target_root" in
    "${CARGO_TARGET_DIR%/}"/*) ;;
    *) echo "TARGET_ROOT must be a fresh CARGO_TARGET_DIR subdirectory" >&2; exit 65 ;;
esac

mkdir -p "$output_root/builds/artifacts" "$output_root/identity" \
    "$output_root/runtime" "$output_root/configs" "$output_root/probes" "$target_root" \
    "$AIPERF_PLUGIN_CAPTURE_TMPDIR"
case "$AIPERF_PLUGIN_CAPTURE_TMPDIR" in
    "$target_root"/*) ;;
    *) echo "capture TMPDIR must be owned by TARGET_ROOT" >&2; exit 65 ;;
esac
required_free_bytes=${AIPERF_CAPTURE_REQUIRED_FREE_BYTES:-75161927680}
available_free_bytes=$(require_free_bytes "$target_root" "$required_free_bytes")
TMPDIR=$AIPERF_PLUGIN_CAPTURE_TMPDIR
export TMPDIR
printf 'required_free_bytes=%s\navailable_free_bytes=%s\ntmpdir=%s\n' \
    "$required_free_bytes" "$available_free_bytes" "$AIPERF_PLUGIN_CAPTURE_TMPDIR" \
    >"$output_root/identity/storage-capacity.txt"

projection_list=$target_root/measurement-source-projection.txt
cp "$harness_root/rust/benchmarks/plugin-baseline-measurement-source-projection.txt" \
    "$projection_list"
probe_source=$target_root/probe-source
base_source=$target_root/base-source
base_archive=$target_root/baseline-source.tar
mkdir -p "$base_source" "$probe_source"
run_owned 300 git-archive git -C "$baseline_source" archive --format=tar \
    "$baseline_revision" >"$base_archive"
require_output nonempty git-archive "$base_archive"
run_owned 300 extract-base tar -xf "$base_archive" -C "$base_source"
require_output nonempty extract-base "$base_source/rust/Cargo.lock"
run_owned 300 extract-probe tar -xf "$base_archive" -C "$probe_source"
require_output nonempty extract-probe "$probe_source/rust/Cargo.lock"
normalize_source_modes() {
    run_owned 300 normalize-directories find "$1" -type d -exec chmod 0755 {} +
    run_owned 300 normalize-executables find "$1" -type f -perm /0111 \
        -exec chmod 0755 {} +
    run_owned 300 normalize-files find "$1" -type f ! -perm /0111 \
        -exec chmod 0644 {} +
}
normalize_source_modes "$base_source"
normalize_source_modes "$probe_source"
while IFS= read -r projected_path; do
    mkdir -p "$probe_source/$(dirname "$projected_path")"
    cp "$harness_root/$projected_path" "$probe_source/$projected_path"
done <"$projection_list"
normalize_source_modes "$probe_source"

tool_target=$target_root/evidence-tools
run_owned 1800 evidence-tool-build sh -c \
    'cd "$1"; CARGO_TARGET_DIR=$2 cargo build --locked -p aiperf-bench-tools --bin evidence_digest' \
    sh "$probe_source/rust" "$tool_target"
digest_tool=$tool_target/debug/evidence_digest
require_output executable evidence-tool-build "$digest_tool"

digest() {
    run_owned 300 digest "$digest_tool" "$1"
}

capture_identity() {
    identity=$output_root/identity
    cp "$base_archive" "$identity/source-tree.tar"
    cp "$base_source/rust/Cargo.lock" "$identity/baseline-Cargo.lock"
    cp "$probe_source/rust/Cargo.lock" "$identity/Cargo.lock"
    run_owned 300 effective-source-archive tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -C "$probe_source" -cf "$identity/effective-source-tree.tar" .
    require_output nonempty effective-source-archive "$identity/effective-source-tree.tar"
    run_owned 300 source-projection-diff sh -c \
        'git diff --no-index -- "$1" "$2" || [ "$?" -eq 1 ]' sh \
        "$base_source" "$probe_source" >"$identity/harness.patch"
    require_output nonempty source-projection-diff "$identity/harness.patch"
    cp "$probe_source/rust/scripts/capture-plugin-baseline.sh" \
        "$identity/capture-plugin-baseline.sh"
    cp "$probe_source/rust/scripts/plugin-baseline-owned-command.sh" \
        "$identity/plugin-baseline-owned-command.sh"
    cp "$probe_source/rust/bench-tools/src/bin/evidence_digest.rs" \
        "$identity/evidence_digest.rs"
    cp "$probe_source/rust/benchmarks/plugin-baseline-invalidations.tsv" \
        "$identity/invalidations.tsv"
    cp "$probe_source/rust/benchmarks/exporter-observable-policy.json" \
        "$identity/exporter-observable-policy.json"
    cp "$probe_source/rust/benchmarks/exporter-static-calibration-corpus.json" \
        "$identity/exporter-static-calibration-corpus.json"
    cp "$identity/invalidations.tsv" "$identity/invalidated-attempts.txt"
    cp "$projection_list" "$identity/measurement-source-projection.txt"
    run_owned 300 source-projection-archive tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -C "$probe_source" -cf "$identity/measurement-source-projection.tar" \
        -T "$identity/measurement-source-projection.txt"
    require_output nonempty source-projection-archive \
        "$identity/measurement-source-projection.tar"
    run_owned 600 cargo-metadata sh -c \
        'cd "$1"; cargo metadata --locked --format-version 1' sh \
        "$probe_source/rust" >"$target_root/cargo-metadata.raw.json"
    require_output nonempty cargo-metadata-raw "$target_root/cargo-metadata.raw.json"
    run_owned 300 cargo-metadata-normalize "$digest_tool" normalize-cargo-receipt \
        metadata "$probe_source" "$target_root/cargo-metadata.raw.json" \
        "$identity/cargo-metadata.json"
    require_output nonempty cargo-metadata "$identity/cargo-metadata.json"
    run_owned 600 cargo-tree-workspace sh -c \
        'cd "$1"; cargo tree --locked --workspace --edges normal,build --prefix depth' sh \
        "$probe_source/rust" >"$target_root/cargo-tree-workspace.raw.txt"
    require_output nonempty cargo-tree-workspace-raw \
        "$target_root/cargo-tree-workspace.raw.txt"
    run_owned 300 cargo-tree-workspace-normalize "$digest_tool" normalize-cargo-receipt \
        tree "$probe_source" "$target_root/cargo-tree-workspace.raw.txt" \
        "$identity/cargo-tree-workspace.txt"
    require_output nonempty cargo-tree-workspace "$identity/cargo-tree-workspace.txt"
    run_owned 600 cargo-tree-cli sh -c \
        'cd "$1"; cargo tree --locked -p aiperf-cli --edges normal,build --prefix depth' sh \
        "$probe_source/rust" >"$target_root/cargo-tree-cli.raw.txt"
    require_output nonempty cargo-tree-cli-raw "$target_root/cargo-tree-cli.raw.txt"
    run_owned 300 cargo-tree-cli-normalize "$digest_tool" normalize-cargo-receipt \
        tree "$probe_source" "$target_root/cargo-tree-cli.raw.txt" \
        "$identity/cargo-tree-cli.txt"
    require_output nonempty cargo-tree-cli "$identity/cargo-tree-cli.txt"
    rustc_identity="$(rustc --version);LLVM $(rustc -vV | sed -n 's/^LLVM version: //p')"
    run_owned 300 topology "$digest_tool" topology "$capture_generation" \
        "$baseline_revision" "$rustc_identity" \
        "$(rustc -vV | sed -n 's/^host: //p')" release "$identity/Cargo.lock" \
        "$identity/cargo-metadata.json" "$identity/cargo-tree-workspace.txt" \
        "$identity/cargo-tree-cli.txt" >"$identity/package-topology.json"
    require_output nonempty topology "$identity/package-topology.json"

    {
        echo "baseline_revision=$baseline_revision"
        echo "harness_revision=$baseline_revision plus exact measurement-source-projection.tar"
        rustc -Vv
        cargo -V
        echo "rustc_sysroot=$(rustc --print sysroot)"
        echo "rustc_target=$(rustc -vV | sed -n 's/^host: //p')"
        echo "cargo_profile=release"
        echo "python=$(python3 --version 2>&1)"
        echo "tini=$(tini --version 2>&1)"
        echo "kernel=$(uname -srvmo)"
        microcode=$(sed -n 's/^microcode[[:space:]]*: //p' /proc/cpuinfo | sed -n '1p')
        [ -n "$microcode" ] || microcode=unavailable:no-cpuinfo-microcode
        echo "microcode=$microcode"
        echo "allocator_provider=mimalloc 0.1.52; libmimalloc-sys 0.1.49; aiperf global allocator"
        echo "frequency_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable:no-cpufreq-interface)"
        echo "isolated_cpus=$(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo unavailable:no-isolated-cpus-interface)"
        echo "cpuset_effective=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo unavailable:no-cgroup-cpuset)"
        echo "mock_placement=taskset CPUs 0-3; client taskset CPUs 4-7 or 4-71; same paper-rig pod"
        echo "client_affinity_single=4-7"
        echo "client_affinity_concurrent=4-71"
        echo "mock_affinity=0-3"
        echo "firmware_vendor=$(cat /sys/class/dmi/id/bios_vendor 2>/dev/null || echo unavailable)"
        echo "firmware_version=$(cat /sys/class/dmi/id/bios_version 2>/dev/null || echo unavailable)"
        echo "firmware_date=$(cat /sys/class/dmi/id/bios_date 2>/dev/null || echo unavailable)"
    } >"$identity/environment.txt"
    env | LC_ALL=C sort >"$identity/admitted-environment.txt"
    lscpu --json >"$identity/lscpu.json"
    lscpu --extended >"$identity/lscpu-extended.txt"
    sed -n '/^processor[[:space:]]*: 0$/,/^$/p' /proc/cpuinfo >"$identity/cpu0.txt"
    cat /proc/meminfo >"$identity/meminfo.txt"
    cat /proc/self/status >"$identity/process-status.txt"
    find /sys/devices/system/node -maxdepth 2 -type f \
        \( -name cpulist -o -name meminfo \) -print -exec cat {} \; \
        >"$identity/numa-topology.txt" 2>&1
    digest "$identity/source-tree.tar" >"$identity/source-tree.digest"
    digest "$identity/effective-source-tree.tar" >"$identity/effective-source-tree.digest"
    digest "$identity/baseline-Cargo.lock" >"$identity/baseline-cargo-lock.digest"
    digest "$identity/Cargo.lock" >"$identity/cargo-lock.digest"
    digest "$identity/cargo-metadata.json" >"$identity/cargo-metadata.digest"
    digest "$identity/cargo-tree-workspace.txt" >"$identity/cargo-tree-workspace.digest"
    digest "$identity/cargo-tree-cli.txt" >"$identity/cargo-tree-cli.digest"
    digest "$identity/capture-plugin-baseline.sh" >"$identity/harness.digest"
    digest "$identity/evidence_digest.rs" >"$identity/digest-tool-source.digest"
    digest "$identity/measurement-source-projection.txt" \
        >"$identity/measurement-source-projection-list.digest"
    digest "$identity/measurement-source-projection.tar" \
        >"$identity/measurement-source-projection.digest"
}

assert_effective_source_unchanged() {
    identity=$output_root/identity
    after=$target_root/effective-source-tree-after.tar
    run_owned 300 effective-source-final-archive \
        tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -C "$probe_source" -cf "$after" .
    require_output nonempty effective-source-final-archive "$after"
    cmp "$identity/effective-source-tree.tar" "$after"
    cmp "$identity/Cargo.lock" "$probe_source/rust/Cargo.lock"
}

run_build() {
    name=$1
    command=$2
    artifact=$3
    build_target=$target_root/build-$name
    build_dir=$output_root/builds/$name
    mkdir -p "$build_target" "$build_dir"
    start=$(date +%s%N)
    run_owned 7200 "build-$name-clean" sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 sh -c "$3"' sh \
        "$probe_source/rust" "$build_target" "$command" \
        >"$build_dir/clean.log" 2>&1
    require_output nonempty "build-$name-clean" "$build_dir/clean.log"
    end=$(date +%s%N)
    clean_ns=$((end - start))
    start=$(date +%s%N)
    run_owned 7200 "build-$name-second" sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 sh -c "$3"' sh \
        "$probe_source/rust" "$build_target" "$command" \
        >"$build_dir/second.log" 2>&1
    require_output nonempty "build-$name-second" "$build_dir/second.log"
    end=$(date +%s%N)
    second_ns=$((end - start))
    require_output nonempty "build-$name-artifact" "$build_target/release/$artifact"
    cp "$build_target/release/$artifact" "$output_root/builds/artifacts/$name-$artifact"
    {
        echo "command=$command"
        echo "target_dir=$build_target"
        echo "first_build_kind=isolated_clean_target"
        echo "first_build_nanoseconds=$clean_ns"
        echo "second_build_nanoseconds=$second_ns"
        digest "$output_root/builds/artifacts/$name-$artifact"
        digest "$build_dir/clean.log"
        digest "$build_dir/second.log"
    } >"$build_dir/summary.txt"
}

capture_builds() {
    run_build default "cargo build --locked -p aiperf-cli --release" aiperf
    run_build engine "cargo build --locked -p aiperf-runtime --release --features engine" libaiperf_runtime.rlib
    run_build grpc "cargo build --locked -p aiperf-cli --release --features grpc" aiperf
    run_build parquet "cargo build --locked -p aiperf-cli --release --features parquet" aiperf
    run_build dynosim "cargo build --locked -p aiperf-cli --release --features dynosim" aiperf
    run_build full "cargo build --locked -p aiperf-cli --release --features full" aiperf

    mock_target=$target_root/mock-server
    mkdir -p "$mock_target"
    run_owned 7200 mock-server-build sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 cargo build --locked -p aiperf-mock-server --release' \
        sh "$probe_source/rust" "$mock_target" \
        >"$output_root/builds/mock-server.log" 2>&1
    require_output nonempty mock-server-build "$output_root/builds/mock-server.log"
    require_output executable mock-server-build "$mock_target/release/aiperf-mock-server"
    cp "$mock_target/release/aiperf-mock-server" \
        "$output_root/builds/artifacts/aiperf-mock-server"
    digest "$output_root/builds/artifacts/aiperf-mock-server" \
        >"$output_root/builds/mock-server.digest"
}

write_configs() {
    run_owned_with_stdin 60 write-configs python3 - "$output_root/configs" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

def config(name, endpoint, streaming, requests, concurrency, workers, transport="http", records=False):
    url = "grpc://127.0.0.1:18081" if transport == "grpc" else "http://127.0.0.1:18080"
    artifacts = "\n  artifacts: {records: [jsonl]}" if records else ""
    text = f'''schemaVersion: "2.0"
randomSeed: 20260826
benchmark:
  models: [mock-model]
  endpoint:
    urls: ["{url}"]
    type: {endpoint}
    streaming: {str(streaming).lower()}
    waitForModelTimeout: 0.0
  dataset:
    type: synthetic
    entries: 256
    prompts: {{isl: 32, osl: 32}}
  phases:
    - {{name: profiling, type: concurrency, requests: {requests}, concurrency: {concurrency}}}
  gpuTelemetry: {{enabled: false}}
  serverMetrics: {{enabled: false}}
  transport: {{type: {transport}}}{artifacts}
runtime: {{workers: {workers}, dispatch: global, ui: none}}
'''
    (root / f"{name}.yaml").write_text(text)

config("http-nonstream-c1", "chat", False, 1000, 1, 1)
config("http-nonstream-c64", "chat", False, 64000, 64, 1)
config("http-streaming-c1", "chat", True, 1000, 1, 1)
config("http-streaming-c64", "chat", True, 64000, 64, 1)
config("grpc-unary-c1", "kserve_v2_infer", False, 1000, 1, 1, "grpc")
config("grpc-unary-c64", "kserve_v2_infer", False, 64000, 64, 1, "grpc")
config("grpc-streaming-c1", "kserve_v2_vlm", True, 1000, 1, 1, "grpc")
config("grpc-streaming-c64", "kserve_v2_vlm", True, 64000, 64, 1, "grpc")
config("http-streaming-workers4", "chat", True, 64000, 64, 4)
config("otlp-disabled-capture", "chat", True, 64000, 64, 1)
config("otlp-enabled-capture", "chat", True, 64000, 64, 1)
PY
    for config_name in http-nonstream-c1 http-nonstream-c64 \
        http-streaming-c1 http-streaming-c64 grpc-unary-c1 grpc-unary-c64 \
        grpc-streaming-c1 grpc-streaming-c64 http-streaming-workers4 \
        otlp-disabled-capture otlp-enabled-capture
    do
        require_output nonempty write-configs "$output_root/configs/$config_name.yaml"
    done
}

run_scenario() {
    name=$1
    cores=$2
    export_level=$3
    expected_budget=$4
    shift 4
    scenario=$output_root/runtime/$name
    mkdir -p "$scenario/artifacts"
    run_owned_with_stdin 930 "runtime-$name" python3 - \
        "$scenario/resource.json" "$scenario/process.log" "$cores" \
        "$output_root/builds/artifacts/default-aiperf" profile \
        --config "$output_root/configs/$name.yaml" \
        --artifact-dir "$scenario/artifacts" --export-level "$export_level" "$@" <<'PY'
import json, resource, subprocess, sys, time

resource_path, log_path, cores, *command = sys.argv[1:]
before = resource.getrusage(resource.RUSAGE_CHILDREN)
started = time.perf_counter_ns()
with open(log_path, "wb") as log:
    completed = subprocess.Popen(["taskset", "-c", cores, *command], stdout=log,
                                 stderr=subprocess.STDOUT)
    timed_out = False
    try:
        exit_status = completed.wait(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out = True
        completed.terminate()
        try:
            completed.wait(timeout=10)
        except subprocess.TimeoutExpired:
            completed.kill()
            completed.wait()
        exit_status = 124
finished = time.perf_counter_ns()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
measurement = {
    "command": ["taskset", "-c", cores, *command],
    "exit_status": exit_status,
    "timed_out": timed_out,
    "end_to_end_nanoseconds": finished - started,
    "user_cpu_nanoseconds": round((after.ru_utime - before.ru_utime) * 1_000_000_000),
    "system_cpu_nanoseconds": round((after.ru_stime - before.ru_stime) * 1_000_000_000),
    "maximum_resident_set_kib": after.ru_maxrss,
}
with open(resource_path, "w") as output:
    json.dump(measurement, output, indent=2, sort_keys=True)
    output.write("\n")
raise SystemExit(exit_status)
PY
    require_output nonempty "runtime-$name-resource" "$scenario/resource.json"
    require_output file "runtime-$name-process-log" "$scenario/process.log"
    require_output nonempty "runtime-$name-report" \
        "$scenario/artifacts/profile_export_aiperf.json"
    run_owned_with_stdin 60 "runtime-report-$name" python3 - \
        "$name" "$scenario/resource.json" \
        "$scenario/artifacts/profile_export_aiperf.json" \
        "$expected_budget" <<'PY'
import json, math, sys
name, resource_path, path, expected_budget = sys.argv[1:]
expected_budget = int(expected_budget)
resource = json.load(open(resource_path))
if resource.get("exit_status") != 0 or resource.get("timed_out") is not False:
    raise SystemExit(f"{name}: invalid runtime process receipt: {resource}")
for field in (
    "command",
    "end_to_end_nanoseconds",
    "user_cpu_nanoseconds",
    "system_cpu_nanoseconds",
    "maximum_resident_set_kib",
):
    if field not in resource:
        raise SystemExit(f"{name}: runtime process receipt lacks {field}")
report = json.load(open(path))
duration = report["benchmark_duration"]["avg"]
if duration < 30:
    raise SystemExit(f"{name}: duration {duration} is below 30 seconds")
errors = report.get("error_summary", [])
if errors:
    raise SystemExit(f"{name}: errors: {errors}")
for metric in ("request_count", "completed_request_count"):
    observed = report[metric]
    for statistic in ("avg", "min", "max"):
        if observed[statistic] != expected_budget:
            raise SystemExit(
                f"{name}: {metric}.{statistic}={observed[statistic]} "
                f"does not equal budget {expected_budget}"
            )
if name.startswith("http-streaming") or name.startswith("otlp-") or name == "exporter-100k":
    chunks = report["http_req_chunks_received"]
    if chunks["min"] != 32 or chunks["max"] != 32:
        raise SystemExit(f"{name}: expected exactly 32 transport chunks, got {chunks}")
for metric in ("request_throughput", "output_token_throughput", "time_to_first_token"):
    value = report[metric]["avg"]
    if not math.isfinite(value):
        raise SystemExit(f"{name}: non-finite {metric}")
print(json.dumps({"name": name, "duration_seconds": duration}, sort_keys=True))
PY
    digest "$scenario/artifacts/profile_export_aiperf.json" >"$scenario/summary.digest"
    digest "$scenario/process.log" >"$scenario/process.digest"
}

capture_runtime() {
    write_configs
    mock=$output_root/builds/artifacts/aiperf-mock-server
    setsid taskset -c 0-3 "$mock" --host 127.0.0.1 --port 18080 --grpc-port 18081 \
        --workers 4 --no-tokenizer --fixed-output-tokens 30 --ttft 5 --itl 1 \
        --ttft-jitter-cv 0 --itl-jitter-cv 0 \
        >"$output_root/runtime/mock-server.log" 2>&1 &
    mock_pid=$!
    run_owned_with_stdin 60 mock-readiness python3 - <<'PY'
import socket, time
for port in (18080, 18081):
    for _ in range(400):
        try:
            with socket.create_connection(("127.0.0.1", port), .1):
                break
        except OSError:
            time.sleep(.05)
    else:
        raise SystemExit(f"mock port {port} did not become ready")
PY
    run_scenario http-nonstream-c1 4-7 summary 1000
    run_scenario http-nonstream-c64 4-71 summary 64000
    run_scenario http-streaming-c1 4-7 summary 1000
    run_scenario http-streaming-c64 4-71 summary 64000
    run_scenario grpc-unary-c1 4-7 summary 1000
    run_scenario grpc-unary-c64 4-71 summary 64000
    run_scenario grpc-streaming-c1 4-7 summary 1000
    run_scenario grpc-streaming-c64 4-71 summary 64000
    run_scenario http-streaming-workers4 4-71 summary 64000
    run_scenario otlp-disabled-capture 4-71 summary 64000
    run_scenario otlp-enabled-capture 4-71 summary 64000 \
        --otel-url http://127.0.0.1:18080/v1/metrics \
        --otel-resource-attributes benchmark.scenario=plugin-baseline
    terminate_owned_group "$mock_pid" mock-server
    mock_pid=
}

capture_probes() {
    exporter=$output_root/runtime/exporter-100k
    mkdir -p "$exporter"
    exporter_identity_blake3=$(digest "$output_root/identity/experiment-identity.json")
    exporter_build_artifact_blake3=$(digest \
        "$output_root/builds/artifacts/engine-libaiperf_runtime.rlib")
    exporter_build_receipt_blake3=$(digest "$output_root/builds/engine/summary.txt")
    run_owned 3600 exporter-probe env \
        "AIPERF_EXPORTER_PROBE_ROOT=$exporter" \
        "AIPERF_EXPORTER_EXPERIMENT_IDENTITY_BLAKE3=$exporter_identity_blake3" \
        "AIPERF_EXPORTER_BUILD_ARTIFACT_BLAKE3=$exporter_build_artifact_blake3" \
        "AIPERF_EXPORTER_BUILD_RECEIPT_BLAKE3=$exporter_build_receipt_blake3" \
        sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 taskset -c 4-71 cargo test --locked --release -p aiperf-runtime --features engine --lib engine::records::tests::exporter_capture_allocation_and_duration_baseline -- --exact --nocapture --test-threads=1' \
        sh "$probe_source/rust" "$target_root/probes" \
        >"$exporter/process.log" 2>&1
    require_output nonempty exporter-probe "$exporter/process.log"
    run_owned_with_stdin 60 exporter-observation python3 - \
        "$exporter/process.log" "$exporter/observation.json" <<'PY'
import json, sys

log_path, observation_path = sys.argv[1:]
prefix = "AIPERF_ALLOCATION_SAMPLE "
samples = [
    json.loads(line[len(prefix):])
    for line in open(log_path, encoding="utf-8")
    if line.startswith(prefix)
]
if len(samples) != 1 or samples[0].get("path") != "exporter_capture":
    raise SystemExit(f"expected exactly one exporter observation, found {samples!r}")
sample = samples[0]
if sample.get("iterations") != 1_600_000:
    raise SystemExit("exporter observation does not contain exactly one 1600000-row sample")
if "warmup_samples" in sample or "retained_samples" in sample:
    raise SystemExit("exporter command must emit one sample, not aggregate harness repetitions")
if sample.get("exporter_interval_nanoseconds", 0) < 30_000_000_000:
    raise SystemExit("exporter-only retained interval is shorter than 30 seconds")
receipts = sample.get("repetition_receipts")
if not isinstance(receipts, list) or len(receipts) != 16:
    raise SystemExit("exporter observation must contain exactly 16 repetition receipts")
required = {
    "schema_version", "experiment_identity_blake3", "attempt_ordinal", "scenario_id",
    "pair_id", "member", "repetition_ordinal", "corpus_blake3", "processed_records",
    "observable_kind", "raw_observable_blake3", "comparison_observable_blake3",
    "provenance_receipt_blake3", "active_duration_ns", "build_artifact_blake3",
    "build_receipt_blake3",
}
comparison = receipts[0].get("comparison_observable_blake3")
for ordinal, receipt in enumerate(receipts):
    if set(receipt) != required:
        raise SystemExit(f"exporter receipt {ordinal} field set mismatch")
    if receipt.get("repetition_ordinal") != ordinal or receipt.get("processed_records") != 100_000:
        raise SystemExit(f"exporter receipt {ordinal} schedule mismatch")
    if receipt.get("pair_id") != "task1-static-calibration" or receipt.get("member") != "static":
        raise SystemExit(f"exporter receipt {ordinal} calibration binding mismatch")
    if receipt.get("comparison_observable_blake3") != comparison:
        raise SystemExit(f"exporter receipt {ordinal} comparison observable mismatch")
with open(observation_path, "w", encoding="utf-8") as output:
    json.dump(sample, output, indent=2, sort_keys=True)
    output.write("\n")
PY
    require_output nonempty exporter-observation "$exporter/observation.json"
    digest "$exporter/process.log" >"$exporter/process.digest"
    digest "$exporter/observation.json" >"$exporter/summary.digest"

    run_owned 3600 allocation-probe sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 cargo test --locked --release -p aiperf-runtime --bench chat_dispatch_bench required_plugin_allocation_baselines -- --exact --nocapture --test-threads=1' \
        sh "$probe_source/rust" "$target_root/probes" \
        >"$output_root/probes/allocation-probes.log" 2>&1
    require_output nonempty allocation-probe "$output_root/probes/allocation-probes.log"
    run_owned 3600 response-reduction-probe sh -c \
        'cd "$1"; CARGO_TARGET_DIR=$2 cargo test --locked --release -p aiperf-runtime --features engine --lib transport::reduce::tests::response_reduction_allocation_baseline -- --exact --nocapture --test-threads=1' \
        sh "$probe_source/rust" "$target_root/probes" \
        >>"$output_root/probes/allocation-probes.log" 2>&1
    require_output nonempty response-reduction-probe \
        "$output_root/probes/allocation-probes.log"
    digest "$output_root/probes/allocation-probes.log" \
        >"$output_root/probes/allocation-probes.digest"
}

capture_name=$(basename "$(dirname "$output_root")")
capture_generation=${capture_name#task1-}
capture_generation=${capture_generation%-final}

capture_identity
capture_builds
run_owned 300 experiment-identity "$digest_tool" experiment-identity \
    "$output_root/identity/experiment-identity.json" "$output_root/identity" \
    "$output_root/builds/artifacts/aiperf-mock-server" \
    "$output_root/builds/artifacts/default-aiperf" \
    "$output_root/builds/artifacts/engine-libaiperf_runtime.rlib" \
    "$output_root/builds/artifacts/grpc-aiperf" \
    "$output_root/builds/artifacts/parquet-aiperf" \
    "$output_root/builds/artifacts/dynosim-aiperf" \
    "$output_root/builds/artifacts/full-aiperf"
require_output nonempty experiment-identity \
    "$output_root/identity/experiment-identity.json"
capture_runtime
capture_probes
run_owned 300 measurement-results "$digest_tool" measurement-results \
    "$capture_generation" "$output_root" \
    "$output_root/identity/measurement-results.json"
require_output nonempty measurement-results \
    "$output_root/identity/measurement-results.json"
assert_effective_source_unchanged

manifest=$output_root/../evidence-manifest.json
release_tag=native-plugin-baseline-caa3ff6f-${capture_name#task1-}
bundle=$output_root/../aiperf-${release_tag}.tar.gz
locator=$output_root/../bundle-locator.json
run_owned 300 evidence-manifest "$digest_tool" manifest "$output_root" >"$manifest"
require_output nonempty evidence-manifest "$manifest"
run_owned 300 evidence-verify "$digest_tool" verify "$manifest" "$output_root"
chmod -R a-w "$output_root"
chmod a-w "$manifest"
run_owned 1800 evidence-bundle sh -c \
    'tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$1" -cf - "$2" "$3" | gzip -n' \
    sh "$(dirname "$output_root")" "$(basename "$output_root")" \
    "$(basename "$manifest")" >"$bundle"
require_output nonempty evidence-bundle "$bundle"
verification_root=$target_root/completed-bundle-verification
mkdir "$verification_root"
verify_completed_bundle "$digest_tool" "$capture_generation" "$bundle" "$manifest" \
    "$verification_root" "$output_root/../bundle-verification.json"
digest "$manifest" >"$manifest.digest"
digest "$bundle" >"$bundle.digest"
run_owned 300 locator "$digest_tool" locator \
    https://github.com/ajcasagrande/rust-native-plugin-lab \
    "$release_tag" "$bundle" "$manifest" "$locator"
require_output nonempty locator "$locator"
run_owned 300 locator-verify "$digest_tool" verify-locator "$locator"
echo "manifest=$manifest"
echo "bundle=$bundle"
echo "locator=$locator"
cat "$bundle.digest"
