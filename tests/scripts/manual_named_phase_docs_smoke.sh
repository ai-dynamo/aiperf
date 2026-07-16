#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Manual smoke checks for the named-phase/adaptive-scale docs touched by AIP-1004/1005/1006.
# Run this from the ai-dynamo-aiperf repo after port-forwarding a model endpoint, e.g.:
#   kubectl -n inguyen-ft port-forward svc/inguyen-qwen3-vllm 8000:8000
#   tests/scripts/manual_named_phase_docs_smoke.sh --profile
#
# By default this runs config validation/expansion only. Add --profile to send tiny requests.

set -euo pipefail

ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
AIPERF_CMD="${AIPERF_CMD:-uv run aiperf}"
WORKDIR="${WORKDIR:-$(mktemp -d /tmp/aiperf-docs-smoke.XXXXXX)}"
RUN_PROFILE=0
KEEP_WORKDIR=0

usage() {
  cat <<EOF
Usage: $0 [--profile] [--keep-workdir]

Environment:
  ENDPOINT_URL  OpenAI-compatible endpoint URL.
                Default: ${ENDPOINT_URL}
  MODEL         Model name to put in generated configs.
                Default: ${MODEL}
  AIPERF_CMD    Command prefix for aiperf.
                Default: ${AIPERF_CMD}
  WORKDIR       Directory for generated configs/artifacts.
                Default: mktemp under /tmp

Examples:
  ENDPOINT_URL=http://127.0.0.1:8000/v1/chat/completions \\
  MODEL=Qwen/Qwen3-0.6B \\
  $0

  $0 --profile --keep-workdir
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      RUN_PROFILE=1
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -d .git || ! -d src/aiperf ]]; then
  echo "Run this from the ai-dynamo-aiperf repo root." >&2
  exit 2
fi

mkdir -p "$WORKDIR"
if [[ "$KEEP_WORKDIR" -eq 0 ]]; then
  trap 'rm -rf "$WORKDIR"' EXIT
fi

run() {
  echo
  echo "+ $*"
  "$@"
}

run_aiperf() {
  echo
  echo "+ ${AIPERF_CMD} $*"
  # shellcheck disable=SC2086
  ${AIPERF_CMD} "$@"
}

write_yaml_config_doc_basic() {
  cat > "$WORKDIR/yaml-config-basic.yaml" <<YAML
schemaVersion: "2.0"

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 500
    prompts: {isl: 512, osl: 128}
  phases:
    - {name: warmup, kind: warmup, type: concurrency, concurrency: 8, requests: 50}
    - {name: profiling, kind: profiling, type: concurrency, requests: 500}
  artifacts:
    dir: ${WORKDIR}/artifacts/yaml-config-basic

sweep:
  type: grid
  parameters:
    concurrency: [8, 16]
    requests: [100]
YAML
}

write_yaml_config_minimal() {
  cat > "$WORKDIR/minimal.yaml" <<YAML
schemaVersion: "2.0"

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 16
    prompts: {isl: 32, osl: 8}
  phases: {type: concurrency, concurrency: 1, requests: 4}
YAML
}

write_named_phases_doc_example() {
  cat > "$WORKDIR/named-phases.yaml" <<YAML
schemaVersion: "2.0"

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 32
    prompts: {isl: 32, osl: 8}
  phases:
    - name: warmup
      kind: warmup
      type: concurrency
      requests: 2
      concurrency: 1
    - name: low_cancel_1
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
      cancellation: {rate: 5, delay: 0}
    - name: storm_1
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
      cancellation: {rate: 50, delay: 0}
    - name: recovery_1
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
      cancellation: {rate: 0, delay: 0}
YAML
}

write_adaptive_scale_doc_example() {
  cat > "$WORKDIR/adaptive-scale.yaml" <<YAML
schemaVersion: "2.0"

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 64
    prompts: {isl: 32, osl: 8}
  phases:
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 4
      duration: 5
      adaptive_scale:
        enabled: true
        control:
          variable: concurrency
          min: 1
          max: 4
        assessment_period: 1
        min_completed_requests: 1
        sustain_duration: 1
        strategy:
          type: ramp_until_fail
          step_policy: fixed_percent_step
          step_percent: 100
      sla:
        request_latency:
          p95:
            le: 10000
YAML
}

write_sweep_paths_doc_example() {
  cat > "$WORKDIR/sweep-paths.yaml" <<YAML
schemaVersion: "2.0"

sweep:
  type: grid
  parameters:
    phases.storm_1.cancellation.rate: [5, 50]
    phases.1.concurrency: [2, 4]
    phases.profiling.duration: [5]

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 16
    prompts: {isl: 32, osl: 8}
  phases:
    - name: warmup
      kind: warmup
      type: concurrency
      concurrency: 1
      requests: 2
    - name: low
      kind: profiling
      type: concurrency
      concurrency: 1
      requests: 4
      cancellation: {rate: 0, delay: 0}
    - name: storm_1
      kind: profiling
      type: concurrency
      concurrency: 1
      requests: 4
      cancellation: {rate: 0, delay: 0}
    - name: profiling
      kind: profiling
      type: concurrency
      concurrency: 1
      duration: 2
YAML
}

write_legacy_profiling_fallback_sweep() {
  cat > "$WORKDIR/legacy-profiling-fallback-sweep.yaml" <<YAML
schemaVersion: "2.0"

sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [2, 4]

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 16
    prompts: {isl: 32, osl: 8}
  phases:
    - name: storm_1
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
YAML
}

write_ambiguous_legacy_profiling_sweep() {
  cat > "$WORKDIR/ambiguous-profiling-sweep.yaml" <<YAML
schemaVersion: "2.0"

sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [2]

benchmark:
  model: ${MODEL}
  endpoint:
    url: ${ENDPOINT_URL}
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 16
    prompts: {isl: 32, osl: 8}
  phases:
    - name: low
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
    - name: storm
      kind: profiling
      type: concurrency
      requests: 4
      concurrency: 1
YAML
}

write_yaml_config_doc_basic
write_yaml_config_minimal
write_named_phases_doc_example
write_adaptive_scale_doc_example
write_sweep_paths_doc_example
write_legacy_profiling_fallback_sweep
write_ambiguous_legacy_profiling_sweep

echo "Generated doc smoke configs in: $WORKDIR"
echo "Endpoint: $ENDPOINT_URL"
echo "Model:    $MODEL"

# Commands copied/adapted from docs/tutorials/yaml-config.md:
run_aiperf config validate "$WORKDIR/yaml-config-basic.yaml"
run_aiperf config expand "$WORKDIR/yaml-config-basic.yaml"
run_aiperf config expand "$WORKDIR/yaml-config-basic.yaml" --full
run_aiperf config expand "$WORKDIR/yaml-config-basic.yaml" --index 0 --full

run_aiperf config validate "$WORKDIR/minimal.yaml"
run_aiperf config validate "$WORKDIR/named-phases.yaml"
run_aiperf config validate "$WORKDIR/adaptive-scale.yaml"
run_aiperf config validate "$WORKDIR/sweep-paths.yaml"
run_aiperf config expand "$WORKDIR/sweep-paths.yaml" --full
run_aiperf config validate "$WORKDIR/legacy-profiling-fallback-sweep.yaml"
run_aiperf config expand "$WORKDIR/legacy-profiling-fallback-sweep.yaml" --full

# Negative example for the documented ambiguous legacy phases.profiling.* behavior.
echo
if run_aiperf config validate "$WORKDIR/ambiguous-profiling-sweep.yaml"; then
  echo "Expected ambiguous phases.profiling.* validation to fail, but it passed." >&2
  exit 1
else
  echo "Ambiguous phases.profiling.* validation failed as expected."
fi

if [[ "$RUN_PROFILE" -eq 1 ]]; then
  echo
  echo "Running tiny profile examples against $ENDPOINT_URL"
  run_aiperf profile \
    --config "$WORKDIR/minimal.yaml" \
    --tokenizer builtin \
    --extra-inputs ignore_eos:true \
    --ui none \
    --output-artifact-dir "$WORKDIR/artifacts/minimal-profile"

  run_aiperf profile \
    --config "$WORKDIR/adaptive-scale.yaml" \
    --tokenizer builtin \
    --extra-inputs ignore_eos:true \
    --ui none \
    --output-artifact-dir "$WORKDIR/artifacts/adaptive-scale-profile"

  echo
  echo "Adaptive artifact files:"
  run find "$WORKDIR/artifacts/adaptive-scale-profile" -maxdepth 5 -type f | sort
fi

cat <<EOF

Doc smoke complete.
Generated files: $WORKDIR
EOF
if [[ "$KEEP_WORKDIR" -eq 0 ]]; then
  echo "Temporary workdir will be removed on exit. Re-run with --keep-workdir to inspect files."
fi
