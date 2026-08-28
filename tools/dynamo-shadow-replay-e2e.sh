#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end harness for the streaming Dynamo shadow-replay pipeline.
#
# Stands up a real S3 server (MinIO), publishes the local Dynamo request-trace
# corpus into a bucket, runs `aiperf profile` with a Config-v2 `dataset_streams`
# + `shadow_replay` selection reading that bucket, and points the resulting
# endpoint traffic at a local `aiperf-mock-server`.
#
# Usage:
#   tools/dynamo-shadow-replay-e2e.sh [--no-build]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MINIO_BIN="/tmp/minio"
MINIO_DATA="/tmp/minio-data"
MINIO_HOST="127.0.0.1"
MINIO_PORT="19000"
MINIO_ENDPOINT="http://${MINIO_HOST}:${MINIO_PORT}"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
BUCKET="dynamo-traces"
BUCKET_PREFIX="traces/"

MOCK_HOST="127.0.0.1"
MOCK_PORT="18090"
MOCK_URL="http://${MOCK_HOST}:${MOCK_PORT}"

TRACE_SRC="${HOME}/.aiperf/datasets/dynamo-request-traces/request-trace/v1"
CONFIG_PATH="/tmp/dynamo-shadow-replay-e2e.yaml"
OUT_DIR="/tmp/dynamo-shadow-e2e-out"
BUILD_LOG="/tmp/aiperf-streaming-build.log"
MINIO_LOG="/tmp/minio-e2e.log"
MOCK_LOG="/tmp/aiperf-mock-server-e2e.log"

AIPERF_BIN="${REPO_ROOT}/rust/target/debug/aiperf"
MOCK_BIN="${REPO_ROOT}/rust/target/debug/aiperf-mock-server"

DO_BUILD=1
for arg in "$@"; do
  case "${arg}" in
    --no-build) DO_BUILD=0 ;;
    *)
      echo "[e2e] unknown argument: ${arg}" >&2
      echo "[e2e] usage: $0 [--no-build]" >&2
      exit 2
      ;;
  esac
done

MINIO_PID=""
MOCK_PID=""

log() { echo "[e2e] $*"; }
die() { echo "[e2e] FAIL: $*" >&2; exit 1; }

cleanup() {
  local status=$?
  set +e
  if [[ -n "${MOCK_PID}" ]] && kill -0 "${MOCK_PID}" 2>/dev/null; then
    log "stopping aiperf-mock-server (pid ${MOCK_PID})"
    kill "${MOCK_PID}" 2>/dev/null
    wait "${MOCK_PID}" 2>/dev/null
  fi
  if [[ -n "${MINIO_PID}" ]] && kill -0 "${MINIO_PID}" 2>/dev/null; then
    log "stopping MinIO (pid ${MINIO_PID})"
    kill "${MINIO_PID}" 2>/dev/null
    wait "${MINIO_PID}" 2>/dev/null
  fi
  rm -rf "${MINIO_DATA}" "${OUT_DIR}"
  exit "${status}"
}
trap cleanup EXIT

# Poll `cmd` once a second until it succeeds, up to 30 seconds.
wait_for() {
  local label="$1"
  shift
  local waited=0
  while (( waited < 30 )); do
    if "$@" >/dev/null 2>&1; then
      log "${label} is ready after ${waited}s"
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  die "${label} did not become ready within 30s"
}

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------

if [[ ! -x "${MINIO_BIN}" ]]; then
  cat >&2 <<EOF
[e2e] FAIL: MinIO server binary not found or not executable at ${MINIO_BIN}

Download it with:

  curl -fsSL https://dl.min.io/server/minio/release/linux-amd64/minio -o ${MINIO_BIN}
  chmod +x ${MINIO_BIN}
EOF
  exit 1
fi

S3_CLI=""
if command -v aws >/dev/null 2>&1; then
  S3_CLI="aws"
elif command -v mc >/dev/null 2>&1; then
  S3_CLI="mc"
else
  cat >&2 <<'EOF'
[e2e] FAIL: neither the `aws` CLI nor the MinIO client `mc` is available.

Install one of:

  pip install awscli
  curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /tmp/mc && chmod +x /tmp/mc && export PATH=/tmp:$PATH
EOF
  exit 1
fi
log "using ${S3_CLI} for bucket creation and upload"

if [[ ! -d "${TRACE_SRC}" ]]; then
  die "Dynamo trace corpus not found at ${TRACE_SRC}"
fi
TRACE_COUNT="$(find "${TRACE_SRC}" -type f | wc -l)"
if (( TRACE_COUNT == 0 )); then
  die "no trace files found under ${TRACE_SRC}; populate the corpus before running this harness"
fi
log "found ${TRACE_COUNT} trace file(s) under ${TRACE_SRC}"

# ---------------------------------------------------------------------------
# 1. Start MinIO
# ---------------------------------------------------------------------------

log "starting MinIO on ${MINIO_ENDPOINT} (data dir ${MINIO_DATA})"
rm -rf "${MINIO_DATA}"
mkdir -p "${MINIO_DATA}"
MINIO_ROOT_USER="${MINIO_ACCESS_KEY}" \
MINIO_ROOT_PASSWORD="${MINIO_SECRET_KEY}" \
  "${MINIO_BIN}" server "${MINIO_DATA}" \
  --address "${MINIO_HOST}:${MINIO_PORT}" \
  >"${MINIO_LOG}" 2>&1 &
MINIO_PID=$!
log "MinIO pid ${MINIO_PID}, log ${MINIO_LOG}"

sleep 1
kill -0 "${MINIO_PID}" 2>/dev/null || die "MinIO exited immediately; see ${MINIO_LOG}"

wait_for "MinIO health" curl -fsS "${MINIO_ENDPOINT}/minio/health/live"

# ---------------------------------------------------------------------------
# 2. Create bucket and upload traces
# ---------------------------------------------------------------------------

export AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}"
export AWS_DEFAULT_REGION="us-east-1"
export AWS_REGION="us-east-1"
# A forward proxy would swallow loopback S3 traffic.
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

if [[ "${S3_CLI}" == "aws" ]]; then
  log "creating bucket s3://${BUCKET}"
  aws s3 --endpoint-url "${MINIO_ENDPOINT}" mb "s3://${BUCKET}" \
    || log "bucket s3://${BUCKET} already exists, continuing"

  log "uploading traces to s3://${BUCKET}/${BUCKET_PREFIX}"
  aws s3 --endpoint-url "${MINIO_ENDPOINT}" cp "${TRACE_SRC}/" \
    "s3://${BUCKET}/${BUCKET_PREFIX}" --recursive \
    || die "trace upload failed"

  UPLOADED="$(aws s3 --endpoint-url "${MINIO_ENDPOINT}" ls \
    "s3://${BUCKET}/${BUCKET_PREFIX}" --recursive | wc -l)"
else
  log "configuring mc alias 'e2eminio'"
  mc alias set e2eminio "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" \
    || die "mc alias set failed"

  log "creating bucket ${BUCKET}"
  mc mb --ignore-existing "e2eminio/${BUCKET}" || die "mc mb failed"

  log "uploading traces to e2eminio/${BUCKET}/${BUCKET_PREFIX}"
  mc cp --recursive "${TRACE_SRC}/" "e2eminio/${BUCKET}/${BUCKET_PREFIX}" \
    || die "trace upload failed"

  UPLOADED="$(mc ls --recursive "e2eminio/${BUCKET}/${BUCKET_PREFIX}" | wc -l)"
fi

if (( UPLOADED == 0 )); then
  die "no objects present under s3://${BUCKET}/${BUCKET_PREFIX} after upload"
fi
log "uploaded ${UPLOADED} object(s) to s3://${BUCKET}/${BUCKET_PREFIX}"

# ---------------------------------------------------------------------------
# 3. Build aiperf
# ---------------------------------------------------------------------------

if (( DO_BUILD == 1 )); then
  log "building aiperf-cli with --features streaming-s3 (log ${BUILD_LOG})"
  set +e
  (
    set -o pipefail
    cd "${REPO_ROOT}/rust" && cargo build -p aiperf-cli --features streaming-s3 2>&1 | tee "${BUILD_LOG}"
  )
  build_status=$?
  set -e
  if (( build_status != 0 )); then
    die "cargo build failed (exit ${build_status}); see ${BUILD_LOG}"
  fi
  log "building aiperf-mock-server"
  ( cd "${REPO_ROOT}/rust" && cargo build -p aiperf-mock-server ) \
    || die "aiperf-mock-server build failed"
else
  log "--no-build: skipping cargo build"
fi

[[ -x "${AIPERF_BIN}" ]] || die "aiperf binary missing at ${AIPERF_BIN} (drop --no-build?)"
[[ -x "${MOCK_BIN}" ]] || die "aiperf-mock-server binary missing at ${MOCK_BIN} (drop --no-build?)"

# ---------------------------------------------------------------------------
# 4. Start aiperf-mock-server
# ---------------------------------------------------------------------------

log "starting aiperf-mock-server on ${MOCK_URL} (log ${MOCK_LOG})"
"${MOCK_BIN}" --fast --port "${MOCK_PORT}" >"${MOCK_LOG}" 2>&1 &
MOCK_PID=$!
log "aiperf-mock-server pid ${MOCK_PID}"

sleep 1
kill -0 "${MOCK_PID}" 2>/dev/null || die "aiperf-mock-server exited immediately; see ${MOCK_LOG}"

# Any HTTP answer proves the listener is bound; the route need not be a 2xx.
wait_for "aiperf-mock-server" curl -sS -o /dev/null "${MOCK_URL}/v1/models"

# ---------------------------------------------------------------------------
# 5. Write the Config-v2 document
# ---------------------------------------------------------------------------

log "writing Config-v2 document ${CONFIG_PATH}"
cat >"${CONFIG_PATH}" <<EOF
schemaVersion: "2.0"
benchmark:
  model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  endpoint:
    type: chat
    url: "${MOCK_HOST}:${MOCK_PORT}"
    streaming: true
  dataset_streams:
    items:
      - id: dynamo-s3
        source:
          id: s3
          config:
            bucket: ${BUCKET}
            prefix: ${BUCKET_PREFIX}
            policy:
              mode: lossy_window
              max_keys: 1024
            page_max_keys: 1000
            max_pages_per_pass: 16
            max_unsealed_generations: 1024
            max_attempts: 3
            base_backoff_ns: 50000000
            max_backoff_ns: 2000000000
            poll_interval_ns: 500000000
            endpoint_url: ${MINIO_ENDPOINT}
            region: us-east-1
            force_path_style: true
        format:
          id: streaming_dynamo_trace
          config:
            max_record_bytes: 1048576
            max_chunk_bytes: 262144
            max_block_hashes_per_record: 4096
            max_block_size: 512
            max_input_length: 1048576
            emit_tool_events: false
        session_program:
          id: conversation
        limits:
          acquired_partitions: 4
          decoded_fragments: 256
          decoded_bytes: 8388608
          state_memory: 8388608
          state_disk: 0
  shadow_replay:
    stream: dynamo-s3
    actions:
      request:
        id: scheduled_request
        config:
          max_active_actions: 8
          is_streaming: true
    time:
      mode: relative
    ordering:
      watermark: source_order
      late: drop
    overload:
      mode: backpressure
    checkpoint:
      mode: none
  phases:
    type: concurrency
    requests: 10
    concurrency: 4
EOF

# ---------------------------------------------------------------------------
# 6. Run the profile
# ---------------------------------------------------------------------------

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

log "running aiperf profile"
set +e
"${AIPERF_BIN}" profile --config "${CONFIG_PATH}" --artifact-dir "${OUT_DIR}"
profile_status=$?
set -e
if (( profile_status != 0 )); then
  die "aiperf profile exited ${profile_status}"
fi

# ---------------------------------------------------------------------------
# 7. Verify output
# ---------------------------------------------------------------------------

log "verifying artifacts under ${OUT_DIR}"
REPORT="$(find "${OUT_DIR}" -type f -name '*_aiperf.json' | sort | head -1)"
if [[ -z "${REPORT}" ]]; then
  die "no *_aiperf.json artifact found under ${OUT_DIR}"
fi
log "inspecting ${REPORT}"

python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert 'requests' in d or 'summary' in d, f'unexpected keys: {list(d)[:5]}'" "${REPORT}" \
  || die "report ${REPORT} did not contain the expected top-level keys"

log "PASS: shadow replay read ${UPLOADED} S3 object(s) and produced ${REPORT}"
