#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark a KServe v2 gRPC embedding endpoint (a non-LLM Triton python-backend
# embedder) with AIPerf. Measures request latency + throughput (no TTFT/ITL/token
# metrics — meaningless for embeddings), the classic-Triton-vs-Dynamo comparison.
#
# Two modes:
#   (default) spin up the aiperf-mock-server as an FP32 embedder and run against it.
#   --target grpc://HOST:PORT   run against a real Triton/Dynamo gRPC frontend instead.
#
# Runner:
#   (default) run `aiperf profile` from the container image ($IMAGE).
#   --local   run the host-installed `aiperf` (pip/maturin) instead of Docker.
#
# Usage:
#   ./run-embeddings-e2e.sh                          # mock + image
#   ./run-embeddings-e2e.sh --local                  # mock + host aiperf
#   ./run-embeddings-e2e.sh --duration 60s --concurrency 64
#   ./run-embeddings-e2e.sh --target grpc://1.2.3.4:8001 --model clip-l14 \
#       --input query --output text_embeddings --duration 2m --concurrency 64
#
# Time-based by default: --duration (30, 30s, 2m) --grace (drain window)
#   --concurrency (in-flight cap) --workers (thread-per-core dispatch threads;
#   defaults to nproc — the single biggest throughput lever) --pool (synthetic
#   prompts, recycled). --full = all metrics.
# Env overrides: IMAGE, DIM, TOKENIZER, ARTIFACT_DIR.
set -euo pipefail

# ---- defaults -------------------------------------------------------------
IMAGE="${IMAGE:-nvcr.io/nvidian/dynamo-dev/aiperf:rust-emb-20260715-214809-10b4b4056}"
TARGET=""                         # empty => start a local mock
MODEL="clip-l14"
INPUT_NAME="query"                # Triton input tensor (TYPE_STRING)
OUTPUT_NAME="text_embeddings"     # Triton output tensor (TYPE_FP32)
DURATION="30s"                    # time-based run length (e.g. 30, 30s, 2m)
GRACE="5s"                        # wait for in-flight requests after duration
CONCURRENCY=512
WORKERS="$(nproc 2>/dev/null || echo 8)"   # thread-per-core: dispatch threads (each its own runtime+channel)
POOL=5000                         # synthetic prompt pool (recycled over the duration)
DIM="${DIM:-768}"                 # mock embedding width (mock mode only)
TOKENIZER="${TOKENIZER:-cl100k_base}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/tmp/aiperf-emb/artifacts}"
USE_LOCAL=0
SHOW_FULL=0                       # --full => print AIPerf's entire metric table

# ---- args -----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)      TARGET="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --input)       INPUT_NAME="$2"; shift 2 ;;
    --output)      OUTPUT_NAME="$2"; shift 2 ;;
    --duration)    DURATION="$2"; shift 2 ;;
    --grace)       GRACE="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --pool)        POOL="$2"; shift 2 ;;
    --dim)         DIM="$2"; shift 2 ;;
    --tokenizer)   TOKENIZER="$2"; shift 2 ;;
    --image)       IMAGE="$2"; shift 2 ;;
    --local)       USE_LOCAL=1; shift ;;
    --full)        SHOW_FULL=1; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

WORKDIR="$(dirname "$ARTIFACT_DIR")"
mkdir -p "$ARTIFACT_DIR"
MOCK_PID=""
cleanup() { [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---- mock (only when no external --target given) --------------------------
if [[ -z "$TARGET" ]]; then
  TARGET="grpc://127.0.0.1:8001"
  MOCK_BIN="./rust/target/release/aiperf-mock-server"
  echo ">> starting FP32 embedding mock on gRPC :8001 (dim=$DIM)"
  if [[ -x "$MOCK_BIN" ]]; then
    "$MOCK_BIN" --fast --grpc-port 8001 --grpc-embedding-dim "$DIM" >/tmp/aiperf-emb-mock.log 2>&1 &
  else
    cargo run --release -p aiperf-mock-server -- \
      --fast --grpc-port 8001 --grpc-embedding-dim "$DIM" >/tmp/aiperf-emb-mock.log 2>&1 &
  fi
  MOCK_PID=$!
  # wait for the gRPC listener
  for _ in $(seq 1 60); do
    if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':8001'; then break; fi
    sleep 1
  done
  (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':8001' \
    || { echo "mock failed to listen on :8001 — see /tmp/aiperf-emb-mock.log" >&2; exit 1; }
fi

# ---- config ---------------------------------------------------------------
CFG="$WORKDIR/emb.yaml"
cat > "$CFG" <<YAML
schemaVersion: "2.0"
benchmark:
  models: [${MODEL}]
  tokenizer:
    name: ${TOKENIZER}
  endpoint:
    urls: ["${TARGET}"]
    type: kserve_v2_embeddings
    streaming: false
    waitForModelTimeout: 0.0
    extra:
      v2_input_name: ${INPUT_NAME}
      v2_output_name: ${OUTPUT_NAME}
  dataset:
    type: synthetic
    entries: ${POOL}
    prompts: { isl: 32, osl: 16 }
  phases:
    - { name: profiling, type: concurrency, duration: ${DURATION}, grace_period: ${GRACE}, concurrency: ${CONCURRENCY} }
  gpuTelemetry: { enabled: false }
  serverMetrics: { enabled: false }
  transport: { type: grpc }
  runtime: { ui: none, workers: ${WORKERS} }
YAML

echo ">> target=$TARGET model=$MODEL in=$INPUT_NAME out=$OUTPUT_NAME duration=$DURATION grace=$GRACE concurrency=$CONCURRENCY workers=$WORKERS"

# ---- run ------------------------------------------------------------------
if [[ "$USE_LOCAL" -eq 1 ]]; then
  aiperf profile --config "$CFG" --artifact-dir "$ARTIFACT_DIR"
else
  docker run --rm --network host -v "$WORKDIR":/work "$IMAGE" \
    "aiperf profile --config /work/$(basename "$CFG") --artifact-dir /work/$(basename "$ARTIFACT_DIR")"
fi

# ---- results table --------------------------------------------------------
# AIPerf writes its full metric table to profile_export_console.txt regardless of
# `runtime.ui`. For an embedder that whole table is mostly token/decode/prefill
# rows that are always 0, so by default we render only the metrics that mean
# something for a non-tokenizing model. Pass --full for AIPerf's complete table.
CONSOLE="$ARTIFACT_DIR/profile_export_console.txt"
JSON="$ARTIFACT_DIR/profile_export_aiperf.json"
echo
if [[ "$SHOW_FULL" -eq 1 ]]; then
  [[ -s "$CONSOLE" ]] && cat "$CONSOLE" || echo ">> no console table at $CONSOLE (run may have failed)"
elif [[ -s "$JSON" ]]; then
  python3 - "$JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
# Only the metrics meaningful for a non-tokenizing (embedding) endpoint.
WANT = [
    ("request_latency",        "Request Latency"),
    ("effective_latency",      "Effective Latency"),
    ("credit_to_start_latency","Scheduling Delay"),
    ("request_throughput",     "Request Throughput"),
    ("input_token_throughput", "Input Token Throughput"),
    ("input_sequence_length",  "Input Sequence Length"),
    ("effective_concurrency",  "Effective Concurrency"),
    ("benchmark_duration",     "Benchmark Duration"),
    ("completed_request_count","Completed Requests"),
    ("request_count",          "Request Count"),
    ("request_error_rate",     "Request Error Rate"),
]
cols = ["avg", "min", "max", "p50", "p90", "p99", "std"]
def fmt(v):
    if v is None: return "—"
    return f"{v:,.2f}" if abs(v) >= 1000 else f"{v:.2f}"
rows = []
for key, label in WANT:
    m = d.get(key)
    if not isinstance(m, dict): continue
    unit = m.get("unit", "")
    rows.append([f"{label} ({unit})"] + [fmt(m.get(c)) for c in cols])
w0 = max(len(r[0]) for r in rows)
wc = 12
head = "Metric".ljust(w0) + "".join(c.rjust(wc) for c in cols)
print("  NVIDIA AIPerf — embedding endpoint")
print("  " + head)
print("  " + "-" * len(head))
for r in rows:
    print("  " + r[0].ljust(w0) + "".join(c.rjust(wc) for c in r[1:]))
print("\n  (token/decode/prefill metrics omitted — always 0 for a non-tokenizing model; use --full for all)")
PY
else
  echo ">> no results at $JSON (run may have failed)"
fi
echo ">> artifacts: $ARTIFACT_DIR  (profile_export_aiperf.{json,csv}, profile_export.jsonl)"
