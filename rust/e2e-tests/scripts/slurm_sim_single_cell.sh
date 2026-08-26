#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Simulate a 2-task SLURM allocation (1 controller + 1 cell) on loopback and drive a
# real native cellular run through `aiperf slurm run` against aiperf-mock-server. This
# is the degenerate single-cell case: `cell_count == 1`, which the cross-host launcher
# promotion engages (`AIPERF_CELL_LAUNCHER=slurm` fires the controller at `cells >= 1`)
# even though a same-host `--cells 1` run would stay a lone single process. The
# bootstrap contract is NOT hand-set: `aiperf slurm generate` mints this run's
# per-rank material and the exports below are adopted verbatim from its script.
# There is no SLURM here, so only the SLURM_* placement env is set by hand, exactly
# as srun would: rank 0 = controller, rank 1 = the sole cell, SLURM_JOB_NODELIST ->
# 127.0.0.1 (via the AIPERF_SLURM_CONTROLLER_HOST override so the coordinate is
# IPv4 loopback, not ::1).
#
# Each task's stderr is captured to its own file so a cell that dies is visible.
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="$ROOT/rust/target/debug/aiperf"
MOCK="$ROOT/rust/target/debug/aiperf-mock-server"
WORK="$(mktemp -d /tmp/aiperf-slurm-1cell-sim.XXXXXX)"
PORT=8971
CTRL_PORT=9541
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

echo "root=$ROOT work=$WORK"

# --- mock server -----------------------------------------------------------
"$MOCK" --port "$PORT" --host 127.0.0.1 --fast >"$WORK/mock.log" 2>&1 &
MOCK_PID=$!
cleanup() { kill "$MOCK_PID" 2>/dev/null; }
trap cleanup EXIT
for _ in $(seq 1 50); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
  sleep 0.2
done

# --- generated job script: mints this run's per-rank bootstrap material ------
CONFIG="$WORK/benchmark.yaml"
printf 'benchmark: {}\n' >"$CONFIG"
"$BIN" slurm generate --config "$CONFIG" --cells 1 \
  --controller-port "$CTRL_PORT" --run-dir "$WORK/run" \
  --output "$WORK/job.sbatch" || { echo "slurm generate failed"; exit 1; }
echo "===== generated job.sbatch ====="; cat "$WORK/job.sbatch"
# Adopt the generated launcher/port/bootstrap exports verbatim, as srun would.
eval "$(grep '^export ' "$WORK/job.sbatch")"

# --- shared SLURM allocation env (2 tasks: rank 0 controller + rank 1 cell) --
COMMON_ENV=(
  "SLURM_JOB_ID=525252"
  "SLURM_NTASKS=2"
  "SLURM_JOB_NODELIST=127.0.0.1"
  "AIPERF_SLURM_CONTROLLER_HOST=127.0.0.1"
)

PROFILE_ARGS="--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url http://127.0.0.1:$PORT --endpoint-type chat \
  --request-count 20 --concurrency 4 --random-seed 42 \
  --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8 --output-tokens-stddev 0 \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --artifact-dir $WORK/artifacts --ui simple"

# --- launch the sole cell task (rank 1) first so it is dialing when rank 0 binds
env "${COMMON_ENV[@]}" SLURM_PROCID=1 SLURM_NODEID=1 \
  "$BIN" slurm run >"$WORK/cell-1.log" 2>&1 &
CELL_1_PID=$!

# --- launch controller (rank 0) --------------------------------------------
env "${COMMON_ENV[@]}" SLURM_PROCID=0 SLURM_NODEID=0 \
  "$BIN" slurm run $PROFILE_ARGS >"$WORK/controller.log" 2>&1 &
CTRL_PID=$!

wait "$CTRL_PID"; CTRL_RC=$?
echo "controller exit=$CTRL_RC"

echo "===== controller.log (tail) ====="; tail -n 25 "$WORK/controller.log"
echo "===== cell-1.log (tail) ====="; tail -n 20 "$WORK/cell-1.log"

REPORT="$WORK/artifacts/native-v2.json"
if [ -f "$REPORT" ]; then
  echo "===== REPORT FOUND ====="
else
  echo "===== NO REPORT at $REPORT ====="
fi
echo "WORKDIR=$WORK"
exit $CTRL_RC
