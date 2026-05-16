#!/usr/bin/env bash
# Test the endpoint readiness checker end-to-end.
#
# 1. Start aiperf profile with --ready-check-timeout pointing at a port
#    where nothing is running yet.
# 2. Sleep while aiperf retries the readiness check.
# 3. Start the mock server — aiperf should detect it and proceed.
set -euo pipefail

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
URL="http://localhost:${PORT}"
READY_TIMEOUT=60
DELAY=15

cleanup() {
    echo "--- cleaning up ---"
    [[ -n "${AIPERF_PID:-}" ]] && kill "$AIPERF_PID" 2>/dev/null || true
    [[ -n "${MOCK_PID:-}" ]]   && kill "$MOCK_PID"   2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Starting aiperf profile (ready-check-timeout=${READY_TIMEOUT}s) ==="
echo "    Endpoint: ${URL} (nothing listening yet)"

PYTHONUNBUFFERED=1 uv run aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url "${URL}" \
    --request-count 5 \
    --concurrency 1 \
    --ready-check-timeout "$READY_TIMEOUT" \
    --workers-max 1 \
    2>&1 | sed 's/^/[aiperf] /' &
AIPERF_PID=$!

echo "=== aiperf started (pid=${AIPERF_PID}), sleeping ${DELAY}s before starting mock server ==="
sleep "$DELAY"

echo "=== Starting mock server on port ${PORT} ==="
PYTHONUNBUFFERED=1 uv run aiperf-mock-server --port "$PORT" --fast --no-tokenizer \
    2>&1 | sed 's/^/[mock]   /' &
MOCK_PID=$!

echo "=== Waiting for aiperf to finish ==="
wait "$AIPERF_PID"
AIPERF_EXIT=$?

echo "=== aiperf exited with code ${AIPERF_EXIT} ==="
exit "$AIPERF_EXIT"
