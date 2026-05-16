#!/usr/bin/env bash
# Run a 30-worker 5-minute benchmark against mock server in Kind cluster.
# Usage: ./dev/run-30w-bench.sh <run_number>
set -euo pipefail

RUN=${1:?Usage: run-30w-bench.sh <run_number>}
LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run-${RUN}.log"

echo "=== Run $RUN starting at $(date -Iseconds) ===" | tee "$LOG"

cd "$(dirname "$0")/.."

# Deploy
uv run aiperf kube profile \
  --model-names mock-model \
  --image aiperf:dev \
  --url "http://aiperf-mock-server.default.svc.cluster.local:8000/v1" \
  --endpoint-type chat \
  --concurrency 1000 \
  --benchmark-duration 300 \
  --benchmark-grace-period 30 \
  --warmup-request-count 30 \
  --tokenizer gpt2 \
  --image-pull-policy Never \
  --workers-max 30 \
  --skip-endpoint-check \
  --skip-preflight \
  -d -v 2>&1 | tee -a "$LOG"

# Extract job-id and namespace from output
JOB_ID=$(grep -oP 'Job ID:\s+\K\w+' "$LOG" | head -1 || true)
NS="aiperf-${JOB_ID}"

if [ -z "$JOB_ID" ]; then
  echo "ERROR: Could not extract job ID" | tee -a "$LOG"
  exit 1
fi

echo "Job ID: $JOB_ID, Namespace: $NS" | tee -a "$LOG"

# Wait for pods to be ready (up to 3 minutes)
echo "Waiting for pods to be ready..." | tee -a "$LOG"
for i in $(seq 1 36); do
  READY=$(kubectl --context kind-aiperf get pods -n "$NS" --no-headers 2>/dev/null | grep -c "Running" || true)
  TOTAL=$(kubectl --context kind-aiperf get pods -n "$NS" --no-headers 2>/dev/null | wc -l || true)
  echo "  Pods: $READY/$TOTAL running (attempt $i/36)" | tee -a "$LOG"
  if [ "$READY" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    break
  fi
  sleep 5
done

# Monitor controller logs until completion or timeout (7 minutes)
echo "Monitoring controller logs..." | tee -a "$LOG"
CONTROLLER_POD=$(kubectl --context kind-aiperf get pods -n "$NS" -l job-name="${NS}-controller-0" --no-headers 2>/dev/null | awk '{print $1}' | head -1)

if [ -z "$CONTROLLER_POD" ]; then
  echo "ERROR: Could not find controller pod" | tee -a "$LOG"
  kubectl --context kind-aiperf get pods -n "$NS" 2>&1 | tee -a "$LOG"
  exit 1
fi

# Stream controller logs with timeout
timeout 480 kubectl --context kind-aiperf logs -f "$CONTROLLER_POD" -n "$NS" 2>&1 | tee -a "$LOG" || true

# Check final status
echo "" | tee -a "$LOG"
echo "=== Final pod status ===" | tee -a "$LOG"
kubectl --context kind-aiperf get pods -n "$NS" --no-headers 2>/dev/null | tee -a "$LOG" || true

# Check for success markers
if grep -q "Benchmark completed" "$LOG" 2>/dev/null || grep -q "Waiting for API subprocess" "$LOG" 2>/dev/null; then
  echo "=== Run $RUN: SUCCESS ===" | tee -a "$LOG"
  EXIT=0
else
  echo "=== Run $RUN: FAILED ===" | tee -a "$LOG"
  # Dump worker logs on failure
  echo "=== Worker pod logs ===" >> "$LOG"
  for pod in $(kubectl --context kind-aiperf get pods -n "$NS" -l job-name="${NS}-workers-0" --no-headers 2>/dev/null | awk '{print $1}'); do
    echo "--- $pod ---" >> "$LOG"
    kubectl --context kind-aiperf logs "$pod" -n "$NS" --tail=50 2>&1 >> "$LOG" || true
  done
  EXIT=1
fi

# Cleanup namespace
kubectl --context kind-aiperf delete namespace "$NS" --wait=false 2>/dev/null || true

echo "=== Run $RUN finished at $(date -Iseconds) ===" | tee -a "$LOG"
exit $EXIT
