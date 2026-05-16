---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Monitoring and Troubleshooting
---

# Monitoring and Troubleshooting

AIPerf provides several tools for monitoring running benchmarks, diagnosing problems, and retrieving logs. This guide covers how to use each one.

---

## Watching a Running Benchmark

The `watch` command gives you a live dashboard of your benchmark with status, metrics, pod health, and self-debugging diagnostics:

```bash
aiperf kube watch
```

This auto-detects the most recently deployed benchmark and starts a Rich TUI that refreshes every 2 seconds. It shows:

- Job phase and progress
- Request throughput and latency
- Pod status across all worker and controller pods
- Automated diagnosis (stall detection, error rate warnings, health classification)
- Kubernetes events

### Watch a Specific Job

```bash
aiperf kube watch my-benchmark
```

### Watch All Running Benchmarks

```bash
aiperf kube watch --all
```

### JSON Output for Automation

For CI pipelines or AI agents, use NDJSON output (one JSON object per line per refresh):

```bash
aiperf kube watch --output json
```

Include live log lines in the JSON stream:

```bash
aiperf kube watch --output json --follow-logs
```

### Adjust Refresh Rate

```bash
aiperf kube watch --interval 5
```

---

## Attaching to a Benchmark

The `attach` command connects to a running benchmark via port-forward and streams live progress updates via WebSocket:

```bash
aiperf kube attach
```

This is what `aiperf kube profile` does automatically after deploying. Use `attach` to reconnect after detaching or from a different terminal.

Attach to a specific job:

```bash
aiperf kube attach my-benchmark
```

Press **Ctrl+C** to cancel the benchmark. AIPerf sends a cancellation signal (`spec.cancel=true`) to the cluster, and the operator cleans up the resources.

---

## Listing Jobs

See all benchmark jobs and their status:

```bash
aiperf kube list
```

```
NAME                 NAMESPACE           PHASE      PROGRESS  AGE
qwen3-benchmark      aiperf-benchmarks   Running    67%       3m
llama-throughput      aiperf-benchmarks   Completed  100%      15m
mistral-test         aiperf-benchmarks   Failed     0%        20m
```

### Filter by Status

```bash
aiperf kube list --running
aiperf kube list --completed
aiperf kube list --failed
```

### Wide Output

Show additional columns like model name, endpoint, and error messages:

```bash
aiperf kube list --wide
```

### Live Refresh

Watch the list with automatic updates:

```bash
aiperf kube list --watch
aiperf kube list --watch --interval 10
```

---

## Reading Logs

Get logs from all pods associated with a benchmark:

```bash
aiperf kube logs
```

### Specific Job

```bash
aiperf kube logs my-benchmark
```

### Specific Container

Each benchmark pod has containers named `control-plane` (the controller) and `worker-pod-manager` (the worker processes). To see logs from a specific container:

```bash
aiperf kube logs --container control-plane
```

### Follow Logs in Real-Time

```bash
aiperf kube logs -f
```

### Last N Lines

```bash
aiperf kube logs --tail 100
```

### Save Logs to Files

Save all pod logs to a directory (one file per pod):

```bash
aiperf kube logs --output ./my-logs
```

---

## Debugging Failed Benchmarks

The `debug` command runs a one-shot diagnostic analysis of your deployment:

```bash
aiperf kube debug -n aiperf-benchmarks
```

It inspects:

- **Pod states** -- Identifies CrashLoopBackOff, OOMKilled, ImagePullBackOff, Unschedulable, and other problem states
- **Kubernetes events** -- Shows recent warning events with suggestions
- **Node resources** -- Reports CPU, memory, and GPU allocatable vs. capacity for each node
- **Container logs** -- With `--verbose`, fetches recent logs from problem pods

### Debug a Specific Job

```bash
aiperf kube debug --job-id my-benchmark
```

### Verbose Mode

Includes container logs from pods with problems:

```bash
aiperf kube debug --job-id my-benchmark --verbose
```

### All Namespaces

Scan every namespace that has AIPerf deployments:

```bash
aiperf kube debug --all-namespaces
```

### Sample Output

```
Diagnostic Report: aiperf-benchmarks

POD                              STATUS    RESTARTS  NODE       ISSUES
aiperf-bench-controller-0-0      Running   0         node-1     0
aiperf-bench-worker-0-0          Running   3         node-2     1

Problems Found
[aiperf-bench-worker-0-0] OOMKilled (container: worker)
  Suggestion: Container was killed due to out-of-memory. Increase memory limits.

Node Resources
NODE      READY  CPU      MEMORY      GPU  PRESSURE
node-1    Yes    8/16     32Gi/64Gi   2/4  -
node-2    Yes    4/8      8Gi/16Gi    1/2  MemoryPressure

Summary
Pods: 2 total, 2 running, 1 with issues
Warning events: 3
Nodes under pressure: node-2
```

---

## Pre-Flight Checks

Before deploying, validate that the cluster is ready:

```bash
aiperf kube preflight
```

This checks:

- Cluster connectivity and API version
- JobSet CRD installed
- RBAC permissions in the target namespace
- Resource quotas and node capacity
- Image accessibility
- Endpoint reachability (if specified)

### With Specific Parameters

```bash
aiperf kube preflight \
  --image nvcr.io/nvidia/aiperf:latest \
  --endpoint-url http://my-server:8000 \
  --workers 20 \
  --namespace my-benchmarks
```

### JSON Output

For CI/CD pipelines:

```bash
aiperf kube preflight -o json
```

Returns a structured JSON object with pass/fail/warn status for each check, suitable for automated gating.

---

## Retrieving Results

After a benchmark completes, get the results:

```bash
aiperf kube results
```

This downloads the full results package via the controller pod's API. The results include:

- `profile_export_aiperf.json` -- Summary metrics
- `request_records.jsonl` -- Per-request timing data
- Server metrics and other exported files

### From the Operator Storage

Even after pods are deleted, results are stored on the operator's PVC. This is the default:

```bash
aiperf kube results my-benchmark
```

### Summary Only

Download only the summary results (faster):

```bash
aiperf kube results --summary-only
```

### Direct From Pods

If you want to fetch results directly from the running benchmark pods (tries the controller API first, falls back to `kubectl cp`):

```bash
aiperf kube results --from-pods
```

### Shut Down After Download

Free up cluster resources by shutting down the API service after downloading (only takes effect with `--from-pods`):

```bash
aiperf kube results --from-pods --shutdown
```

### Save to Custom Directory

```bash
aiperf kube results --output ./my-results
```

---

## Common Issues and Solutions

### Pods Stuck in Pending

**Symptom:** `aiperf kube list` shows `Pending` and pods never start.

**Diagnosis:**
```bash
aiperf kube debug --job-id my-benchmark
```

**Common causes:**
- Insufficient resources (CPU, memory, GPU) -- check node capacity in the debug output
- Missing node selectors or tolerations -- pods may be targeting nodes that don't exist
- Kueue quota exhausted -- check your ClusterQueue capacity

### ImagePullBackOff

**Symptom:** Pods fail with `ImagePullBackOff`.

**Fix:** Verify the image exists and pull secrets are configured:
```bash
# Check preflight
aiperf kube preflight --image your-image:tag

# Add pull secrets
aiperf kube profile ... --image-pull-secrets my-registry-secret
```

### OOMKilled

**Symptom:** Worker pods restart with `OOMKilled` status.

**Fix:** Reduce concurrency per worker pod so each pod uses less memory:
```yaml
spec:
  connectionsPerWorker: 50    # reduce from default 100
```

Increasing per-pod memory requests/limits is done on the operator deployment via the `AIPERF_K8S_*` environment variables (see `values.yaml`).

### Benchmark Timeout

**Symptom:** Job transitions to `Failed` with timeout error.

**Fix:** Increase or disable the timeout:
```yaml
spec:
  timeoutSeconds: 3600    # 1 hour
  # or
  timeoutSeconds: 0       # no timeout
```

### Endpoint Unreachable

**Symptom:** Operator reports endpoint health check failure.

**Diagnosis:** Verify the Dynamo frontend is reachable from inside the cluster:
```bash
kubectl run curl-test --rm -it --image=curlimages/curl -- \
  curl -s http://dynamo-agg-frontend.dynamo-server.svc:8000/v1/models
```

**Fix:** Ensure the Dynamo deployment is healthy (`kubectl get pods -n dynamo-server`), and the URL in your config uses the correct frontend service DNS name: `http://{deploy-name}-frontend.{namespace}.svc:8000/v1`.

### Stale Namespaces

If old benchmark namespaces are accumulating, the watchdog will warn you. Clean up with:

```bash
kubectl delete namespace aiperf-benchmarks-old
```

---

## Web Dashboard

The AIPerf operator includes a built-in web dashboard for comprehensive monitoring and analysis of your benchmarks.

Access it by port-forwarding to the operator:

```bash
kubectl port-forward -n aiperf-system deploy/aiperf-operator 8081:8081
```

Then open [http://localhost:8081](http://localhost:8081) in your browser.

### Dashboard Features

- **Dashboard Tab** -- Overview with KPI cards, active jobs count, and throughput trends across all jobs
- **Jobs Tab** -- Sortable table of all benchmark jobs with phase filters (Running, Completed, Failed)
- **Job Detail Page** -- Live metrics, charts, phase progress bar, and pod status for a single job
- **Leaderboard Tab** -- Rank benchmark runs by any metric (throughput, latency percentiles, etc.)
- **Compare Tab** -- Side-by-side comparison of multiple jobs to identify performance differences
- **History Tab** -- Time-series charts showing how metrics evolve across all your benchmark runs

### Quick Navigation

Use **Ctrl+K** to open the command palette and quickly jump to any job or page. Search by job name or view recent benchmarks.

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Kubernetes Configuration](configuration.md) -- All CRD fields and deployment options
- [Production Deployments](production.md) -- CI/CD, Kueue, and GitOps workflows
