---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Monitoring and Troubleshooting
---

# Monitoring and Troubleshooting

AIPerf provides several tools for monitoring running benchmarks, diagnosing problems, and retrieving logs. This guide covers how to use each one.

---

## Attaching to a Benchmark

The `attach` command connects to a running benchmark via port-forward and streams live progress updates via WebSocket:

```bash
aiperf kube attach
```

`aiperf kube profile` follows the run automatically after deploying unless you
pass `--detach`, but in operator mode (the default) it polls the AIPerfJob CR
rather than port-forwarding. Use `attach` to get the WebSocket progress stream,
to reconnect after detaching, or to watch from a different terminal.

Attach to a specific job:

```bash
aiperf kube attach my-benchmark
```

Press **Ctrl+C** to detach. This only closes your local stream; the benchmark
keeps running in the cluster. To stop the benchmark, run `aiperf kube cancel
<job>`, which patches `spec.cancel: true` so the operator tears down the JobSet
and stamps `status.phase=Cancelled`.

---

## Listing Jobs

See all benchmark jobs and their status:

```bash
aiperf kube list
```

```
NAME               NAMESPACE          PHASE      WORKERS  PROGRESS  THROUGHPUT  LATENCY   AGE
qwen3-benchmark    aiperf-benchmarks  Running      10/10       67%    142.3 rps  318.7 ms  3m
llama-throughput   aiperf-benchmarks  Completed    10/10      100%    137.9 rps  330.1 ms  15m
mistral-test       aiperf-benchmarks  Failed         0/4         -            -         -  20m
```

`WORKERS` is `ready/total`, `THROUGHPUT` is requests per second, and `LATENCY`
is the p99 request latency. A dash means the CR has not published that value.

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

The controller pod runs `control-plane` (the SystemController) alongside per-service containers (`dataset-manager`, `timing-manager`, `records-manager`, `api`, `event-bus-proxy`, `results-sidecar`, and optionally `gpu-telemetry-manager` / `server-metrics-manager`). Worker pods run `worker-group-manager`. To see logs from a specific container:

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

- **Pod states** -- Identifies CrashLoopBackOff, ImagePullBackOff, ErrImagePull, OOMKilled, CreateContainerConfigError, RunContainerError, and Unschedulable, each with a suggested fix
- **Kubernetes events** -- Shows the most recent warning events (or all recent events with `--verbose`)
- **Node resources** -- Reports CPU, memory, and GPU allocatable vs. capacity for each node
- **Benchmark diagnostics** -- When `--job-id` targets a specific job, runs the metric detectors in `aiperf.kubernetes.benchmark_diagnosis` over `status.liveMetrics`: high error rate, high tail latency (p99 above a multiple of the average), and a stalled job (Pending too long, or Running with neither throughput nor completed requests). Thresholds are the `AIPERF_K8S_DIAGNOSIS_*` environment variables. The section is omitted when nothing tripped.
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
aiperf-bench-workers-0-0         Running   3         node-2     1

Problems Found
[aiperf-bench-workers-0-0] OOMKilled (previous) (container: worker-0)
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

It runs these checks in order, and stops early only if cluster connectivity
fails:

- Cluster connectivity and Kubernetes version
- Namespace exists (or can be created)
- RBAC permissions in the target namespace
- JobSet CRD installed and JobSet controller running
- Resource quotas and node resources
- Referenced secrets and image-pull access
- Network policies and DNS resolution
- Endpoint connectivity (when `--endpoint-url` is given)

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

By default this downloads the full results package from the operator's PVC storage (use `--from-pods` to pull from the benchmark pods instead). The results include:

- `profile_export_aiperf.json` -- Summary metrics
- `profile_export.jsonl` -- Per-request timing data
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

If you want to fetch results directly from the running benchmark pods (via the controller API):

```bash
aiperf kube results --from-pods
```

The default `--all` path uses the controller API and nothing else — if that
call fails, the download fails. The `kubectl cp` fallback applies only to
`--summary-only`, which tries the API first and then copies from the
`control-plane` container.

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

### Expired Local Kubernetes Login

When an interactive `aiperf kube` command receives HTTP 401 or a recognized
`kubectl` logged-out response, it pauses and reloads the selected kubeconfig
until your normal credential provider works again. Complete the usual login
(for example, your OIDC, cloud, or access-proxy login) in another terminal.
AIPerf does not launch the provider or request credentials itself. Press
**Ctrl+C** to stop waiting.

This behavior is limited to terminals with interactive input and output.
Operator pods, in-cluster service accounts, redirected output, and CI fail
immediately so unattended work cannot hang. HTTP 403 is also returned
immediately because it means the authenticated identity lacks RBAC permission,
not that the login expired. Missing credential-provider executables, malformed
plugins, TLS failures, and unreachable API servers remain ordinary errors.

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

In operator mode, inspect the durable startup diagnosis directly:

```bash
kubectl get aiperfjob my-benchmark \
  -o jsonpath='{.status.startupIssue}{"\n"}{.status.conditions[?(@.type=="WorkersReady")]}{"\n"}'
```

Temporary capacity shortages, pending PVC binding, Kueue admission, and unknown
scheduler reasons remain retryable even when `timeoutSeconds: 0`. Stable image,
container-configuration, crash-loop, node-selector, untolerated-taint, and
volume-affinity blockers fail only after the critical startup grace period.
Tune warning and critical thresholds with
`AIPERF_K8S_WATCHDOG_PENDING_THRESHOLD_SECONDS` and
`AIPERF_K8S_WATCHDOG_PENDING_CRITICAL_THRESHOLD_SECONDS` on the operator
container.

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

`spec.connectionsPerWorker` is immutable after creation, so changing it means
creating a new AIPerfJob.

The per-pod memory budget is resolved by the process that renders the JobSet, so
it is set on the operator container rather than in the CR. The chart has no
values key for it; patch the Deployment directly:

```bash
kubectl set env -n aiperf-system deploy/aiperf-operator \
  AIPERF_K8S_WORKER_POD_MEMORY=8Gi
```

Per job, `spec.resourceMode` selects the QoS shape: `burstable` (the default)
sets requests without limits so containers are not cgroup-OOM-killed for
exceeding their request, `guaranteed` sets `requests == limits`, and `none`
omits both. It is also immutable after creation.

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

The CLI-side watchdog that runs while `aiperf kube profile` follows a benchmark
warns when it finds more than two leftover `aiperf-*` namespaces. Remove the
finished benchmarks inside one with `aiperf kube cleanup`, or delete the
namespace outright:

```bash
aiperf kube cleanup --namespace aiperf-benchmarks-old --force
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
- **Sweeps Tab** -- Table of AIPerfSweeps with per-sweep drill-down (variation curves, Pareto, children)
- **Launch Tab** -- Read-only YAML helper for scaffolding a manifest (copy-only; browser submission is disabled)
- **Leaderboard Tab** -- Rank benchmark runs by any metric (throughput, latency percentiles, etc.)
- **Compare Tab** -- Side-by-side comparison of multiple jobs to identify performance differences
- **History Tab** -- Time-series charts showing how metrics evolve across all your benchmark runs

See [Web Dashboard](dashboard-ui.md) for the full page-by-page reference.

### Quick Navigation

Use **Ctrl+K** to open the command palette and quickly jump to any job or page. Search by job name or view recent benchmarks.

---

## Operator Prometheus Metrics

The kopf operator container exposes a Prometheus `/metrics` endpoint from an
in-process daemon thread (`src/aiperf/operator/metrics.py`). The port is
`AIPERF_METRICS_PORT` (`OperatorEnvironment.METRICS_PORT`, default **9090**; set
to `0` to disable). This is separate from the results-server on 8081.

```bash
kubectl port-forward -n aiperf-system deploy/aiperf-operator 9090:9090
curl http://localhost:9090/metrics
```

Exposed series:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `aiperf_operator_handler_duration_seconds` | Histogram | `handler` | Wall-clock duration of each instrumented kopf reconcile handler. |
| `aiperf_operator_handler_total` | Counter | `handler`, `outcome` | Reconcile-handler invocations by outcome. |
| `aiperf_operator_completion_claim_races_total` | Counter | — | Lost `try_claim_completion` races (concurrent ticks contending for the completion claim). |

`outcome` is one of four values: `success` (returned normally), `retry` (raised
`kopf.TemporaryError`, so kopf will re-dispatch), `fatal` (raised
`kopf.PermanentError`, so kopf stops retrying and the CR is stuck), and `error`
(anything else, including `CancelledError`, `KeyboardInterrupt`, and
`SystemExit`). `retry` and `fatal` are separated so you can alert on stuck CRs
without false positives from transient apiserver hiccups.

Only kopf reconcile handlers are instrumented (via the `@track_handler("name")`
decorator); helper functions are not.

### Benchmark Metrics from the Controller Pod

Separately from the operator's own reconcile metrics, each benchmark's
controller pod serves its live benchmark metrics in Prometheus exposition
format from the `api` container at `/metrics` on the API port (default 9090).
The controller pod is annotated with `prometheus.io/scrape: "true"`,
`prometheus.io/port`, and `prometheus.io/path: /metrics`, so an
annotation-based Prometheus scrape config picks it up without extra
configuration.

Set `serviceMonitor.enabled=true` in the chart to have a Prometheus Operator
`ServiceMonitor` scrape the Service's `metrics` port at `/metrics`. It is off by
default, and is skipped when `operator.metrics.port` is `0` or when the
`monitoring.coreos.com/v1` CRDs are absent. The repository ships no Grafana
dashboards or `PrometheusRule` alerts.

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Kubernetes Configuration](configuration.md) -- All CRD fields and deployment options
- [Production Deployments](production.md) -- CI/CD, Kueue, and GitOps workflows
