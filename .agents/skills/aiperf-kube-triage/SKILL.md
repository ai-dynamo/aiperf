---
name: aiperf-kube-triage
description: Use when an AIPerf benchmark on Kubernetes is stuck, failing, crash-looping, OOM-killed, pending, queued, stalled, showing a high error rate, or its results cannot be retrieved - diagnosing an AIPerfJob or AIPerfSweep from CR status, pod state, and logs.
---

# Triaging an AIPerf Kubernetes Run

Classify first, then follow the branch. Do not read logs before you know the
phase — the phase determines which logs matter.

**Related skills:** `aiperf-kube-run` (lifecycle and exit codes),
`aiperf-kube-setup` (cluster/operator install problems).

The machine-parseable playbook with full JSON schemas and jq/python snippets is
`references/debugging.md` (bundled with this skill). This skill is the routing
layer.

## Step 1: classify

```bash
kubectl get aiperfjob <NAME> -n <NS> -o json | jq '{
  phase: .status.phase, subPhase: .status.subPhase,
  workers: .status.workers, error: .status.error,
  conditions: .status.conditions }'

aiperf kube debug -j <NAME> -n <NS> --verbose
```

`aiperf kube debug` reports pod states, recent events, node resources, and a
slice of logs from any pod with problems. `-A` inspects all namespaces. For a
sweep child, pass `--variation <idx>` (long form only — `-v` is `--verbose`
here) and `-t <trial>`.

## Step 2: branch on `status.phase`

| Phase / symptom | Likely cause | Next command |
|---|---|---|
| `Pending` > ~60 s | Unschedulable pods | `kubectl get pods -n <NS> -l aiperf.nvidia.com/job-id=<ID>` then read `PodScheduled` reason |
| `Queued` | Kueue has not admitted the workload | `kubectl get workloads -n <NS>` and read `.status.conditions[?(@.type=="QuotaReserved")]` |
| `Initializing` > ~120 s | Image pull, ConfigMap, or ZMQ connection probe | `aiperf kube logs <ID> --container control-plane --tail 50` |
| `Running`, restarts > 3 | Crash loop | `aiperf kube logs <ID> --container <c> --tail 100` |
| `Running`, OOM-killed pod | Node memory pressure, *not* a container limit (see below) | `kubectl describe node` for pressure; raise `AIPERF_K8S_WORKER_POD_MEMORY` / `AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY` on the **operator** deployment |
| `Running`, `requestsCompleted` flat | Stalled benchmark | check endpoint reachability from inside the cluster |
| `Running`, `request_error_rate.avg > 5` | Endpoint rejecting requests | inspect worker logs and server-side errors |
| `Failed` | See failure table below | `kubectl ... .status.error` + controller logs |
| `Cancelled` | Someone (or `spec.cancel`) stopped it | no action |
| `Completed` | Done | `aiperf kube results` |

`request_error_rate.avg` in `status.liveMetrics.metrics` is the **only** error
signal published while a job runs — `error_request_count` is `ERROR_ONLY` and is
filtered out of live metrics. Do not conclude "no errors" from its absence.

## Step 3: failure patterns

| `status.error` contains | Root cause | Fix |
|---|---|---|
| `preflight` | Cluster validation failed | `aiperf kube preflight -o json`, fix failing checks |
| `endpoint`, `health check` | Server unreachable from the cluster | resolve the URL from inside a pod, not from your laptop |
| `timeout` | Exceeded `spec.timeoutSeconds` | raise it, or set `0` |
| `ConfigMap`, `size` | Config over the 1 MiB object limit | shrink the config / move data to a mounted file |
| `image`, `pull` | Image not accessible | check tag and `imagePullSecrets`; on Kind, `kind load` + `pullPolicy: Never` |
| `RBAC`, `forbidden` | Missing permissions | check service account, Role, RoleBinding |

Scheduling messages map directly to fixes: `Insufficient cpu/memory` -> lower
`--total-workers`; `nvidia.com/gpu` -> no free GPU nodes; `didn't match Pod's
node affinity/selector` -> fix `spec.podTemplate.nodeSelector`; `had untolerated
taint` -> add `spec.podTemplate.tolerations`; `quota` -> namespace ResourceQuota
exhausted.

## Rules that bite

- **A benchmark that looks hung at high concurrency is often CPU starvation of a
  single service, not a deadlock.** The records-manager and system-controller
  default to `75m` CPU (`AIPERF_K8S_RECORDS_MANAGER_CPU`,
  `AIPERF_K8S_SYSTEM_CONTROLLER_CPU`). At very high request rates a pegged core
  starves the event loop, heartbeats expire, and the CR freezes mid-phase. Raise
  the CPU before hunting for a logic bug.
- **The default `resourceMode` is `burstable`, so pods have requests and no
  limits.** `spec.resourceMode` defaults to `burstable`
  and emits `requests` only. That means
  a container cannot be cgroup-OOM-killed for exceeding its memory budget and
  cannot be CFS-throttled for exceeding its CPU budget — the `75m`/`150m`
  numbers are scheduling hints, not ceilings. Two consequences: an `OOMKilled`
  container is evidence of *node* memory pressure or kubelet eviction, not of a
  too-small `AIPERF_K8S_*_MEMORY`; and a starving service is losing a fair-share
  race against co-tenants, which raising the request fixes only indirectly (by
  moving the pod or reserving more share). Set `resourceMode: guaranteed` if you
  actually want the env budgets enforced as limits, and expect real OOM kills
  once you do.
- **Heartbeat expiry is a symptom, not a cause.** `AIPERF_K8S_CONTROLLER_HEARTBEAT_EXPIRY_SECONDS`
  defaults to 30 s and must be at least twice the interval; widening it hides
  the starvation instead of fixing it.
- **Workers pass two startup probes, and they fail differently.** The PUB/SUB
  connection probe fails *closed*: the pod exits on purpose so Kubernetes
  restarts it with a fresh ZMQ context
  (`AIPERF_K8S_JOBSET_WORKER_CONNECTION_PROBE_TIMEOUT`, backoff limit 20). A
  few early restarts on first deploy are expected, not a bug. The credit-return
  PUSH probe fails *open*: credit dispatch rides the DEALER but returns ride a
  separate PUSH/PULL fan-in, and on budget expiry
  (`AIPERF_WORKER_RETURN_PROBE_BUDGET`, default 30 s;
  `AIPERF_WORKER_RETURN_PROBE_RETRY_DELAY`, default 0.1 s) the worker announces
  dispatchability anyway and logs a warning. So a dead return path produces a
  stalled run with **zero restarts** -- grep worker logs for
  `Credit-return channel still has no peer` before blaming the endpoint.
- **Operator-mode results are harvested only on terminal phase.** "results
  missing" mid-run is correct behavior; use `--from-pods` to read the live
  controller instead.
- **On a successful run the pods do not last 300 s — they are deleted at
  once.** `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` (300 s) is only the
  fallback: once the operator has harvested results it calls
  its post-success cleanup and deletes the JobSet immediately. The TTL window is real only for failed runs and for partial
  harvests that are being retried. Never plan to "grab the logs after it
  finishes" — stream them with `--follow` during the run, or accept that a
  clean success leaves you only the PVC artifacts. Direct (operator-less) mode
  is the opposite: `AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS` defaults to
  28800 s (8 h) so pods stay around for manual retrieval.
- **`debug`, `cancel`, `delete`, `list` exit 0 on a missing target** by design.
  Never gate a script on their exit status; use `attach` or `logs`.
- **First deploy attempts flake more than steady state** (ConfigMap propagation,
  ZMQ probe). One clean retry before deep investigation is a legitimate step.

## Diagnosis thresholds

`aiperf kube debug` findings are tunable via `AIPERF_K8S_DIAGNOSIS_*`:
`STALLED_PENDING_THRESHOLD_SECONDS` (60), `STALLED_RUNNING_THRESHOLD_SECONDS`
(30), `HIGH_ERROR_RATE_THRESHOLD` (0.05), `FAIL_ABOVE_ERROR_RATE` (1.0),
`HIGH_LATENCY_P99_MULTIPLIER` (10). Full env-var table in
`references/debugging.md`.

## Log collection

```bash
aiperf kube logs <ID> -n <NS> -o ./triage --ignore-not-found   # per-pod files
aiperf kube logs <ID> --container control-plane --follow
```

Containers worth naming: `control-plane` (SystemController), `api`,
`event-bus-proxy`, worker containers. Save logs to a directory before the TTL
reaps the pods.
