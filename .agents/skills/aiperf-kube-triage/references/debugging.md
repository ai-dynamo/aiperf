---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# AIPerf Kubernetes Triage Reference

Terminal reference for diagnosing an `AIPerfJob` / `AIPerfSweep`. Every command
is machine-parseable. Classify first, then run only that branch's block.

## 1. Triage snapshot

```bash
kubectl get aiperfjob <NAME> -n <NS> -o json | jq '{
  phase: .status.phase, subPhase: .status.subPhase,
  currentPhase: .status.currentPhase, workers: .status.workers,
  requestsCompleted: .status.requestsCompleted,
  startupIssue: .status.startupIssue,
  error: .status.error, conditions: .status.conditions }'

aiperf kube debug -j <NAME> -n <NS> --verbose     # pods, events, node pressure, log slices
```

`aiperf kube debug` flags: `-j/--job-id`, `-n/--namespace`, `-v/--verbose`,
`-A/--all-namespaces`, `--variation <idx>` (long form only — `-v` is `--verbose`
here), `-t/--trial`. It has no JSON output mode.

| Phase / symptom | Go to |
|---|---|
| `Completed` | Collect results (§9) |
| `Failed` | §3 Failed job |
| `Cancelled` | Intentional or `spec.cancel: true`. No action. |
| `Pending` > ~60 s | §4 Stuck Pending |
| `Queued` | §5 Kueue not admitted |
| `Initializing` > ~120 s | §6 Stuck Initializing |
| `Running`, restarts > 3 | §7 Crash loop |
| `Running`, container `OOMKilled` | §8 OOM kills |
| `Running`, `status.phases.*.requestsCompleted` flat and `request_throughput` 0 | §10 Stalled |
| `Running`, `request_error_rate.avg > 5` (percent) | §11 High error rate |
| `Running`, none of the above | Healthy. `aiperf kube attach`. |

## 2. CR `.status` schema

`phase` (enum `Pending|Queued|Initializing|Running|Completed|Failed|Cancelled`),
`jobId`, `jobSetName`, `startTime`, `completionTime`, `observedGeneration`,
`error`, `conditions[]`,
`workers{ready,total,dispatchable,routerConnected,readyRecordProcessors,declaredRecordProcessors,readyPods,totalPods,degradedPods}`,
`phases{<name>:{requestsCompleted,…}}`,
`requestsCompleted`, `requestsTotal`, `requestsPerSecond`,
`currentPhase` (inner benchmark stage),
`subPhase` (controller lifecycle: `initializing|configuring|ready|profiling|processing|stopping|shutdown`; removed on terminal phase),
`liveMetrics.metrics{<tag>:{avg,p50,p90,p99,min,max,std,count,sum,unit}}`,
`serverMetrics`, `liveSummary`, `summary`, `results`, `resultsPath`, `runEpoch`,
`resultsTtlDays`,
`startupIssue{fingerprint,podName,containerName,reason,message,category,terminalAfterThreshold,firstObservedTime,warningEmitted}`
where `category` is `ContainerConfig|CrashLoop|ImagePull|SchedulingConstraint|SchedulingDelay`.

`subPhase` is mirrored to the annotation `aiperf.nvidia.com/system-state`. Pod
selector `aiperf.nvidia.com/job-id=<JOB_ID>` (also `app=aiperf`,
`aiperf.nvidia.com/{name,parent,trial}`). Containers — controller pod:
`control-plane`, `dataset-manager`, `timing-manager`, `records-manager`, `api`,
`gpu-telemetry-manager`, `server-metrics-manager`, `results-sidecar`,
`event-bus-proxy`; worker pod: `worker-group-manager` plus worker and
record-processor containers.

**Immutable spec fields** (a "fix" to any of these means deleting and recreating
the CR, not patching it): `image`, `imagePullPolicy`, `benchmark`, `resourceMode`,
`connectionsPerWorker`, `ttlSecondsAfterFinished`, `resultsTtlDays`,
`keepFailedPods`, `podTemplate`, `scheduling`, `schemaVersion`, `sweep`,
`multiRun`, `plot`, `variables`, `randomSeed`, `noSweepTable`,
`skipEndpointCheck`, `failurePolicy`. Only `timeoutSeconds` and `cancel` are
patchable in place.

## 3. Failed job

```bash
kubectl get aiperfjob <NAME> -n <NS> -o json | jq '.status | {phase, error, conditions}'

aiperf kube logs <JOB_ID> -n <NS> --container control-plane --tail 50
aiperf kube debug -j <JOB_ID> -n <NS> --verbose
```

| `status.error` contains | Root cause | Fix |
|---|---|---|
| `preflight` | Cluster validation failed | `aiperf kube preflight -o json`, apply `hints` |
| `endpoint`, `health check` | Server unreachable *from inside the cluster* | resolve the URL from a pod, not your laptop |
| `timeout` | Exceeded `spec.timeoutSeconds` | patch it higher, or `0` for no timeout |
| `ConfigMap`, `size` | Config over the 1 MiB object limit | shrink config / mount data as a file |
| `image`, `pull` | Image not accessible | check tag and `imagePullSecrets`; on Kind, `kind load` + `imagePullPolicy: Never` |
| `RBAC`, `forbidden` | Missing permissions | check ServiceAccount, Role, RoleBinding |

## 4. Stuck Pending

```bash
kubectl get pods -n <NS> -l aiperf.nvidia.com/job-id=<JOB_ID> -o json | jq -c '
  .items[] | {pod: .metadata.name} + (.status.conditions[]
  | select(.type=="PodScheduled" and .status=="False") | {reason, message})'

kubectl get nodes -o json | jq -c '.items[] | {node: .metadata.name,
  cpu: .status.allocatable.cpu, memory: .status.allocatable.memory,
  gpu: (.status.allocatable["nvidia.com/gpu"] // "0")}'
```

| Scheduling message | Fix |
|---|---|
| `Insufficient cpu` / `Insufficient memory` | lower `--total-workers`; recreate |
| `nvidia.com/gpu` | no free GPU nodes — wait or add capacity |
| `didn't match Pod's node affinity/selector` | fix `spec.podTemplate.nodeSelector` (immutable — recreate) |
| `had untolerated taint` | add `spec.podTemplate.tolerations` (immutable — recreate) |
| `quota` | namespace ResourceQuota exhausted |

## 5. Kueue not admitted (`Queued`)

The operator stamps `kueue.x-k8s.io/queue-name` (+ `kueue.x-k8s.io/priority-class`)
on the JobSet and creates it `spec.suspend: true` whenever a queue name resolves
from `spec.scheduling.queueName` or the operator-side
`AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME`. It does **not** read the namespace
annotation `kueue.x-k8s.io/default-queue-name` — that only defaults the queue on
Kueue's side and satisfies operator preflight. While suspended and labelled the
phase is `Queued`; on admission it walks `Initializing` -> `Running`.

```bash
kubectl get workloads -n <NS> -o json | jq -c '.items[] | {
  workload: .metadata.name,
  admitted: ([.status.conditions[]? | select(.type=="Admitted" and .status=="True")] | length > 0),
  conditions: [.status.conditions[]? | {type, status, message}]}'

kubectl get clusterqueues -o json | jq -c '.items[] | {name: .metadata.name,
  flavors: .status.flavorsReservation, pending: .status.pendingWorkloads}'
kubectl describe workload -n <NS>   # QuotaReserved/Admitted messages name the over-quota resource
kubectl get workloads -A -o wide    # admitted workloads currently holding the quota
```

- Compare `ClusterQueue.spec.resourceGroups.*.nominalQuota` to what the JobSet
  requests: each worker pod requests the `AIPERF_K8S_WORKER_POD_CPU` /
  `_MEMORY` budget (`150m` / `4Gi`) split across its containers.
- Operator preflight `FAIL` `Kueue LocalQueue '<name>' not found` (or "Kueue is
  not installed, but LocalQueue '<name>' was explicitly requested"): the queue
  does not exist in the namespace. `SKIP` means no queue was requested at all.
  This check is operator-side only; client-side `preflight` omits it.
- Operator preflight `WARN` `Kueue is installed but no queue configured`: set
  `--queue-name` / `spec.scheduling.queueName`, or `kubectl annotate namespace
  <NS> kueue.x-k8s.io/default-queue-name=<queue>`.
- No preemption despite `priorityClass`: the `WorkloadPriorityClass` value must
  exceed the admitted workloads', and the ClusterQueue must enable
  `spec.preemption.reclaimWithinCohort` / `withinClusterQueue`.
- `spec.scheduling` is immutable — priority cannot be raised on a queued job.

## 6. Stuck Initializing

```bash
kubectl get pods -n <NS> -l aiperf.nvidia.com/job-id=<JOB_ID> -o json | jq -c '
  .items[] as $p | $p.status.containerStatuses[]? | select(.ready == false) | {
    pod: $p.metadata.name, container: .name, restarts: .restartCount,
    waiting_reason: .state.waiting.reason, waiting_message: .state.waiting.message}'
```

| Waiting reason | Fix |
|---|---|
| `ContainerCreating` | normal, image pulling — wait |
| `ImagePullBackOff` | bad image or missing pull secret |
| `CrashLoopBackOff` | §7 |
| `CreateContainerConfigError` | missing ConfigMap/Secret/volume — read pod events |

Worker pods run a PUB/SUB connection probe that **fails closed**: the pod exits
so Kubernetes restarts it with a fresh ZMQ context
(`AIPERF_K8S_JOBSET_WORKER_CONNECTION_PROBE_TIMEOUT` 60 s, backoff limit 20). A
few early restarts on first deploy are expected, not a bug.

## 7. Crash loop

```bash
kubectl logs -n <NS> <POD> --previous -c <CONTAINER> --tail=50
kubectl get pod -n <NS> <POD> -o json | jq -c '.status.containerStatuses[]
  | select(.lastState.terminated) | {container: .name,
    exit_code: .lastState.terminated.exitCode, reason: .lastState.terminated.reason,
    message: .lastState.terminated.message}'
```

| Exit code | Meaning |
|---|---|
| 137 | SIGKILL — node memory pressure / eviction / external kill (§8) |
| 1 | application error: bad config, missing model, endpoint unreachable |
| 2 | Python import/syntax error — wrong image version |

## 8. OOM kills

```bash
kubectl get pod -n <NS> <POD> -o json | jq -c '
  {limits: [.spec.containers[] | {name, memory: (.resources.limits.memory // "none")}],
   oom: [.status.containerStatuses[] | select(.lastState.terminated.reason=="OOMKilled") | .name]}'
kubectl describe node <NODE> | grep -i pressure
```

`spec.resourceMode` defaults to **`burstable`**: requests, **no limits**. A
container therefore cannot be cgroup-OOM-killed for exceeding its memory budget
nor CFS-throttled for exceeding its CPU budget — the `AIPERF_K8S_*` numbers are
scheduling hints, not ceilings. An `OOMKilled` container is evidence of *node*
memory pressure or kubelet eviction, not of a too-small `AIPERF_K8S_*_MEMORY`.
Use `resourceMode: guaranteed` (requests == limits) to enforce the budgets, and
expect real OOM kills once you do.

Fixes, in order:

1. Lower `spec.connectionsPerWorker` (default `100`) — immutable, recreate.
2. Lower `spec.benchmark.runtime.workersPerPod` to fan the same cluster-wide
   worker total across more pods — `spec.benchmark` is immutable, recreate.
3. Raise `AIPERF_K8S_WORKER_POD_MEMORY` (default `4Gi`) **on the operator**, not
   in `spec.podTemplate.env` — the operator process renders the JobSet:
   `kubectl set env -n aiperf-system deploy/aiperf-operator AIPERF_K8S_WORKER_POD_MEMORY=8Gi`

## 9. Collect results

```bash
aiperf kube results <JOB_ID> -n <NS>                    # operator PVC (works after pods are gone)
aiperf kube results <JOB_ID> --from-pods --summary-only # live controller API, kubectl cp fallback
aiperf kube results <JOB_ID> --output ./artifacts --run <epoch>
aiperf kube logs <JOB_ID> -n <NS> -o ./triage --ignore-not-found   # <dir>/logs/<pod>.log
aiperf kube logs <JOB_ID> --container control-plane --follow
```

Results are harvested only on a **terminal** phase, so "results missing" mid-run
is correct — use `--from-pods`. On a *successful* run the operator deletes the
JobSet immediately after harvesting; the 300 s
`AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` window is only a fallback for
failed runs and retried partial harvests. Stream logs during the run; do not
plan to collect them after. Direct (operator-less) mode keeps pods 8 h
(`AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS`).

```bash
# profile_export_aiperf.json: top-level keys are metric tags, each an object of
# {unit, avg, p50, p90, p99, min, max, std, count, sum}.
jq '{throughput: .request_throughput.avg, latency_avg: .request_latency.avg,
     latency_p99: .request_latency.p99, ttft_avg: .time_to_first_token.avg,
     itl_avg: .inter_token_latency.avg, otps: .output_token_throughput.avg,
     requests: .request_count.avg, errors: .error_request_count.avg,
     error_rate_pct: .request_error_rate.avg}' ./artifacts/profile_export_aiperf.json
```

`request_count` counts valid requests only; `completed_request_count` is
successes + errors. `error_request_count` is omitted entirely on a clean run.

## 10. Stalled benchmark

```bash
kubectl run aiperf-curl-test --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s -o /dev/null -w '{"http_code":%{http_code},"time_total":%{time_total}}' <ENDPOINT_URL>/models
aiperf kube logs <JOB_ID> --container control-plane --tail 30 2>&1 | grep -iE "error|timeout|refused|unreachable"
```

| Symptom | Fix |
|---|---|
| `http_code: 0` / connection refused | wrong endpoint URL or frontend not running; resolve the in-cluster service DNS name |
| `http_code: 200` but no progress | workers not connecting — check ZMQ in controller logs |
| curl times out | NetworkPolicy blocking, or the server is still loading the model |

Two more stall causes that produce **zero restarts**:

- **Credit-return channel dead.** Credit dispatch rides a DEALER but credit
  *returns* ride a separate PUSH/PULL fan-in. That probe fails **open**: on
  budget expiry (`AIPERF_WORKER_RETURN_PROBE_BUDGET` 30 s,
  `AIPERF_WORKER_RETURN_PROBE_RETRY_DELAY` 0.1 s) the worker announces itself
  dispatchable anyway and only logs a warning. Grep worker logs for
  `Credit-return channel still has no peer` before blaming the endpoint.
- **CPU starvation of one service.** The records-manager and system-controller
  request only `75m`. At high request rates a pegged core starves the event
  loop, heartbeats expire, and the CR freezes mid-phase. Raise
  `AIPERF_K8S_RECORDS_MANAGER_CPU` on the operator before hunting a logic bug.
  Widening `AIPERF_K8S_CONTROLLER_HEARTBEAT_EXPIRY_SECONDS` (30 s, must stay at
  least twice the 10 s interval) hides the starvation instead of fixing it.

## 11. High error rate

```bash
kubectl get aiperfjob <NAME> -n <NS> -o jsonpath='{.status.liveMetrics.metrics}' | python3 -m json.tool
```

`request_error_rate` has unit **percent** (`100 * errors / completed`) and is the
authoritative live error signal; `error_request_count` carries the `ERROR_ONLY`
flag and is filtered out of `status.liveMetrics` and `/api/metrics`, so its
absence never means "no errors". The `AIPERF_K8S_DIAGNOSIS_*` thresholds are
0..1 fractions, so `request_error_rate` is divided by 100 before comparison.

| Error rate | Likely cause |
|---|---|
| 5-20% | endpoint overloaded — lower phase concurrency (recreate; `spec.benchmark` is immutable) |
| 20-50% | model or endpoint errors — check server logs for 500/503 |
| >50% | endpoint down or misconfigured; verify the model name the server serves |
| 100% | wrong URL or auth required — fix URL or inject a key via a Secret |

## 12. Preflight JSON

`aiperf kube preflight -o json` (also `-i/--image`, `--image-pull-secret`,
`--secret`, `-e/--endpoint-url`, `-w/--workers`):

```json
{"passed": true, "has_warnings": false,
 "checks": [{"name": "Cluster Connectivity", "status": "pass",
             "message": "Connected to Kubernetes cluster",
             "details": [], "hints": [], "duration_ms": 45.2}]}
```

`status` is one of `pass` (no action), `fail` (blocking — apply `hints[0]`),
`warn` (review, non-blocking), `skip` (not applicable), `info` (context only).
`passed` is false iff any check is `fail`; `has_warnings` true iff any is `warn`.
Gate a script on `jq -e '.passed'` and print the `fail` checks' `hints`.

## 13. Validate JSON

`aiperf kube validate -o json [-s/--strict] <FILE>...` — exits `1` if any file
fails; `--strict` promotes warnings to errors.

```json
[{"path": "benchmark.yaml", "passed": true, "errors": [],
  "warnings": ["Unknown spec fields (did you mean to put these under spec.benchmark?): foo"]}]
```

## 14. Tunable environment variables

Resource and JobSet variables are read by the process that *renders* the JobSet,
so set them on the operator deployment (`kubectl set env -n aiperf-system
deploy/aiperf-operator KEY=VALUE`); `spec.podTemplate.env` has no effect on
container resources. Default operator namespace `aiperf-system`, default
benchmark namespace `aiperf-benchmarks`.

| Variable | Default | What it does |
|---|---|---|
| `AIPERF_K8S_WORKER_POD_CPU` / `_MEMORY` | `150m` / `4Gi` | worker-pod budget (workers + record processors + WGM) |
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` / `_MEMORY` | `75m` / `192Mi` | control-plane container |
| `AIPERF_K8S_RECORDS_MANAGER_CPU` / `_MEMORY` | `75m` / `256Mi` | raise CPU first at high concurrency |
| `AIPERF_K8S_TIMING_MANAGER_CPU` / `_MEMORY` | `50m` / `192Mi` | timing manager |
| `AIPERF_K8S_DATASET_MANAGER_CPU` / `_MEMORY` | `50m` / `256Mi` | dataset manager |
| `AIPERF_K8S_EVENT_BUS_PROXY_CPU` / `_MEMORY` | `50m` / `64Mi` | XPUB/XSUB proxy sidecar |
| `AIPERF_K8S_SWEEP_CONTROLLER_CPU` / `_MEMORY` | `75m` / `512Mi` | sweep controller (BoTorch needs the memory) |
| `AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED` | `true` | run the event-bus proxy sidecar |
| `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` | `300` | fallback JobSet TTL (success deletes immediately) |
| `AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS` | `28800` | direct-mode pod retention |
| `AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT` | `20` | worker restart budget |
| `AIPERF_K8S_JOBSET_WORKER_CONNECTION_PROBE_TIMEOUT` | `60.0` | PUB/SUB probe (fails closed) |
| `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` | `""` | operator-wide default Kueue queue |
| `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_PRIORITY_CLASS` | `""` | operator-wide default priority class |
| `AIPERF_K8S_CONTROLLER_HEARTBEAT_INTERVAL_SECONDS` | `10.0` | controller heartbeat cadence |
| `AIPERF_K8S_CONTROLLER_HEARTBEAT_EXPIRY_SECONDS` | `30.0` | must be >= 2x the interval |
| `AIPERF_K8S_DIAGNOSIS_STALLED_PENDING_THRESHOLD_SECONDS` | `60.0` | Pending -> "stalled" finding |
| `AIPERF_K8S_DIAGNOSIS_STALLED_RUNNING_THRESHOLD_SECONDS` | `30.0` | Running with no progress -> "stalled" |
| `AIPERF_K8S_DIAGNOSIS_HIGH_ERROR_RATE_THRESHOLD` | `0.05` | fraction, not percent |
| `AIPERF_K8S_DIAGNOSIS_FAIL_ABOVE_ERROR_RATE` | `1.0` | fraction at which a finished run is called Failed |
| `AIPERF_K8S_DIAGNOSIS_HIGH_LATENCY_P99_MULTIPLIER` | `10.0` | p99 / avg tail-latency flag |
| `AIPERF_K8S_WATCHDOG_PENDING_CRITICAL_THRESHOLD_SECONDS` | `90.0` | operator watchdog pending critical |
| `AIPERF_K8S_WATCHDOG_CRASHLOOP_RESTART_THRESHOLD` | `2` | restarts before crash-loop is declared |
| `AIPERF_K8S_WATCH_DEFAULT_TIMEOUT_SECONDS` | `600` | CLI watch/attach timeout |
| `AIPERF_K8S_RESULTS_DOWNLOAD_TIMEOUT_SECONDS` | `300.0` | artifact download timeout |
| `AIPERF_K8S_RESULTS_KUBECTL_COPY_TIMEOUT_SECONDS` | `1800.0` | `kubectl cp` fallback timeout |
| `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` | `false` | enable to `ps`/`py-spy` across containers in a pod |
| `AIPERF_WORKER_RETURN_PROBE_BUDGET` | `30.0` | credit-return probe budget (fails **open**) |
| `AIPERF_WORKER_RETURN_PROBE_RETRY_DELAY` | `0.1` | credit-return probe retry interval |

## 15. Quick commands

| Task | Command |
|---|---|
| Phase + error | `kubectl get aiperfjob <NAME> -n <NS> -o jsonpath='{.status.phase} {.status.error}'` |
| All jobs | `kubectl get aiperfjobs -A -o json` |
| Pods for a job | `kubectl get pods -n <NS> -l aiperf.nvidia.com/job-id=<ID> -o json` |
| Events | `kubectl get events -n <NS> --sort-by=.lastTimestamp -o json` |
| Controller logs | `aiperf kube logs <ID> --container control-plane --tail 50` |
| Worker logs | `aiperf kube logs <ID> --container worker-group-manager --tail 50` |
| Live monitor | `aiperf kube attach <ID> -n <NS>` |
| Cancel | `kubectl patch aiperfjob <NAME> -n <NS> --type=merge -p '{"spec":{"cancel":true}}'` |
| Delete | `kubectl delete aiperfjob <NAME> -n <NS>` |

`debug`, `cancel`, `delete`, `list` exit `0` on a missing target by design — never
gate a script on their exit status; use `attach` or `logs`. First deploy attempts
flake more than steady state (ConfigMap propagation, ZMQ probe); one clean retry
before deep investigation is legitimate.
