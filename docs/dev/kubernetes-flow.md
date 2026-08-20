---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Kubernetes Flow
---

# Kubernetes Flow End-to-End

This document describes the complete flow from user command to benchmark completion when running AIPerf on Kubernetes.

## Overview

```mermaid
flowchart TB
    subgraph WS["User workstation"]
        CLI["aiperf kube profile --model Qwen/Qwen3-0.6B --total-workers 10"]
        SUB["Submit AIPerfJob CR<br/>(or direct manifests if no operator)"]
        CLI --> SUB
    end

    SUB -->|kubernetes_asyncio API| JS

    subgraph K8S["Kubernetes cluster"]
        subgraph JS["JobSet: aiperf-{job_id}"]
            CTRL["Controller pod<br/>SystemController / TimingMgr<br/>DatasetMgr / RecordsMgr / API"]
            W0["Worker pod 0"]
            W1["Worker pod 1"]
            WN["Worker pod N"]
            W0 --> CTRL
            W1 --> CTRL
            WN --> CTRL
        end
        CM["ConfigMap: benchmark config"]
        CM --> CTRL
    end
```

`WorkerGroupManager` is the readiness and capacity authority for each Kubernetes worker pod. Local multiprocessing is managed directly by `MultiProcessServiceManager` and `WorkerManager`.

## 1. CLI Entry Point

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B --url http://server:8000 --image aiperf:latest --total-workers 10
```

CLI commands defined in `src/aiperf/cli_commands/kube/`:

| Command | Purpose |
|---------|---------|
| `init` | Generate a starter configuration template |
| `validate` | Validate AIPerfJob and AIPerfSweep YAML files against the CRD schema |
| `profile` | Run a benchmark in Kubernetes |
| `sweep` | Run a parameter sweep or multi-run benchmark in Kubernetes |
| `generate` | Generate Kubernetes YAML manifests |
| `attach` | Attach to a running benchmark and stream progress |
| `list` | List benchmark jobs and their status; pass `--watch` for live updates |
| `logs` | Retrieve logs from benchmark pods |
| `results` | Retrieve benchmark results |
| `show` | Render an AIPerfJob CR with Jinja2/env-vars resolved |
| `debug` | Run diagnostic analysis on a deployment |
| `preflight` | Run pre-flight checks against the target cluster |
| `dashboard` | Open the operator results server UI in your browser |

## 2. Deployment Generation

The deployment logic in `src/aiperf/cli_commands/kube/profile.py` auto-detects whether the AIPerfJob CRD is installed. If the operator is present, `deploy_via_operator()` (in `profile_deploy.py`) submits an `AIPerfJob` custom resource and the operator reconciles it; otherwise `deploy_direct()` (in `profile_deploy_direct.py`) creates the manifests (ConfigMap, Role, RoleBinding, JobSet) directly. `--operator` forces operator mode without the cluster-scoped CRD probe; `--no-operator` forces direct mode. The operator path creates the default CLI-owned namespace when no namespace flag is supplied, but treats an explicit `--namespace` as pre-provisioned and performs only namespaced custom-resource operations.

```mermaid
flowchart TD
    A["1. Resolve benchmark config<br/>from CLI flags or AIPerfJob CR YAML"] --> B
    B["2. Configure ServiceConfig<br/>service_run_type = KUBERNETES<br/>dataset API base URL = controller DNS"] --> C
    C{"3. AIPerfJob CRD present?"}
    C -->|yes, or --operator| D["Operator mode:<br/>create AIPerfJob CR, operator reconciles"]
    C -->|no, or --no-operator| E["Direct mode:<br/>create ConfigMap + RBAC + JobSet directly"]
```

`ServiceRunType.KUBERNETES` is a generated plugin enum member — it comes from
`src/aiperf/plugin/plugins.yaml` via the generated `src/aiperf/plugin/enums.py`,
which is never hand-edited.

## 3. Kubernetes Resources

### Resource Creation Order

```mermaid
flowchart TD
    NS["Namespace (if auto-generated)"] --> ROLE[Role]
    ROLE --> RB[RoleBinding]
    ROLE --> CM[ConfigMap]
    RB --> CM
    CM --> JS[JobSet]
    JS --> CTRL["controller (1 pod)"]
    JS --> W["workers (N pods)"]
```

### Same-name recreation fencing

Operator-managed RBAC, ConfigMaps, and JobSets are adopted after an
`AlreadyExists` response only when their controller owner reference matches the
current AIPerfJob or AIPerfSweep name and immutable UID. A resource owned by an
older incarnation, or a matching resource already being deleted, remains on a
retry path until Kubernetes garbage collection finishes. A deterministic name
occupied by an unrelated or ownerless resource fails permanently instead of
being adopted.

Sweep child rollups and the sweep-controller pod apply the same UID fence to
parent status writes. Resource-version and JSON Patch test operations prevent a
delayed child event or stale controller pod from updating a newly recreated,
same-named AIPerfSweep.

AIPerfJob recovery callbacks also pin the parent resource version and accept
JobSet state only when its controller owner API version, kind, name, and UID
match the callback's immutable parent UID. Timeout, startup-failure, and result
salvage cleanup re-read that ownership and delete with the JobSet UID as a
precondition. A stable startup-blocker deadline first atomically adds
`aiperf.nvidia.com/startup-failure-claimed` with JSON Patch tests for the
parent UID, resource version, spec cancellation state, phase, exact
`status.startupIssue`, and annotation map. The annotation-map test makes this
failure-cleanup claim mutually exclusive with the durable completion claim;
only the winner may enter JobSet deletion. A matching persisted failure claim
resumes cleanup after an operator restart. Pod watches and stale-heartbeat
recovery only persist critical startup diagnosis; the cached-state deadline is
the sole path that may claim, delete, and terminalize that blocker. JobSet
failure events use the same ownership fence before a direct status patch.
Event status changes are rebased onto the fenced live status so every condition
type not demonstrably changed by the event survives concurrent controller
writes. Pod events resolve the full Pod to batch Job to JobSet to AIPerfJob
controller-owner chain before restart reporting, startup diagnosis, or
controller-termination salvage; transient owner-read failures request a
bounded kopf retry instead of dropping the one-shot event.

For an operator-managed AIPerfJob, the immutable owner UID is injected into
controller-pod services as `AIPERF_JOB_UID`. Completion and progress writers
read the live target, verify ownership, and use JSON Patch `test` operations on
the AIPerfJob or JobSet UID before adding annotations. If the named resource was
recreated between the read and patch, the API server rejects the whole atomic
patch and the stale controller cannot mutate the replacement.

Cancellation and completion callbacks use the same UID in their process-local
cancellation, progress-client, and completion-claim keys. JobSet deletion uses
the validated JobSet UID as a precondition, and terminal status merge patches
carry the live parent resource version. Ready/latest and runs-index publication
recheck the parent UID; the SQLite latest row advances only when `latest.txt`
accepted the same epoch. A delayed callback therefore cannot close a
replacement job's client, delete its JobSet, or publish terminal state for it.

AIPerfSweep deletion uses sweep-name, sweep-UID, and, when available, run-epoch
labels only to discover candidate child jobs. Before setting
`spec.cancel=true`, the handler requires an `AIPerfSweep` owner reference whose
name and immutable UID match the deleting parent. A standalone AIPerfJob cannot
be cancelled merely by carrying user-writable sweep tracking labels.

### Pod Architecture

Each control-plane service runs in its own container in the controller pod
(sibling containers, not subprocesses). Workers and record processors
likewise each run in their own container inside a worker pod.

**Controller pod** (container names from the `Containers` class in `src/aiperf/kubernetes/constants.py`):

| Container | Role |
|---|---|
| `control-plane` | SystemController (orchestration) |
| `dataset-manager` | Generates prompts, serves dataset |
| `timing-manager` | Schedules requests, issues credits |
| `records-manager` | Aggregates results from workers |
| `api` | WebSocket + HTTP on port 9090 |
| `gpu-telemetry-manager` | GPU metrics via DCGM (optional) |
| `server-metrics-manager` | Prometheus metrics (optional) |
| `results-sidecar` | Serves exported results after controller exit (port 9091) |
| `event-bus-proxy` | XPUB/XSUB ZMQ proxy sidecar (`AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED=true` by default; isolates pub/sub I/O from the SystemController process) |

Per-container health ports run 8080-8088 (`AIPERF_K8S_PORT_*_HEALTH`); the API
service is on 9090 and the results sidecar on 9091.

**Worker pod (x N):**

| Container | Role |
|---|---|
| `worker-group-manager` | Group-local readiness, dataset download, proxy |
| `worker-0` .. `worker-{N}` | One worker per container; each makes LLM API calls |
| `record-processor-0` .. `record-processor-{M}` | One record processor per container; each computes metrics per record |

### RBAC Permissions

The Role rules are the `_RULES` class-var on the resource builder in
`src/aiperf/kubernetes/resources.py`:

| API group | Resources | Verbs |
|---|---|---|
| `aiperf.nvidia.com` | `aiperfjobs`, `aiperfjobs/status` | get, list, watch, patch, update |
| `""` (core) | `configmaps` | get, list, watch, create, update, patch, delete |
| `""` (core) | `pods`, `pods/log` | get, list, watch |
| `""` (core) | `services`, `endpoints` | get, list, watch, create, delete |
| `""` (core) | `events` | get, list, watch, create, patch |
| `batch` | `jobs` | get, list, watch |
| `jobset.x-k8s.io` | `jobsets` | get, list, watch, create, update, patch, delete |
| `jobset.x-k8s.io` | `jobsets/status` | get, list, watch |

## 4. Inter-Pod Communication

### Network Topology

```mermaid
flowchart TD
    DNS["Kubernetes DNS<br/>{jobset}-controller-0-0.{jobset}.{ns}.svc.cluster.local"]
    DNS --> W0["Worker pod 0"]
    DNS --> W1["Worker pod 1"]
    DNS --> WN["Worker pod N"]
```

Each worker pod resolves the controller through that headless-service DNS name,
injected via the ZMQ controller-host environment variable
(`AIPERF_K8S_PORT_*` settings in `src/aiperf/kubernetes/environment.py`).

### Communication Channels

| Channel | Port | Protocol | Purpose |
|---------|------|----------|---------|
| ZMQ Event Bus | IPC + TCP | ZMQ PUB/SUB | Message broadcasting |
| API Service | 9090 | HTTP/WS | WebSocket streaming, Dataset API |
| Health | 8080 | HTTP | Kubernetes probes |

### Dual-Bind ZMQ Configuration

```mermaid
flowchart TD
    CP["Controller pod"]
    CP --> IPC["IPC socket: event_bus_proxy_frontend.ipc<br/>used by control-plane services in the same pod"]
    CP --> TCP["TCP sockets: 0.0.0.0:5663 (frontend), :5664 (backend)<br/>used by worker pods (external)"]
```

Port defaults live on the `AIPERF_K8S_PORT_` settings in
`src/aiperf/kubernetes/environment.py`.

## 5. Dataset Transfer

In Kubernetes mode, the DatasetManager streams conversations directly to zstd-compressed files.
Workers download the compressed files via HTTP and decompress locally for memory-mapped access.

### Metadata Synchronization

The API Service waits for dataset metadata before serving files:

```mermaid
sequenceDiagram
    participant DM as DatasetManager
    participant API as API Service
    participant WGM as WorkerGroupManager

    DM->>DM: stream_writer.write() (zstd streaming to .zst)
    DM->>DM: finalize() + compress index
    DM->>API: DatasetConfiguredNotification (ZMQ pub/sub)
    Note over API: sets the dataset-configured event<br/>and caches the client metadata
    WGM->>API: GET /api/dataset/data (Accept-Encoding: zstd)
    Note over API: waits on the dataset-configured event,<br/>serves paths from the metadata
    API-->>WGM: stream .zst as-is (Content-Encoding: zstd)
    Note over WGM: decompress, then mmap locally
    WGM->>API: GET /api/dataset/index
    API-->>WGM: stream .zst as-is
    Note over WGM: decompress
```

Net effect: only `.zst` files exist on the control plane, file serving is
metadata-driven, and each worker pod ends up with local `.dat` files for mmap
access.

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **DatasetManager** | Streams to `.zst`, broadcasts `DatasetConfiguredNotification` with `MemoryMapClientMetadata` |
| **API Service** | Waits for notification via `asyncio.Event`, serves files using paths from metadata |
| **WorkerGroupManager** | Downloads via HTTP, decompresses within its Kubernetes worker pod, then exposes dataset readiness and current-state snapshots to sibling workers |

### Benefits

| Approach | Disk on Controller | Transfer | CPU Overhead |
|----------|-------------------|----------|--------------|
| **compress_only mode** | Compressed only | Passthrough | Compress once, decompress distributed |
| On-the-fly compression | Uncompressed + compressed | Re-compress per request | High on controller |

### Files Created

**Controller (DatasetManager):**
```
{mmap_base}/aiperf_mmap_{benchmark_id}/
├── dataset.dat.zst   # zstd-compressed conversations (streaming write)
└── index.dat.zst     # zstd-compressed byte offset index
```

**Workers (after download):**
```
{mmap_base}/aiperf_mmap_{benchmark_id}/
├── dataset.dat       # Decompressed conversations (mmap target)
└── index.dat         # Decompressed index
```

## 6. Benchmark Execution Flow

```mermaid
flowchart LR
    A[Pods start] --> B[Dataset ready]
    B --> C[Timing credits]
    C --> D[Workers execute]
    D --> E[Metrics compute]
    E --> F[Records aggregate]
    F --> G[Results export]
```

### Detailed Steps

1. **Pods Start** - Control-plane services register with `SystemController`, and each worker pod brings up one `WorkerGroupManager` as the controller-facing authority for that group
2. **DatasetManager** - Generates prompts, serves via HTTP at `/api/dataset`
3. **WorkerGroupManager** - Downloads dataset files once per Kubernetes worker pod, publishes current state, and makes sibling workers dispatchable only after readiness converges
4. **TimingManager** - Schedules requests, issues credits to workers
5. **Workers** - Make LLM API calls once their `WorkerGroupManager` reports group-local readiness, then generate raw records
6. **RecordProcessor** - Computes metrics (latency, TTFT, throughput)
7. **RecordsManager** - Aggregates results from all workers

### Service Discovery

`KubernetesServiceManager` in `src/aiperf/controller/kubernetes_service_manager.py`:

| Method | Behavior |
|--------|----------|
| `run_service()` | For service types in `EXTERNAL_K8S_SERVICES`, spawns nothing — records the expected instance count with `ServiceRegistry` and waits for the sibling container/worker pod to register. Anything else falls through to `MultiProcessServiceManager.run_service` and spawns a subprocess. |
| `stop_service()` | No-op for externally managed pods (they get shutdown over the control channel and exit on their own); otherwise delegates to the multiprocess manager. |
| `shutdown_all_services()` | Stops any locally managed subprocesses and releases the `kubernetes_asyncio` API client. |
| `check_pods_healthy()` / `_monitor_worker_pods()` | Polls worker-pod status so a crash-looping or evicted pod surfaces as a controller-side failure. |

`wait_for_all_services_registration()` is inherited from `BaseServiceManager` and is what actually blocks until every expected service has registered over ZMQ.

`SystemController` consumes both halves: `_verify_pods_healthy()` gates `PROFILE_START` on `check_pods_healthy()` (catching a pod that registered and then died), and `_watch_pod_failure_abort()` waits on `pod_failure_abort_event` so a mid-run breach of `AIPERF_POD_FAILURE_ABORT_THRESHOLD_PERCENT` cancels the benchmark through the same path as Ctrl+C. `BaseServiceManager` supplies inert defaults, so non-Kubernetes modes need no branch at the call site.

### Cross-Pod Clock Correction

Controller and worker pods have independent clocks. Every credit carries the controller's `issued_at_ns`, so each receipt is a one-way offset sample (`received - issued`) fed to `ClockOffsetTracker`, which min-filters a 20-sample window (NTP RFC 5905 clock filter) to reject transit jitter. At startup a worker sends `TimePing` on its credit DEALER; `StickyCreditRouter._handle_time_ping` echoes `TimePong` verbatim so RTT is measured entirely against the worker's own clock. Every `RequestRecord` leaving a worker is stamped with `clock_offset_ns`; the contract is `controller_time = worker_time - clock_offset_ns`. Both sampling and probing are gated on Kubernetes mode (`Worker._tracks_clock_offset`) — in local mode both sides share one clock, so records carry `None`.

## 7. Results Collection

### Data Flow

```mermaid
flowchart LR
    W["Workers (raw records)"] --> RP[RecordProcessor]
    RP --> RM[RecordsManager]
    RM --> API["API Service (port 9090)"]
```

In operator mode, results are served by the `results-server` container inside the operator deployment (port 8081), which reads from the operator PVC. In direct mode (no operator) or when the operator PVC is unavailable, results can be copied from the controller pod's `results-sidecar` or via `kubectl cp`.

The same `results-server` container also hosts every `/api/v1/*` router for the operator (jobs, sweeps, results, config, admin, analytics, dashboard_proxy) — there is no separate FastAPI app in the operator container, which only runs kopf (`/healthz` on port 8080 and the Prometheus `/metrics` endpoint on port 9090). The sweep-controller's empty-summary fallback (`K8sChildJobExecutor._fetch_summary_from_operator`) targets this container too via `AIPERF_OPERATOR_BASE_URL` (default `http://aiperf-operator.aiperf-system:8081`); it does not point at the operator container's port 8080.

### Retrieval Methods

```bash
# Operator mode (default): fetch via operator results-server
aiperf kube results {job_id}

# Direct mode / fallback: copy from the controller pod
aiperf kube results {job_id} --from-pods
kubectl cp <controller-pod>:/results ./results -n <namespace>
```

## Results layout and history

Each AIPerfJob submission lands in its own artifact directory keyed by the CR's
**creationTimestamp epoch** (seconds since 1970). Re-creating a CR with the
same name never overwrites prior results.

On-disk shape under the operator's results PVC (`AIPERF_RESULTS_DIR`, default
`/data`):

```text
<base>/<namespace>/<name>/
  <epoch-A>/   ← run A artifacts
  <epoch-B>/   ← run B artifacts
  latest.txt   ← pointer to the epoch of the most recent successful run
```

The pointer file is written atomically (staged write + `os.replace`) at the
single success gate in `handlers/completion.py`, alongside `status.resultsPath`
and `status.runEpoch`. A retention pass (env `AIPERF_RESULTS_RETAIN_RUNS`,
default 10) trims older run dirs on every successful completion; the just-written
epoch is always protected from deletion.

### HTTP API

Non-epoch routes now require an explicit run epoch and return `409 Conflict` if
omitted. Callers must use the `/runs/<epoch>` form; there is no implicit
"latest run" fallback (`_require_epoch_for_results` raises `HTTPException(409)`
for all non-epoch routes in `src/aiperf/operator/routers/results_files.py`).

| Route | Behavior |
|---|---|
| `GET /api/v1/results/<ns>/<name>` | **409** — epoch required |
| `GET /api/v1/results/<ns>/<name>.zip` | **409** — epoch required |
| `GET /api/v1/results/<ns>/<name>/<filename>` | **409** — epoch required |
| `GET /api/v1/results/<ns>/<name>/runs/<epoch>` | List files from the pinned run |
| `GET /api/v1/results/<ns>/<name>/runs/<epoch>.zip` | Zip bundle of the pinned run |
| `GET /api/v1/results/<ns>/<name>/runs/<epoch>/<filename>` | Download one file from the pinned run |

`<epoch>` is validated against `EPOCH_RE` (`^\d{9,20}$`, in `src/aiperf/operator/results_layout.py`) before any disk access.

### Edge cases

- **Rapid delete + resubmit within the same wall-clock second** produces colliding
  epoch keys; the second submission overwrites the first. This matches the
  legacy `EPOCH=$(date +%s)` semantics and is intentional.
- **A delayed older completion never rolls the pointer backward.**
  `write_latest` reads the current pointer first and no-ops when it already
  names a numerically newer epoch.
- **`latest.txt` points at a missing directory** (corruption, manual delete):
  default-route requests return 404 until the next successful completion
  rewrites the pointer. Historical routes still work.

### Runs/sweep index writes

The operator maintains a SQLite index at `<RESULTS.DIR>/.aiperf_index.sqlite` that mirrors disk state for fast queries. Writes happen at fixed handler points:

```mermaid
sequenceDiagram
    participant K as kopf
    participant O as operator
    participant FS as PVC
    participant DB as runs_index

    K->>O: on_create(AIPerfJob)
    O->>FS: save_job_spec_file
    O->>DB: upsert_run_created (Pending)

    Note over O: phase transitions (Running, Aggregating, ...)
    O->>DB: upsert_run_phase

    K->>O: completion observed
    O->>FS: download results, write ready marker
    O->>DB: upsert_run_completed + set_latest

    K->>O: on_delete or retention
    O->>FS: rm -rf run dir
    O->>DB: delete_run
```

Read sites (`results_layout.list_runs_async`, `results_db.ResultsDB`, `routers/results_files.py`) consult the index first and fall back to disk only when a row is missing, firing a lazy backfill in the background.

CR deletion wins every race with completion. The delete handler first records a
sticky cancellation event. The result harvester waits on that event alongside
metrics requests, primary and sidecar downloads, and retry backoff, so deletion
interrupts long I/O instead of waiting for its timeout. Metrics and completed
downloads from earlier boundaries remain available for recovery, but completion
does not publish terminal status or a latest pointer after cancellation.
Completion also rechecks cancellation after retention and every index await. If
cancellation lands while an upsert is in flight, the completion writer deletes
that exact epoch after the upsert returns; it cannot recreate an orphan row after
delete cleanup has already passed. Set cancellation flags are never evicted to
satisfy the progress-client cache bound, including when a large sweep deletes
thousands of children concurrently.

## 8. Completion & Cleanup

### Lifecycle

```mermaid
flowchart LR
    Deploy --> Running --> Complete --> TTL[TTL expires] --> Deleted
```

### Pending pod reconciliation

The Pod watch classifies each changed Pod body directly and records the
highest-priority startup blocker in `status.startupIssue` without listing the
JobSet's Pods. A healthy update for the same Pod clears the blocker. The
fingerprint and `firstObservedTime` make the grace period survive operator
restarts. The same diagnostic is exposed through `WorkersReady=False`, and a
Kubernetes warning event is emitted after
`AIPERF_K8S_WATCHDOG_PENDING_THRESHOLD_SECONDS`.

A bounded deadline handler re-evaluates only an already-cached
`status.startupIssue`; it does not list Pods or run the broad recovery engine.
This lets an unchanged blocker reach its warning or failure threshold even
when the controller heartbeat remains healthy and no further Pod event occurs.
Before deleting the JobSet it re-reads and validates the exact parent UID,
resource version, non-terminal phase, cancellation state, and blocker
fingerprint. It revalidates again after deletion and commits through an atomic
UID/resource-version/phase/full-startup-issue JSON fence, so a concurrent
recovery or terminal transition remains authoritative.

Image pull, container configuration, repeated crash-loop, and structural
scheduling failures may transition the AIPerfJob to `Failed` only after the
blocker remains unchanged for
`AIPERF_K8S_WATCHDOG_PENDING_CRITICAL_THRESHOLD_SECONDS`. Capacity shortages,
pending PVC binding, unknown scheduler reasons, and Kueue suspension remain
retryable: they are visible in status but do not auto-fail when the job timeout
is disabled. The operator checks both regular and init-container statuses and
`PodScheduled=False` conditions, using its shared `k8s_client` context.

### Lifecycle surface: `status.subPhase`

The controller pushes `status.subPhase` from its `SystemState`
(`src/aiperf/common/enums/enums.py`) directly to the AIPerfJob at least every
10 seconds, including quiet states with unchanged progress. It is distinct
from `status.phase` (the operator's own view) and `status.currentPhase` (the
per-benchmark stage), and is cleared on terminal transitions.

Focused watches maintain the operator-owned coarse lifecycle while the broad
recovery engine is gated. JobSet `replicatedJobsStatus` changes update worker
counts and `WorkersReady`, promoting Pending or Queued jobs to Initializing
when workers start. Controller `subPhase` transitions promote the job to
Running at profiling or later states. Event-authored status commits re-read the
live parent, reject terminal or cancelled jobs, and JSON-patch-test both the
resource version and expected live phase so a stale JobSet or Pod callback
cannot reverse a concurrent completion or cancellation. Resource-version or
JSON-test conflicts raise a bounded Kopf retry; the retried watch handler
re-reads the parent and rebuilds its status update, so one-shot readiness and
healthy-Pod clears are not lost to controller heartbeat writes. The subphase
watch uses this same direct fence and never writes through an ordinary Kopf
merge patch.

`status.currentPhase` preserves the user-provided phase name, such as
`cache_prime` or `steady_state_profile`; it is not restricted to the legacy
`warmup` and `profiling` names. Each `status.phases.<name>` entry includes
`phaseName`, the closed semantic `phaseKind` (`warmup` or `profiling`),
`phaseIndex`, and `profilingIndex`. The operator uses `phaseKind` for lifecycle
behavior: warmup-kind phases remain in Initializing, while a profiling-kind
phase promotes the job to Running and gates completion regardless of its name.
For stable `kubectl get` columns, the operator also projects the latest
profiling-kind phase's counters to top-level `status.requestsCompleted`,
`status.requestsTotal`, and `status.requestsPerSecond`; printer columns never
assume a phase is literally named `profiling`.

The controller-side `ProgressRouter` mirrors progress annotations to both the
JobSet and AIPerfJob and patches the AIPerfJob status. It also refreshes
the UID-fenced `aiperf.nvidia.com/controller-heartbeat` annotation on every
push.

`status.currentPhase` is written by that same push
(`_push_aiperfjob_status` in `src/aiperf/api/routers/progress.py`): it names the
most recently started phase, mirroring `JobProgress.current_phase`, and prefers
phases with an explicit identity over legacy aggregate entries. The written
value is always a key of the `status.phases` map it is pushed with — a phase
that has started but not yet sent a request is normally omitted from that map,
so the push emits a zeroed entry for it rather than naming a missing key.
Consumers such as `_requests_progress_percent` silently fall back to
alphabetized iteration (which resolves to warmup's 100%) on a miss, so an
unresolvable pointer is worse than omitting the key entirely.

The status push is fenced against terminal transitions. kopf clears
`currentPhase` and `subPhase` when it stamps a terminal phase
(`StatusBuilder.set_phase`), and a push that passed the UID fence just before
that could otherwise resurrect both keys. When the CR already carries a
`status.phase`, the push therefore goes out as `application/json-patch+json`
with a leading `test` op on `/status/phase`, so the apiserver rather than
wall-clock ordering settles the race. A rejection whose body identifies the
failed `test` op is logged at debug and dropped — that is the fence working;
any other rejection is logged at warning with the response body and re-raised,
so a CRD schema violation in the payload cannot masquerade as a lost race and
silently stop status updates. A push observing an already-terminal phase is
skipped outright, payload and all, rather than being trimmed to the non-racy
keys, so it can never overwrite the completion handler's final `summary`. Only a
CR with no `status.phase` at all still uses the plain merge patch — there is no
terminal value to race against yet, and a `test` op on an absent path would
fail with 422.

`status.serverMetrics` rides on the same push. It is the dashboard's
non-WebSocket fallback: `job-detail.js` prefers the live per-job socket's
`serverSummary` and falls back to the CR value when the socket is down, which is
a common port-forward failure mode. The status subresource has a 1.5 MB
apiserver object ceiling that neither the WebSocket frame nor the in-memory REST
cache has, so the CR carries a **curated subset**, not the full export.
`project_server_metrics_for_cr`
(`src/aiperf/kubernetes/server_metrics_projection.py`) emits an explicit
allow-list of the ~20 metric names that
`src/aiperf/operator/ui/components/server-metrics/helpers.js` (`backendMetric`)
actually renders, and per series only its `endpoint_url`, `labels`, and the
`avg`/`max`/`rate`/`p99_estimate`/`count` stats — never raw samples, buckets, or
timeslices. It is an allow-list rather than a full copy minus excludes so it
cannot silently regrow as new server metrics are added; adding a metric to
`backendMetric` requires adding it to `CURATED_METRIC_NAMES` too.

Three limits bound the write. `AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_SERIES`
and `AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_LABELS` are cardinality sanity
bounds; overflow drops the offending metric **whole**, with a debug log. Nothing
is truncated — labels are the series identity, and a trimmed series list or label
set would decode as a valid-but-wrong aggregate rather than as missing data.
`MAX_SERIES` is counted per metric across all endpoints, so it must clear the
worker or GPU count of the largest deployment: a per-worker metric such as
`dynamo_component_kvstats_gpu_cache_usage_percent` otherwise vanishes whole and
takes its dashboard tile with it.

`AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_BYTES` is the authoritative backstop,
because the cardinality caps bound how *many* labels a series carries but not
how long each label string is. An over-budget projection carries no metrics.
This matters more than it looks: exceeding the apiserver's 1.5 MB object ceiling
rejects the whole status patch, `_write_status_patch` re-raises, and
`_patch_aiperfjob_status` swallows it at debug — so every other status update
(`phases`, `liveMetrics`, `resultsExported`, `controllerFailure`) stops silently
too. The projected value is also `scrub_non_finite`-cleaned, because a single NaN
gauge is an invalid JSON number that would reject the same patch the same way.

Note how the two limits interact before raising either. `MAX_SERIES` is only the
per-metric bound; the *total* is what `MAX_BYTES` sees. With all 20 allow-listed
metrics present, the byte budget binds at roughly 85 series per metric — well
below the 256 default — so on an all-20-metric or many-endpoint deployment
`MAX_BYTES` is the limit that actually fires. Raising `MAX_SERIES` past that
point does not buy more data; it hands control to `MAX_BYTES`, whose overflow
costs the **whole** panel rather than one metric. Raise `MAX_BYTES` alongside it,
or accept the per-metric drop.

Overflow does **not** omit the key. Omitting it would leave whatever snapshot
last fit sitting in the CR indefinitely — stale values indistinguishable from
live ones, which is the failure the snapshot semantic exists to prevent, and one
the panel cannot self-diagnose (`SummaryStrip` renders only a duration, never an
absolute `end_time`, so a frozen scrape window looks live). Instead the overflow
writes `{summary, metrics: {}, projection_dropped: true, projection_message}`,
which replaces the stale value, and `ServerMetricsSection` renders that flag as
an explicit "collected but too large to carry" card rather than the "no server
metrics collected" empty state — an operator who loses the panel should not go
debug their exporter. It is logged at warning, not debug. `summary` is kept only
if the marker itself fits, since endpoint URLs are unbounded too.

`status.serverMetrics` is a **snapshot** of the latest scrape, not an
accumulation. The push's JSON-patch path normally pre-resolves each key through
`_merge_patch_value` (RFC 7386) so a fenced write stays equivalent to the merge
patch it stands in for, but that recursively unions dicts — and this key is a
map of metric name to dict-valued stats, so a metric that stops being projected
would linger indefinitely with stale values indistinguishable from live ones.
The caps are a disappearance generator by design, so `serverMetrics` is listed in
`_SNAPSHOT_STATUS_KEYS` and emitted unresolved, letting the `add` op replace the
member outright.

Two consequences worth knowing. The CR fallback's per-endpoint details table
lists fewer source-metric rows than `server_metrics_export.json` does. And a
backend chip lights only when at least one of that backend's exposed metrics is
in the allow-list: `detectBackends` scans metric names on whatever payload it is
given, so any backend whose metrics do not intersect `CURATED_METRIC_NAMES` loses
its chip on the CR path. KVBM is the guaranteed case — `detectBackends` keys it
off a `kvbm_*` prefix and no `kvbm_*` metric appears in `backendMetric` — but it
is a class of gap, not a single instance. The live WebSocket path and the final
`server_metrics_export.json` are unaffected and keep the full payload.

The operator's recurring watchdog inspects only this cached parent body while
the heartbeat is fresh; broad JobSet, Pod, sidecar, and results recovery
runs only after heartbeat expiry or when an explicit `timeoutSeconds` deadline
is due. A controller service error is pushed as `status.controllerFailure`
before that controller exits; the operator fences and terminalizes the exact
parent as Failed, and never lets a later sidecar artifact salvage reinterpret
that explicit failure as a successful completion.

One transition is easy to misread: `SystemState.PROCESSING` is set when the
SystemController handles `CreditsCompleteMessage` — that is, when request
*dispatch* finishes and only record aggregation remains. It is **not** set at
profile completion; the whole aggregation phase happens while `subPhase` reads
`processing`.

For the same reason, the job-timeout drain guard in `_check_job_timeout` keys off
`status.resultsExported` — pushed by the controller only once every exporter has
flushed — and never off `status.currentPhase`. `currentPhase` is a pointer into
`status.phases` and carries user-supplied phase names, so a benchmark phase named
`processing` would otherwise bypass the timeout. A timed-out run is deferred to
the completion handler only when the completion claim is already held or
`resultsExported` is true; a run whose aggregation or export hangs still fails on
the deadline.

### Completion Signals

- Controller receives `ALL_RECORDS_RECEIVED` message
- Results available via API service
- Services shut down cleanly

For `exportLevel: raw`, result publication has an additional acknowledged
barrier before shutdown. While ZMQ and the group-local lifecycle channels are
still live, `SystemController` sends `FINALIZE_ARTIFACTS` to the exact set of
registered `WorkerGroupManager` service IDs. Each manager requires its exact
declared record-processor set to flush successfully, stops those processors,
waits for their exact shutdown notices, and uploads every materialized RAW
JSONL file. The controller API stages each upload under a temporary name,
fsyncs it, and atomically renames it before returning the size acknowledgement.
Only then does the manager acknowledge the controller command.

Any missing service, rejected RAW row, timeout, flush error, HTTP failure, or
size mismatch fails the barrier. The controller skips aggregation/export and
withholds both the
results-ready marker and `ResultsExportedMessage`, so an incomplete RAW result
set cannot be advertised as authoritative. Empty processors are valid and may
produce no file; completion is proven by service acknowledgements rather than
filename counts or file-size stability polling.

The controller performs final export before broadcasting service shutdown or
stopping its message bus. After the durable marker commits, the still-running
API service can therefore receive `ResultsExportedMessage`; the marker remains
the authoritative recovery signal if that live notification is lost.

### Cleanup Options

```bash
# Automatic (TTL-based)
ttlSecondsAfterFinished: 300  # Pods auto-delete after 5 minutes

# Manual cleanup (operator mode): delete the AIPerfJob CR
kubectl delete aiperfjob <name> -n <namespace>

# Manual cleanup (direct mode): delete the JobSet
kubectl delete jobset <name> -n <namespace>
```

## 9. Configuration

### CLI Options

```bash
aiperf kube profile \
  --image myregistry.io/aiperf:latest \
  --namespace benchmarks \
  --total-workers 10 \
  --ttl-seconds 300 \
  --kubeconfig ~/.kube/prod-config \
  --node-selector '{"nvidia.com/gpu": "A100"}' \
  --tolerations '[{"key":"nvidia.com/gpu","operator":"Exists"}]' \
  --image-pull-secrets registry-creds \
  --env-from-secrets 'OPENAI_API_KEY=llm-api-key/api-key'
```

Sensitive endpoint fields never rely on the ConfigMap copy. JSON
serialization redacts them, and `aiperf service --benchmark-run` restores them
from the Secret-backed `AIPERF_INJECTED_API_KEY`/`OPENAI_API_KEY`,
`AIPERF_INJECTED_HEADERS`, and `AIPERF_INJECTED_ENDPOINT_URLS` environment
variables. Generation and operator reconciliation fail closed when the
corresponding `valueFrom.secretKeyRef` mapping is absent.
`aiperf service` requires `--benchmark-run` and never resolves per-container
benchmark flags.

### Environment Variables

Resource limits configured via `src/aiperf/kubernetes/environment.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` | 75m | System controller container CPU (request and limit) |
| `AIPERF_K8S_DATASET_MANAGER_MEMORY` | 256Mi | Dataset manager container memory (request and limit) |
| `AIPERF_K8S_WORKER_POD_CPU` | 150m | Worker pod CPU (request and limit) |
| `AIPERF_K8S_WORKER_POD_MEMORY` | 4Gi | Worker pod memory (request and limit) |
| `AIPERF_K8S_PORT_API_SERVICE` | 9090 | API service port |
| `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` | 300 | TTL after completion |

### AIPerfSweep handlers

The kopf operator registers sweep-lifecycle handlers in `src/aiperf/operator/main.py`. Seven registrations are on the parent `AIPerfSweep` CRD; one more watches child `AIPerfJob`s to roll their status up into the parent:

- `@kopf.on.create AIPerfSweep` (handler in `handlers/sweep/create.py`) — validates the workload through the canonical Config-v2 mapping loader and `AIPerfSweepSpec`, computes `totalVariations`/`maxTotalRuns`, sets `status.runEpoch` to a collision-safe decimal key derived from `metadata.creationTimestamp` and immutable `metadata.uid`, provisions a namespace-scoped ServiceAccount/Role/RoleBinding for the sweep-controller pod, and creates a single-replica JobSet that runs `python -m aiperf.sweep_controller.main`.
- `@kopf.on.update AIPerfSweep field=spec.cancel` (handler in `handlers/sweep/lifecycle.py`) — mirrors the cancel signal into `status.conditions[Cancelling]` and advances `status.observedGeneration`, including terminal/no-op updates, so GitOps clients can distinguish an acknowledged spec change. The sweep-controller pod observes `spec.cancel` directly via its own poll and propagates it to the current child.
- `@kopf.on.update AIPerfSweep field=spec.ttlSecondsAfterFinished` — acknowledges the other mutable parent control immediately. The reaper timer reads the latest TTL; create-time execution fields are immutable after admission.
- `@kopf.on.field AIPerfSweep field=status.aggregation.phase new=Complete` plus `@kopf.on.resume` — triggers `handlers/sweep/_aggregate_fetch.fetch_sweep_aggregate_to_disk` to pull the cross-variation aggregate off the sweep-controller's `emptyDir` results-sidecar before the JobSet is reaped, and resumes an interrupted harvest after an operator restart. The fetch reports `(downloaded, listed)` counts; a partial harvest (`downloaded < listed`) or a missing/unparsable `aggregate.json` raises `kopf.TemporaryError` so the JobSet — and with it the only other copy of the artifacts — stays alive for re-harvest. During commit, the operator materializes every child `sweep.json` backlink on its PVC from the canonical `children.json` manifest before status publication. Delayed callbacks carry the parent CR's immutable UID, verify the live parent and exact JobSet owner API version/kind/name/UID with `controller: true`, and publish status with a JSON Patch UID test before advancing `latest.txt` or the runs index. Only a full harvest with a parseable `aggregate.json` and durable child lineage on the PVC publishes the operator-backed `aggregateRef`, flips `resultsAvailable` to true, and deletes that exact JobSet with its resource UID as a delete precondition. A same-name replacement makes the old callback a no-op; transient reads and status-validation failures against the current owner retry.
- `@kopf.on.delete AIPerfSweep` — cooperatively cancels only AIPerfJobs whose exact owner kind/name/UID matches the deleting sweep; sweep labels narrow discovery but never establish ownership. Kubernetes owner-reference GC tears down the sweep-controller JobSet and RBAC.
- `@kopf.timer` `cleanup_old_sweeps` — TTL reaper for terminal `AIPerfSweep`s, evaluated at the operator monitor cadence rather than the daily result-retention cadence. A completed aggregate is not eligible until the operator-backed result reference is published, so even `ttlSecondsAfterFinished: 0` cannot delete the only `emptyDir` copy during harvest. Parent deletion uses the timer body's immutable UID as a Kubernetes delete precondition, so a stale timer cannot reap a same-name replacement.
- `@kopf.on.field AIPerfJob field=status.phase` (handler in `handlers/sweep/child_rollup.py`) — this one is on **child AIPerfJobs**, not the AIPerfSweep CRD: for AIPerfJob children whose `ownerReferences` include an `AIPerfSweep`, it recomputes the parent's `runStates`/`currentChildRef`/`lastChildEvent`. Standalone AIPerfJobs are no-ops.

Sweep **result** retention is process-level rather than a CR timer. After the
runs-index bootstrap and once per day, the operator scans durable sweep epoch
directories, reads `aggregate.json.specSnapshot.resultsTtlDays` (falling back
to `AIPERF_RESULTS_TTL_DAYS` for legacy archives), removes expired archives
and SQLite rows, and reconciles `latest.txt`. This continues after the parent
CR's default 300-second lifecycle TTL has elapsed; bootstrap reverse-pruning
repairs an index delete interrupted by an operator crash.

The AIPerfJob CRD likewise permits only runtime-control edits after creation:
`spec.cancel` and `spec.timeoutSeconds`. Their dedicated field handlers advance
`status.observedGeneration` only after the edit is consumed successfully; a
failed JobSet deletion leaves a cancel edit unacknowledged for kopf to retry.

### Sweep plan convergence

Kubernetes and local sweeps use the same Config-v2 planning path. The kube CLI
keeps the post-environment, pre-Jinja template leaves in the submitted CR. The
operator and sweep-controller validate a rendered copy through
`load_config_from_mapping`, while retaining that raw envelope so each variation
can render its own values. The sweep-controller then calls
`build_benchmark_plan`; its only plan adaptation is attaching the
Kubernetes-only `failurePolicy`. Adaptive sweeps instantiate their planner
through the shared `build_search_planner` factory, and
`K8sChildJobExecutor` supplies the cluster execution backend behind the same
`RunExecutor` protocol used by local sweeps.

## CRD Generator

Both CRDs (`aiperfjobs.aiperf.nvidia.com`, `aiperfsweeps.aiperf.nvidia.com`)
are auto-generated by `tools/generate_crd.py` from the `AIPerfJobSpec` and
`AIPerfSweepSpec` Pydantic models (`src/aiperf/kubernetes/crd_models.py`).
Both inherit `AIPerfWorkloadSpec`, which composes the complete `AIPerfConfig`
envelope (`benchmark`, `sweep`, `multiRun`, `variables`, and related fields)
with the Kubernetes deployment surface. **Never edit the rendered YAML in
`deploy/helm/aiperf-operator/templates/crd*.yaml` directly** — the next
regeneration overwrites it.

### Generator pipeline

1. **JSON Schema walk** (`_convert_schema`) — recursively converts the
   Pydantic-emitted JSON Schema into K8s-compatible OpenAPI v3, resolving
   `$ref`, collapsing `anyOf`-with-null into nullables, and falling back to
   `x-kubernetes-preserve-unknown-fields: true` at narrow shorthand
   boundaries (`models`, `endpoint.urls`, top-level
   `model`/`dataset`/`warmup`/`profiling`, `sweep`).
2. **Type-on-marker pass** (`_ensure_type_on_preserve_unknown`) — defaults
   `type: object` on every node carrying `x-kubernetes-preserve-unknown-fields:
   true`. K8s structural-schema validation rejects the marker without a
   declared type, AND CEL field access compiles only on typed nodes.
3. **Shape-detector decorators** (`_decorate_*_node`) — each helper detects
   its target node by a unique fingerprint of property keys (e.g. an
   endpoint node has `urls` + `apiKey` + `connectionReuse`; a runtime node
   has `apiPort` + `apiHost` + `workersPerPod`). The walker
   (`_walk_dict_apply`) calls every decorator on every dict node, so the
   same set of CEL rules fires on both AIPerfJob's `spec.benchmark` and
   AIPerfSweep's `spec.benchmark` from a single pass.
4. **Kind-specific attachment** — sweep-only rules are added in
   `_build_aiperfsweep_crd_from_schema` by direct property indexing after the
   walker runs: `has(self.sweep)` makes the sweep block required on AIPerfSweep
   (the inverse `!has(self.sweep)` fires on AIPerfJob), plus `oldSelf == self`
   immutability on `spec.sweep` / `spec.multiRun`. The old Tier-1D rule that
   forbade `sweep`/`multi_run` inside the per-child benchmark was removed —
   `AIPerfJobSpec.benchmark` is typed as `BenchmarkConfig` (no such fields), so
   the generated structural schema enforces it at the apiserver without CEL.

### Adding a CEL rule

The user-facing catalog of every rule lives in
[`docs/kubernetes/crd-validation.md`](../kubernetes/crd-validation.md). To
add a new one:

1. Identify the **shape** the rule applies to (benchmark, endpoint,
   runtime, multiRun) and pick the matching `_decorate_*_node`
   helper. If your target is a brand new shape, write a new shape detector
   modelled on the existing ones.
2. Append a `{"rule": ..., "message": ...}` entry to that helper's
   `_add_validation_rules(...)` call. Tag the rule with the tier label
   (1A/1B/.../4O) in a comment so future-you can grep back to the
   brainstorm in `docs/kubernetes/crd-validation.md`.
3. Add a structural assertion in
   `tests/unit/operator/test_aiperfsweep_crd_generation.py` (the existing
   tests follow a "rule string is in `rules` set" pattern).
4. Regenerate with `uv run python tools/generate_crd.py` and confirm
   idempotency with `tools/generate_crd.py --check`.
5. Round-trip on a real apiserver (`kind create cluster && kubectl apply
   --dry-run=server -f crd.yaml`). The K8s apiserver compiles CEL at
   CRD-install time and rejects rules that reference undeclared fields
   (`undefined field 'X'`) or opaque preserve-unknown items.

CEL constraints worth remembering:

- `has(self.X)` requires X to be declared in the schema. Anything hidden
  inside a `x-kubernetes-preserve-unknown-fields: true` blob is invisible.
- Array items emitted as opaque preserve-unknown blobs cannot be
  dereferenced. Heterogeneous Pydantic discriminated unions
  (`phases[]`, `datasets[]`) end up opaque, so item-internal invariants
  (phase-name uniqueness, phase→dataset reference integrity, "seamless not
  on first") stay enforced in the operator's `@model_validator` decorators
  on `AIPerfConfig` rather than at the apiserver.
- `oldSelf` is only available in transition rules and triggers on
  `kubectl edit` / `kubectl patch`. Use `!has(oldSelf.X) || oldSelf.X ==
  self.X` for "first-set freezes after that" semantics (e.g.
  `artifacts.benchmarkId`, `scheduling.queueName`).

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **JobSet API** | Orchestrates controller + workers as atomic unit |
| **Dual-bind ZMQ** | IPC for in-pod speed, TCP for cross-pod reach |
| **API-based results** | Retrievable via API service or kubectl cp |
| **Dataset HTTP API** | Avoids shared volume complexity |
| **WebSocket streaming** | Real-time progress to local CLI |
| **Container-per-service** | One container per service; failure isolation and per-container resources |
