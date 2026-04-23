---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Kubernetes Flow
---

# Kubernetes Flow End-to-End

This document describes the complete flow from user command to benchmark completion when running AIPerf on Kubernetes.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER WORKSTATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  $ aiperf kube profile --model Qwen/Qwen3-0.6B --workers-max 10             │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌──────────────────────┐                           │
│                          │ Submit AIPerfJob CR  │                           │
│                          │ (or direct manifests │                           │
│                          │  if no operator)     │                           │
│                          └──────────┬───────────┘                           │
│                                     │                                       │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │ kubernetes_asyncio API
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUBERNETES CLUSTER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                    JobSet: aiperf-benchmark-{job_id}              │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │  ┌─────────────────────┐    ┌─────────────────────────────────┐  │     │
│   │  │   Controller Pod    │    │         Worker Pods (N)         │  │     │
│   │  │                     │    │                                 │  │     │
│   │  │  ┌───────────────┐  │    │  ┌─────────┐  ┌─────────┐      │  │     │
│   │  │  │ SystemCtrl    │  │◄───┼──│ Worker  │  │ Worker  │ ...  │  │     │
│   │  │  │ WorkerMgr     │  │    │  │ Pod 0   │  │ Pod 1   │      │  │     │
│   │  │  │ TimingMgr     │  │    │  └─────────┘  └─────────┘      │  │     │
│   │  │  │ DatasetMgr    │  │    │                                 │  │     │
│   │  │  │ RecordsMgr    │  │    └─────────────────────────────────┘  │     │
│   │  │  │ API Service   │  │                                         │     │
│   │  │  └───────────────┘  │                                         │     │
│   │  └─────────────────────┘                                         │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│   ┌───────────────────────┐                                               │
│   │ ConfigMap: config     │                                               │
│   │ - run_config.json     │                                               │
│   └───────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

`WorkerGroupManager` is the universal readiness and capacity authority for worker groups across local and Kubernetes mode. This document focuses on the Kubernetes deployment, where each worker group maps to one worker pod, but the group-local startup and dataset contract is intentionally the same one used by local worker groups.

## 1. CLI Entry Point

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B --url http://server:8000 --image aiperf:latest --workers-max 10
```

CLI commands defined in `src/aiperf/cli_commands/kube/`:

| Command | Purpose |
|---------|---------|
| `init` | Generate a starter configuration template |
| `validate` | Validate AIPerfJob YAML files against the CRD schema |
| `profile` | Run a benchmark in Kubernetes |
| `generate` | Generate Kubernetes YAML manifests |
| `attach` | Attach to a running benchmark and stream progress |
| `list` | List benchmark jobs and their status |
| `logs` | Retrieve logs from benchmark pods |
| `results` | Retrieve benchmark results |
| `debug` | Run diagnostic analysis on a deployment |
| `watch` | Watch a running benchmark with live status and diagnostics |
| `preflight` | Run pre-flight checks against the target cluster |
| `dashboard` | Open the operator results server UI in your browser |

## 2. Deployment Generation

The deployment logic in `src/aiperf/cli_commands/kube/profile.py` auto-detects whether the AIPerfJob CRD is installed. If the operator is present, `_deploy_via_operator()` submits an `AIPerfJob` custom resource and the operator reconciles it; otherwise `_deploy_direct()` creates the manifests (ConfigMap, Role, RoleBinding, JobSet) directly. `--no-operator` forces direct mode.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         aiperf kube profile                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. Resolve benchmark config    from CLI flags or AIPerfJob CR YAML        │
│                                                                            │
│  2. Configure ServiceConfig     service_run_type = KUBERNETES              │
│                                 dataset_api_base_url = controller DNS      │
│                                                                            │
│  3. Detect operator             query for AIPerfJob CRD                    │
│                                                                            │
│  4. Operator mode:              create AIPerfJob CR (operator reconciles)  │
│     Direct mode:                create ConfigMap + RBAC + JobSet directly  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## 3. Kubernetes Resources

### Resource Creation Order

```
Namespace (if auto-generated)
    │
    ▼
  Role ──────────────► RoleBinding
    │                      │
    │   ┌──────────────────┘
    ▼   ▼
ConfigMap ──────────────► JobSet
                             │
                             ├──► controller (1 pod)
                             │
                             └──► workers (N pods)
```

### Pod Architecture

Each control-plane service runs in its own container in the controller pod
(sibling containers, not subprocesses). Workers and record processors
likewise each run in their own container inside a worker pod.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROLLER POD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Containers (one per service):                                              │
│    control-plane          SystemController (orchestration)                  │
│    dataset-manager        Generates prompts, serves dataset                 │
│    timing-manager         Schedules requests, issues credits                │
│    records-manager        Aggregates results from workers                   │
│    api                    WebSocket + HTTP on port 9090                     │
│    gpu-telemetry-manager  GPU metrics via DCGM (optional)                   │
│    server-metrics-manager Prometheus metrics (optional)                     │
│    results-sidecar        Serves exported results after controller exit     │
│    event-bus-proxy        XPUB/XSUB ZMQ proxy sidecar (AIPERF_K8S_EVENT_     │
│                           BUS_SIDECAR_ENABLED=true by default; isolates     │
│                           pub/sub I/O from the SystemController process)    │
│                                                                             │
│  Per-container health ports (8080-8088); API on 9090; results-sidecar 9091  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKER POD (x N)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Containers:                                                                │
│    worker-group-manager   Group-local readiness, dataset download, proxy    │
│    worker-0..N            Each worker makes LLM API calls (one per ctnr)    │
│    record-processor-0..M  Each computes metrics per record (one per ctnr)   │
│                                                                             │
│  Per-container health ports (starting at 8080)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RBAC Permissions

Role grants access to: `configmaps`, `pods`, `pods/log`, `services`, `endpoints`, `events`, `jobs`, `jobsets`

## 4. Inter-Pod Communication

### Network Topology

```
                              Kubernetes DNS
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    │                               ▼                               │
    │   ┌───────────────────────────────────────────────────────┐   │
    │   │  {jobset}-controller-0-0.{jobset}.{ns}.svc.cluster.local  │
    │   └───────────────────────────────────────────────────────┘   │
    │                               │                               │
    │               ┌───────────────┼───────────────┐               │
    │               │               │               │               │
    │               ▼               ▼               ▼               │
    │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   │
    │   │  Worker Pod 0 │   │  Worker Pod 1 │   │  Worker Pod N │   │
    │   │               │   │               │   │               │   │
    │   │  AIPERF_ZMQ_  │   │  AIPERF_ZMQ_  │   │  AIPERF_ZMQ_  │   │
    │   │  CONTROLLER_  │   │  CONTROLLER_  │   │  CONTROLLER_  │   │
    │   │  HOST=...     │   │  HOST=...     │   │  HOST=...     │   │
    │   └───────────────┘   └───────────────┘   └───────────────┘   │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
```

### Communication Channels

| Channel | Port | Protocol | Purpose |
|---------|------|----------|---------|
| ZMQ Event Bus | IPC + TCP | ZMQ PUB/SUB | Message broadcasting |
| API Service | 9090 | HTTP/WS | WebSocket streaming, Dataset API |
| Health | 8080 | HTTP | Kubernetes probes |

### Dual-Bind ZMQ Configuration

```
Controller Pod (services within same pod)
    │
    ├── IPC Socket: ipc:///aiperf/ipc/event_bus_proxy_frontend.ipc
    │   └── Used by: SystemController, WorkerMgr, TimingMgr, etc.
    │
    └── TCP Socket: tcp://0.0.0.0:5663 (frontend), :5664 (backend)
        └── Used by: Worker pods (external)
```

## 5. Dataset Transfer

In Kubernetes mode, the DatasetManager streams conversations directly to zstd-compressed files.
Workers download the compressed files via HTTP and decompress locally for memory-mapped access.

### Metadata Synchronization

The API Service waits for dataset metadata before serving files:

```
DatasetManager                      API Service                    WorkerGroupManager
     │                                   │                               │
     │  stream_writer.write()            │                               │
     │  (zstd streaming to .zst)         │                               │
     │                                   │                               │
     │  finalize() + compress index      │                               │
     │                                   │                               │
     │                                   │                               │
     │  DatasetConfiguredNotification    │                               │
     │  ═══════════════════════════════► │                               │
     │  (via ZMQ pub/sub)                │                               │
     │  • data_file_path                 │ _dataset_configured.set()     │
     │  • index_file_path                │ _dataset_client_metadata=...  │
     │  • compressed_data_file_path      │                               │
     │  • compressed_index_file_path     │                               │
     │                                   │                               │
     │                                   │◄─── GET /api/dataset/data ────┤
     │                                   │     Accept-Encoding: zstd     │
     │                                   │                               │
     │                                   │ wait_for(_dataset_configured) │
     │                                   │ use metadata.compressed_*     │
     │                                   │                               │
     │                                   │──── stream .zst as-is ───────►│
     │                                   │     Content-Encoding: zstd    │
     │                                   │                               │
     │                                   │                     decompress │
     │                                   │                     ──────────►│
     │                                   │                     mmap local │
     │                                   │                               │
     │                                   │◄─── GET /api/dataset/index ───┤
     │                                   │                               │
     │                                   │──── stream .zst as-is ───────►│
     │                                   │                               │
     │                                   │                     decompress │
     │                                   │                     ──────────►│
     │                                   │                               │
     ▼                                   ▼                               ▼
  Only .zst files                  Metadata-driven                Local .dat files
  on control plane                 file serving                   for mmap access
```

### Key Components

| Component | Responsibility |
|-----------|----------------|
| **DatasetManager** | Streams to `.zst`, broadcasts `DatasetConfiguredNotification` with `MemoryMapClientMetadata` |
| **API Service** | Waits for notification via `asyncio.Event`, serves files using paths from metadata |
| **WorkerGroupManager** | Downloads via HTTP, decompresses locally, then exposes group-local dataset readiness and current-state snapshots to sibling workers using the same readiness contract local mode uses |

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

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Pods      │    │   Dataset   │    │   Timing    │    │   Workers   │
│   Start     │───►│   Ready     │───►│   Credits   │───►│   Execute   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│   Results   │◄───│   Records   │◄───│   Metrics   │◄─────────┘
│   Export    │    │   Aggregate │    │   Compute   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Detailed Steps

1. **Pods Start** - Control-plane services register with `SystemController`, and each worker pod brings up one `WorkerGroupManager` as the controller-facing authority for that group
2. **DatasetManager** - Generates prompts, serves via HTTP at `/api/dataset`
3. **WorkerGroupManager** - Downloads dataset files once per group, publishes group-local current state, and makes sibling workers dispatchable only after group readiness converges
4. **TimingManager** - Schedules requests, issues credits to workers
5. **Workers** - Make LLM API calls once their `WorkerGroupManager` reports group-local readiness, then generate raw records
6. **RecordProcessor** - Computes metrics (latency, TTFT, throughput)
7. **RecordsManager** - Aggregates results from all workers

### Service Discovery

`KubernetesServiceManager` in `src/aiperf/controller/kubernetes_service_manager.py`:

| Method | Behavior |
|--------|----------|
| `run_service()` | No-op (pods created by JobSet) |
| `stop_service()` | No-op (JobSet manages lifecycle) |
| `wait_for_all_services_registration()` | Waits for ZMQ registration |
| `wait_for_all_services_start()` | Waits for RUNNING state |

## 7. Results Collection

### Data Flow

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Workers   │────────►│  Records    │────────►│  Records    │
│  (records)  │         │  Processor  │         │  Manager    │
└─────────────┘         └─────────────┘         └──────┬──────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   API Service    │
                                              │   (port 9090)    │
                                              └─────────────────┘
```

In operator mode, results are served by the `results-server` container inside the operator deployment (port 8081), which reads from the operator PVC. In direct mode (no operator) or when the operator PVC is unavailable, results can be copied from the controller pod's `results-sidecar` or via `kubectl cp`.

### Retrieval Methods

```bash
# Operator mode (default): fetch via operator results-server
aiperf kube results {job_id}

# Direct mode / fallback: copy from the controller pod
aiperf kube results {job_id} --from-pods
kubectl cp <controller-pod>:/results ./results -n <namespace>
```

## 8. Completion & Cleanup

### Lifecycle

```
Deploy ──► Running ──► Complete ──► TTL Expires ──► Deleted
```

### Completion Signals

- Controller receives `ALL_RECORDS_RECEIVED` message
- Results available via API service
- Services shut down cleanly

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
  --workers-max 10 \
  --ttl-seconds 300 \
  --kubeconfig ~/.kube/prod-config \
  --node-selector '{"nvidia.com/gpu": "A100"}' \
  --tolerations '[{"key":"nvidia.com/gpu","operator":"Exists"}]' \
  --image-pull-secrets registry-creds \
  --env-vars 'HF_TOKEN:my-token,API_KEY:abc123'
```

### Environment Variables

Resource limits configured via `src/aiperf/kubernetes/environment.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` | 500m | System controller container CPU (request and limit) |
| `AIPERF_K8S_DATASET_MANAGER_MEMORY` | 2Gi | Dataset manager container memory (request and limit) |
| `AIPERF_K8S_WORKER_POD_CPU` | 4000m | Worker pod CPU (request and limit) |
| `AIPERF_K8S_WORKER_POD_MEMORY` | 12Gi | Worker pod memory (request and limit) |
| `AIPERF_K8S_PORT_API_SERVICE` | 9090 | API service port |
| `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` | 300 | TTL after completion |

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **JobSet API** | Orchestrates controller + workers as atomic unit |
| **Dual-bind ZMQ** | IPC for in-pod speed, TCP for cross-pod reach |
| **API-based results** | Retrievable via API service or kubectl cp |
| **Dataset HTTP API** | Avoids shared volume complexity |
| **WebSocket streaming** | Real-time progress to local CLI |
| **Container-per-service** | One container per service; failure isolation and per-container resources |
