# Kubernetes Flow End-to-End

This document describes the complete flow from user command to benchmark completion when running AIPerf on Kubernetes.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER WORKSTATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  $ aiperf kube profile --model Qwen/Qwen3-0.6B --workers 10                 │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─────────────────┐                                │
│                          │  Generate Job   │                                │
│                          │  ID & Manifests │                                │
│                          └────────┬────────┘                                │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │ kubectl apply
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
│   │ - user_config.json    │                                               │
│   │ - service_config.json │                                               │
│   └───────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. CLI Entry Point

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B --url http://server:8000 --image aiperf:latest --workers 10
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

## 2. Deployment Generation

The deployment logic in `src/aiperf/cli_commands/kube/profile.py` (`_deploy_direct()`) orchestrates deployment:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         run_kubernetes_deployment()                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. Generate Job ID              job_id = uuid4().hex[:8]  → "a1b2c3d4"   │
│                                                                            │
│  2. Configure ServiceConfig      service_run_type = KUBERNETES             │
│                                  zmq_dual_bind = IPC + TCP                 │
│                                  dataset_api_base_url = controller:9090    │
│                                                                            │
│  3. Calculate Pod Distribution   workers=100, workers_per_pod=10 → 10 pods │
│                                                                            │
│  4. Generate Manifests           resources.py + jobset.py                  │
│                                                                            │
│  5. Apply to Cluster             kubectl apply -f manifests                │
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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROLLER POD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Container: control-plane                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Subprocess Tree                                │ │
│  │                                                                        │ │
│  │   SystemController (main process)                                      │ │
│  │        │                                                               │ │
│  │        ├── WorkerManager          Manages worker lifecycle             │ │
│  │        ├── TimingManager          Schedules requests, issues credits   │ │
│  │        ├── DatasetManager         Generates prompts, serves dataset    │ │
│  │        ├── RecordsManager         Aggregates results from workers      │ │
│  │        ├── API Service            WebSocket + HTTP on port 9090        │ │
│  │        ├── GPUTelemetryManager    GPU metrics via DCGM (optional)      │ │
│  │        └── ServerMetricsManager   Prometheus metrics (optional)        │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Ports: 8080 (health), 9090 (API)                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKER POD (x N)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Container: worker-pod-manager                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Subprocess Tree                                │ │
│  │                                                                        │ │
│  │   WorkerPodManager (main process)                                      │ │
│  │        │                                                               │ │
│  │        ├── Worker[0]              Makes LLM API calls                  │ │
│  │        ├── Worker[1]                                                   │ │
│  │        ├── ...                                                         │ │
│  │        ├── Worker[N]                                                   │ │
│  │        │                                                               │ │
│  │        └── RecordProcessor        Computes metrics per record          │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Port: 8080 (health)                                                        │
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
DatasetManager                      API Service                    WorkerPodManager
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
| **WorkerPodManager** | Downloads via HTTP, decompresses locally, notifies workers via `DatasetDownloadedNotification` |

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

1. **Pods Start** - Services register with SystemController via ZMQ
2. **DatasetManager** - Generates prompts, serves via HTTP at `/api/dataset`
3. **TimingManager** - Schedules requests, issues credits to workers
4. **Workers** - Make LLM API calls, generate raw records
5. **RecordProcessor** - Computes metrics (latency, TTFT, throughput)
6. **RecordsManager** - Aggregates results from all workers

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

Results are available via the API service or by copying files from the controller pod.

### Retrieval Methods

```bash
# Via API service (preferred)
aiperf kube results {job_id}

# Copy from pod
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

# Manual cleanup
aiperf kube delete {job_id}                    # Delete JobSet + resources
aiperf kube delete {job_id} --delete-namespace # Also delete namespace
aiperf kube cancel {job_id}                    # Stop running benchmark
```

## 9. Configuration

### CLI Options

```bash
aiperf kube profile \
  --image myregistry.io/aiperf:latest \
  --namespace benchmarks \
  --workers 10 \
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
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` | 250m | System controller container CPU (request and limit) |
| `AIPERF_K8S_DATASET_MANAGER_MEMORY` | 512Mi | Dataset manager container memory (request and limit) |
| `AIPERF_K8S_WORKER_POD_CPU` | 3350m | Worker pod CPU (request and limit) |
| `AIPERF_K8S_WORKER_POD_MEMORY` | 6144Mi | Worker pod memory (request and limit) |
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
| **Subprocess model** | Single container per pod, multiple processes |
