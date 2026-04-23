---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Kubernetes Configuration
---

# Kubernetes Configuration Reference

This guide covers all the ways to configure AIPerf benchmarks on Kubernetes -- from the AIPerfJob custom resource fields to CLI flags and Helm chart settings.

---

## AIPerfJob Custom Resource

An `AIPerfJob` is a Kubernetes custom resource that tells the operator what benchmark to run. Here is the full structure:

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-benchmark
  namespace: aiperf-benchmarks  # optional, defaults to aiperf-benchmarks
spec:
  # Benchmark configuration (what to measure)
  benchmark:
    models: ["Qwen/Qwen3-0.6B"]
    endpoint:
      urls: ["http://dynamo-agg-frontend.dynamo-server.svc:8000/v1"]
      streaming: true
    datasets:
      main:
        type: synthetic
        entries: 1000
        prompts:
          isl: { mean: 512, stddev: 0 }
          osl: { mean: 128, stddev: 0 }
    phases:
      profiling:
        type: concurrency
        concurrency: 50
        requests: 500

  # Container image (required)
  image: "nvcr.io/nvidia/aiperf:latest"

  # Pod resource mode
  resourceMode: guaranteed       # "guaranteed", "burstable", or "none"

  # Worker scaling
  connectionsPerWorker: 100       # max concurrent connections per worker process

  # Lifecycle
  ttlSecondsAfterFinished: 300    # seconds to keep pods after completion
  timeoutSeconds: 0               # benchmark timeout (0 = no timeout)

  # Cancel a running benchmark
  cancel: false                   # set to true to cancel

  # Pod customization
  podTemplate:
    nodeSelector:
      nvidia.com/gpu.product: "A100"
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
    imagePullSecrets:
      - my-registry-secret
    env:
      - name: AIPERF_HTTP_CONNECTION_LIMIT
        value: "200"
    volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
    volumeMounts:
      - name: model-cache
        mountPath: /root/.cache/huggingface

  # Kueue scheduling
  scheduling:
    queueName: my-queue
    priorityClass: high-priority
```

---

## Spec Fields Reference

### Benchmark Configuration (`spec.benchmark`)

The `benchmark` section mirrors the standard AIPerf YAML config. Any field you use in a local `aiperf profile` run works here.

| Field | Type | Description |
|-------|------|-------------|
| `models` | list[string] | Model name(s) served by the endpoint |
| `endpoint.urls` | list[string] | Inference server URLs |
| `endpoint.streaming` | bool | Enable streaming responses |
| `endpoint.type` | string | Endpoint type (default: `chat`) |
| `datasets` | map | Named dataset configurations |
| `phases` | map | Named load phases (warmup, profiling, etc.) |

See the [YAML Config Reference](../tutorials/yaml-config.md) for the complete set of benchmark fields.

### Load Phases (`spec.benchmark.phases`)

Each phase defines a load pattern:

```yaml
phases:
  warmup:
    type: concurrency
    concurrency: 10
    requests: 10
    exclude_from_results: true    # don't include warmup in final metrics
  profiling:
    type: concurrency
    concurrency: 50
    requests: 500
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | required | Load type: `concurrency`, `constant`, `poisson`, `gamma`, `user_centric`, or `fixed_schedule` |
| `concurrency` | int | - | Number of concurrent requests |
| `requests` | int | - | Total requests to send |
| `duration` | float | - | Phase duration in seconds (alternative to `requests`) |
| `exclude_from_results` | bool | `false` | Exclude this phase from final metrics |

### Deployment Options (spec top level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | AIPerf container image |
| `imagePullPolicy` | string | - | `Always`, `IfNotPresent`, or `Never` (Helm default: `IfNotPresent`) |
| `resourceMode` | string | `guaranteed` | Pod CPU/memory mode. `guaranteed` keeps requests==limits (Guaranteed QoS); `burstable` sets requests only, no limits (Burstable QoS); `none` omits CPU/memory requests and limits for both controller and worker pods. |
| `connectionsPerWorker` | int | 100 | Max concurrent connections per worker process |
| `ttlSecondsAfterFinished` | int | 300 | Seconds to keep pods after completion |
| `timeoutSeconds` | int | 0 | Benchmark timeout in seconds (0 = no timeout) |
| `cancel` | bool | `false` | Set to `true` to cancel a running benchmark |

### Pod Template (`spec.podTemplate`)

Customize the pods that run your benchmark:

| Field | Type | Description |
|-------|------|-------------|
| `nodeSelector` | map | Node labels to constrain scheduling |
| `tolerations` | list | Tolerations for tainted nodes |
| `imagePullSecrets` | list[string] | Secret names for private registries |
| `env` | list | Extra environment variables |
| `volumes` | list | Additional volume definitions |
| `volumeMounts` | list | Additional volume mounts |
| `annotations` | map | Extra pod annotations |
| `labels` | map | Extra pod labels |
| `serviceAccountName` | string | Custom service account |

### Scheduling (`spec.scheduling`)

For clusters using [Kueue](https://kueue.sigs.k8s.io/) for resource management:

| Field | Type | Description |
|-------|------|-------------|
| `queueName` | string | Kueue LocalQueue name for gang-scheduling |
| `priorityClass` | string | Kueue WorkloadPriorityClass for scheduling priority |

---

## CLI Flags

When using `aiperf kube profile`, you can set deployment options via CLI flags. These override values in a config file:

| Flag | Maps To | Default | Description |
|------|---------|---------|-------------|
| `--image` | `spec.image` | required | Container image |
| `--image-pull-policy` | `spec.imagePullPolicy` | - | Image pull policy (Helm default: `IfNotPresent`) |
| `--workers-max` | - | 10 | Total workers (distributed across pods) |
| `--name` | `metadata.name` | auto-generated | Job name (DNS label, max 40 chars) |
| `--namespace` | `metadata.namespace` | `aiperf-benchmarks` | Target namespace |
| `--ttl-seconds` | `spec.ttlSecondsAfterFinished` | 300 | TTL after completion |
| `--node-selector` | `spec.podTemplate.nodeSelector` | `{}` | Node selector labels |
| `--tolerations` | `spec.podTemplate.tolerations` | `[]` | Pod tolerations |
| `--queue-name` | `spec.scheduling.queueName` | - | Kueue queue name |
| `--priority-class` | `spec.scheduling.priorityClass` | - | Kueue priority class |
| `--image-pull-secrets` | `spec.podTemplate.imagePullSecrets` | `[]` | Pull secret names |
| `--env-vars` | `spec.podTemplate.env` | `{}` | Extra env vars |
| `--env-from-secrets` | `spec.podTemplate.env` | `{}` | Env vars from secrets |
| `--service-account` | `spec.podTemplate.serviceAccountName` | - | Pod service account |
| `--detach` | - | `false` | Exit after deploying |
| `--dry-run` | - | `false` | Print CR without submitting |
| `--no-operator` | - | `false` | Deploy without operator |
| `--skip-endpoint-check` | - | `false` | Skip endpoint health check |
| `--no-wait` | - | `false` | Don't wait for pods ready |
| `--attach-port` | - | 0 (ephemeral) | Local port for port-forward |

---

## Helm Chart Configuration

The operator Helm chart is configured via `values.yaml`. Key settings:

### Operator

```yaml
operator:
  replicas: 1
  resources:
    requests: { cpu: 250m, memory: 256Mi }
    # No limits set by default (burstable QoS) so the operator can scale
    # memory/CPU with high-concurrency runs.
  env:
    monitorInterval: "10.0"         # seconds between status checks
    monitorInitialDelay: "5.0"      # delay before first status check
    jobTimeoutSeconds: "0"          # 0 = no timeout
    podRestartThreshold: "3"        # restarts before warning events
    resultsTtlDays: "30"            # days to keep results on PVC
    resultsMaxRetries: "5"          # retries for fetching results
    resultsRetryDelay: "2.0"        # delay between result fetch retries
    endpointCheckTimeout: "10.0"    # endpoint health check timeout
    resultsCompressOnDisk: "true"   # store results as zstd on PVC
```

### Storage

Results are stored on a PVC so they survive pod deletion:

```yaml
storage:
  enabled: false
  size: 1Ti
  storageClassName: ""    # empty = cluster default
  accessMode: "ReadWriteOnce"
```

### Results Server

A sidecar that serves stored results via HTTP (used by `aiperf kube results` by default):

```yaml
resultsServer:
  port: 8081
  resources:
    requests: { cpu: 100m, memory: 512Mi }
    limits: { cpu: 500m, memory: 1Gi }
```

### Benchmark Namespace

```yaml
benchmarkNamespace:
  create: true
  name: "aiperf-benchmarks"
```

### Default Image

The default image used for benchmark jobs if not specified in the CR:

```yaml
defaults:
  image: "nvcr.io/nvidia/aiperf:latest"
  imagePullPolicy: "IfNotPresent"
```

---

## Configuration Patterns

### Combining CLI Flags with Config Files

CLI flags override config file values. This is useful for changing deployment settings without editing the YAML:

```bash
# Use config file for benchmark settings, override image and workers
aiperf kube profile \
  --config benchmark.yaml \
  --image my-registry/aiperf:v2.0 \
  --workers-max 20 \
  --namespace production
```

### Validating Before Deploying

Check your config file for errors before submitting:

```bash
# Validate YAML structure and fields
aiperf kube validate benchmark.yaml

# Strict mode fails on unknown fields
aiperf kube validate --strict benchmark.yaml

# JSON output for CI
aiperf kube validate -o json benchmark.yaml
```

Preview what will be submitted without deploying:

```bash
aiperf kube profile --config benchmark.yaml --image aiperf:latest --dry-run
```

### Memory Estimation

AIPerf prints a memory estimate before deploying. This helps you right-size your pods:

```bash
aiperf kube generate --operator --config benchmark.yaml --image aiperf:latest
```

The memory estimate is printed to stderr. It accounts for dataset size, number of workers, connection pools, and record buffers.

### Multiple Phases

Use multiple phases to warm up before measuring:

```yaml
phases:
  warmup:
    type: concurrency
    concurrency: 10
    requests: 20
    exclude_from_results: true
  low_load:
    type: concurrency
    concurrency: 25
    requests: 250
  high_load:
    type: concurrency
    concurrency: 100
    requests: 500
```

Phases run in order. Metrics from phases with `exclude_from_results: true` are not included in the final report.

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Monitoring and Troubleshooting](monitoring.md) -- Live monitoring and debugging
- [Production Deployments](production.md) -- CI/CD, Kueue, and GitOps workflows
- [YAML Config Reference](../tutorials/yaml-config.md) -- Complete benchmark configuration options
