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
      - name: main
        type: synthetic
        entries: 1000
        prompts:
          isl: { mean: 512, stddev: 0 }
          osl: { mean: 128, stddev: 0 }
    phases:
      - name: profiling
        type: concurrency
        concurrency: 50
        requests: 500
    artifacts:
      autoPlot: true
      plotRequired: false

  # Config-v2 envelope field (a sibling of benchmark, not nested inside it)
  plot:
    visualization:
      single_run_defaults: [ttft_over_time]
      single_run_plots:
        ttft_over_time:
          type: scatter
          x: request_number
          y: time_to_first_token
          title: TTFT over time

  # Container image (defaults to the chart/AIPerf image when omitted)
  image: "nvcr.io/nvidia/aiperf:latest"

  # Pod resource mode
  resourceMode: burstable        # "burstable" (default), "guaranteed", or "none"

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
      - {name: my-registry-secret}
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
| `datasets` | list | Named dataset configurations (each entry has a `name`) |
| `phases` | list | Ordered load phases (warmup, profiling, etc.), each with a `name` |

See the [YAML Config Reference](../tutorials/yaml-config.md) for the complete set of benchmark fields.

> **Shorthand siblings.** The apiserver also accepts the singular shortcuts
> from AIPerf CLI YAML — `model:` (string/list/object), `dataset:` (single
> dict), and top-level `warmup:` / `profiling:` (phase dicts) — and the
> operator hoists them into the canonical `models`/`datasets`/`phases`
> shapes before validation. Mixing the canonical and shorthand form for the
> same slot (e.g. both `datasets:` and `dataset:`) is rejected at admission.
> Full rule catalog: [CRD Validation Rules](crd-validation.md).

`datasets` and `phases` are lists, not maps. Kubernetes alphabetizes the keys
of object-typed CRD fields at storage time, so phase ordering is only
preserved because it is expressed as a list.

### Plot envelope (`spec.plot`)

Kubernetes preserves the Config-v2 `plot:` envelope field and runs it after
the benchmark's exporters have finished, before the results-sidecar ready
marker is written. The resolved envelope is saved as
`.aiperf-plot-config.yaml` beside the run artifacts, so a later
`aiperf plot <run-directory>` uses the same visualization configuration.

Setting `plot:` implies `spec.benchmark.artifacts.autoPlot: true` unless
`autoPlot: false` was explicitly authored. With the default
`plotRequired: false`, rendering failures produce a warning and the exported
benchmark artifacts still become ready. With `plotRequired: true`, rendering
is part of the completion transaction: a failure leaves results unready and
the controller exits non-zero. Inline plot mappings are the portable form for
hand-authored CRs; `aiperf kube generate -f config.yaml` resolves a file-backed
`plot: ./plots/config.yaml` before submitting the CR.

### Load Phases (`spec.benchmark.phases`)

Each phase defines a load pattern:

```yaml
phases:
  - name: warmup
    kind: warmup
    type: concurrency
    concurrency: 10
    requests: 10
  - name: profiling
    kind: profiling
    type: concurrency
    concurrency: 50
    requests: 500
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Unique phase identity used in status and artifact paths |
| `kind` | string | inferred for canonical names | Semantic role: `warmup` or `profiling`; warmup metrics are kept phase-scoped and excluded from profiling aggregates |
| `type` | string | required | Load type: `concurrency`, `constant`, `poisson`, `gamma`, `user_centric`, or `fixed_schedule` |
| `concurrency` | int | - | Number of concurrent requests |
| `requests` | int | - | Total requests to send |
| `duration` | float | - | Phase duration in seconds (alternative to `requests`) |

### Deployment Options (spec top level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | Installed chart image (operator) or `nvcr.io/nvidia/aiperf:latest` (direct mode) | AIPerf container image |
| `imagePullPolicy` | string | - | `Always`, `IfNotPresent`, or `Never` (Helm default: `IfNotPresent`) |
| `resourceMode` | string | `burstable` | Pod CPU/memory mode. `burstable` (default) sets requests only, no limits (Burstable QoS) so the controller can grow during aggregation without being OOM-killed by cgroup; `guaranteed` keeps requests==limits (Guaranteed QoS); `none` omits CPU/memory requests and limits for both controller and worker pods. |
| `connectionsPerWorker` | int | 100 | Max concurrent connections per worker process |
| `ttlSecondsAfterFinished` | int | 300 | Seconds to keep pods after completion |
| `timeoutSeconds` | int | 0 | Benchmark timeout in seconds (0 = no timeout) |
| `cancel` | bool | `false` | Set to `true` to cancel a running benchmark |
| `keepFailedPods` | bool | `false` | Preserve pods on failure for debugging (overrides `ttlSecondsAfterFinished`) |
| `resultsTtlDays` | int | - | Override operator-level `AIPERF_RESULTS_TTL_DAYS` for this job or sweep archive |
| `skipEndpointCheck` | bool | `false` | Skip the operator-side endpoint reachability probe before deploying |

### Pod Template (`spec.podTemplate`)

Customize the pods that run your benchmark:

| Field | Type | Description |
|-------|------|-------------|
| `nodeSelector` | map | Node labels to constrain scheduling |
| `tolerations` | list | Tolerations for tainted nodes |
| `imagePullSecrets` | list[object] | Secret references for private registries, K8s `LocalObjectReference` shape: `[{name: my-secret}]` |
| `env` | list | Extra environment variables |
| `volumes` | list | Additional volume definitions |
| `volumeMounts` | list | Additional volume mounts |
| `annotations` | map | Extra pod annotations |
| `labels` | map | Extra pod labels |
| `serviceAccountName` | string | Custom service account |
| `containerSecurityContext` | map | SecurityContext applied to every container in the controller and worker pods |

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
| `--image` | `spec.image` | YAML, installed chart, or direct-mode default | Explicit container-image override; an image authored in workload YAML remains authoritative when this flag is omitted |
| `--image-pull-policy` | `spec.imagePullPolicy` | - | Image pull policy (Helm default: `IfNotPresent`) |
| `--total-workers` | `spec.benchmark.runtime.workers` | 10 | Exact worker target (distributed across pods); when omitted, a YAML-authored `runtime.workers` wins before automatic sizing |
| `--name` | `metadata.name` | auto-generated | Job name (DNS label, max 40 chars) |
| `--namespace` | `metadata.namespace` | `aiperf-benchmarks` | Target namespace |
| `--ttl-seconds` | `spec.ttlSecondsAfterFinished` | 300 | TTL after completion |
| `--node-selector` | `spec.podTemplate.nodeSelector` | `{}` | Node selector labels |
| `--tolerations` | `spec.podTemplate.tolerations` | `[]` | Pod tolerations |
| `--queue-name` | `spec.scheduling.queueName` | - | Kueue queue name |
| `--priority-class` | `spec.scheduling.priorityClass` | - | Kueue priority class |
| `--image-pull-secrets` | `spec.podTemplate.imagePullSecrets` | `[]` | Pull secret names |
| `--env-vars` | `spec.podTemplate.env` | `{}` | Non-sensitive extra env vars |
| `--env-from-secrets` | `spec.podTemplate.env` | `{}` | Env vars from Kubernetes Secrets; required for endpoint API keys, sensitive headers, and credentialed URLs |
| `--service-account` | `spec.podTemplate.serviceAccountName` | - | Pod service account |
| `--detach` | - | `false` | Exit after deploying |
| `--dry-run` | - | `false` | Print CR without submitting |
| `--operator` | - | `false` | Deploy through the operator without probing the cluster-scoped AIPerfJob CRD |
| `--no-operator` | - | `false` | Deploy without operator |
| `--skip-endpoint-check` | - | `false` | Skip endpoint health check |
| `--no-wait` | - | `false` | Don't wait for pods ready |
| `--attach-port` | - | 0 (ephemeral) | Local port for port-forward |

Benchmark CLI flags use the same precedence for plain AIPerf config files and
`AIPerfJob` CR input: explicitly passed flags override YAML, while omitted CLI
defaults do not rewrite authored values. This applies to `profile` and
`generate`; `kube sweep` applies the same benchmark overrides before it builds
the `AIPerfSweep` template. Kubernetes deployment flags merge into the CR
deployment subtree, so, for example, `--node-selector gpu=true` does not erase
an unrelated YAML `podTemplate.affinity` or `podTemplate.volumes` block. An
explicit list-valued flag still replaces the corresponding YAML list.

Operator mode is auto-detected when neither mode flag is passed. On a
multi-tenant cluster where users can create namespaced `AIPerfJob` resources
but cannot read cluster-scoped CRDs, pass `--operator` explicitly. When an
explicit `--namespace` is also supplied, `profile` assumes the namespace was
pre-provisioned and does not attempt to create it. `--operator` and
`--no-operator` are mutually exclusive.

When an `AIPerfJob` CR is passed to `profile --no-operator` or
`generate --no-operator`, direct mode preserves the fields that JobSet can
represent, including `imagePullPolicy`, `resourceMode`, `keepFailedPods`,
`ttlSecondsAfterFinished`, `podTemplate`, and `scheduling`. The operator still
owns CR lifecycle fields such as `timeoutSeconds`, `resultsTtlDays`, `cancel`,
and `failurePolicy`; they have no direct-mode reconciler.

Some benchmark runtime fields are intentionally Kubernetes-managed:
`artifacts.dir` is fixed to the mounted `/results` volume, and the operator
sets the service run type, API bind, dataset-service URL, UI, and ZMQ transport
needed for cross-pod operation. Other runtime fields, including
`workers`, `workersPerPod`, `recordProcessors`,
`recordProcessorsPerPod`, and `statsInterval`, remain user-configurable. A
total `recordProcessors` value must divide evenly across identical worker pods;
otherwise set `recordProcessorsPerPod` explicitly.

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

`operator.replicas` is fixed at `1`. The kopf process and runs index have one
authoritative writer and do not use leader election, so the chart rejects
multi-replica values instead of presenting unsafe pseudo-HA.

### Storage

Results are stored on a PVC so they survive pod deletion:

```yaml
storage:
  enabled: true           # default — PVC-backed; set false for ephemeral emptyDir
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

The `resultsServer` chart block only exposes `port` and `resources` today.

The results-server also hosts optional POST routes that create or cancel
`AIPerfJob` resources. These are governed by two environment variables read on
the results-server container: `AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED`
(default `false`) and `AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN` (default empty —
fails closed). When disabled, the read-only APIs stay exposed while those
mutating POSTs return 403, so serving the results-server does not grant write
access through the operator ServiceAccount. The index-rebuild route is mounted
disabled and always returns 503; restart the operator pod to run the single-writer
startup rebuild.

The bundled chart does **not** template these two variables (there is no `resultsServer.mutatingRoutes` value and no token-secret projection). To turn the routes on, set both env vars directly on the `results-server` container — e.g. via a deployment patch or a customized chart template — then have clients send `Authorization: Bearer <token>` on protected POST requests. The browser dashboard never receives this token and keeps create/cancel controls disabled; use `aiperf kube` or `kubectl` from an authenticated terminal for those mutations.

### `dashboard`

Optional Plotly Dash sidecar for the operator Pod. Default off.

| Key                              | Default      | Description                                                                   |
|----------------------------------|--------------|-------------------------------------------------------------------------------|
| `dashboard.enabled`              | `false`      | Whether to add the dashboard container and surface the "Plots ↗" SPA link.   |
| `dashboard.port`                 | `8082`       | Pod-local HTTP port. `results-server` reverse-proxies `/dashboard/*` here.    |
| `dashboard.resources.requests`   | `cpu: 100m, memory: 1Gi` | Resource requests. Leave generous so the build has memory.        |
| `dashboard.resources.limits`     | `{}`         | Empty by default = no limit. Set `memory:` to enforce a ceiling.             |

See [`dashboard-ui.md`](dashboard-ui.md#isolated-plotly-dashboard-sidecar-opt-in) for the full architecture.

### Benchmark Namespace

```yaml
benchmarkNamespace:
  create: true
  name: "aiperf-benchmarks"
```

`benchmarkNamespace.create` controls only whether the chart creates the
namespace. The chart always installs its benchmark `Role` and `RoleBinding` in
`benchmarkNamespace.name`, including when the namespace already exists and
`create` is `false`. `aiperf kube setup` uses that mode because it creates the
resolved `--namespace` before invoking Helm.

### Default Image

The default image used for benchmark jobs if not specified in the CR:

```yaml
defaults:
  image: "" # empty = "<image.repository>:<image.tag|Chart.AppVersion>"
  imagePullPolicy: "IfNotPresent"
```

When `defaults.image` is empty (the chart default), the chart computes the
benchmark image as `<image.repository>:<image.tag | Chart.AppVersion>`, so
overriding `image.tag` automatically propagates to benchmark pods. Set
`defaults.image` explicitly to decouple the benchmark image from the operator
image.

### Ingress

Expose the results-server HTTP API outside the cluster via a Kubernetes `Ingress`. Disabled by default -- results are reachable via `ClusterIP` + `kubectl port-forward`.

```yaml
ingress:
  enabled: false
  className: ""                    # IngressClass name (e.g. "nginx"); empty uses cluster default
  annotations: {}                  # annotations applied to the Ingress
  hosts:
    - host: aiperf.example.com
      paths:
        - path: /
          pathType: Prefix         # backend port defaults to resultsServer.port; override with portNumber
  tls: []                          # optional list of {hosts, secretName}
```

### NetworkPolicy

Restrict pod traffic to/from the operator. Disabled by default -- no restrictions applied. When enabled, ingress is allowed from the benchmark namespace on the health (8080) and results (`resultsServer.port`) ports, plus DNS, the K8s API server, and the benchmark namespace on egress.

```yaml
networkPolicy:
  enabled: false
  allowedNamespaces: []            # extra namespaces allowed to reach the operator and reachable on egress
  allowedIngressCIDRs: []          # CIDR allow-list for external scrapers (Prometheus, ingress controllers)
```

### Kueue

```yaml
kueue:
  # When set, the benchmark namespace is annotated with
  # kueue.x-k8s.io/default-queue-name so all AIPerf jobs are admitted through
  # Kueue even without an explicit --queue-name flag. Left empty, it falls
  # back to kueue.localQueueName when createQueues is true.
  defaultQueueName: ""

  # Optionally let the chart provision the Kueue objects themselves
  # (ResourceFlavor + ClusterQueue + LocalQueue). Off by default so the
  # chart renders on clusters without Kueue CRDs.
  createQueues: false
  flavorName: "default-flavor"
  clusterQueueName: "aiperf-cluster-queue"
  localQueueName: "aiperf-local-queue"
  resources:
    cpu: "1000"
    memory: "4Ti"
    gpu: ""            # empty = omit nvidia.com/gpu from the quota entirely
```

See [Kueue Integration](kueue.md) for the full gang-scheduling walkthrough.

---

## Configuration Patterns

### Combining CLI Flags with Config Files

CLI flags override config file values. This is useful for changing deployment settings without editing the YAML:

```bash
# Use config file for benchmark settings, override image and workers
aiperf kube profile \
  --config benchmark.yaml \
  --image my-registry/aiperf:v2.0 \
  --total-workers 20 \
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
  - name: warmup
    kind: warmup
    type: concurrency
    concurrency: 10
    requests: 20
  - name: low_load
    kind: profiling
    type: concurrency
    concurrency: 25
    requests: 250
  - name: high_load
    kind: profiling
    type: concurrency
    concurrency: 100
    requests: 500
```

Phases run in order. Every phase keeps phase-scoped results; only phases with
`kind: profiling` contribute to profiling aggregates. Canonical names
`warmup` and `profiling` infer their matching kinds, while custom names require
an explicit `kind`.

---

## Resource Mode

`spec.resourceMode` controls the QoS class Kubernetes assigns to benchmark pods. The three modes differ only in how `requests` and `limits` are emitted onto the manifest — the underlying resource budget is the same in every case.

| Mode | Behavior | K8s QoS class | When to use |
|---|---|---|---|
| `burstable` (default) | `requests` only; no `limits`. | Burstable | Default. Cost-sensitive clusters, development, and any benchmark where the controller's aggregation phase may temporarily allocate beyond the request — limits-free pods are not OOM-killed by cgroup. Controller pods stay Burstable by default in the operator's own `values.yaml` for the same reason. |
| `guaranteed` | `requests == limits` for CPU and memory. | Guaranteed | Production benchmarks where pods must not be evicted under pressure and noisy-neighbor behavior is unacceptable. Use this mode when you have measured the controller's peak memory and want a hard ceiling. |
| `none` | Neither `requests` nor `limits`. | BestEffort | Environments where CPU/memory admission control is disabled (e.g. CI `kind` clusters with tight node budgets, or when an external scheduler handles admission). The resource-dependent preflight checks ("Node Resources", "Per-Node Schedulability", "Resource Quotas", "Memory Estimation") are auto-skipped.** |

The mode applies to both controller-pod and worker-pod containers; there is no per-container override. OOMKill semantics follow the QoS class — `guaranteed` pods will not be evicted for resource pressure, `burstable` pods may be throttled, and `none`/BestEffort pods can be evicted first.

---

## Tunable Environment Variables (`AIPERF_K8S_*`)

These variables tune the operator and individual benchmark pods. Set them on the operator deployment (`operator.env.*` in `values.yaml`) to affect every subsequent job, or on `spec.podTemplate.env` to affect one CR only.

### Resource sizing (per-container CPU / memory)

Every control-plane container, the event-bus proxy sidecar, the results sidecar, and the worker pod have a paired `_CPU` / `_MEMORY` variable. Defaults are low burstable requests so many tiny jobs can start concurrently; raise them for very large concurrency or high-token workloads.

| Variable | Default | Applies to |
|---|---|---|
| `AIPERF_K8S_SYSTEM_CONTROLLER_CPU` / `_MEMORY` | `75m` / `192Mi` | SystemController container |
| `AIPERF_K8S_TIMING_MANAGER_CPU` / `_MEMORY` | `50m` / `192Mi` | TimingManager container |
| `AIPERF_K8S_DATASET_MANAGER_CPU` / `_MEMORY` | `50m` / `256Mi` | DatasetManager container |
| `AIPERF_K8S_RECORDS_MANAGER_CPU` / `_MEMORY` | `75m` / `256Mi` | RecordsManager container (raise to 4000m+ for >500k concurrency) |
| `AIPERF_K8S_API_CPU` / `_MEMORY` | `75m` / `256Mi` | API container (WebSocket + HTTP) |
| `AIPERF_K8S_GPU_TELEMETRY_MANAGER_CPU` / `_MEMORY` | `25m` / `192Mi` | GPU telemetry container |
| `AIPERF_K8S_SERVER_METRICS_MANAGER_CPU` / `_MEMORY` | `25m` / `192Mi` | Server-metrics container |
| `AIPERF_K8S_RESULTS_SIDECAR_CPU` / `_MEMORY` | `25m` / `192Mi` | Results sidecar (fallback retrieval path) |
| `AIPERF_K8S_EVENT_BUS_PROXY_CPU` / `_MEMORY` | `50m` / `64Mi` | Event-bus XPUB/XSUB proxy sidecar |
| `AIPERF_K8S_WORKER_POD_CPU` / `_MEMORY` | `150m` / `4Gi` | Worker pod (workers + record processors + WPM) |

### Architecture toggles

| Variable | Default | Purpose |
|---|---|---|
| `AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED` | `true` | Run the XPUB/XSUB event-bus proxy as a dedicated sidecar container. Set to `false` to revert to the pre-sidecar behavior where `SystemController` hosts the proxy in-process. Only disable if you are explicitly testing the legacy path. |
| `AIPERF_K8S_RECORD_PROCESSOR_SCALE_FACTOR` | `1` | Workers per record processor inside each worker pod. `1` means one RP per worker (maximum fairness); higher values amortize RP overhead across more workers. |
| `AIPERF_K8S_RECORD_PROCESSOR_CPU_REQUEST` | (unset) | Optional per-RP CPU request override. When unset, RP CPU is derived from the worker-pod budget. |

### JobSet and lifecycle

| Variable | Default | Purpose |
|---|---|---|
| `AIPERF_K8S_JOBSET_TTL_SECONDS_AFTER_FINISHED` | `300` | Seconds to keep pods after JobSet completion. Override per-CR via `spec.ttlSecondsAfterFinished`. |
| `AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS` | `28800` (8h) | TTL applied when `--no-operator` is used, giving you time to pull results from pod-local storage. |
| `AIPERF_K8S_JOBSET_CONTROLLER_BACKOFF_LIMIT` | `0` | Controller-job retry count. Default `0` — fail fast when the controller crashes. |
| `AIPERF_K8S_JOBSET_WORKER_BACKOFF_LIMIT` | `20` | Worker-job retry count. Higher than controller to absorb transient pod-startup flakes. |
| `AIPERF_K8S_JOBSET_WORKER_CONNECTION_PROBE_TIMEOUT` | `60.0` | Seconds a worker waits for the PUB/SUB connection probe before exiting so K8s restarts it. |

### Health probes

| Variable | Default | Purpose |
|---|---|---|
| `AIPERF_K8S_HEALTH_STARTUP_PERIOD_SECONDS` | `5` | Startup probe interval. |
| `AIPERF_K8S_HEALTH_STARTUP_FAILURE_THRESHOLD` | `30` | Consecutive failures before the container is killed during startup. Raise for slow-starting workloads (large tokenizers, cold container images). |
| Other `AIPERF_K8S_HEALTH_*` | see code | Liveness/readiness intervals, timeouts, thresholds. |

### Ports

All health and service ports are overridable via `AIPERF_K8S_PORT_*` (e.g. `AIPERF_K8S_PORT_API_SERVICE=9090`, `AIPERF_K8S_PORT_RESULTS_SIDECAR=9091`, `AIPERF_K8S_PORT_SYSTEM_CONTROLLER_HEALTH=8080`). Consult `src/aiperf/kubernetes/environment.py::_PortSettings` for the full list — changing these is rarely necessary.

The complete, generated reference for every `AIPERF_*` variable (including non-k8s ones) lives in [`../environment-variables.md`](../environment-variables.md).

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Monitoring and Troubleshooting](monitoring.md) -- Live monitoring and debugging
- [Production Deployments](production.md) -- CI/CD, Kueue, and GitOps workflows
- [CRD Validation Rules](crd-validation.md) -- Apiserver-side CEL invariants and shorthand acceptance
- [Preflight Checks](preflight.md) -- What the operator validates before admitting a CR
- [Memory Estimator](memory-estimator.md) -- How per-component memory estimates drive resource requests
- [Direct Mode](direct-mode.md) -- Trade-offs when running `--no-operator`
- [User-defined output files](user-files.md) -- `artifacts.user_files` for templated sidecar files
- [YAML Config Reference](../tutorials/yaml-config.md) -- Complete benchmark configuration options
