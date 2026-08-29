# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# aiperf-operator Chart Configuration Reference

Terminal reference for the `aiperf-operator` Helm chart: every value group and
default, production hardening, and Kueue integration. Chart `version: 0.7.0`,
`appVersion: "0.13.0"`, `kubeVersion: ">=1.24.0-0"`.

## Pinned dependency versions

| Dependency | Pinned version | Notes |
|---|---|---|
| JobSet | `v0.8.0` | Hard dependency. The chart grants `jobset.x-k8s.io` RBAC but does **not** install it. |
| Kueue | `v0.10.1` | Optional. Pinned for the project's own install tooling. |
| NVIDIA device plugin | `v0.17.0` | Optional, for GPU-resident benchmarks. |

```bash
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml
kubectl -n jobset-system wait --for=condition=available \
  --timeout=120s deployment/jobset-controller-manager
```

## Schema strictness — closed maps

The chart's values JSON schema sets `additionalProperties: false` on the root
and on `image`, `podTemplate`, `chaos`, `operator`, `operator.metrics`,
`resultsServer`, `dashboard`, `rbac`, `storage`, `serviceAccount`,
`kueue`, `ingress`, `networkPolicy`, `serviceMonitor`,
`defaults`, `tests`, `tests.image`, and object entries of
`serverMetricsDiscoveryNamespaces`. An unknown key in any of those is a
**render-time validation error**.

Free-form (extensible) maps: `operator.resources`, `.podAnnotations`,
`.nodeSelector`, `.affinity`, `resultsServer.resources`, `dashboard.resources`,
`serviceAccount.annotations`, `ingress.annotations`, `serviceMonitor.labels`
/`.tlsConfig`/`.bearerTokenSecret`, `kueue.resources`, `tests.resources`.

**`operator.env` is effectively closed even though the schema does not mark it
so** — the Deployment renders exactly nine env vars from nine named keys; any
other key there renders nothing.

## Value matrix

### `image` / registry

| Key | Default | Effect |
|---|---|---|
| `image.repository` | `nvcr.io/nvidia/aiperf` | Image for every operator-pod container (operator, results-server, dashboard). |
| `image.tag` | `""` | Falls back to `Chart.AppVersion` (`0.13.0`). Also propagates to benchmark pods unless `defaults.image` is set. |
| `image.pullPolicy` | `IfNotPresent` | |
| `imagePullSecrets` | `[]` | List of `{name}`. Operator pod only — benchmark pods need the secret in the benchmark namespace via the CR's pod template. |
| `nameOverride` | `""` | Replaces chart name in resource names/labels. |
| `fullnameOverride` | `""` | Replaces the full release name in resource names. |

### `operator` — the kopf container

| Key | Default | Effect |
|---|---|---|
| `operator.replicas` | `1` | Any other value calls `fail` at render time. No leader election exists. Deployment `strategy.type: Recreate`. |
| `operator.resources` | `requests: {cpu: 250m, memory: 256Mi}`, no limits | Burstable on purpose; high-concurrency runs need headroom. |
| `operator.podAnnotations` | `{}` | |
| `operator.nodeSelector` | `{}` | |
| `operator.tolerations` | three entries (see below) | Replaces, does not merge. |
| `operator.affinity` | `{}` | |
| `operator.topologySpreadConstraints` | `[]` | |
| `operator.priorityClassName` | `""` | |
| `operator.watchNamespaces` | `[]` | Empty renders `--all-namespaces`; each entry renders `--namespace=<ns>`. Does **not** narrow RBAC. |
| `operator.tmpSizeLimit` | `""` | `sizeLimit` on the `/tmp` emptyDir. Unbounded when empty; exceeding evicts the pod (k8s 1.25+). |
| `operator.metrics.port` | `9090` | `0` removes the metrics container port, the Service `metrics` port, and the NetworkPolicy metrics rule, and blocks the ServiceMonitor from rendering at all. `AIPERF_METRICS_PORT` is set to this value regardless. |

Default tolerations:

```yaml
operator:
  tolerations:
    - {key: dedicated, operator: Equal, value: user-workload, effect: NoSchedule}
    - {key: dedicated, operator: Equal, value: user-workload, effect: NoExecute}
    - {key: kubernetes.io/arch, operator: Exists, effect: NoSchedule}
```

Deliberately absent: `nvidia.com/gpu:NoSchedule`. The operator is a
control-plane pod and must not consume GPU nodes.

#### `operator.env` — the nine renderable keys

| Key | Default | Env var |
|---|---|---|
| `monitorInterval` | `"10.0"` | `AIPERF_OPERATOR_MONITOR_INTERVAL` |
| `monitorInitialDelay` | `"5.0"` | `AIPERF_OPERATOR_MONITOR_INITIAL_DELAY` |
| `jobTimeoutSeconds` | `"0"` (no timeout) | `AIPERF_JOB_TIMEOUT_SECONDS` |
| `podRestartThreshold` | `"3"` | `AIPERF_POD_RESTART_THRESHOLD` |
| `resultsTtlDays` | `"30"` | `AIPERF_RESULTS_TTL_DAYS` |
| `resultsMaxRetries` | `"5"` | `AIPERF_RESULTS_MAX_RETRIES` |
| `resultsRetryDelay` | `"2.0"` | `AIPERF_RESULTS_RETRY_DELAY` |
| `endpointCheckTimeout` | `"10.0"` | `AIPERF_ENDPOINT_CHECK_TIMEOUT` |
| `resultsCompressOnDisk` | `"true"` | `AIPERF_RESULTS_COMPRESS_ON_DISK` (zstd on the PVC) |

Also templated unconditionally: `PYTHONUNBUFFERED=1`,
`AIPERF_RESULTS_DIR=<storage.mountPath>`, `AIPERF_METRICS_PORT`,
`AIPERF_K8S_SHARE_PROCESS_NAMESPACE`, `AIPERF_DASHBOARD_PORT`,
`AIPERF_OPERATOR_BASE_URL=http://<fullname>.<ns>:<resultsServer.port>`.

**Any other `AIPERF_K8S_*` tunable (pod sizing, JobSet TTL, Kueue defaults,
probes) has no chart value.** Apply it to the live Deployment after install:

```bash
kubectl -n aiperf-system set env deployment/aiperf-operator -c operator \
  AIPERF_K8S_WORKER_POD_MEMORY=8Gi AIPERF_K8S_RECORDS_MANAGER_CPU=4000m
```

### `resultsServer` (FastAPI sidecar) and `dashboard` (Plotly Dash sidecar)

results-server hosts the entire `/api/v1/*` surface (jobs, sweeps, results,
config, admin, analytics, dashboard proxy); the kopf container serves only
`/healthz` on 8080. The PVC is mounted read-write in results-server on purpose:
the SQLite runs index is in WAL mode and a WAL reader must create the `-shm`
sidecar; read-only is enforced at the SQLite layer instead.

| Key | Default |
|---|---|
| `resultsServer.port` | `8081` — also baked into `AIPERF_OPERATOR_BASE_URL`, the Service `results` port, the Ingress default backend port, and stamped CR status URLs. |
| `resultsServer.resources` | `requests {cpu 100m, memory 512Mi}`, `limits {cpu 500m, memory 1Gi}` |
| `dashboard.enabled` | `false` |
| `dashboard.port` | `8082` — pod-local only, not on the Service |
| `dashboard.resources` | `requests {cpu 100m, memory 1Gi}`, `limits {}` |

Toggling it changes **three** containers: it adds the dashboard container (PVC
read-only) and flips `AIPERF_DASHBOARD_PORT` on the operator container plus
`AIPERF_DASHBOARD_PORT` and `AIPERF_DASHBOARD_PROXY_ENABLED` on results-server.
Request path: client → Ingress/port-forward → results-server `:8081` →
`localhost:<port>`.

### `podTemplate` and `chaos` — benchmark-pod overrides

| Key | Default | Effect |
|---|---|---|
| `podTemplate.shareProcessNamespace` | `false` | Renders `spec.shareProcessNamespace: true` on JobSet pods. Weakens isolation; test-only. |
| `chaos.controllerHttpUrlOverride` | `""` | Collapses per-CR controller routing to one URL. Never in production. |
| `chaos.apiserverServiceHostOverride` | `""` | Overwrites `KUBERNETES_SERVICE_HOST`. Wrong value = total apiserver loss. |
| `chaos.apiserverServicePortOverride` | `""` | Overwrites `KUBERNETES_SERVICE_PORT`. |
| `chaos.apiserverTlsServerNameOverride` | `""` | Preserves TLS verification when dialing an L4 proxy. |

### `storage`

| Key | Default | Effect |
|---|---|---|
| `storage.enabled` | `true` | `false` skips the PVC entirely and swaps the `results` volume to `emptyDir`. Results then die with the pod. |
| `storage.size` | `1Ti` | PVC request. |
| `storage.storageClassName` | `""` | Omitted from the PVC when empty → cluster default class. |
| `storage.mountPath` | `/data` | Mounted in every operator-pod container; also `AIPERF_RESULTS_DIR`. |
| `storage.accessMode` | `ReadWriteOnce` | Sufficient for the mandatory single replica. `ReadWriteMany` available if the provisioner requires it. |
| `storage.emptyDirSizeLimit` | `""` | Only used when `storage.enabled=false`. |

Sizing: budget by retained runs; `resultsCompressOnDisk=true` stores zstd and
`operator.env.resultsTtlDays` (30) bounds growth. `Recreate` strategy means a
`ReadWriteOnce` PVC never deadlocks on volume attach during upgrade.

### `serviceAccount`, `rbac`, and namespaces

| Key | Default | Effect |
|---|---|---|
| `serviceAccount.create` | `true` | |
| `serviceAccount.name` | `""` | **Required** when `create=false`: the template calls `required` and fails the render rather than silently binding the namespace `default` SA and 403-ing on every reconcile. |
| `serviceAccount.annotations` | `{}` | e.g. IRSA / workload-identity. |
| `rbac.create` | `true` | Gates ClusterRole **and** ClusterRoleBinding **and** the whole benchmark-RBAC template. |
| `benchmarkRbacNamespaces` | `[]` | Extra namespaces beyond the release namespace that get the benchmark Role/RoleBinding. **Not created** — they must already exist. De-duplicated against the release namespace. |
| `serverMetricsDiscoveryNamespaces` | `[]` | Existing inference namespaces. **Not created.** Each gets only a `pods: get/list/watch` Role bound to subjects from the benchmark namespaces. |

```yaml
serverMetricsDiscoveryNamespaces:
  - dynamo-server                    # binds `default` SA of every benchmark ns
  - namespace: dynamo-staging
    serviceAccounts: [aiperf-bench]  # binds custom SAs instead
```

### `ingress`, `networkPolicy`, `serviceMonitor`, `defaults`, `tests`

| Key | Default | Effect |
|---|---|---|
| `ingress.enabled` | `false` | Otherwise ClusterIP + port-forward. |
| `ingress.className` | `""` | Cluster default IngressClass. |
| `ingress.annotations` | `{}` | |
| `ingress.hosts` | `[{host: aiperf.example.com, paths: [{path: /, pathType: Prefix}]}]` | Per-path `pathType` defaults to `Prefix`; backend port defaults to `resultsServer.port`, overridable per path with `portNumber`. |
| `ingress.tls` | `[]` | List of `{hosts, secretName}`. |
| `networkPolicy.enabled` | `false` | See hardening below. |
| `networkPolicy.allowedNamespaces` | `[]` | Extra namespaces on both ingress and egress. |
| `networkPolicy.allowedIngressCIDRs` | `[]` | `ipBlock` allow-list for external scrapers. |
| `serviceMonitor.enabled` | `false` | Renders only when also `monitoring.coreos.com/v1` exists **and** `operator.metrics.port > 0`; silent no-op otherwise. |
| `serviceMonitor.interval` | `30s` | |
| `serviceMonitor.scrapeTimeout` | `10s` | |
| `serviceMonitor.honorLabels` | `true` | |
| `serviceMonitor.scheme` | `http` | |
| `serviceMonitor.labels` | `{}` | e.g. `release: prometheus-stack`. |
| `serviceMonitor.tlsConfig` | `{}` | |
| `serviceMonitor.bearerTokenSecret` | `{}` | `{name, key}`. |
| `defaults.image` | `""` | Benchmark-pod image; computed `<image.repository>:<image.tag\|AppVersion>` when empty. |
| `defaults.imagePullPolicy` | `IfNotPresent` | |
| `tests.enabled` | `true` | Gates the `helm test` hook pods **and** their dedicated SA/Role/RoleBinding/ClusterRole/ClusterRoleBinding. Deliberately *not* gated on `rbac.create`. With it false, `helm test` prints `TEST SUITE: None` and exits 0. |
| `tests.image.repository` | `alpine/k8s` | Must provide `kubectl` and `curl`. |
| `tests.image.tag` | `"1.33.11"` | Pinned, not `latest`. |
| `tests.image.pullPolicy` | `IfNotPresent` | |
| `tests.resources` | `requests {cpu 50m, memory 64Mi}`, `limits {cpu 200m, memory 128Mi}` | |

## Ports summary

| Port | Container | Exposed on Service |
|---|---|---|
| 8080 | operator (kopf `--liveness`, `/healthz`) | yes, `health` |
| `resultsServer.port` = 8081 | results-server, all `/api/v1/*` | yes, `results` |
| `operator.metrics.port` = 9090 | operator Prometheus `/metrics` | yes, `metrics` (when > 0) |
| `dashboard.port` = 8082 | dashboard | **no** — proxied through 8081 |

## Production hardening

### RBAC scoping and what narrowing breaks

The ClusterRole grants, cluster-wide:

| API group | Resources | Verbs |
|---|---|---|
| `apiextensions.k8s.io` | customresourcedefinitions | get, list, watch |
| `aiperf.nvidia.com` | aiperfjobs(+/status,/finalizers) | full |
| `aiperf.nvidia.com` | aiperfsweeps(+/status,/finalizers) | full |
| `jobset.x-k8s.io` | jobsets | create, delete, get, list, patch, update, watch |
| `jobset.x-k8s.io` | jobsets/status | get, list, watch |
| `kueue.x-k8s.io` | localqueues | get, list |
| `batch` | jobs | get, list, watch |
| `apps` | deployments | get, list, watch |
| `""` | serviceaccounts | get, list, watch, create |
| `""` | resourcequotas | get, list |
| `""` | secrets | **get only** |
| `networking.k8s.io` | networkpolicies | get, list, watch |
| `""` | configmaps | create, delete, get, list, patch, update, watch |
| `""` | services, endpoints | create, delete, get, list, watch |
| `rbac.authorization.k8s.io` | roles, rolebindings | create, delete, get, list, watch |
| `""` | namespaces | get, list, watch |
| `""` | pods, pods/log | **get, list, watch only** |
| `""` | nodes | get, list |
| `""` | events | get, list, watch, create, patch |

Load-bearing omissions — do not "fix" them:

- **No `pods: patch`.** Pod-restart detection is an event handler precisely
  because a field handler would need a diff-base annotation on every observed
  Pod. The `pods`/`pods/log` rule must never gain `create`, `delete`, `patch`,
  or `update`; check the rendered ClusterRole with
  `helm template aiperf-operator "$CHART" | yq 'select(.kind=="ClusterRole").rules'`.
- **`secrets: get` without `list`/`watch`.** Preflight reads referenced pull
  secrets by name; `list` would expose every secret in the cluster.
- **No `coordination.k8s.io/leases`.** No leader election exists — hence
  `replicas: 1`.

What breaks when you narrow:

| Narrowing | Consequence |
|---|---|
| `operator.watchNamespaces: [ns…]` | Only limits kopf's watch. RBAC stays cluster-wide. CRs elsewhere are ignored, not rejected. |
| `rbac.create=false` | Removes ClusterRole + ClusterRoleBinding **and** the per-namespace benchmark Role/RoleBinding **and** the `-metrics-discovery` Roles/RoleBindings. Recreating only the first four silently loses cross-namespace server-metrics discovery. You must also set `serviceAccount.name`. |
| Drop `kueue.x-k8s.io/localqueues` | The queue preflight check always 403s and degrades to WARN. |
| Drop `resourcequotas` / `secrets` / `networkpolicies` | Corresponding preflight checks degrade; jobs still submit. |
| `rbac.create=false` + `tests.enabled=false` | The only combination that emits **zero** cluster-scoped RBAC objects. |

Benchmark-namespace Role (bound to the `default` SA of each benchmark
namespace): `pods get/list/watch`; `jobsets get/list/watch/**patch**`;
`aiperfjobs`+`/status` `get/list/watch/**patch**`. `patch` not `update` on
purpose — `update` (PUT) would let a benchmark pod replace an entire JobSet
spec.

Pod security is hard-coded, not configurable: pod-level `runAsNonRoot: true`,
uid/gid/fsGroup `1000`, `seccompProfile: RuntimeDefault`; per-container
`allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`,
`capabilities.drop: [ALL]`; `terminationGracePeriodSeconds: 30`.

### NetworkPolicy

With `networkPolicy.enabled=true`, one policy targeting the operator pod
(`Ingress` + `Egress`):

- **Ingress** on 8080, `resultsServer.port`, and `operator.metrics.port` (when
  > 0) from: the release namespace (needed for `helm test` to reach
  `/healthz`), every `benchmarkRbacNamespaces` entry,
  and every `networkPolicy.allowedNamespaces` entry — all matched via the
  `kubernetes.io/metadata.name` label, so namespace auto-labelling must be on.
- **Ingress** on the same ports from each `allowedIngressCIDRs` `ipBlock`. Use
  this for a Prometheus or ingress controller that is not namespace-selectable.
- **Egress**: UDP/TCP 53 to `kube-system`; TCP 443 and 6443 to **anywhere**
  (no standard apiserver selector exists); all ports to the benchmark
  namespace, `benchmarkRbacNamespaces`, and `allowedNamespaces`.

Consequences: an inference endpoint on a non-443 port in a namespace on none of
those lists is unreachable from the operator, and a Prometheus outside the
allow-lists cannot scrape even with `serviceMonitor.enabled=true`.

### Resource sizing

Operator pod: `250m`/`256Mi` requested, no limits — raise the request, not a
limit. Benchmark pod containers are sized by `AIPERF_K8S_*` env vars on the
operator Deployment (`kubectl set env`, above), not by chart values; requests ==
limits (Guaranteed QoS). Defaults:

| Container (env prefix `AIPERF_K8S_<NAME>_`) | CPU | Memory |
|---|---|---|
| `SYSTEM_CONTROLLER` / `RECORDS_MANAGER` / `API` | 75m | 192Mi / 256Mi / 256Mi |
| `SWEEP_CONTROLLER` (imports torch+BoTorch) | 75m | 512Mi |
| `TIMING_MANAGER` / `DATASET_MANAGER` | 50m | 192Mi / 256Mi |
| `GPU_TELEMETRY_MANAGER` / `SERVER_METRICS_MANAGER` / `RESULTS_SIDECAR` | 25m | 192Mi |
| `EVENT_BUS_PROXY` | 50m | 64Mi |
| `WORKER_POD` (workers + record processors + WPM) | 150m | 4Gi |

Known cliff: `AIPERF_K8S_RECORDS_MANAGER_CPU` at the 75m default pegs one core
above roughly 500k requests; the event loop starves, heartbeats are missed, and
the CR freezes in `Pending`. Set `4000m` or more for very large runs. Raise
`AIPERF_K8S_WORKER_POD_MEMORY` for memory-heavy datasets or extreme concurrency
(1.8-3 GiB working set per pod measured at 10K concurrency).

### High availability and upgrades

No HA by design: `replicas` must be `1`, `strategy: Recreate`, no leader
election. During a rollout the old pod terminates before the new one starts;
in-flight benchmarks keep running (they are JobSet pods) and are reconciled on
restart. Durability comes from the PVC plus a CR annotation gating completion
work with an atomic test-and-set patch, so restarting mid-completion cannot
double-fire.

```bash
helm upgrade aiperf-operator "$CHART" \
  -n aiperf-system -f values-prod.yaml --wait
```

The CRDs and the benchmark namespaces carry `helm.sh/resource-policy: keep`, so
`helm uninstall` leaves running jobs and their CRs intact. Corollary: CRD schema
changes are **not** applied by `helm upgrade` — apply the new CRDs explicitly
with `kubectl apply --server-side` when the chart's CRD schema changed.

`topologySpreadConstraints` with one replica constrains which topology the
single pod may land in; it cannot spread anything. Use `priorityClassName:
system-cluster-critical` to protect the operator from preemption.

## Kueue

### Install

```bash
kubectl apply --server-side \
  -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml
kubectl -n kueue-system wait --for=condition=available \
  --timeout=300s deployment/kueue-controller-manager
```

Kueue must have its JobSet integration enabled (`jobset.x-k8s.io/jobset` in the
controller's `integrations.frameworks`).

### Queue objects

Set `kueue.createQueues=true` to have the chart provision the trio. It renders
only when the cluster already exposes `kueue.x-k8s.io/v1beta1` — otherwise a
silent no-op, so install Kueue first.

| Key | Default | Effect |
|---|---|---|
| `kueue.createQueues` | `false` | Gates ResourceFlavor + ClusterQueue + LocalQueue. |
| `kueue.flavorName` | `default-flavor` | ResourceFlavor name (no node labels/taints). |
| `kueue.clusterQueueName` | `aiperf-cluster-queue` | `namespaceSelector: {}` — all namespaces. |
| `kueue.localQueueName` | `aiperf-local-queue` | Created in the chart's release namespace. |
| `kueue.resources.cpu` | `"1000"` | ClusterQueue nominalQuota. |
| `kueue.resources.memory` | `"4Ti"` | ClusterQueue nominalQuota. |
| `kueue.resources.gpu` | `""` | **Empty omits `nvidia.com/gpu` from `coveredResources` entirely.** Set e.g. `"64"` to add GPU quota. |
| `kueue.defaultQueueName` | `""` | Sets `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` on the operator container. Auto-fills from `localQueueName` when `createQueues=true`. |

Equivalent by hand — note the GPU resource the chart omits unless
`kueue.resources.gpu` is set:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata: {name: default-flavor}
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata: {name: aiperf-cluster-queue}
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: default-flavor
          resources:
            - {name: cpu, nominalQuota: "1000"}
            - {name: memory, nominalQuota: "4Ti"}
            - {name: nvidia.com/gpu, nominalQuota: "64"}
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata: {name: aiperf-local-queue, namespace: aiperf-system}
spec: {clusterQueue: aiperf-cluster-queue}
```
### The three ways to bind a job to a queue

| # | Mechanism | Who applies it | Operator stamps `queue-name` label + `spec.suspend: true`? |
|---|---|---|---|
| 1 | CR `spec.scheduling.queueName` (CLI `--queue-name`) | operator, per job | **yes** |
| 2 | `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME` on the operator container | operator, cluster-wide fallback | **yes** |
| 3 | Namespace annotation `kueue.x-k8s.io/default-queue-name` applied by hand | Kueue's own webhook | **no** |

Chart `kueue.defaultQueueName` drives mechanism 2, so setting it is
sufficient: **yes**.

Mechanism 2 *is* what chart `kueue.defaultQueueName` sets (`operator.env` is a
fixed nine-key map and does not carry it). To set it without re-running Helm:

```bash
kubectl -n aiperf-system set env deployment/aiperf-operator -c operator \
  AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME=aiperf-local-queue \
  AIPERF_K8S_JOBSET_KUEUE_DEFAULT_PRIORITY_CLASS=
```

Mechanism 3 is the sharp edge. The operator never reads the annotation, so it
adds no `kueue.x-k8s.io/queue-name` label and no `spec.suspend: true`; admission
depends entirely on Kueue's webhook and the operator's phase reporting (which
keys on the label) will not surface `Queued`. The chart never applies the
annotation, so this mechanism only exists if you apply it yourself:

```bash
kubectl annotate namespace <your-benchmark-namespace> \
  kueue.x-k8s.io/default-queue-name=aiperf-local-queue --overwrite
```

### What the operator does with queue labels

- Resolution is `spec.scheduling.queueName` → env default → none. The label and
  `spec.suspend: true` use the *same* resolver, so a queue-labelled but
  unsuspended JobSet (which would bypass gang admission) cannot occur.
- `spec.scheduling.priorityClass` → `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_PRIORITY_CLASS`
  → none, stamped as `kueue.x-k8s.io/priority-class`.
- Gang semantics: controller pod + all worker pods admitted atomically or not.
- When the JobSet carries the queue label, is suspended, and the CR phase is
  `Pending` or `Queued`, the reported phase is `Queued`.

### Preflight matrix for the queue check

| Situation | Result |
|---|---|
| `queueName` set, LocalQueue exists in the namespace | PASS |
| `queueName` set, LocalQueue 404 and Kueue installed | FAIL — create the queue or drop `queueName` |
| `queueName` set, LocalQueue 404 and Kueue absent | FAIL (fails closed — an explicitly queued JobSet would stay suspended forever) |
| `queueName` set, other HTTP error | WARN |
| No `queueName`, Kueue not installed | SKIP |
| No `queueName`, Kueue installed, namespace has the default-queue annotation | PASS |
| No `queueName`, Kueue installed, no annotation | WARN — job bypasses gang-scheduling and quota |

## Reference install

```bash
helm upgrade --install aiperf-operator "$CHART" \
  -n aiperf-system --create-namespace \
  -f values-prod.yaml --wait --timeout 5m
```

```yaml
# values-prod.yaml
image: {repository: registry.example.com/aiperf, tag: "0.13.0"}
imagePullSecrets: [{name: regcred}]
operator:
  priorityClassName: system-cluster-critical
  nodeSelector: {node-pool: control}
  tolerations: [{key: dedicated, operator: Equal, value: control, effect: NoSchedule}]
  env: {resultsTtlDays: "14"}
  tmpSizeLimit: "512Mi"
storage: {size: 2Ti, storageClassName: fast-rwo}
benchmarkRbacNamespaces: [my-benchmarks]
serverMetricsDiscoveryNamespaces: [dynamo-server]
kueue:
  createQueues: true
  resources: {cpu: "2000", memory: "8Ti", gpu: "64"}
networkPolicy: {enabled: true, allowedIngressCIDRs: ["10.0.0.0/8"]}
serviceMonitor: {enabled: true, labels: {release: prometheus-stack}}
dashboard:
  enabled: true
  resources: {requests: {cpu: 100m, memory: 1Gi}, limits: {memory: 4Gi}}
```

Verify:

```bash
kubectl -n aiperf-system rollout status deployment/aiperf-operator
kubectl -n aiperf-system get pod -l app.kubernetes.io/component=operator \
  -o jsonpath='{.items[0].spec.containers[*].name}'   # operator results-server [dashboard]
helm test aiperf-operator -n aiperf-system
kubectl -n aiperf-system port-forward svc/aiperf-operator 8081:8081 &
curl -s localhost:8081/healthz
```
