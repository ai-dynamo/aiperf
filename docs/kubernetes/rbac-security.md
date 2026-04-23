---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: RBAC and Security
---

# RBAC and Security

AIPerf separates cluster-wide operator authority from per-namespace benchmark
authority. The operator Deployment runs under a ServiceAccount bound to a
`ClusterRole` so a single operator replica can watch `AIPerfJob` custom
resources across the cluster, create JobSets in benchmark namespaces, and
manage leader-election Leases. Benchmark pods (controllers, workers, record
processors) run under the `default` ServiceAccount of their namespace, bound
to a namespace-scoped `Role` that grants only the verbs those pods need to
discover peers, patch their JobSet status, and read the parent `AIPerfJob`.

The split lets cluster admins pre-provision the operator's cluster-wide
RBAC once (under a security review), then hand developers a `Role` template
that cannot escalate beyond the benchmark namespace. Developers never touch
cluster-scoped resources; operators never grant benchmark pods permissions
they don't need.

```mermaid
flowchart TB
    subgraph ClusterScope[Cluster scope]
        CR[ClusterRole<br/>aiperf-operator]
        CRB[ClusterRoleBinding]
        CR -.bound by.-> CRB
    end

    subgraph OperatorNS[Operator namespace]
        OperSA[ServiceAccount<br/>aiperf-operator]
        OperPod[Operator Deployment pod]
        OperPod -->|runs as| OperSA
    end

    subgraph BenchNS[Benchmark namespace]
        BRole[Role<br/>aiperf-operator-benchmark]
        BRB[RoleBinding]
        DefSA[ServiceAccount<br/>default]
        CtlPod[Controller pod]
        WkPod[Worker pod]
        BRole -.bound by.-> BRB
        BRB -.subject.-> DefSA
        CtlPod -->|runs as| DefSA
        WkPod -->|runs as| DefSA
    end

    CRB -.subject.-> OperSA
    OperPod -->|creates JobSet in| BenchNS
    OperPod -->|watches AIPerfJob across| ClusterScope
```

The operator owns JobSet/ConfigMap/Role creation in the benchmark namespace.
Benchmark pods only read what the operator placed there and patch their own
status.

## Operator ClusterRole catalog

The operator's `ClusterRole` is rendered from
`deploy/helm/aiperf-operator/templates/clusterrole.yaml`. Every rule exists
to support a concrete operator responsibility; nothing is granted
speculatively.

| API group | Resources | Verbs | Purpose | Source |
|---|---|---|---|---|
| `apiextensions.k8s.io` | `customresourcedefinitions` | `get, list, watch` | kopf CRD discovery at startup | `clusterrole.yaml:12-14` |
| `aiperf.nvidia.com` | `aiperfjobs`, `aiperfjobs/status`, `aiperfjobs/finalizers` | `get, list, watch, create, update, patch, delete` | Reconcile AIPerfJob CRs, patch status/phase, manage finalizers | `clusterrole.yaml:17-19` |
| `jobset.x-k8s.io` | `jobsets` | `create, delete, get, list, patch, update, watch` | Create/own the controller + worker JobSet for each AIPerfJob | `clusterrole.yaml:22-24` |
| `jobset.x-k8s.io` | `jobsets/status` | `get, list, watch` | Observe JobSet readiness and roll it up to AIPerfJob status | `clusterrole.yaml:25-27` |
| `kueue.x-k8s.io` | `workloads` | `get, list, watch` | Surface Kueue admission state on the AIPerfJob status | `clusterrole.yaml:30-32` |
| `batch` | `jobs` | `get, list, watch` | Monitor the Jobs that JobSet creates under the hood | `clusterrole.yaml:35-37` |
| `apps` | `deployments` | `get, list, watch` | Preflight checks for the JobSet controller and Kueue controller | `clusterrole.yaml:40-42` |
| `""` (core) | `serviceaccounts`, `resourcequotas`, `secrets` | `get, list, watch` | Preflight verifies custom SA, ResourceQuota headroom, and imagePullSecrets | `clusterrole.yaml:47-49` |
| `networking.k8s.io` | `networkpolicies` | `get, list, watch` | Preflight checks whether the benchmark namespace has a restrictive NetworkPolicy | `clusterrole.yaml:52-54` |
| `""` (core) | `configmaps` | `create, delete, get, list, patch, update, watch` | Create `run_config.json` ConfigMap consumed by every benchmark pod | `clusterrole.yaml:57-59` |
| `""` (core) | `services`, `endpoints` | `create, delete, get, list, watch` | Headless Service for pod DNS; endpoint monitoring | `clusterrole.yaml:62-64` |
| `rbac.authorization.k8s.io` | `roles`, `rolebindings` | `create, delete, get, list, watch` | Create the per-namespace benchmark `Role`/`RoleBinding` on first deploy | `clusterrole.yaml:67-69` |
| `""` (core) | `namespaces` | `get, list, watch` | Resolve the benchmark namespace referenced by AIPerfJob | `clusterrole.yaml:72-74` |
| `""` (core) | `pods`, `pods/log` | `get, list, watch` | Surface pod status, restart counts, and logs in `aiperf kube logs` | `clusterrole.yaml:77-79` |
| `""` (core) | `nodes` | `get, list` | Count GPUs for the cluster endpoint served by the API sidecar | `clusterrole.yaml:82-84` |
| `""` (core) | `events` | `get, list, watch, create, patch` | Emit Kubernetes events and let benchmark pods read them for UI display | `clusterrole.yaml:87-89` |
| `coordination.k8s.io` | `leases` | `create, delete, get, list, patch, update, watch` | kopf leader election when running more than one operator replica | `clusterrole.yaml:92-94` |

The binding
(`deploy/helm/aiperf-operator/templates/clusterrolebinding.yaml:8-17`) connects
this `ClusterRole` to the operator `ServiceAccount` in the release namespace.

## Benchmark-namespace Role catalog

`deploy/helm/aiperf-operator/templates/benchmark-rbac.yaml` renders a `Role`
and a matching `RoleBinding` in every namespace where benchmark pods may
run. By default this is both `default` and the configured
`benchmarkNamespace.name` (default `aiperf-benchmarks`). The `RoleBinding`
subject is the namespace's `default` ServiceAccount — benchmark pods never
need a custom SA.

| API group | Resources | Verbs | Purpose | Source |
|---|---|---|---|---|
| `""` (core) | `pods` | `get, list, watch` | Workers discover controller and peer record processors via pod labels | `benchmark-rbac.yaml:17-19` |
| `jobset.x-k8s.io` | `jobsets` | `get, list, watch, patch, update` | Controller patches JobSet status to signal graceful completion | `benchmark-rbac.yaml:22-24` |
| `aiperf.nvidia.com` | `aiperfjobs`, `aiperfjobs/status` | `get, list, watch, patch, update` | Controller reads AIPerfJob spec and patches per-phase progress fields | `benchmark-rbac.yaml:27-29` |

Notice that benchmark pods have no create/delete permission on any
resource, no access to Secrets, no cross-namespace visibility, and no access
to Events. They can only read what the operator placed there and patch the
status subresources they own.

## `rbac.create=false` workflow

In security-sensitive clusters, RBAC is usually reviewed and committed
separately from workload charts. Set `rbac.create=false` and
`serviceAccount.create=false` to make the operator chart use a
pre-provisioned `ServiceAccount` and skip both the `ClusterRole` + `ClusterRoleBinding`
and the per-namespace `Role` + `RoleBinding`.

```bash
helm install aiperf-operator ./deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --set rbac.create=false \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aiperf-operator-sa
```

The cluster admin applies the RBAC tree out-of-band. A minimal example
mirroring the chart defaults:

```yaml
---
# ServiceAccount in the operator namespace
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aiperf-operator-sa
  namespace: aiperf-system
---
# ClusterRole — copy the rules from clusterrole.yaml verbatim
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: aiperf-operator
rules:
- apiGroups: ["aiperf.nvidia.com"]
  resources: ["aiperfjobs", "aiperfjobs/status", "aiperfjobs/finalizers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# ... (remaining rules from clusterrole.yaml) ...
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: aiperf-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: aiperf-operator
subjects:
- kind: ServiceAccount
  name: aiperf-operator-sa
  namespace: aiperf-system
---
# Per-benchmark-namespace Role + RoleBinding (apply once per namespace)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: aiperf-operator-benchmark
  namespace: aiperf-benchmarks
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["jobset.x-k8s.io"]
  resources: ["jobsets"]
  verbs: ["get", "list", "watch", "patch", "update"]
- apiGroups: ["aiperf.nvidia.com"]
  resources: ["aiperfjobs", "aiperfjobs/status"]
  verbs: ["get", "list", "watch", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: aiperf-operator-benchmark
  namespace: aiperf-benchmarks
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: aiperf-operator-benchmark
subjects:
- kind: ServiceAccount
  name: default
  namespace: aiperf-benchmarks
```

Under `rbac.create=false`, the operator still tries to call
`create roles` and `create rolebindings` on the first reconcile of a new
namespace (see the `rbac.authorization.k8s.io` rule in the `ClusterRole`
catalog). If the admin pre-provisions the per-namespace `Role` and
`RoleBinding` in every intended benchmark namespace, the operator's
idempotent create returns `AlreadyExists` and reconciliation proceeds.
If you intend to run in a single fixed namespace only, you can trim those
verbs from the admin-owned `ClusterRole` without functional regression.

The helper
`deploy/helm/aiperf-operator/templates/_helpers.tpl`'s
`aiperf-operator.serviceAccountName` template resolves to `.Values.serviceAccount.name`
when `serviceAccount.create` is false, so the operator Deployment will
reference the pre-provisioned SA as expected.

## Pod `securityContext`

Every container in both the controller and worker pods receives a
hardened `securityContext` assembled by
`build_security_context()` in `src/aiperf/kubernetes/jobset_helpers.py:19-47`.
The base context applied to every container is:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

This context is applied to:

- The five control-plane containers in the controller pod
  (`jobset_builder.py:181`)
- The event-bus proxy sidecar (`jobset_builder.py:229`)
- The results-serving sidecar (`jobset_builder.py:256`)
- Each worker and record-processor container

### Overriding via `podTemplate.containerSecurityContext`

The CRD schema at
`deploy/helm/aiperf-operator/templates/crd.yaml:177-180` exposes
`spec.podTemplate.containerSecurityContext` as a free-form object. Keys
supplied there merge on top of the base context. `capabilities` is merged
shallowly (user keys update the drop/add lists); all other keys are replaced.

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: hardened-run
spec:
  podTemplate:
    containerSecurityContext:
      runAsUser: 65534
      runAsGroup: 65534
      capabilities:
        drop: ["ALL"]
        add: []
  benchmark:
    models: [meta-llama/Meta-Llama-3-8B-Instruct]
    endpoint: {type: chat, url: https://llm.example.com/v1}
```

### What `readOnlyRootFilesystem: true` requires

The root filesystem is read-only. AIPerf writes to four locations, which
must be provided as writable volumes (the JobSet builder already mounts
these as `emptyDir` volumes on every container via
`build_shared_volumes()` in `src/aiperf/kubernetes/jobset_helpers.py`):

- `/aiperf/ipc` — ZMQ IPC socket files (controller mode)
- `/aiperf/datasets` — shared dataset mmap files (dataset-manager writes, API serves to workers)
- `/results` — exported metrics, profile artifacts
- `/tmp` — general scratch (HuggingFace cache, tempfile)

If you supply an alternate `podTemplate.volumeMounts`, ensure each of these
paths is covered by a writable volume. Otherwise the controller will fail
to bind IPC sockets on startup and pods will CrashLoopBackOff.

## NetworkPolicy expectations

The chart does not ship a `NetworkPolicy` template — policy is
site-specific. AIPerf's control plane relies on a small, fixed set of
flows. When you enforce a default-deny policy in the benchmark namespace,
allow:

| Source | Destination | Ports | Reason |
|---|---|---|---|
| Worker pods | Controller pod | 5557, 5564, 5661/5662, 5663/5664, 5665/5666, 5667, 5668 (TCP) | ZMQ records push/pull (5557), credit router (5564), dataset-manager proxy DEALER/ROUTER (5661/5662), event-bus XPUB/XSUB (5663/5664), raw-inference PUSH/PULL (5665/5666), control ROUTER/DEALER (5667), credit-return router (5668). See `src/aiperf/config/zmq.py:243-289,547-571`. |
| All benchmark pods | Controller pod | 8080-8088 (TCP) | Health and readiness probes (`src/aiperf/kubernetes/environment.py:180-215`) |
| Client / ingress | API service | 9090 (TCP) | UI dispatch, progress streaming (`environment.py:195-196`) |
| Client / ingress | Results sidecar | 9091 (TCP) | Post-run result downloads (`environment.py:197-203`) |
| Controller + worker pods | LLM inference endpoint | 443 (TCP) | Outbound HTTPS to the endpoint under benchmark |
| All benchmark pods | `kube-dns` (UDP 53) | 53 | Pod DNS lookups for headless Service peer discovery |
| Operator pod | Kube API server | 443/6443 (TCP) | Watch CRDs, create JobSets, patch status |

Sample allow-list in the benchmark namespace:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aiperf-benchmark-allow
  namespace: aiperf-benchmarks
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: aiperf
  policyTypes: [Ingress, Egress]
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: aiperf
    ports:
    - {protocol: TCP, port: 5557}
    - {protocol: TCP, port: 5564}
    - {protocol: TCP, port: 5661}
    - {protocol: TCP, port: 5662}
    - {protocol: TCP, port: 5663}
    - {protocol: TCP, port: 5664}
    - {protocol: TCP, port: 5665}
    - {protocol: TCP, port: 5666}
    - {protocol: TCP, port: 5667}
    - {protocol: TCP, port: 5668}
  - from: []
    ports:
    - {protocol: TCP, port: 9090}
    - {protocol: TCP, port: 9091}
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: aiperf
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - {protocol: UDP, port: 53}
  - to: []
    ports:
    - {protocol: TCP, port: 443}
```

The final egress rule to `0.0.0.0/0:443` is the LLM endpoint reach; lock
it down to the endpoint's `Service`/`Endpoints` selector or an
`ipBlock.cidr` matching your inference gateway when the target is on-cluster.

## Least-privilege recipe

A hardened rollout checklist:

1. **Pre-provision operator RBAC** — apply the `ServiceAccount`,
   `ClusterRole`, and `ClusterRoleBinding` under admin review. Install the
   chart with `rbac.create=false` and `serviceAccount.create=false`.
2. **Dedicated benchmark namespace per team** — never share the benchmark
   namespace between teams. Benchmark pods have read access to every pod
   in the namespace (`benchmark-rbac.yaml:17-19`), so coexisting unrelated
   workloads leak metadata.
3. **Distinct ServiceAccounts per namespace** — the benchmark `RoleBinding`
   targets `default`. If multiple apps must coexist, rename the benchmark
   SA (e.g. `aiperf-benchmark-sa`) and set `spec.podTemplate.serviceAccountName`
   on the AIPerfJob instead of binding to `default`.
4. **`runAsNonRoot: true` everywhere** — the base context already enforces
   this. Do not override with `runAsUser: 0`.
5. **`readOnlyRootFilesystem: true`** — keep the default; cover the three
   writable paths above with `emptyDir` volumes.
6. **Drop all Linux capabilities** — the base context drops `ALL`. If a
   sidecar demands `NET_BIND_SERVICE` for a low port, add via
   `containerSecurityContext.capabilities.add` rather than reverting the
   drop.
7. **`seccompProfile: RuntimeDefault`** — default-applied. Do not downgrade
   to `Unconfined`.
8. **NetworkPolicy default-deny + allow-list** — apply the sample above
   and lock the LLM endpoint egress to a CIDR or selector.
9. **ResourceQuota on the benchmark namespace** — AIPerf preflight reads
   quotas (`clusterrole.yaml:47-49`) and will fail fast if headroom is
   insufficient.
10. **Audit logging** — enable Kubernetes audit logging on the
    `aiperf.nvidia.com` and `jobset.x-k8s.io` API groups. Operator
    reconciliation events, including status patches and JobSet creation,
    appear under the operator SA and are easy to attribute.
11. **Image pull secrets** — reference secrets via `imagePullSecrets` in
    `values.yaml` rather than mounting credentials into pods.
12. **PodSecurity admission** — enforce the `restricted` profile on the
    benchmark namespace. AIPerf's default container `securityContext`
    already satisfies it.

With this profile, a compromised benchmark pod cannot escape the
namespace, cannot read cluster-wide resources, cannot write outside its
emptyDir volumes, cannot add capabilities, and can only reach the LLM
endpoint and its own peers.
