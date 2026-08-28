---
name: aiperf-kube-setup
description: Use when preparing a real multi-node Kubernetes cluster for AIPerf - installing or upgrading the aiperf-operator Helm chart, installing JobSet, wiring a private registry pull secret, choosing node placement and a storage class for the results PVC, or scoping the operator to namespaces.
---

# Preparing a Real Cluster for AIPerf

Everything that happens once per cluster, before any benchmark is submitted.
This targets a real multi-node cluster (shared dev, DGXC, on-prem, cloud):
tainted node pools, a private registry, a non-default storage class, RBAC you
do not fully own.

**Related skills:** `aiperf-kube-run` (submit a benchmark once this is done),
`aiperf-kube-triage` (operator installed but jobs misbehave).

## 1. Inventory the cluster before you install

Never carry another cluster's manifest over unread. Node pool labels, taints,
arch, storage classes, and pull-secret names differ per cluster and are the
top source of "installed fine, every job Pending".

```bash
CTX=<your-context>                     # always pass --context explicitly
kubectl --context "$CTX" get nodes -L kubernetes.io/arch -L nvidia.com/gpu.product
kubectl --context "$CTX" get nodes -o custom-columns=\
NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu,TAINTS:.spec.taints
kubectl --context "$CTX" get storageclass
kubectl --context "$CTX" get crd jobsets.jobset.x-k8s.io
```

| Requirement | Verify |
|---|---|
| Kubernetes v1.24+, kubectl authenticated | `kubectl cluster-info` |
| JobSet controller (hard dependency of operator mode) | `kubectl get crd jobsets.jobset.x-k8s.io` |
| NVIDIA device plugin, if benchmarking GPU-resident servers | GPU column above is non-empty |
| Helm v3 and the AIPerf CLI locally | `helm version`, `aiperf --version` |
| Registry access for the operator image | `docker login <registry>` |

Record four things before writing any values file: the node pool label you may
schedule on, the taints you must tolerate, the storage class name, and the
pull-secret name.

## 2. Install JobSet

The chart grants RBAC for `jobset.x-k8s.io` but does **not** install JobSet.
AIPerf pins JobSet `v0.8.0`. Install that version unless you have a reason to
diverge, and keep the pin and this step in step with each other.

```bash
kubectl --context "$CTX" apply --server-side \
  -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml
kubectl --context "$CTX" -n jobset-system \
  wait --for=condition=available --timeout=120s deployment/jobset-controller-manager
```

## 3. Namespaces and the registry pull secret

Two namespaces: one for the operator, one for benchmark pods. Pre-create both
when cluster policy forbids chart-managed namespace creation.

```bash
NS_OP=aiperf-system
NS_BENCH=aiperf-benchmarks
kubectl --context "$CTX" create ns "$NS_OP"
kubectl --context "$CTX" create ns "$NS_BENCH"

for ns in "$NS_OP" "$NS_BENCH"; do
  kubectl --context "$CTX" -n "$ns" create secret docker-registry regcred \
    --docker-server=<registry> --docker-username=<user> --docker-password=<token>
done
```

The secret is needed in **both**: the operator namespace pulls the operator
image, the benchmark namespace pulls the benchmark image for every job pod.

## 4. Install the operator from a values file

Use a checked-in overlay, not a pile of `--set` flags — the placement and
storage fields are the ones you will revisit.

```yaml
# aiperf-operator-values.yaml
image:
  repository: <registry>/aiperf
  tag: <immutable-tag>
imagePullSecrets:
  - name: regcred
operator:
  nodeSelector: {nodeGroup: <cpu-pool>}
  tolerations: []                 # replace the chart defaults; see below
  watchNamespaces: [aiperf-benchmarks]
storage:
  size: 1Ti
  storageClassName: <class>
  accessMode: "ReadWriteOnce"
benchmarkNamespace:
  create: false                   # you created it in step 3
  name: "aiperf-benchmarks"
```

```bash
# The chart ships inside the aiperf source tree; there is no published Helm
# repo. Clone github.com/ai-dynamo/aiperf and point CHART at it, or use your
# own packaged/OCI copy if your org republishes it.
CHART=deploy/helm/aiperf-operator     # chart version 0.7.0

helm --kube-context "$CTX" upgrade --install aiperf-operator "$CHART" \
  --namespace "$NS_OP" -f aiperf-operator-values.yaml --wait --timeout 5m
```

Other knobs worth knowing, all top-level keys in the chart's `values.yaml`:

| Value | Default | Why you'd change it |
|---|---|---|
| `defaults.image` | computed from `image.*` | decouple benchmark image from operator image |
| `benchmarkRbacNamespaces` | `[]` | benchmarks run in more than one namespace |
| `serverMetricsDiscoveryNamespaces` | `[]` | scrape server metrics from an existing inference namespace |
| `kueue.createQueues` / `kueue.defaultQueueName` | `false` / `""` | provision a ResourceFlavor + ClusterQueue + LocalQueue; see the Kueue rule below |
| `serviceMonitor.enabled` | `false` | Prometheus Operator scraping of operator metrics |
| `operator.priorityClassName` | `""` | keep the operator off the eviction list |
| `operator.env` | — | operator lifecycle/results tunables only (9 fixed keys) |
| `rbac.create` / `networkPolicy.enabled` | `true` / `false` | hardening (read the RBAC rule below before flipping `rbac.create`) |
| `dashboard.enabled` | `false` | Plotly results UI |

Full values matrix, hardening posture, and CI patterns:
`references/configuration.md` (bundled with this skill).

## Rules that bite

- **`kueue.defaultQueueName` does not by itself route benchmarks through
  Kueue.** It only stamps a `kueue.x-k8s.io/default-queue-name` annotation onto
  the Namespace the chart renders — so it is a no-op when
  `benchmarkNamespace.create: false` **and** `benchmarkNamespace.name` is absent
  from `benchmarkRbacNamespaces` — the template renders the primary name only
  under `create`, but renders every `benchmarkRbacNamespaces` entry
  unconditionally, and annotates whichever rendered namespace matches
  `benchmarkNamespace.name`.
  What actually puts the `kueue.x-k8s.io/queue-name` label on the JobSet is
  `spec.scheduling.queueName` on the AIPerfJob, or the operator-level default
  `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME`. Set one of those two if you want every
  job admitted through a queue. AIPerf pins Kueue `v0.10.1`; the chart-created
  ClusterQueue omits `nvidia.com/gpu` from `coveredResources` unless you set
  `kueue.resources.gpu`, so GPU benchmarks are not quota-gated by default.

- **The chart's default tolerations are not universal.** `operator.tolerations`
  ships with `dedicated=user-workload` (`NoSchedule` + `NoExecute`) and
  `kubernetes.io/arch Exists`. On a cluster that taints differently they are
  inert, and the operator pod goes `Pending` with no obvious cause. Set the
  list to your cluster's taints or to `[]` on an untainted pool. Tolerations
  permit; they do not attract — pair them with `operator.nodeSelector`.
- **Chart `imagePullSecrets` covers the operator and `helm test` pods only.**
  Benchmark pods take theirs from the CR (`spec.podTemplate.imagePullSecrets`)
  or `--image-pull-secrets`. Setting only the chart value yields a healthy
  operator and `ImagePullBackOff` on every job.
- **`storage.storageClassName` defaults to `""`, and the PVC then omits the
  field entirely** — which resolves to the cluster's *default* StorageClass. On
  a cluster with no default class the PVC pends forever and `helm --wait` times
  out with nothing obviously wrong. Name the class explicitly, or set
  `storage.enabled=false` to fall back to an emptyDir bounded by
  `storage.emptyDirSizeLimit` (results then die with the pod — test clusters
  only).
- **A `WaitForFirstConsumer` storage class makes the PVC pend until the
  operator pod schedules.** That is normal; a PVC pending *after* the pod is
  Running means the class or zone is wrong.
- **`accessMode: ReadWriteOnce` binds results to one node.** Fine for the
  required single operator replica, but node failure strands the volume — use
  an RWX class if you need results to survive it.
- **`benchmarkNamespace.create: true` fails when the namespace already
  exists**, and `create: false` does not buy you an escape hatch via
  `benchmarkRbacNamespaces`: that list is rendered into Namespace objects
  unconditionally (the chart's benchmark-namespace template gates only the
  primary name on `create`), so listing an existing namespace re-triggers the same
  ownership failure. Either Helm-adopt the namespace (label
  `app.kubernetes.io/managed-by: Helm` plus annotations
  `meta.helm.sh/release-name` and `meta.helm.sh/release-namespace`) or leave
  it out of the chart and apply the benchmark Role/RoleBinding out of band.
- **`operator.watchNamespaces` does not narrow RBAC.** The ClusterRole still
  grants cluster-wide reads. Narrowing it is not a one-flag change:
  `rbac.create=false` gates the *whole* benchmark-RBAC template, so it also
  deletes the benchmark Role/RoleBinding that gives job pods `pods
  get/list/watch`, `jobsets patch`, and `aiperfjobs/status patch` — the install
  succeeds and then every benchmark fails. It also does not remove cluster
  scope on its own: the `helm test` ClusterRole/ClusterRoleBinding still
  render, so a zero-ClusterRole install needs `rbac.create=false` **and**
  `tests.enabled=false`, with both the operator and the benchmark RBAC
  reapplied namespace-scoped out of band.
- **Mutable tags silently deploy old code.** A rebuilt `:latest` with
  `pullPolicy: IfNotPresent` reuses whatever is on the node. Use immutable
  tags, and confirm what actually rolled out.
- **Multi-arch matters on mixed pools.** An amd64-only image lands
  `exec format error` on arm64 GPU nodes rather than a pull failure.
- **The per-container CPU defaults are sized for small runs.**
  `AIPERF_K8S_RECORDS_MANAGER_CPU` and `AIPERF_K8S_SYSTEM_CONTROLLER_CPU`
  default to `75m`; past a few hundred thousand requests the records manager
  pegs a core and heartbeats miss. Raise them before a large campaign, not
  after it stalls. They are read by the process that renders the JobSet, so
  they must be set on the **operator container** -- `spec.podTemplate.env` has
  no effect on container resources, and `operator.env` is a closed 9-key map
  (monitor/timeout/results tunables) that silently drops anything else:

  ```bash
  kubectl set env -n "$NS_OP" deploy/aiperf-operator -c operator \
    AIPERF_K8S_RECORDS_MANAGER_CPU=4000m
  ```

  Pass `-c operator` or the var also lands on the results-server container.
  The chart templates the operator env list in full and has no `extraEnv`, so
  **the next `helm upgrade` silently reverts this** — re-apply it after every
  upgrade, or the campaign that worked last month stalls this month.
- **Upgrading the chart does not restart running benchmarks.** In-flight
  AIPerfJobs keep their old controller image.

## Verify the install

```bash
kubectl --context "$CTX" get crd | grep aiperf.nvidia.com   # aiperfjobs + aiperfsweeps
kubectl --context "$CTX" -n "$NS_OP" get pods,pvc           # 2/2 ready (3/3 with dashboard), PVC Bound
kubectl --context "$CTX" -n "$NS_OP" logs deploy/aiperf-operator -c operator --tail=20
helm --kube-context "$CTX" test aiperf-operator -n "$NS_OP"
```

`helm test` pulls `alpine/k8s` from Docker Hub, not from your registry, and the
chart's `imagePullSecrets` holds credentials for the wrong host. On an
air-gapped or Docker-Hub-blocked cluster this step fails for reasons unrelated
to the operator — install with `tests.enabled=false` and verify by hand.

Then the online check, with the arguments a real job will use:

```bash
aiperf kube preflight --kube-context "$CTX" -n "$NS_BENCH" \
  --image <registry>/aiperf:<tag> --image-pull-secret regcred \
  --endpoint-url http://<svc>.<ns>.svc.cluster.local:8000 \
  --workers 16 -o json | jq '.checks[] | select(.status == "fail" or .status == "warn")'
```

`preflight` covers apiserver connectivity, API versions, and RBAC honestly.
Read the other three checks as existence tests, not reachability tests:

- **Image**: parses the reference. Supplying `--image-pull-secret`
  short-circuits it to `PASS` without contacting the registry, so a typo'd tag
  or an unpushed local build sails through.
- **Endpoint**: for a `.svc` host it only confirms the Service *object*
  exists; anything else returns `INFO` unverified. A Service with zero ready
  endpoints or a wrong port passes.
- **Node capacity**: sums `allocatable` across Ready nodes and compares it to
  the worker projection. It ignores current pod usage, taints, `nodeSelector`,
  and ResourceQuota, so a full or fully-tainted cluster passes and every pod
  then goes Pending. It also reads `AIPERF_K8S_*` from the *CLI's* environment,
  not the operator's, so a raised records-manager CPU is invisible to it.

A clean preflight means "nothing is obviously missing", not "this run will
schedule". `CheckStatus` also emits
`skip` and `info` for benign cases, which is why the filter above matches
`fail`/`warn` rather than everything that is not `pass`. Re-run it
whenever the worker count or endpoint changes — capacity is the check that
goes stale. Finish with a small benchmark from `aiperf-kube-run` before
handing the cluster to anyone else.

The `AIPerfJob`/`AIPerfSweep` CRDs shipped in the chart are generated from the
operator's Pydantic models. Never hand-edit `templates/crd-aiperfjob.yaml` or
`templates/crd-aiperfsweep.yaml` in a chart checkout — an upgrade overwrites
them. From an aiperf source checkout, regenerate with
`uv run python tools/generate_crd.py` (`--check` verifies without writing).

## Local Kind clusters

Out of scope here: use it for correctness and lifecycle work, not for
placement, storage, or performance. The differences are a locally built image
(`kind load` plus `image.pullPolicy=Never`, re-loaded after every rebuild), no
pull secret, and the default storage class. Everything above about tolerations,
node pools, and RWX still needs re-deriving on the real cluster.
