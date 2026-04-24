---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Kueue Integration
---

# Kueue Integration

AIPerf integrates with [Kueue](https://kueue.sigs.k8s.io/) to provide
gang-scheduling, quota management, and prioritization for benchmark JobSets.
This page explains what Kueue buys you, how to install and configure it, and
how AIPerfJobs bind to Kueue resources.

---

## What Kueue Is

Kueue is a Kubernetes-native job queueing system maintained by the upstream
`kubernetes-sigs` project. It sits between job controllers (JobSet, Job,
`RayJob`, etc.) and the cluster scheduler, admitting workloads only when the
cluster has enough free capacity in a named *ClusterQueue* (cluster-scoped
quota pool) addressed through a *LocalQueue* (namespace-scoped handle). A
workload either runs in full or stays suspended; Kueue never schedules a
partial JobSet.

Useful upstream entry points:

- [Overview](https://kueue.sigs.k8s.io/docs/overview/)
- [Concepts: ClusterQueue, LocalQueue, ResourceFlavor](https://kueue.sigs.k8s.io/docs/concepts/)
- [JobSet integration](https://kueue.sigs.k8s.io/docs/tasks/run/jobsets/)
- [Workload Priority Classes](https://kueue.sigs.k8s.io/docs/concepts/workload_priority_class/)

---

## Why Use Kueue With AIPerf

An AIPerf benchmark runs as a JobSet of several coordinated pods: the
controller, worker replicas, records manager, and optional telemetry
sidecars. All pods must start together, or the controller sits idle while
workers slowly trickle in and the benchmark either fails or produces
misleading numbers.

Kueue solves three problems that matter for benchmark campaigns:

- **Gang scheduling.** Kueue suspends the JobSet until the cluster can admit
  *every* pod at once. On a mixed cluster, this prevents a half-scheduled
  run that would skew latency metrics.
- **Priority and preemption across campaigns.** A
  `WorkloadPriorityClass` lets a smoke-test job jump the queue ahead of a
  long soak-test, or lets a tenant's urgent run preempt an idle reservation.
- **Fair sharing across teams.** A single ClusterQueue with `borrowingLimit`
  rules lets multiple namespaces share GPU capacity without one team
  starving another — useful when several groups run benchmarks on the same
  cluster.

Without Kueue, AIPerfJobs are submitted directly to the JobSet controller
and compete for nodes using default Kubernetes scheduling.

---

## Install

AIPerf pins the tested Kueue version in
[`dev/versions.py`](../../dev/versions.py):

```python
# dev/versions.py
KUEUE_VERSION = "v0.10.1"
```

Follow the upstream
[installation guide](https://kueue.sigs.k8s.io/docs/installation/) for that
release. A typical install applies the released manifest bundle:

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml
```

The `aiperf kube preflight` command verifies that the Kueue CRDs are
present and that any referenced LocalQueue actually exists (see
`src/aiperf/operator/preflight/_infra.py`). If Kueue is not installed, the
check is marked `SKIP` and AIPerfJobs run without gang-scheduling.

---

## Configuring a ClusterQueue and LocalQueue

A minimal setup for a benchmark cluster looks like this. Adjust the
`nominalQuota` values to match your real GPU inventory.

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: default-flavor
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: aiperf-cluster-queue
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: default-flavor
          resources:
            - name: "cpu"
              nominalQuota: 256
            - name: "memory"
              nominalQuota: 1024Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 16
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: aiperf-queue
  namespace: aiperf-benchmarks
spec:
  clusterQueue: aiperf-cluster-queue
```

Optional: add a `WorkloadPriorityClass` for high-priority runs.

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: aiperf-high
value: 1000
description: "High-priority AIPerf benchmark runs"
```

---

## Binding an AIPerfJob to a Queue

There are three ways to route a job to a LocalQueue. All three ultimately
set the `kueue.x-k8s.io/queue-name` label on the JobSet.

### 1. CLI flags

The `aiperf kube profile` command (and every other submit subcommand)
accepts `--queue-name` and `--priority-class`:

```bash
aiperf kube profile \
  --endpoint-url http://my-server:8000/v1 \
  --model Qwen/Qwen3-0.6B \
  --queue-name aiperf-queue \
  --priority-class aiperf-high
```

These flags live in the `Kubernetes Scheduling` group and are defined on
`KubeOptions` in `src/aiperf/config/kube.py`.

### 2. CR YAML

Set the same fields directly in the AIPerfJob spec under `scheduling`:

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: latency-sweep-7f2a
  namespace: aiperf-benchmarks
spec:
  scheduling:
    queueName: aiperf-queue
    priorityClass: aiperf-high
  benchmark:
    models: ["Qwen/Qwen3-0.6B"]
    # ...
```

### 3. Namespace default via Helm

The `aiperf-operator` Helm chart exposes a `kueue.defaultQueueName` value
(see `deploy/helm/aiperf-operator/values.yaml`). When set, the chart
annotates the benchmark namespace with
`kueue.x-k8s.io/default-queue-name`, and every AIPerfJob in that namespace
is admitted through Kueue automatically — no per-job flag required.

```yaml
# values.yaml
kueue:
  defaultQueueName: aiperf-queue
```

The annotation is applied by
`deploy/helm/aiperf-operator/templates/benchmark-namespace.yaml` when the
chart creates the benchmark namespace.

---

## What the Operator Does

```mermaid
flowchart LR
  CLI["aiperf kube profile<br/>--queue-name aiperf-queue"] --> CR["AIPerfJob CR<br/>spec.scheduling.queueName"]
  CR --> OP["Operator<br/>handlers/create.py"]
  OP --> JS["JobSet<br/>metadata.labels:<br/>kueue.x-k8s.io/queue-name<br/>spec.suspend: true"]
  JS --> K["Kueue admission<br/>controller"]
  K -->|admitted| RUN["spec.suspend: false<br/>pods scheduled"]
  K -->|queued| WAIT["Workload pending<br/>quota unavailable"]
  JS --> MON["Operator monitor<br/>Phase: QUEUED"]
  RUN --> MON2["Operator monitor<br/>Phase: INITIALIZING"]
```

Concretely:

1. `src/aiperf/kubernetes/jobset.py` translates
   `spec.scheduling.queueName` and `priorityClass` into JobSet labels
   (`kueue.x-k8s.io/queue-name`, `kueue.x-k8s.io/priority-class`) via
   `KueueLabels` in `src/aiperf/kubernetes/constants.py`.
2. When a `queueName` is set, the JobSet is created with
   `spec.suspend: true`. Kueue unsuspends it once the `Workload` it
   generates has been admitted.
3. `src/aiperf/operator/handlers/monitor.py::_handle_kueue_suspension`
   watches the JobSet's `spec.suspend` field. While the JobSet is suspended
   and carries a Kueue queue label, the operator surfaces the
   `QUEUED` phase on the AIPerfJob status (see `Phase.QUEUED` in
   `src/aiperf/operator/status.py`).
4. Once Kueue admits the workload, the JobSet unsuspends, pods start, and
   the monitor transitions the phase to `INITIALIZING` -> `RUNNING`.

You can observe admission directly with `kubectl`:

```bash
kubectl get workloads -n aiperf-benchmarks
kubectl get aiperfjob latency-sweep-7f2a -n aiperf-benchmarks \
  -o jsonpath='{.status.phase}'
```

---

## Troubleshooting

**Job stuck in `QUEUED` phase.** Kueue has accepted the Workload but has
not admitted it. Check in order:

1. `kubectl describe workload -n aiperf-benchmarks` — look at the
   `QuotaReserved` and `Admitted` conditions. The message usually names
   the resource that is over quota.
2. `kubectl get clusterqueue aiperf-cluster-queue -o yaml` — compare
   `spec.resourceGroups.*.nominalQuota` to what the JobSet requests.
   Benchmark worker pods request the CPU and memory listed in the CR's
   `podTemplate.resources`; see
   [`configuration.md`](configuration.md) for defaults.
3. Another admitted Workload may be holding the quota. List active
   workloads with
   `kubectl get workloads -A -o wide` and cancel stale ones.

**Preflight fails: `Kueue LocalQueue '<name>' not found`.** The
`--queue-name` flag references a LocalQueue that does not exist in the
target namespace. Either create it (see the YAML above) or drop the flag.
The preflight logic distinguishes "CRD missing" (`SKIP`) from "queue
missing" (`FAIL`) in `_verify_kueue_local_queue`.

**Preflight warns: `Kueue is installed but no queue configured`.** Kueue
is present on the cluster but the job bypasses it. Fix by either passing
`--queue-name`, setting `spec.scheduling.queueName` in the CR, or adding
the namespace default:

```bash
kubectl annotate namespace aiperf-benchmarks \
  kueue.x-k8s.io/default-queue-name=aiperf-queue
```

**`priorityClass` set but preemption never happens.** The
`WorkloadPriorityClass` value must be *higher* than currently admitted
workloads, and the ClusterQueue must enable preemption
(`spec.preemption.reclaimWithinCohort` / `withinClusterQueue`). See the
upstream
[preemption guide](https://kueue.sigs.k8s.io/docs/concepts/preemption/).

---

## See Also

- [`getting-started.md`](getting-started.md) — install the operator and
  run your first job.
- [`configuration.md`](configuration.md) — full AIPerfJob spec reference,
  including the `scheduling` block.
- [`production.md`](production.md) — multi-tenant and HA considerations.
