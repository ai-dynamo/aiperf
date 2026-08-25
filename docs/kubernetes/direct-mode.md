---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Direct Mode (no operator)
---

# Direct Mode (no operator)

Direct mode runs an AIPerf benchmark on Kubernetes without requiring the AIPerf
operator (and the `AIPerfJob` CRD it owns) to be installed. The `aiperf kube`
CLI instead creates the underlying Kubernetes resources — `Role`,
`RoleBinding`, `ConfigMap`, and `JobSet`, plus a `Namespace` when
`--namespace` is omitted — directly against the API server via
`kubernetes_asyncio`.

It is triggered explicitly with `--no-operator`:

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://server:8000 --image aiperf:latest --no-operator
```

or automatically when `aiperf kube profile` detects that the `AIPerfJob` CRD
is not installed on the cluster. The CLI prints which mode it chose:

```
AIPerfJob CRD detected, using operator mode
# or
AIPerfJob CRD not found, deploying directly (no operator)
```

## What direct mode is not

Direct mode still uses:

- The same container image (`--image`) and the same JobSet topology (one
  controller pod plus N worker pods).
- The same `ConfigMap` containing the run config.
- The same per-namespace `Role` / `RoleBinding` (`RBACSpec`), which grants the
  benchmark's ServiceAccount read/watch on pods, pod logs, jobs, ConfigMaps,
  Services, Endpoints, and Events, plus `patch` on JobSets — all scoped to the
  benchmark namespace, and with no create, update, or delete verb anywhere. The
  `aiperfjobs` rule is included too and is simply inert without the CRD.
- The same live attach workflow (`aiperf kube attach`), which streams
  WebSocket progress through a port-forward to the controller pod.

What it does not use is the cluster-scoped operator Deployment, the
`AIPerfJob` custom resource, and anything that lives on the operator side —
dashboard UI, cross-job analytics, and the operator PVC that aggregates
results across runs.

You can still use an `AIPerfJob` YAML as the input to direct mode. The CLI
projects its JobSet-compatible deployment fields into the raw manifests, so
pod templates, scheduling, resource mode, failed-pod retention, image policy,
and an authored TTL are preserved. Explicit benchmark and Kubernetes CLI flags
overlay that YAML with the same precedence used in operator mode.

## Trade-off matrix

| Feature                                   | Operator mode                                       | Direct mode                                                                         |
| ----------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Submission object                         | `AIPerfJob` CR                                      | `Role` + `RoleBinding` + `ConfigMap` + `JobSet` (+ `Namespace` if auto-created)     |
| Status surface                            | `AIPerfJob.status.phase` reconciled by the operator | `JobSet` + pod status only (`kubectl get jobset`, `kubectl get pods`)               |
| Web dashboard (port 8081)                 | Served by operator Deployment                       | Not available                                                                       |
| Analytics: leaderboard / compare / history | Served by operator (cross-job view over PVC)       | Not available (no cross-job storage)                                                |
| Results persistence                       | Controller publishes to operator PVC (durable)      | Results live only on the controller pod's ephemeral `emptyDir` volume              |
| Automated TTL / cleanup                   | Operator reconciles terminal CRs and prunes them    | Relies on `JobSet.spec.ttlSecondsAfterFinished` (direct-mode default: 8 hours)      |
| Preflight checks                          | Operator-side preflight before admitting the CR     | None automatically — run `aiperf kube preflight` yourself first                      |
| Parameter sweeps and multi-run orchestration | `AIPerfSweep` + sweep-controller pod              | Not supported; `generate --no-operator` rejects these configs                        |
| RBAC footprint                            | Cluster-scoped ServiceAccount for the operator      | Benchmark namespace only: one `Role` + `RoleBinding` per run                        |
| Multi-user fairness (Kueue)               | Operator submits with Kueue labels end-to-end       | Supported — Kueue labels still flow through the `JobSet` spec                       |
| `aiperf kube results` default path        | Pulls from the operator PVC (works post-pod-GC)     | Requires `--from-pods`; pulls from the controller pod while it's still alive        |
| Concurrent benchmark isolation            | Operator reconciles; CR conflicts rejected          | CLI fails closed on any same-named non-Namespace resource, whatever its phase; you clean up |

## When direct mode is appropriate

Direct mode is the right choice when at least one of these holds:

- You do not have cluster-admin rights and cannot install the AIPerf CRD or
  the operator Deployment.
- You are restricted to a single namespace (e.g. a tenant namespace in a
  shared cluster) and your `RoleBinding` cannot grant cluster-wide access.
- You are running a one-off ad-hoc benchmark and don't need cross-run
  history, leaderboard, or compare views.
- You're running a CI smoke test (`--detach`) against an ephemeral cluster
  (kind, minikube, GitHub Actions kubernetes-in-docker) where installing the
  operator per job is more overhead than the benchmark itself.

Use operator mode when you need durable results across benchmark deletions,
the dashboard or analytics UIs, or centralized multi-user job management.

## End-to-end workflow

The typical direct-mode run is three commands:

```bash
# 1. Deploy the benchmark. --no-operator forces direct mode; it is also
#    picked automatically when the CRD is missing.
aiperf kube profile \
    --model Qwen/Qwen3-0.6B \
    --url http://inference.example.com:8000 \
    --image ghcr.io/nvidia/aiperf:v1.2.3 \
    --total-workers 10 \
    --concurrency 100 \
    --namespace aiperf-bench \
    --no-operator

# 2. (Optional) If you detached, reattach any time before the JobSet TTL
#    expires to stream progress.
aiperf kube attach --namespace aiperf-bench

# 3. Pull results off the controller pod BEFORE the JobSet TTL expires and
#    the pod is garbage-collected. --from-pods is required in direct mode.
#    --shutdown tells the controller API service to exit cleanly so the pod
#    can terminate afterwards.
aiperf kube results --from-pods --shutdown --namespace aiperf-bench
```

On a successful deploy the CLI prints one `Created <Kind>/<name>` line per
resource. All names derive from the benchmark name as `aiperf-<name>` (the
ConfigMap adds a `-config` suffix), where `<name>` is either your `--name` or
the auto-generated `<model>-<endpoint-type>-<phase-type>` slug
(`generate_benchmark_name`). For the run above — `--namespace aiperf-bench` is
explicit, so the Namespace is not created:

```
Created Role/aiperf-<name>
Created RoleBinding/aiperf-<name>
Created ConfigMap/aiperf-<name>-config
Created JobSet/aiperf-<name>
```

A `Created Namespace/aiperf-benchmarks` line is prepended only when
`--namespace` is omitted, in which case the CLI creates the default
benchmark namespace (`aiperf-benchmarks`) itself. An already-existing
namespace is reused and reported as `Namespace/<name> already exists`.

## Dry-run inspection

Preview the manifests without submitting them:

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://server:8000 --image aiperf:latest \
    --no-operator --dry-run > bench.yaml

kubectl apply -f bench.yaml   # equivalent to what the CLI would have done
```

The memory estimate is written to stderr, so `bench.yaml` contains only the
multi-document Kubernetes YAML and can be passed directly to `kubectl apply`.

`--no-operator` is required here and not merely a preference: `--dry-run`
never contacts the cluster, so it cannot detect that the `AIPerfJob` CRD is
absent and would otherwise print the operator CR as JSON instead of the
manifests. The upside of that same design is that this command works with no
cluster reachable at all — no kubeconfig, no network. See
[workflow.md](./workflow.md#--dry-run-fidelity) for the full decision table
and the pre-validation caveat.

This is useful when your cluster requires a GitOps commit or a manual review
before resources can be created.

## Results retrieval

In operator mode, `aiperf kube results` retrieves results from the operator's
PVC — this works even after the benchmark pods have been garbage-collected.

In direct mode there is no operator PVC. Results must be pulled via
`--from-pods`, which port-forwards to the controller pod's API service
(port 9090) and downloads the exported artifacts (`metrics.json`,
`profile_export_aiperf.json`, console exports, parquet files, and checkpoint
data). `--all` is the default; pass `--summary-only` to fetch just the
summary results.

There is one fallback, and only on the `--summary-only` path: if the API
call fails, the CLI retries with `kubectl cp` against the `control-plane`
container's `/results` directory. The default `--all` path has no such
fallback — if the controller API is unreachable, retrieval fails outright.

The controller pod also runs a small results sidecar on port 9091 that serves
the same `/results` volume (see `src/aiperf/kubernetes/results_sidecar.py`)
and outlives the main controller container, but `aiperf kube results
--from-pods` never targets it — only the operator's completion fetch does, so
that fallback is unavailable in direct mode.

**The pod must still exist when you run `aiperf kube results --from-pods`.**
The direct-mode JobSet sets `ttlSecondsAfterFinished` to 8 hours by default
(`K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS`, tunable via
`AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS`), giving you a generous window
before the pod is deleted. Pass `--ttl-seconds` on `profile` to override.

## Cleanup

Operator mode cleans up by deleting the `AIPerfJob` CR; the operator
reconciles the deletion and removes all child resources.

Direct mode has no CR, so cleanup is manual — `aiperf kube delete` and
`aiperf kube cleanup` both operate on `AIPerfJob` / `AIPerfSweep` CRs and
will not find a direct-mode run. Every resource is named
`aiperf-<name>` (the ConfigMap adds `-config`), so once results are safely
pulled:

```bash
kubectl delete jobset     aiperf-<name>        -n <namespace>
kubectl delete configmap  aiperf-<name>-config -n <namespace>
kubectl delete role       aiperf-<name>        -n <namespace>
kubectl delete rolebinding aiperf-<name>       -n <namespace>
```

Or, if you created a dedicated namespace for the run and don't need it
anymore:

```bash
kubectl delete namespace <namespace>
```

The default `ttlSecondsAfterFinished` on the `JobSet` means completed pods
will eventually be reaped automatically, but the `ConfigMap`, `Role`, and
`RoleBinding` will stay until you delete them (they are cheap, but
accumulate).

## Limitations

- **No cross-job analytics.** Leaderboards, compare views, and the run
  history UI are all served by the operator over its PVC; none of them work
  in direct mode.
- **Results are lost if the pod is deleted before you run `results`.**
  Set a longer `--ttl-seconds` on `profile` if you plan to pull results
  well after the run finishes, and prefer running `results --from-pods
  --shutdown` at the end of the workflow.
- **No operator-side admission / validation.** The operator normally
  validates specs, checks quota, and can reject malformed CRs before any
  pod starts. Direct mode skips all of it — including the endpoint check
  (`--skip-endpoint-check` is accepted for CLI parity but is a no-op here).
  Run `aiperf kube preflight` and `aiperf kube validate` yourself before
  submitting.
- **No CR lifecycle reconciler.** `timeoutSeconds`, `resultsTtlDays`, `cancel`,
  and `failurePolicy` require the operator and therefore do not take effect in
  direct mode. JobSet-native `ttlSecondsAfterFinished` and `keepFailedPods`
  still apply.
- **No CR-level status.** `kubectl get aiperfjob` does not work. Use
  `kubectl get jobset` and `kubectl get pods -l app=aiperf` (plus
  `aiperf kube logs` and `aiperf kube list`, which falls back to listing
  JobSets when no `AIPerfJob` CRs are found) to observe the run.
- **Existing resources with the same name.** Direct mode can safely reuse the
  Namespace, but it refuses to adopt an existing Role, RoleBinding, ConfigMap,
  or JobSet because there is no owner CR or immutable run UID that can prove
  the resource belongs to the new invocation. Pass a unique `--name` or delete
  all resources from the prior direct-mode run first.
