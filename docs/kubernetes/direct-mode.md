---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Direct Mode (no operator)
---

# Direct Mode (no operator)

Direct mode runs an AIPerf benchmark on Kubernetes without requiring the AIPerf
operator (and the `AIPerfJob` CRD it owns) to be installed. The `aiperf kube`
CLI instead creates the underlying Kubernetes resources — `Namespace`,
`ConfigMap`, `Role`, `RoleBinding`, and `JobSet` — directly against the API
server via `kubernetes_asyncio`.

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
- The same per-namespace `Role` / `RoleBinding` that lets the controller pod
  read and update its own run state.
- The same live attach workflow (`aiperf kube attach`), which streams
  WebSocket progress through a port-forward to the controller pod.

What it does not use is the cluster-scoped operator Deployment, the
`AIPerfJob` custom resource, and anything that lives on the operator side —
dashboard UI, cross-job analytics, and the operator PVC that aggregates
results across runs.

## Trade-off matrix

| Feature                                   | Operator mode                                       | Direct mode                                                                         |
| ----------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Submission object                         | `AIPerfJob` CR                                      | `Namespace` + `ConfigMap` + `Role` + `RoleBinding` + `JobSet`                       |
| Status surface                            | `AIPerfJob.status.phase` reconciled by the operator | `JobSet` + pod status only (`kubectl get jobset`, `kubectl get pods`)               |
| Web dashboard (port 8081)                 | Served by operator Deployment                       | Not available                                                                       |
| Analytics: leaderboard / compare / history | Served by operator (cross-job view over PVC)       | Not available (no cross-job storage)                                                |
| Results persistence                       | Controller publishes to operator PVC (durable)      | Results live only on the controller pod's ephemeral volume + the inline sidecar     |
| Automated TTL / cleanup                   | Operator reconciles terminal CRs and prunes them    | Relies on `JobSet.spec.ttlSecondsAfterFinished` (direct-mode default: 8 hours)      |
| Preflight checks                          | Operator-side preflight before admitting the CR     | CLI-only preflight (endpoint, image pull, namespace access)                         |
| RBAC footprint                            | Cluster-scoped ServiceAccount for the operator      | Benchmark namespace only: one `Role` + `RoleBinding` per run                        |
| Multi-user fairness (Kueue)               | Operator submits with Kueue labels end-to-end       | Supported — Kueue labels still flow through the `JobSet` spec                       |
| `aiperf kube results` default path        | Pulls from the operator PVC (works post-pod-GC)     | Requires `--from-pods`; pulls from the controller pod while it's still alive        |
| Concurrent benchmark isolation            | Operator reconciles; CR conflicts rejected          | CLI refuses to overwrite a `Running` or `Pending` same-named resource; you clean up |

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
    --workers-max 10 \
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

On a successful deploy the CLI prints a summary of the created resources,
for example:

```
Created Namespace/aiperf-bench
Created Role/aiperf-bench-benchmark-1a2b3c
Created RoleBinding/aiperf-bench-benchmark-1a2b3c
Created ConfigMap/aiperf-bench-benchmark-1a2b3c-config
Created JobSet/aiperf-bench-benchmark-1a2b3c
```

## Dry-run inspection

Preview the manifests without submitting them:

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://server:8000 --image aiperf:latest \
    --no-operator --dry-run > bench.yaml

kubectl apply -f bench.yaml   # equivalent to what the CLI would have done
```

This is useful when your cluster requires a GitOps commit or a manual review
before resources can be created.

## Results retrieval

In operator mode, `aiperf kube results` retrieves results from the operator's
PVC — this works even after the benchmark pods have been garbage-collected.

In direct mode there is no operator PVC. Results must be pulled via
`--from-pods`, which:

1. Port-forwards to the controller pod's API service and downloads the
   exported artifacts (`metrics.json`, `profile_export_aiperf.json`,
   console exports, parquet files, plus checkpoint data under
   `--all`).
2. Falls back to `kubectl cp` from the controller pod's shared `/results`
   volume if the API service is not reachable.

The controller pod runs a small results sidecar that serves those files from
the shared `/results` volume (see `src/aiperf/kubernetes/results_sidecar.py`),
so the files survive even after the main controller container exits.

**The pod must still exist when you run `aiperf kube results --from-pods`.**
The direct-mode JobSet sets `ttlSecondsAfterFinished` to 8 hours by default
(`K8sEnvironment.JOBSET.DIRECT_MODE_TTL_SECONDS`, tunable via
`AIPERF_K8S_JOBSET_DIRECT_MODE_TTL_SECONDS`), giving you a generous window
before the pod is deleted. Pass `--ttl-seconds` on `profile` to override.

## Cleanup

Operator mode cleans up by deleting the `AIPerfJob` CR; the operator
reconciles the deletion and removes all child resources.

Direct mode has no CR, so cleanup is manual. Once results are safely pulled:

```bash
kubectl delete jobset <job-name>      -n <namespace>
kubectl delete configmap <job-name>-config -n <namespace>
kubectl delete role <job-name>        -n <namespace>
kubectl delete rolebinding <job-name> -n <namespace>
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
  pod starts. In direct mode the CLI performs the same pre-flight checks
  it can do locally (endpoint reachability, image name, namespace access),
  but anything that requires cluster-wide state is skipped.
- **No CR-level status.** `kubectl get aiperfjob` does not work. Use
  `kubectl get jobset` and `kubectl get pods -l app=aiperf` (plus
  `aiperf kube logs` and `aiperf kube watch`) to observe the run.
- **Concurrent runs with the same name.** The CLI will refuse to overwrite
  a `Running` or `Pending` benchmark with the same name in the same
  namespace; pass a unique `--name` or delete the prior resources first.
