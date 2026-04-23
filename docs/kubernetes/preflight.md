---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Preflight Checks
---

# Preflight Checks

Preflight checks validate that a Kubernetes cluster is ready to run an AIPerf benchmark
before any resources are created. They surface common failure modes (missing CRDs,
insufficient RBAC, exhausted quotas, malformed manifests) as explicit, actionable
failures instead of cryptic pod errors an hour into a run.

There are two entry points:

- **`aiperf kube preflight`** — ad-hoc CLI check against the target cluster. Does not
  require an AIPerfJob CR.
- **Operator preflight** — runs on every `AIPerfJob` creation, before the operator
  creates the ConfigMap, JobSet, Role, or RoleBinding. On any `FAIL`, the operator
  sets the CR to `Failed` with a `PreflightPassed=False` condition and does not create
  resources.

Both share the same `CheckResult` / `CheckStatus` / `PreflightResults` shapes
(`src/aiperf/kubernetes/preflight.py`). The check sets differ because each has
different inputs: the CLI knows only what the user passed on the command line; the
operator has the full resolved deployment spec.

## Tiered execution

The operator runs checks in tiers. Later tiers only run if earlier tiers pass —
no point probing node capacity if the cluster is the wrong Kubernetes version.
See `src/aiperf/operator/preflight/_checker.py:88` (`OperatorPreflightChecker.run_all`).

```mermaid
flowchart TD
    T1[Tier 1: Cluster compatibility<br/>K8s version, JobSet CRD]
    T2[Tier 2: RBAC permissions]
    T3[Tier 3+: Concurrent checks<br/>infra, resources, workload]
    Admit[Admit: create ConfigMap and JobSet]
    Reject[Reject: CR set to Failed<br/>PreflightPassed=False]

    T1 -->|all pass| T2
    T2 -->|pass| T3
    T3 -->|no FAIL| Admit
    T1 -->|any FAIL| Reject
    T2 -->|FAIL| Reject
    T3 -->|any FAIL| Reject
```

Tier 1 and Tier 2 are sequential and short-circuit: the first failing check aborts the
whole run. Tier 3+ checks are fanned out with `asyncio.gather` and all results are
collected regardless of individual failures — users see every problem in one pass
rather than fix-and-retry whack-a-mole.

The whole sequence is bounded by `AIPERF_PREFLIGHT_TIMEOUT` (default 30 s,
`src/aiperf/operator/environment.py:114`). A timeout is reported as a synthetic
`Preflight Timeout` check with status `FAIL`.

## Operator vs. CLI invocation

| Aspect | Operator (`OperatorPreflightChecker`) | CLI (`CLIPreflightChecker`) |
|---|---|---|
| Trigger | `kopf.on.create` for `AIPerfJob` | `aiperf kube preflight` |
| Inputs | Fully resolved `DeploymentConfig`, `KubernetesDeployment`, `AIPerfConfig` | CLI flags only |
| Check count | 19 | 13 |
| On FAIL | CR → `Failed`, `kopf.PermanentError` raised | CLI exits 1, JSON output on stdout if `-o json` |
| Source | `src/aiperf/operator/preflight/_checker.py` | `src/aiperf/kubernetes/preflight.py:CLIPreflightChecker` |

## Status values

Every check produces one of five statuses (`CheckStatus`,
`src/aiperf/kubernetes/preflight.py`):

| Status | When |
|---|---|
| `pass` | Check confirmed the cluster meets the requirement. |
| `fail` | Check confirmed a blocking problem. Operator rejects the CR; CLI exits 1. |
| `warn` | Potential problem or non-blocking concern (e.g. quota close to limit, tainted-node mismatch). Does not block. |
| `skip` | Check not applicable (e.g. no nodeSelector set, no secrets referenced, Kueue not installed). |
| `info` | Informational only — value reported, no pass/fail judgment (e.g. external endpoint URL was parsed but cannot be dialled from the CLI). |

## CLI command

```
aiperf kube preflight [OPTIONS]
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `-i`, `--image` | string | unset | Container image to inspect. Enables image-registry and pull-secret checks. |
| `-e`, `--endpoint-url` | string | unset | LLM endpoint URL to probe. Enables the endpoint-connectivity check (cluster-service lookup for `*.svc` URLs; informational for external URLs). |
| `-w`, `--workers` | int | 1 | Planned worker pod count. Used to project CPU and memory requirements against node capacity and namespace quotas. |
| `-o`, `--output` | `text`\|`json` | `text` | Output format. `text` prints rich-formatted progress; `json` prints only the machine-parseable `PreflightResults` dict on stdout (all logging is suppressed to `WARNING`). |

Composite flags inherited from `KubeManageOptions` — `--namespace`, `--kubeconfig`,
`--kube-context` — resolve connection and namespace identically to every other
`aiperf kube` subcommand.

Source: `src/aiperf/cli_commands/kube/preflight.py`.

### Example: JSON output

```bash
aiperf kube preflight \
    --namespace aiperf-benchmarks \
    --image nvcr.io/nvidia/aiperf:25.04 \
    --endpoint-url http://vllm.models.svc.cluster.local:8000 \
    --workers 8 \
    -o json
```

```json
{
  "passed": true,
  "has_warnings": true,
  "checks": [
    {
      "name": "Cluster Connectivity",
      "status": "pass",
      "message": "Connected to Kubernetes cluster",
      "details": [],
      "hints": [],
      "duration_ms": 42.1
    },
    {
      "name": "Kubernetes Version",
      "status": "pass",
      "message": "Kubernetes v1.29.3 (1.24+ required)",
      "details": [],
      "hints": [],
      "duration_ms": 18.0
    },
    {
      "name": "RBAC Permissions",
      "status": "pass",
      "message": "All 8 required permissions granted",
      "details": ["  ✓ create configmaps", "  ✓ get pods"],
      "hints": [],
      "duration_ms": 210.6
    },
    {
      "name": "Resource Quotas",
      "status": "warn",
      "message": "Benchmark may exceed resource quota(s)",
      "details": ["ResourceQuota 'team-quota':", "    cpu: 40 / 64"],
      "hints": ["Request a quota increase or reduce worker count"],
      "duration_ms": 33.2
    }
  ]
}
```

## Check catalog — Tier 1 (blocking, cluster compatibility)

### Cluster Connectivity (CLI only)

- **Validates**: `client.VersionApi.get_code()` succeeds against the cluster.
- **Source**: `src/aiperf/kubernetes/preflight_checks.py:check_cluster_connectivity`
- **Fails if**: kubeconfig missing, cluster unreachable, TLS error, auth rejected.
- **Fix**: set `KUBECONFIG`, check `~/.kube/config`, verify VPN/tunnel.

This check is implicit for the operator (the operator is already in-cluster by the
time handlers run). On CLI failure, the remaining checks are skipped.

### Kubernetes Version

- **Validates**: `major.minor >= 1.24` per `MIN_K8S_MAJOR`/`MIN_K8S_MINOR`
  (`src/aiperf/operator/preflight/_common.py`).
- **Source**: `src/aiperf/operator/preflight/_tier1.py:_check_kubernetes_version`,
  `src/aiperf/kubernetes/preflight_checks.py:check_kubernetes_version`
- **Fails if**: cluster runs Kubernetes < 1.24.
- **Fix**: upgrade the control plane. Older versions lack JobSet support and
  sub-resource patch semantics the operator depends on.

### JobSet CRD

- **Validates**: `jobset.x-k8s.io/v1alpha2` CRD is registered and responds to
  `list_cluster_custom_object` (operator) or `read_custom_resource_definition` (CLI).
- **Source**: `src/aiperf/operator/preflight/_tier1.py:_check_jobset_crd`,
  `src/aiperf/kubernetes/preflight_checks.py:check_jobset_crd`
- **Fails if**: the CRD is not installed (HTTP 404).
- **Fix**: install JobSet per the install-hint emitted in the failure message.
  Example: `kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.5.2/manifests.yaml`.

## Check catalog — Tier 2 (blocking, RBAC)

### RBAC Permissions

- **Validates**: every `(verb, resource, group)` in `OPERATOR_RBAC_PERMISSIONS`
  (operator) or `REQUIRED_RBAC_PERMISSIONS` (CLI) resolves `allowed=true` via
  `SelfSubjectAccessReview`.
- **Source**: `src/aiperf/operator/preflight/_tier1.py:_check_rbac_permissions`,
  `src/aiperf/kubernetes/preflight_checks.py:check_rbac_permissions`,
  `src/aiperf/operator/preflight/_common.py:OPERATOR_RBAC_PERMISSIONS`
- **Fails if**: any permission probe returns `allowed=false` or raises. Error message
  lists every missing permission as `<verb> <group>/<resource>`.
- **Fix**: bind a Role or ClusterRole granting the listed verbs on the namespace.
  The operator Helm chart installs these by default
  (`src/aiperf/kubernetes/resources.py`).

Operator permission set (15 verbs over 8 resources): configmaps, roles,
rolebindings, pods, pods/log, events (all core / rbac), jobsets, jobsets/status
(jobset group). CLI requires a subset of 8.

## Check catalog — Tier 3 (concurrent, infra)

### Namespace (CLI only)

- **Validates**: target namespace exists, or the user can create it.
- **Source**: `src/aiperf/kubernetes/preflight_checks.py:check_namespace`
- **Status**:
  - `pass` — namespace exists, or 404 + create permission granted.
  - `fail` — namespace missing and no create permission.
  - `warn` — namespace missing and create-permission probe itself failed.
- **Fix**: create the namespace, or have an admin do so.

### JobSet Controller

- **Validates**: a deployment containing `"jobset"` in its name exists in
  `jobset-system` and has `ready_replicas > 0`.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_jobset_controller`,
  `src/aiperf/kubernetes/preflight_checks.py:check_jobset_controller`
- **Status**:
  - `pass` — deployment found and ready.
  - `warn` — deployment found but not ready.
  - `warn` — deployment not found (CLI: `fail`).
  - `skip` — cannot list `jobset-system` (403).
- **Fix**: `kubectl get pods -n jobset-system` to diagnose; reinstall the JobSet
  controller if missing.

### Service Account (operator only)

- **Validates**: if `spec.podTemplate.serviceAccountName` is set, the named service
  account exists in the target namespace.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_service_account`
- **Status**:
  - `skip` — no custom service account configured (operator uses default).
  - `pass` — service account exists.
  - `fail` — 404 on `read_namespaced_service_account`.
- **Fix**: `kubectl create serviceaccount <name> -n <namespace>`.

### DNS Resolution

- **Validates**: a deployment containing `"coredns"` in its name exists in
  `kube-system` and is ready. Workers resolve the controller's DNS name for ZMQ
  connections, so a broken DNS plane is a silent killer.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_dns`,
  `src/aiperf/kubernetes/preflight_checks.py:check_dns`
- **Status**: `pass` if ready, `warn` if found-but-not-ready or not found,
  `skip` if `kube-system` is inaccessible.
- **Fix**: `kubectl get pods -n kube-system -l k8s-app=kube-dns`.

### Network Policies

- **Validates**: enumerates `NetworkPolicy` objects in the target namespace.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_network_policies`,
  `src/aiperf/kubernetes/preflight_checks.py:check_network_policies`
- **Status**:
  - `pass` — no policies (pod-to-pod traffic unrestricted).
  - `warn` — at least one policy exists; lists all by name. AIPerf pods need
    multi-port TCP intra-namespace traffic, so restrictive policies can silently
    break the ZMQ mesh.
  - `skip` — 403.
- **Fix**: ensure policies allow ingress/egress within the namespace on ZMQ ports.

### Kueue Queue (operator only)

- **Validates**:
  - If `spec.scheduling.queueName` is set: the named `LocalQueue` exists in the
    namespace.
  - If not set and Kueue is installed: the namespace has the
    `kueue.x-k8s.io/default-queue-name` annotation.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_kueue_queue`
- **Status**:
  - `pass` — queue exists, or namespace has default-queue annotation.
  - `fail` — explicit `queueName` set but `LocalQueue` not found.
  - `skip` — Kueue CRD not installed.
  - `warn` — Kueue installed, no queue configured; job will bypass gang-scheduling.
- **Fix**: create the `LocalQueue`, set `scheduling.queueName`, or annotate the
  namespace with `kueue.x-k8s.io/default-queue-name`.

### Pod Security Admission (operator only)

- **Validates**: reads the `pod-security.kubernetes.io/enforce` label on the
  namespace. AIPerf pods run as non-root (UID 1000) with
  `seccomp=RuntimeDefault` and drop all capabilities, so they are compatible with
  `privileged`, `baseline`, and `restricted`.
- **Source**: `src/aiperf/operator/preflight/_infra.py:_check_pod_security_admission`
- **Status**:
  - `pass` — no PSA label, or label is one of the three known-compatible levels.
  - `warn` — unknown PSA level, or namespace lookup failed.
- **Fix**: usually informational; relabel the namespace if the enforced level
  blocks pod creation.

## Check catalog — Tier 3 (concurrent, resources)

All resource checks are short-circuited to `skip` when `spec.resourceMode=none`
(controller and worker CPU/mem requests and limits are intentionally omitted).
See `OperatorPreflightChecker._resource_mode_skip`.

### Node Resources

- **Validates**: sum of allocatable CPU and memory across Ready nodes is at least
  the deployment's estimated requirement (controller pods + `workers * worker-pod`).
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_node_resources`,
  `src/aiperf/kubernetes/preflight_capacity_checks.py:check_node_resources`
- **Status**:
  - `pass` — cluster has sufficient aggregate capacity and at least one node
    can fit the single largest pod (CLI combined check).
  - `warn` — aggregate shortfall. Message includes required vs. available CPU/mem.
  - `fail` (CLI only) — no single node can fit any one pod.
- **Fix**: reduce worker count, add nodes, or right-size the pods via
  `AIPERF_K8S_WORKER_POD_*` and `AIPERF_K8S_CONTROLLER_POD_*` environment variables.

### Node Selector Match (operator only)

- **Validates**: at least one Ready node has labels matching every
  key/value pair in `spec.podTemplate.nodeSelector`.
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_node_selector_match`
- **Status**:
  - `skip` — no `nodeSelector` specified.
  - `pass` — one or more nodes match.
  - `fail` — no Ready nodes match the selector.
- **Fix**: label an existing node (`kubectl label node <name> key=value`) or drop
  the selector from the CR.

### Per-Node Schedulability (operator only)

- **Validates**: at least one Ready node (matching any `nodeSelector`) has
  allocatable capacity to fit the single largest pod in the deployment (max of
  controller-pod and worker-pod CPU/mem).
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_per_node_schedulability`
- **Status**:
  - `pass` — at least one node can fit the largest pod.
  - `fail` — no node can; reports the largest pod size in the message.
- **Fix**: add larger nodes or reduce per-pod requests/limits.

### Resource Quotas

- **Validates**: for each `ResourceQuota` in the namespace, projected CPU and memory
  usage after the benchmark deploys stays under the hard limit.
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_resource_quotas`,
  `src/aiperf/kubernetes/preflight_capacity_checks.py:check_resource_quotas`
- **Status**:
  - `pass` — no quotas, or all quotas have headroom.
  - `info` (CLI) — quotas exist, headroom available; details list each quota.
  - `warn` — at least one quota would be exceeded. Message calls out which
    resource and the overage.
- **Fix**: request a quota increase, reduce worker count, or deploy to a
  different namespace.

### Memory Estimation (operator only)

- **Validates**: runs `aiperf.kubernetes.memory_estimator.estimate_memory` with the
  resolved config, total workers, and connections-per-worker. Flags configurations
  that are predicted to OOM.
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_memory_estimation`
- **Status**:
  - `pass` — estimator returned no warnings.
  - `warn` — estimator produced warnings; hints are the estimator's own
    recommendations.
- **Fix**: follow the estimator hints — usually reduce concurrency, reduce
  dataset size, or raise the pod memory ceiling via `AIPERF_K8S_*_MEMORY`.

### Tolerations (operator only)

- **Validates**: if `spec.podTemplate.tolerations` is set, at least one node has
  a taint whose `key` matches a configured toleration key.
- **Source**: `src/aiperf/operator/preflight/_resources.py:_check_tolerations`
- **Status**:
  - `skip` — no tolerations configured.
  - `pass` — at least one tainted node matches.
  - `warn` — no matching tainted nodes (tolerations may be unnecessary).
- **Fix**: either taint the intended nodes or remove the tolerations from the CR.

## Check catalog — Tier 3 (concurrent, workload)

### Secrets

- **Validates**: every secret referenced by the pod template — `imagePullSecrets`,
  `volumes[].secret.secretName`, and `env[].valueFrom.secretKeyRef.name` — exists
  in the namespace.
- **Source**: `src/aiperf/operator/preflight/_workload.py:_check_secrets`,
  `src/aiperf/kubernetes/preflight_capacity_checks.py:check_secrets`
- **Status**:
  - `skip` — no secrets referenced.
  - `pass` — all secrets readable.
  - `fail` — at least one secret returned 404.
  - `warn` — at least one secret returned 403 (cannot verify).
- **Fix**: `kubectl create secret -n <namespace>` for missing names, or grant
  `get secrets` to the caller.

### Image Reference (operator) / Image Pull (CLI)

- **Validates**: the configured image has a well-formed reference and a pull path
  that is plausibly authenticated.
- **Source**: `src/aiperf/operator/preflight/_workload.py:_check_image_reference`,
  `src/aiperf/kubernetes/preflight_capacity_checks.py:check_image`
- **Warns on**:
  - Implicit `:latest` tag — inconsistent deployments across reconciles.
  - Non-public registry (`PUBLIC_REGISTRIES`: `docker.io`, `registry-1.docker.io`,
    `ghcr.io`, `quay.io`, `nvcr.io`, `registry.k8s.io`) with no
    `imagePullSecrets` configured.
- **Fails on** (operator): empty image.
- **Fix**: pin an explicit tag; add `imagePullSecrets` for private registries.

### ConfigMap Size (operator only)

- **Validates**: the generated benchmark ConfigMap's data payload fits within
  Kubernetes' 1 MiB per-ConfigMap limit (`CONFIGMAP_MAX_SIZE_BYTES`).
- **Source**: `src/aiperf/operator/preflight/_workload.py:_check_configmap_size`
- **Status**:
  - `pass` — size below 1 MiB; exact byte count reported.
  - `fail` — over 1 MiB (API server would reject the Create).
- **Fix**: reduce input-dataset size, move large fixtures to a `PersistentVolume`,
  or drop optional config fields.

### Dry Run (operator only)

- **Validates**: posts the generated JobSet manifest with `dryRun=All`. Catches
  admission-time rejections (OPA/Gatekeeper, validating webhooks, schema errors)
  before the CR is admitted.
- **Source**: `src/aiperf/operator/preflight/_workload.py:_check_dry_run`
- **Status**:
  - `pass` — API server accepts the manifest.
  - `fail` — API server rejects; error body is parsed and surfaced.
  - `warn` — transient network/timeout error.
- **Fix**: read the API server's error message; usually points to a missing
  label, a forbidden field, or an admission-controller rule.

### Endpoint Connectivity (CLI only)

- **Validates**: the `--endpoint-url` is reachable. For `*.svc` / `*.svc.cluster.local`
  hosts, looks up the corresponding `Service` in the cluster. For external URLs,
  only parses the URL and reports `info` — actual egress cannot be tested from the CLI.
- **Source**: `src/aiperf/kubernetes/preflight_checks.py:check_endpoint_connectivity`
- **Status**:
  - `skip` — no `--endpoint-url` passed.
  - `pass` — cluster service exists.
  - `fail` — cluster service not found.
  - `info` — external URL; connectivity verified later at runtime.
- **Fix**: `kubectl get svc -A | grep <service>`.

## Skipping preflight

### `--skip-endpoint-check` on `aiperf kube profile`

`aiperf kube profile` runs a best-effort local endpoint probe before submitting
the CR. Pass `--skip-endpoint-check` to bypass it — useful when the endpoint is
reachable only from inside the cluster and the CLI is running outside.

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B \
    --url http://vllm.models.svc.cluster.local:8000 \
    --image nvcr.io/nvidia/aiperf:25.04 \
    --skip-endpoint-check
```

Source: `src/aiperf/cli_commands/kube/profile.py:178`.

### `spec.resourceMode=none`

Setting `resourceMode: none` on the `AIPerfJob` spec causes the operator's
resource-capacity checks (`Node Resources`, `Per-Node Schedulability`,
`Resource Quotas`, `Memory Estimation`) to return `skip`. Use this when pod
resource requests and limits are intentionally omitted so the scheduler can
fit pods anywhere. See `OperatorPreflightChecker._resource_mode_skip`.

### Per-check skip conditions

Several checks `skip` automatically when they have nothing to validate:

| Check | Skips when |
|---|---|
| Service Account | `serviceAccountName` not set |
| Node Selector Match | `nodeSelector` not set |
| Tolerations | `tolerations` not set |
| Secrets | no secrets referenced |
| Kueue Queue | Kueue CRD not installed |
| Endpoint Connectivity (CLI) | `--endpoint-url` not passed |
| Image Pull (CLI) | `--image` not passed |

### `AIPERF_PREFLIGHT_TIMEOUT`

Increase the whole-preflight timeout if your cluster is slow to respond:

```bash
export AIPERF_PREFLIGHT_TIMEOUT=90
```

Default: 30 s. Range: 0 (exclusive) to 120 s. Set on the operator pod's
environment — not on the CLI. See `src/aiperf/operator/environment.py:114`.

## Exit codes

`aiperf kube preflight` exits based on the aggregated `PreflightResults`:

| Condition | Exit code |
|---|---|
| All checks `pass` / `skip` / `info` | `0` |
| At least one `warn`, no `fail` | `0` (`has_warnings=true` in JSON) |
| At least one `fail` | `1` |

Source: `src/aiperf/cli_commands/kube/preflight.py` — `if not results.passed:
raise SystemExit(1)`. `results.passed` is false iff any check status is `fail`.
Warnings never block.

## Further reading

- [ai-debugging-guide.md](ai-debugging-guide.md) — interpreting `PreflightResults`
  JSON, surfacing failures in CI pipelines, and the full CR status-condition set
  (`PreflightPassed`, `JobSetReady`, `BenchmarkComplete`).
- [configuration.md](configuration.md) — CRD fields consumed by preflight
  (`resourceMode`, `scheduling.queueName`, `podTemplate.nodeSelector`,
  `podTemplate.tolerations`, `podTemplate.serviceAccountName`).
- [production.md](production.md) — recommended RBAC, Kueue setup, and PSA
  configurations for production clusters.
