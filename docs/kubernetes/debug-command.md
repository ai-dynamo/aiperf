---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Debug Command
---

# Debug Command

`aiperf kube debug` produces a one-shot diagnostic snapshot of a benchmark namespace: which pods are in trouble, what Kubernetes events fired recently, what the cluster nodes look like, and — in verbose mode — the tail of every problem pod's logs. It is designed for triage *after* a benchmark has misbehaved, not for live observation.

If you need continuous updates while a benchmark is running, use [`aiperf kube watch`](./monitoring.md#watching-a-running-benchmark) instead.

---

## When to Use Which Command

AIPerf ships three cluster-inspection commands that overlap in theme but serve different moments in the lifecycle:

| Command | Timing | Duration | What it answers |
|---|---|---|---|
| `aiperf kube preflight` | Before deploy | One-shot | "Will my cluster even accept this job?" (CRDs installed, operator healthy, permissions OK, nodes have capacity) |
| `aiperf kube watch` | During a run | Streaming | "What is the job doing right now?" (live metrics, pod phases, events, auto-diagnosis) |
| `aiperf kube debug` | After something went wrong | One-shot | "Why did this job fail or stall?" (container-status problem map, recent events, node pressure, failed-pod logs) |

Use `debug` when:

- A benchmark is stuck in Pending and you want a single report to attach to an issue.
- Pods are CrashLoopBackOff-ing and you want all their log tails in one place.
- The operator's `status.phase` is `Failed` and you want a post-mortem summary without scripting `kubectl` calls.
- An AI agent is triaging a failure and needs a structured, grep-friendly snapshot.

Use `watch` instead when the job is still running and you want to *catch* the failure as it happens.

```mermaid
flowchart LR
    A[Plan a run] --> B[preflight]
    B -->|pass| C[profile / apply]
    C --> D[watch]
    D -->|job failed or stalled| E[debug]
    D -->|job succeeded| F[results]
    E --> G[Fix config / cluster]
    G --> B
```

---

## CLI Reference

```
aiperf kube debug [OPTIONS]
```

All flags are optional. With no flags, `debug` resolves the current namespace the same way the other kube subcommands do (from `KUBECONFIG` / active context) and inspects every AIPerf pod it finds there.

| Flag | Short | Default | Description |
|---|---|---|---|
| `--namespace` | `-n` | current context namespace | Kubernetes namespace to inspect. |
| `--job-id` | `-j` | (auto-resolve) | Specific AIPerf job ID to diagnose. If set, `debug` locates the JobSet across namespaces and restricts the report to pods labelled for that job. |
| `--all-namespaces` | `-A` | `false` | Inspect every namespace that contains at least one AIPerf JobSet. |
| `--verbose` | `-v` | `false` | Fetch logs from problem pods and show all recent events (not just `Warning`s). |
| `--kubeconfig` | | `$KUBECONFIG` | Path to kubeconfig file. |
| `--kube-context` | | current context | Kubernetes context to use. |

### Namespace resolution order

`debug` picks its target(s) using the first rule that matches:

1. `-A` / `--all-namespaces` — list every namespace with an AIPerf JobSet.
2. `-j <job-id>` — find the JobSet with that ID and use its namespace. Exits with an error if no such job exists.
3. `-n <namespace>` — use the given namespace verbatim.
4. Fall back to the namespace resolved from the active kubeconfig context (or `default`).

---

## What It Collects

For each target namespace, `debug` gathers four independent slices of cluster state and renders them as sections in the report. The collection is best-effort: a failing API call in one section does not prevent the others from being displayed.

### 1. Pod problem map

Every AIPerf pod in the namespace (selected by the `app.kubernetes.io/part-of=aiperf` label, or a narrower per-job selector when `-j` is set) is walked for container-status problems. See [`_extract_pod_info`](../../src/aiperf/cli_commands/kube/_debug_extract.py) — each container status is classified into one of:

| State | Severity | Suggested action |
|---|---|---|
| `CrashLoopBackOff` | CRITICAL | Check logs for the root cause. |
| `ImagePullBackOff` | CRITICAL | Verify the image name and registry access. |
| `ErrImagePull` | CRITICAL | Check image name, tag, and pull secrets. |
| `OOMKilled` (current or previous) | CRITICAL | Increase memory limits. |
| `CreateContainerConfigError` | ERROR | Check ConfigMaps, Secrets, and volume mounts. |
| `RunContainerError` | ERROR | Check security context and resource limits. |
| Pending with a waiting `reason` (e.g. `ContainerCreating`, `PodInitializing`) | WARNING | Container is waiting for the given reason. |
| `Unschedulable` (pod-level condition) | CRITICAL | Check node resources, taints/tolerations, and node selectors. |

The report also shows each pod's `phase`, total restart count across init and app containers, and the node it landed on.

### 2. Namespace events

Recent `Event` objects are fetched from the target namespace (CoreV1 `list_namespaced_event`) and sorted newest-first. By default `debug` shows up to 15 `Warning` events; with `-v` it widens to the 30 most-recent events of any type. See `_get_namespace_events` in `src/aiperf/cli_commands/kube/debug.py`.

### 3. Node resources

Cluster-wide node information is collected once per invocation (shared across all namespaces when `-A` is set). For each node the report shows:

- Ready condition
- CPU capacity and allocatable
- Memory capacity and allocatable
- `nvidia.com/gpu` capacity and allocatable (omitted when zero)
- Any active pressure conditions: `MemoryPressure`, `DiskPressure`, `PIDPressure`

See `_get_node_resources` in `src/aiperf/cli_commands/kube/debug.py`.

### 4. Problem pod logs (verbose only)

When `-v` is passed, `debug` calls `read_namespaced_pod_log` on every container of every pod flagged with at least one problem, tailing the last 20 lines. If the API call fails (pod deleted, container not yet started, RBAC denied), the report substitutes `<logs unavailable>` or `<error fetching logs>` and continues. See `_get_problem_pod_logs` in `src/aiperf/cli_commands/kube/debug.py`.

---

## Output Format

`debug` renders a human-oriented Rich report to the terminal. Each namespace produces the same section sequence:

1. `Diagnostic Report: <namespace>` header
2. Pod overview table (pod, status, restarts, node, issues count)
3. `Problems Found` list (or `No problems detected` on a clean namespace)
4. `Warning Events` table (or `Recent Events` table under `-v`)
5. `Node Resources` table
6. `Problem Pod Logs` sections (only under `-v`, only for pods with problems)
7. `Summary` footer (pod counts, warning-event count, nodes under pressure)

The report is rendered by `_print_report` in `src/aiperf/cli_commands/kube/_debug_report.py`.

`debug` does not currently emit machine-readable JSON. If you need structured output for automation, use `aiperf kube watch --output json` for live NDJSON of the same job, or script against `kubectl get events`, `kubectl get pods -o json`, and `kubectl get aiperfjob -o json` directly.

---

## Triage Recipes

The scenarios below are grouped by the symptom you'd see in `aiperf kube list` or `aiperf kube watch`. Each one shows the exact command to run and what to look for in the output.

### Pod OOMKilled mid-run

Symptom: a worker pod restarts repeatedly; `watch` shows request-rate gaps.

```bash
aiperf kube debug -j aiperf-bench-7f2a -v
```

In the report:

- **Problems Found** section — look for `OOMKilled` or `OOMKilled (previous)` rows. The pod name is in brackets.
- **Summary** — `Nodes under pressure:` will list any node reporting `MemoryPressure`; if the OOM pod is pinned there, the node is the root cause rather than your memory request.
- **Problem Pod Logs** — the last 20 lines usually show the allocation site (e.g. a large dataset load).

Fix by raising the container's memory limit or by reducing `dataset.max_items` / `inputs_per_request`.

### ImagePullBackOff on a fresh deployment

Symptom: `aiperf kube list` shows the job in `Pending` for minutes; no worker pods have transitioned to `Running`.

```bash
aiperf kube debug -j aiperf-bench-7f2a
```

In the report:

- **Problems Found** — `ImagePullBackOff` or `ErrImagePull` with container name.
- **Warning Events** — the `Failed` event's `MESSAGE` column contains the actual registry error (unauthorized, manifest not found, network unreachable).

Fix by correcting the image tag in the spec, adding an `imagePullSecrets` reference, or verifying that cluster nodes can reach the registry.

### PVC binding failure

Symptom: pods stuck in `Pending`, but there is no image error.

```bash
aiperf kube debug -n my-benchmark -v
```

In the report:

- **Problems Found** — an `Unschedulable` entry on a pod, severity `CRITICAL`, with a message like `persistentvolumeclaim "aiperf-results" not found` or `0/4 nodes are available: pod has unbound immediate PersistentVolumeClaims`.
- **Recent Events** (verbose) — the `FailedScheduling` events from the scheduler, repeated every few seconds.

Fix by ensuring the StorageClass exists and a provisioner is running; see [configuration.md](./configuration.md#results-storage).

### CrashLoopBackOff in the controller

Symptom: `aiperf kube list` shows the job `Failed` within seconds of deploy.

```bash
aiperf kube debug -j aiperf-bench-7f2a -v
```

In the report:

- **Problems Found** — the controller pod's container is in `CrashLoopBackOff`.
- **Problem Pod Logs** — the tail almost always contains the Python traceback. Common causes: bad dataset URL, missing tokenizer HF token, invalid endpoint URL.

Fix the config and redeploy. If the traceback mentions an AIPerf subsystem, cross-reference it against [ai-debugging-guide.md](./ai-debugging-guide.md).

### Zero pods scheduled

Symptom: `aiperf kube list` shows the job, but `kubectl get pods` in the namespace is empty.

```bash
aiperf kube debug -n my-benchmark -v
```

In the report:

- **Pods table** — `No pods found` warning at the top.
- **Recent Events** (verbose) — look for `JobSet` or `AIPerfJob` `Warning` events. Common causes: the operator failed to admit the CR (check the operator pod itself with `kubectl logs -n aiperf-system deploy/aiperf-operator`), or the JobSet hit a quota.
- **Node Resources** — if every node reports `Ready: No` or pressure, the scheduler is refusing to place anything.

Fix by correcting the CR, clearing quota, or resolving the node-level issue before redeploying.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Report printed successfully. Note: `debug` exits `0` even when problems are found — the report is the payload. |
| `0` | `-j <job-id>` was given, no job exists with that ID. A user-facing `No AIPerf job found with ID` error is printed and `debug` returns cleanly. |
| `0` | `-A` was given and no namespace contained an AIPerf JobSet. A warning is printed. |
| non-zero | Unrecoverable failure (kubeconfig missing, cluster unreachable, unexpected exception). The error is surfaced via the shared `cli_utils.exit_on_error("Error Running Diagnostics")` wrapper, which prints a red panel and propagates the cyclopts exit code. |

If you are wrapping `debug` from a script and need to distinguish "ran clean, found problems" from "ran clean, no problems", parse the **Summary** section — specifically the `Pods: N total, M running, K with issues` line.

---

## See Also

- [`aiperf kube watch`](./monitoring.md#watching-a-running-benchmark) — live dashboard version of the same data.
- [`aiperf kube preflight`](./getting-started.md) — pre-deploy checks that often prevent the failures `debug` diagnoses.
- [AI Debugging Guide](./ai-debugging-guide.md) — structured troubleshooting recipes that use `debug` output as input.
- Source: [`src/aiperf/cli_commands/kube/debug.py`](../../src/aiperf/cli_commands/kube/debug.py), [`_debug_extract.py`](../../src/aiperf/cli_commands/kube/_debug_extract.py), [`_debug_report.py`](../../src/aiperf/cli_commands/kube/_debug_report.py).
