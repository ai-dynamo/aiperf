---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Logs
---

# Logs

`aiperf kube logs` fetches or streams container logs from a specific
AIPerfJob. It targets the controller pod's service containers, the
per-pod worker / record-processor containers, and the sidecars that
surround them. Use it for post-mortem triage of a finished run, for live
streaming during an active benchmark, or for bulk-dumping every pod's
log to disk.

Unlike `aiperf kube debug`, which is a one-shot diagnostic snapshot
across an entire namespace, `logs` targets a single AIPerfJob and
operates at container granularity.

---

## CLI reference

```bash
aiperf kube logs [JOB_ID] [OPTIONS]
```

| Flag / argument | Type | Default | Description |
|---|---|---|---|
| `JOB_ID` (positional) | `str` | last-deployed job | AIPerfJob ID to pull logs from. If omitted, falls back to the job ID stored on disk by the most recent `aiperf kube profile` / `generate` invocation. |
| `-c`, `--container` | `str` | *(all containers on every pod)* | Restrict to a single container name. See the [container catalog](#container-catalog) below. |
| `-f`, `--follow` | flag | `false` | Stream logs in real time. Only one target can be followed at a time; see [follow semantics](#follow-semantics). |
| `--tail` | `int` | *(full log)* | Return only the last N lines. Combine with `-f` to start a stream at the tail. |
| `-o`, `--output` | `Path` | *(stdout)* | Write logs to a directory instead of printing. See [bulk dump](#bulk-dump-to-disk). |
| `--namespace` | `str` | `aiperf-benchmarks` (or cached) | Namespace that holds the AIPerfJob. Composite flag from `KubeManageOptions`. |
| `--kubeconfig` | `Path` | `$KUBECONFIG` / `~/.kube/config` | kubeconfig file to use. Composite flag from `KubeManageOptions`. |
| `--kube-context` | `str` | current context | kubeconfig context to use. Composite flag from `KubeManageOptions`. |

Job resolution works exactly like every other `aiperf kube` subcommand:
if you omit `JOB_ID`, the CLI reads the last-benchmark cache written by
`aiperf kube profile` / `aiperf kube generate` and reuses that ID and
namespace. Supplying `--namespace` on the command line always overrides
the cached namespace.

---

## Container catalog

AIPerf pods run one process per container. The names below are the 12
canonical values declared on `aiperf.kubernetes.constants.Containers`
(see `src/aiperf/kubernetes/constants.py`); feed them to `--container`
verbatim.

### Controller pod

The controller pod runs nine containers:

| Container | Purpose |
|---|---|
| `control-plane` | SystemController and orchestration logic. |
| `event-bus-proxy` | XPUB/XSUB event-bus proxy sidecar. Isolates pub/sub forwarding from the SystemController event loop so large fan-ins of workers and record processors at startup do not starve the control plane. |
| `dataset-manager` | Dataset generation and memory-map serving. |
| `timing-manager` | Request scheduling and credit timing. |
| `records-manager` | Metric record aggregation, export, and storage. |
| `api` | HTTP + WebSocket API surface for monitoring, progress, and result artifacts. |
| `results-sidecar` | Lightweight sidecar serving exported artifacts from `/results`; outlives the main controller on export so the operator can still download results. |
| `gpu-telemetry-manager` | DCGM GPU metrics collection. Present only when GPU telemetry is enabled. |
| `server-metrics-manager` | Prometheus server-side metrics scraping. Present only when server metrics are enabled. |

### Worker pod

Each worker pod runs three kinds of containers. The `<N>` indices are
pod-local and start at `0`:

| Container | Purpose |
|---|---|
| `worker-group-manager` | Per-pod worker-group lifecycle, dataset download, local inference proxy, raw-record upload coordination. One per pod. |
| `worker-<N>` | One request-issuing worker. Count controlled by `workers_per_pod`. |
| `record-processor-<N>` | One record processor. Count defaults to `workers_per_pod / RECORD_PROCESSOR_SCALE_FACTOR`. |

### Deprecated name

`worker-manager` is listed on `Containers` for backwards compatibility
with older manifests; it is **not** placed on any pod rendered by the
current `_JobSetManifestBuilder`. Passing `--container worker-manager`
matches nothing and emits `No matching containers found`.

---

## Default behaviour (no `--container`)

When `--container` is omitted, `aiperf kube logs` enumerates every pod
that carries the AIPerf job label (`aiperf.nvidia.com/job-id=<JOB_ID>`)
and prints the logs of **every container on every pod**, in the order
the API returns them. For a default JobSet with `worker_replicas=N` and
`workers_per_pod=W`, this fans out to:

- the 7–9 controller-pod containers (depending on which optional
  managers are enabled and whether the event-bus proxy sidecar is
  configured), plus
- `N * (1 + W + record_processors_per_pod)` worker-pod containers.

For large benchmarks that total quickly runs to hundreds of containers,
so you almost always want `--container` for interactive triage and
`-o <dir>` for a full capture.

---

## Follow semantics

```bash
aiperf kube logs -f --container control-plane
```

The `-f/--follow` flag streams logs over a long-lived
`read_namespaced_pod_log(follow=True)` connection and prints lines to
stdout as they arrive. Key properties:

- **Single-target streaming.** The implementation breaks out of the
  target loop after the first streamed container (`if follow: break`).
  If you pass `-f` and the resolved target list has more than one
  entry, the CLI prints a warning of the form:

  ```text
  Follow mode streams one container at a time.
  Showing controller-0/control-plane (9 targets total).
  Use --container to select a specific container.
  ```

  and only the first target is followed.

- **No auto-reattach.** When the underlying pod is deleted, restarted,
  or evicted, the stream ends and the command returns. AIPerf does
  not transparently reconnect; re-run the command.

- **Tail-at-start is supported.** `-f --tail 200` begins the stream at
  the last 200 buffered lines, then follows live.

- **Ctrl+C** terminates the stream cleanly; the `finally` block
  releases the underlying HTTP response so the connection does not
  leak.

---

## Output format

Output is plain UTF-8 text, one line per log record, with a short
header per target:

```text
==> controller-0/control-plane <==
<log line 1>
<log line 2>
...
```

Bytes that cannot be decoded as UTF-8 are replaced (`errors="replace"`)
so a single corrupt line never aborts the stream. The format is
intentionally close to `kubectl logs` so the output is easy to pipe
into `grep`, `rg`, or a pager.

---

## Bulk dump to disk

```bash
aiperf kube logs <JOB_ID> -o ./triage-2026-04-22
```

With `-o <DIR>`:

- The CLI creates `<DIR>/logs/` if it does not exist.
- For every pod matching the job label, it shells out to
  `kubectl logs -n <NAMESPACE> <POD_NAME>` (forwarding
  `--kubeconfig` / `--context` when set) and writes the stdout to
  `<DIR>/logs/<POD_NAME>.log`.
- On success, prints `Logs saved to <DIR>/logs/`.

Notes on the bulk dump path specifically:

- It writes **one file per pod**, not per container. Multi-container
  pods land in a single file in the order `kubectl logs` returns them.
- It calls the local `kubectl` binary via `run_command`, so `kubectl`
  must be on `PATH` and able to reach the cluster with the same
  credentials the rest of the `aiperf kube` CLI uses.
- `--follow`, `--container`, and `--tail` are **ignored** in
  `--output` mode; the dump always captures the full buffered log of
  each pod.

---

## Troubleshooting

### `No pods found for job ID: <JOB_ID>`

The job label selector (`app=aiperf,aiperf.nvidia.com/job-id=<JOB_ID>`)
matched nothing in the namespace. Usually one of:

- The job ID is wrong (typo, stale cache). Run `aiperf kube list` to
  see live jobs.
- The JobSet has already been garbage-collected by TTL. Check the
  AIPerfJob CR phase with `aiperf kube list -o wide`; once the CR and
  its JobSet are removed, pod logs are gone from the API server.
- `--namespace` points at the wrong namespace. The default is
  `aiperf-benchmarks`.

### `No matching containers found`

Pods exist for the job but none of them have a container named
`--container <NAME>`. Re-check the spelling against the
[container catalog](#container-catalog). Common mistakes:

- `worker` (not a real container name — workers are indexed:
  `worker-0`, `worker-1`, ...).
- `worker-manager` (deprecated compat name, not present on any pod).
- `record-processor` without an index.

### `Error getting logs: (400)` or `(404)`

Raised by `kubernetes_asyncio` when the API server rejects the pod-log
request. Typical causes:

- The container has not started yet (status `ContainerCreating`,
  `PodInitializing`, or stuck on image pull). Use
  `aiperf kube debug <JOB_ID>` to see pod phases and events.
- The pod was deleted between the list and the read call (race
  between `follow` reconnects and TTL cleanup).
- The container terminated and its log buffer was rotated out of the
  kubelet cache — try `--previous`-style recovery via `kubectl logs
  --previous` against the pod name directly.

### RBAC denials

Log reads require `pods/log` access in the target namespace. If the
CLI prints `Error getting logs: (403)`, confirm your kubeconfig user
or ServiceAccount has the verb; see
[rbac-security.md](./rbac-security.md) for the roles the operator
itself uses and a minimal user-side role for log access.

### Empty output

A container can legitimately produce zero log lines — for example, a
short-lived worker that failed its startup probe before any handler
ran. In follow mode the stream simply stays silent until the container
writes or exits; in buffered mode the CLI prints a single empty line.
Cross-check with `aiperf kube debug` and with container status
(`Waiting` reason / exit code) to rule out a crashed pod versus a
quiet one.

---

## Related commands

- [`aiperf kube debug`](./debug-command.md) — one-shot namespace-wide
  snapshot including pod phases, events, and recent log tails.
- [`aiperf kube attach`](./attach.md) — stream live benchmark progress
  over the controller's WebSocket instead of container logs.
- [`aiperf kube list`](./workflow.md) — list live AIPerfJobs to
  discover the ID you want to pass here.
