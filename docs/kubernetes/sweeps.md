---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Sweeps on Kubernetes
---

# Sweeps on Kubernetes

Parameter sweeps explore a multi-dimensional space (concurrency, ISL/OSL,
models, rates) in a single declarative config. The sweep *config model* is
shared with the local CLI; see
[Parameter Sweeps and Multi-Run Statistics](../tutorials/sweeps.md) for the
generic grid / scenario / magic-list semantics. This page only covers what
is different when a sweep runs under the Kubernetes operator.

> **Status at HEAD (client-orchestrated only).** The CR/config layer parses
> `spec.benchmark.sweep` and the local Python orchestrator expands it into
> N variations in-process, but the Kubernetes operator currently runs
> **exactly one `BenchmarkRun` per `AIPerfJob`** and explicitly discards
> the `sweep` and `multi_run` blocks before submission (see
> [What the operator does today](#what-the-operator-does-today)). Label
> scaffolding for per-variation identification
> (`aiperf.nvidia.com/sweep-run`, `aiperf.nvidia.com/variation-index`,
> `aiperf.nvidia.com/run-index`) is defined in
> [`src/aiperf/kubernetes/constants.py`][constants] but no code path
> stamps those labels yet — they are reserved for the future
> multi-point-per-job implementation. Until then, treat each sweep point
> as its own `AIPerfJob` and aggregate client-side.

[constants]: ../../src/aiperf/kubernetes/constants.py

---

## Big picture

Sweep execution on Kubernetes today is **client-orchestrated**: a loop
outside the cluster submits one `AIPerfJob` per sweep point, each
producing its own JobSet, its own results directory, and (under Kueue)
its own `Workload`.

```mermaid
flowchart LR
  driver[driver<br/>bash loop / CI matrix / script]
  driver -->|kubectl apply -f c10.yaml| j1[AIPerfJob c10]
  driver -->|kubectl apply -f c25.yaml| j2[AIPerfJob c25]
  driver -->|kubectl apply -f c50.yaml| j3[AIPerfJob c50]
  driver -->|kubectl apply -f c100.yaml| j4[AIPerfJob c100]
  j1 --> js1[JobSet c10]
  j2 --> js2[JobSet c25]
  j3 --> js3[JobSet c50]
  j4 --> js4[JobSet c100]
  js1 --> p1[/results/c10/]
  js2 --> p2[/results/c25/]
  js3 --> p3[/results/c50/]
  js4 --> p4[/results/c100/]
  p1 & p2 & p3 & p4 --> api[/api/v1/analytics/compare]
  api --> table[cross-job<br/>comparison table]
```

For a standalone `aiperf profile --config sweep.yaml` run (non-k8s), the
same config expands in-process:

```mermaid
flowchart LR
  yaml[sweep.yaml<br/>sweep.variables = {concurrency: [10, 25, 50]}]
  yaml --> expand[expand_sweep]
  expand --> plan[BenchmarkPlan<br/>3 configs, 3 variations]
  plan --> orch[MultiRunOrchestrator]
  orch --> r1[run 0]
  orch --> r2[run 1]
  orch --> r3[run 2]
  r1 & r2 & r3 --> agg[aggregate/<br/>profile_export_aiperf_sweep.csv]
```

---

## What the operator does today

The `AIPerfJob` CRD declares `spec.benchmark` as a preserve-unknown-fields
block (see [`deploy/helm/aiperf-operator/templates/crd.yaml`][crd]), so a
`sweep:` key inside `spec.benchmark` is **accepted and stored on the CR
unchanged**. The operator then parses it, and in
`build_benchmark_run()` in
[`src/aiperf/operator/spec_converter.py`][spec-conv] around line 194-195
it strips both sweep-related blocks:

```python
run_config.pop("multi_run", None)
run_config.pop("sweep", None)
apply_k8s_runtime_config(run_config, run_id, namespace)
cfg = BenchmarkConfig.model_validate(run_config)
```

The resulting `BenchmarkConfig` has only the base values, so the JobSet
runs a single benchmark point — whatever values sit at the base of the
config, ignoring any variation list.

There is **no top-level `spec.sweepExecution` field** at HEAD. The
canonical CRD allow-list is `_DEPLOYMENT_FIELDS` in
[`src/aiperf/kubernetes/validate.py`][validate]:

```python
_DEPLOYMENT_FIELDS = {
    "image", "imagePullPolicy", "keepFailedPods", "resourceMode",
    "connectionsPerWorker", "timeoutSeconds", "ttlSecondsAfterFinished",
    "resultsTtlDays", "cancel", "podTemplate", "scheduling",
    "skipEndpointCheck",
}
```

If you see `sweepExecution` referenced in older docs, that is stale —
sweeps have never shipped as a first-class top-level spec field; the
design has always been that `sweep` lives inside `spec.benchmark` and is
expanded by AIPerf's config layer.

[crd]: ../../deploy/helm/aiperf-operator/templates/crd.yaml
[spec-conv]: ../../src/aiperf/operator/spec_converter.py
[validate]: ../../src/aiperf/kubernetes/validate.py

---

## Config file schema

The sweep block is defined in
[`src/aiperf/config/sweep.py`][sweep] and attaches to `AIPerfConfig`
(which extends `BenchmarkConfig` with sweep + multi_run), per
[`src/aiperf/config/config.py`][config].

| Field | Type | Notes |
|---|---|---|
| `sweep.type` | `"grid" \| "scenarios"` | Discriminator. |
| `sweep.variables` (grid) | `dict[str, list[Any]]` | Dot-path -> values; Cartesian product. |
| `sweep.runs` (scenarios) | `list[dict]` | Each dict deep-merged onto base. |

Magic-list shorthand (e.g. `phases.profiling.concurrency: [10, 25, 50]`
at the top level, with no explicit `sweep:` key) is also supported and
expands the same way locally; on Kubernetes it is likewise dropped by the
operator today.

See [the tutorial](../tutorials/sweeps.md) for full grid / scenario /
magic-list examples.

[sweep]: ../../src/aiperf/config/sweep.py
[config]: ../../src/aiperf/config/config.py

### Accepted-but-discarded shape (inside `spec.benchmark`)

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: llama8b-saturation
  namespace: aiperf-benchmarks
spec:
  image: nvcr.io/nvidia/aiperf:latest
  resourceMode: guaranteed
  benchmark:
    models:
      - meta-llama/Llama-3.1-8B-Instruct
    endpoint:
      urls: ["http://llama8b:8000/v1/chat/completions"]
      type: chat
      streaming: true
    datasets:
      main:
        type: synthetic
        entries: 2000
        prompts:
          isl: {type: normal, mean: 512, stddev: 50}
          osl: {type: normal, mean: 128, stddev: 25}
    phases:
      profiling:
        type: concurrency
        dataset: main
        requests: 500
        concurrency: 32        # <-- this is what actually runs
        grace_period: 30
    sweep:                     # <-- accepted on the CR, but discarded
      type: grid               #     before the BenchmarkRun is built
      variables:
        phases.profiling.concurrency: [10, 25, 50, 100]
```

Submitting this CR produces a single benchmark at
`phases.profiling.concurrency: 32`. The four sweep points are **not**
materialized.

---

## CLI flags

`aiperf kube profile` exposes **no `--sweep-*` flags**. Sweep is expressed
only through a config file or through `spec.benchmark.sweep` on a CR, and
is discarded by the operator either way. The full flag set at HEAD
([`src/aiperf/cli_commands/kube/profile.py`][profile-cli]):

- `-d, --detach`
- `--no-wait`
- `--attach-port <int>`
- `--skip-endpoint-check`
- `--dry-run`
- `--no-operator`

Plus the shared `KubeOptions` fields from
[`src/aiperf/config/kube.py`][kube-opts]: `--namespace`, `--kubeconfig`,
`--kube-context`, `--name`, `--image`, `--image-pull-policy`,
`--workers-max`, `--ttl-seconds`, `--node-selector`, `--tolerations`,
`--queue-name`, `--priority-class`, `--annotations`, `--labels`,
`--image-pull-secrets`, `--env-vars`, `--env-from-secrets`,
`--secret-mounts`, `--service-account`.

There is no in-CLI override mechanism for benchmark values
(no `--override` / `--set`), so to vary `concurrency` between jobs you
render one YAML per variation (see the recipe below).

[profile-cli]: ../../src/aiperf/cli_commands/kube/profile.py
[kube-opts]: ../../src/aiperf/config/kube.py

---

## Output layout

### Local (non-k8s) sweep

`aiperf profile --config sweep.yaml` writes under `artifacts.dir`:

```
artifacts/saturation/
  variation_0000_concurrency_10/
    profile_runs/
      run_0001/
  variation_0001_concurrency_25/
    ...
  aggregate/
    profile_export_aiperf_sweep.csv    # one row per variation
    profile_export_aiperf_aggregate.json
```

The CSV exporter is
[`AggregateSweepCsvExporter`][agg-csv] (`profile_export_aiperf_sweep.csv`).

[agg-csv]: ../../src/aiperf/exporters/aggregate/aggregate_sweep_csv_exporter.py

### Kubernetes (one job per point)

Each `AIPerfJob` writes to `/results/{job-id}/` on the operator PVC. For
an N-point sweep expressed as N `AIPerfJob`s, you get N sibling job
directories:

```
/results/
  llama8b-c10/        # AIPerfJob llama8b-c10
    profile_export_aiperf.csv
    profile_export_aiperf.json
  llama8b-c25/
  llama8b-c50/
  llama8b-c100/
```

There is **no cross-job `profile_export_aiperf_sweep.csv`** on the
operator PVC — that file is only produced by the local
`MultiRunOrchestrator`. To get a sweep-style view of N jobs, either
download each per-job CSV and combine client-side, or use the
`/api/v1/analytics/compare` endpoint (see
[results-api.md](results-api.md)).

---

## Retrieving results

Every `AIPerfJob` is individually addressable via the results server
(see [`src/aiperf/operator/results_server.py`][results-server]):

- `GET /api/v1/results/{namespace}/{job_id}` — list files
- `GET /api/v1/results/{namespace}/{job_id}/profile_export_aiperf.csv` — download
- `GET /api/v1/analytics/summary/{namespace}/{job_id}` — DuckDB summary
- `GET /api/v1/analytics/compare?jobs=ns/a,ns/b,ns/c` — cross-job table
- `GET /api/v1/analytics/leaderboard` — rank runs by metric
- `GET /api/v1/analytics/history` — metric over time

`analytics/compare` is the primary sweep-aggregation entry point today.

[results-server]: ../../src/aiperf/operator/results_server.py

---

## Kueue interaction

One `AIPerfJob` maps to one JobSet and therefore one Kueue `Workload`, so
a client-orchestrated sweep of N sibling jobs produces **N Workloads**
queued independently. Control ordering with
`spec.scheduling.priorityClass` and target a LocalQueue with
`spec.scheduling.queueName`; see [kueue.md](kueue.md).

If and when the operator learns to materialize sweep sub-jobs inside one
`AIPerfJob`, each variation will likely still become its own JobSet and
therefore its own Workload — the mapping **one benchmark point = one
Workload** is a property of the JobSet boundary, not of who submits it.

---

## Recipe: concurrency sweep over 4 points

Target: sweep `phases.profiling.concurrency` over `[10, 25, 50, 100]` with
`requests: 500` each, against the same endpoint. Because `aiperf kube
profile` does not accept per-value overrides, we render one YAML per
variation and submit each.

### 1. Template (`tpl.yaml`)

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: llama8b-c__C__
  namespace: aiperf-benchmarks
spec:
  image: nvcr.io/nvidia/aiperf:latest
  resourceMode: guaranteed
  benchmark:
    models:
      - meta-llama/Llama-3.1-8B-Instruct
    endpoint:
      urls: ["http://llama8b:8000/v1/chat/completions"]
      type: chat
      streaming: true
    datasets:
      main:
        type: synthetic
        entries: 2000
        prompts:
          isl: {type: normal, mean: 512, stddev: 50}
          osl: {type: normal, mean: 128, stddev: 25}
    phases:
      profiling:
        type: concurrency
        dataset: main
        requests: 500
        concurrency: __C__
        grace_period: 30
```

### 2. Render and submit one CR per concurrency value

```bash
set -euo pipefail
NS=aiperf-benchmarks
for C in 10 25 50 100; do
  sed "s/__C__/$C/g" tpl.yaml > "c${C}.yaml"
  kubectl apply -f "c${C}.yaml"
done
```

Each applied manifest creates one `AIPerfJob` CR; after the operator
reconciles, each becomes one JobSet and (with Kueue) one Workload.

### 3. Wait for completion

```bash
aiperf kube list --namespace aiperf-benchmarks
aiperf kube watch llama8b-c100 --namespace aiperf-benchmarks
```

### 4. Aggregate with `/analytics/compare`

Port-forward the results API (see [results-api.md](results-api.md) for
deployment-specific details), then:

```bash
curl -sSf "http://localhost:8081/api/v1/analytics/compare?jobs=\
aiperf-benchmarks/llama8b-c10,\
aiperf-benchmarks/llama8b-c25,\
aiperf-benchmarks/llama8b-c50,\
aiperf-benchmarks/llama8b-c100" | jq
```

Or pull the per-job CSVs and combine client-side:

```bash
for C in 10 25 50 100; do
  curl -sSf \
    "http://localhost:8081/api/v1/results/aiperf-benchmarks/llama8b-c${C}/profile_export_aiperf.csv" \
    > "llama8b-c${C}.csv"
done
```

---

## Troubleshooting

### "I put `sweep:` in `spec.benchmark` but only one run executed"

Expected at HEAD. `build_benchmark_run()` in
[`src/aiperf/operator/spec_converter.py`][spec-conv] calls
`run_config.pop("sweep", None)` (and the same for `multi_run`) before
constructing the `BenchmarkRun`. Submit one `AIPerfJob` per variation
until multi-variation execution lands.

### "My output is missing `profile_export_aiperf_sweep.csv`"

That file is only produced by the local `MultiRunOrchestrator`. Per-job
Kubernetes runs produce `profile_export_aiperf.csv` and
`profile_export_aiperf.json` only. Combine them client-side (see the
recipe above) or use `/api/v1/analytics/compare`.

### "I only see one point's results on the PVC"

Check that the N `AIPerfJob`s actually applied — each must have a
**distinct** `metadata.name` (the templating loop above handles this).
Re-applying the same CR name updates the existing CR in place and does
**not** create a new job directory.

### "Sweep labels are missing on my pods"

`aiperf.nvidia.com/sweep-run`, `aiperf.nvidia.com/variation-index`, and
`aiperf.nvidia.com/run-index` are defined as label *keys* in
[`src/aiperf/kubernetes/constants.py`][constants] but no code path
currently sets them — they are scaffolding reserved for the future
multi-point-per-job implementation. Filter jobs by
`aiperf.nvidia.com/job-id` or `aiperf.nvidia.com/name` instead, or add
your own label via `spec.labels` / `--labels`.

### "`sweepExecution` is mentioned somewhere — should I use it?"

No. There is no top-level `spec.sweepExecution` in the CRD and it is not
in `_DEPLOYMENT_FIELDS` in
[`src/aiperf/kubernetes/validate.py`][validate]. If and when
multi-variation execution ships, the access path will remain
`spec.benchmark.sweep` (the shared config model).

### "Sweep runs slow" (when orchestrated as N jobs)

Each `AIPerfJob` spins up its own JobSet — controller, worker, timing-
manager, and records-manager pods all restart per variation. For N sweep
points, image pull and controller init add a fixed cost N times.
Mitigations:

- Pre-pull the image on all target nodes (DaemonSet with an init
  container referencing the image, or `imagePullPolicy: IfNotPresent`
  after a single priming run).
- Use the same `spec.scheduling.queueName` so Kueue admits jobs in
  sequence without re-scheduling onto cold nodes.
- Set `spec.ttlSecondsAfterFinished` high enough that completed jobs do
  not GC before you download results.
- Keep sweep variations coarse — N=4 is cheap, N=40 is a long serial
  workflow at today's JobSet-per-point granularity.

When sweeps are materialized inside a single `AIPerfJob`, pod reuse
across variations will reduce this overhead.

---

## Cross-reference

- [Generic sweep semantics (tutorial)](../tutorials/sweeps.md) — grid vs.
  scenarios, multi-run statistics, magic-list shorthand.
- [results-api.md](results-api.md) — per-job result download and the
  `/analytics/compare` endpoint used for client-side sweep aggregation.
- [kueue.md](kueue.md) — queue / priority controls for sibling jobs.
- [validate.md](validate.md) — CR validation rules and the authoritative
  `_DEPLOYMENT_FIELDS` allow-list.
