---
name: aiperf-kube-sweep
description: Use when running a parameter sweep, multi-run confidence trial, or adaptive/Bayesian search on Kubernetes with aiperf kube sweep and the AIPerfSweep CR - authoring sweep parameter axes, addressing child variations, or collecting sweep aggregates.
---

# Sweeps on Kubernetes

`aiperf kube sweep` submits one `AIPerfSweep` CR. A dedicated sweep-controller
pod runs the orchestration loop and creates **one child `AIPerfJob` per
(variation, trial)** — the kopf operator itself stays sweep-agnostic.

**Related skills:** `aiperf-kube-run` (single-job lifecycle, shared flags),
`aiperf-kube-triage` (a child is stuck). Reference:
`docs/tutorials/sweeps.md#running-sweeps-on-kubernetes`,
`docs/kubernetes/adaptive-search.md`.

## Quick start

```yaml
# sweep.yaml
benchmark:
  models: [Qwen/Qwen3-0.6B]
  endpoint:
    urls: [http://server:8000/v1/chat/completions]
    type: chat
    streaming: true
  datasets:
    - name: main
      type: synthetic
  phases:
    - name: profiling
      type: poisson
      duration: 120
      concurrency: 8
      rate: 10
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [4, 8, 16, 32]
```

```bash
aiperf kube sweep --config sweep.yaml --image aiperf:latest --dry-run   # preview
aiperf kube sweep --config sweep.yaml --image aiperf:latest --total-workers 64 --detach
kubectl get aiperfsweeps -n aiperf-benchmarks
kubectl get aiperfjobs -l aiperf.nvidia.com/sweep=<sweep-name>
aiperf kube results <sweep-name>            # whole sweep
aiperf kube results <sweep-name> -v 7 -t 2  # one child
```

**`aiperf kube validate` does not accept this file.** `validate` dispatches on a
`kind:` field and only handles CR-format `AIPerfJob` / `AIPerfSweep` manifests; a
Config-v2 `benchmark:`/`sweep:` document has no `kind` and fails with
`kind: expected one of ['AIPerfJob', 'AIPerfSweep'], got 'None'`. Preview
Config-v2 sweep input with `--dry-run` instead, and reserve `validate` for
manifests you would `kubectl apply`.

## Parameter path rules

- Dotted paths are rooted **inside** the `benchmark:` block:
  `phases.profiling.concurrency`. A redundant `benchmark.` prefix is rejected
  by the validator.
- Bare names (`concurrency`, `rate`, `requests`, `duration`, ...) are sugar for
  `phases.profiling.<name>`.
- `variables.<name>` is the one envelope-level escape; it rewrites the top-level
  Jinja `variables:` block per variation.

## What gets created

| Object | Naming / location |
|---|---|
| Parent CR | `kubectl get aiperfsweeps` |
| Sweep-controller JobSet | one per sweep |
| Child `AIPerfJob` per (variation, trial) | `<sweep>-v<idx:02d>[-t<trial:01d>]`, labeled `aiperf.nvidia.com/sweep=<sweep>` |
| Child results | `<base>/<ns>/<sweep>-v07-t2/<child-epoch>/` |
| Sweep aggregate | `<base>/<ns>/sweeps/<sweep>/<sweep-epoch>/` — `aggregate.json`, `children.json`, `sweep_aggregate/` |

Children address by index everywhere: `results`, `logs`, and `debug` all take
`--variation` (0..199) and `--trial` (0..9); `--trial` requires `--variation`.
On `debug`, `--variation` must be spelled long — `-v` is `--verbose` there.

## Rules that bite

- **Credential-bearing axes are rejected before anything is created.** API keys,
  tokens, passwords, credential-bearing URLs, every `endpoint.headers.*` axis,
  and credential-like `variables.*` names are refused for grid, zip, scenario,
  adaptive-search, Sobol, and Latin Hypercube sweeps. Keep credentials fixed and
  inject them via the Secret-backed endpoint credential env vars
  (`docs/kubernetes/rbac-security.md`).
- **Never nest an in-process sweep inside an operator-managed run.** With
  `AIPERF_OPERATOR_MANAGED=1` set in a controller pod, any `plan.is_sweep` hard-
  fails on purpose. The sweep controller strips the parent `multiRun` block from
  each child so a child executes exactly one run instead of recursing.
- **The cardinality contract is one AIPerfJob and one controller pod per
  variation.** Sizing `--total-workers` is per child, not per sweep — a 32-way
  grid at 64 workers wants capacity for 32 concurrent controller+worker sets, or
  Kueue admission control to serialize them.
- **The aggregate is inlined into CR status only up to ~600 KB**
  (`AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES`; the apiserver rejects
  patches over ~1 MiB with HTTP 413). Past that the sweep-controller drops
  `confidence` from status and the full document is served only from the
  results sidecar / PVC. Read big aggregates via `aiperf kube results`, never
  from `kubectl get -o json`.
- **Plotting runs once, in the sweep-controller, after cross-variation
  aggregation.** With `plotRequired: true` a plot failure fails aggregation and
  withholds the ready marker (nothing is harvested); with `false` it is logged
  and the aggregate still becomes ready.
- **A partial sweep download fails `aiperf kube results` (exit 1)** even though
  the successfully downloaded children remain on disk.
- **CR object-map keys get alphabetized by the apiserver.** Anything order-
  sensitive in a sweep spec must be a list of named entries, not a mapping.

## Adaptive / Bayesian search

The sweep-controller instantiates the same `BayesianSearchPlanner` plugin the
in-process path uses; the K8s executor creates one `AIPerfJob` per iteration, so
iterations are sequential by construction. Convergence flags
(`--convergence-metric`, `--convergence-threshold`, `--convergence-stat`,
`--confidence-level`, `--bo-constraint-mode`, SLO flags) are accepted by
`aiperf kube sweep` exactly as by the local path. See
`docs/kubernetes/adaptive-search.md`.

## Cancelling and cleanup

```bash
aiperf kube cancel <sweep-name>            # whole sweep
aiperf kube cancel <sweep-name> -v 7       # one variation
aiperf kube cancel <name> --kind sweep     # when a job shares the name
aiperf kube delete <sweep-name> --kind sweep --force
```

Deleting the parent garbage-collects children through ownerReferences; results
already harvested to the operator PVC survive.
