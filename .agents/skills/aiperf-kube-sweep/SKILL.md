---
name: aiperf-kube-sweep
description: Use when running a parameter sweep, multi-run confidence trial, or adaptive/Bayesian search on Kubernetes with aiperf kube sweep and the AIPerfSweep CR - authoring sweep parameter axes, addressing child variations, or collecting sweep aggregates.
---

# Sweeps on Kubernetes

`aiperf kube sweep` submits one `AIPerfSweep` CR. A dedicated sweep-controller
pod runs the orchestration loop and creates **one child `AIPerfJob` per
(variation, trial)**, sequentially. The kopf operator has its own
`AIPerfSweep` handlers (spec validation, RBAC, the controller JobSet, child
status roll-up) but stays planner-agnostic — the adaptive/BO loop lives only
in the sweep-controller pod.

**Related skills:** `aiperf-kube-run` (single-job lifecycle, shared flags),
`aiperf-kube-triage` (a child is stuck). Full grid/zip/scenario and adaptive-
search reference: `references/adaptive-search.md` (bundled with this skill).

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
kubectl get aiperfsweeps -n my-benchmarks
kubectl get aiperfjobs -l aiperf.nvidia.com/sweep=<sweep-name>
aiperf kube results <sweep-name>            # whole sweep
aiperf kube results <sweep-name> -v 7 -t 2  # one child
```

**`aiperf kube validate` accepts this file.** A bare Config-v2 document with
no `apiVersion`/`kind` is detected, checked against the `AIPerfSweep` contract
(inferred from the presence of `sweep:`), and passes with a warning naming the
contract it used. Only the wrapped-CR form additionally validates deployment
fields (`image`, `podTemplate`, worker counts, credential transport), so
validate the CR you would `kubectl apply` when those matter.

**`--dry-run` does not expand anything.** It renders the `AIPerfSweep` CR,
prints it as JSON, and returns before touching the cluster. It will not show you the variation
list, the child count, or whether the sweep is over the cardinality caps. To
see the variation count before you submit, expand it locally (the in-process
`aiperf profile` sweep path prints the plan) or read the caps below and do the
multiplication yourself.

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
  (`AIPERF_INJECTED_API_KEY`, `AIPERF_INJECTED_HEADERS`,
  `AIPERF_INJECTED_ENDPOINT_URLS`), sourced from a Secret you create in the
  benchmark namespace.
- **Never nest an in-process sweep inside an operator-managed run.** With
  `AIPERF_OPERATOR_MANAGED=1` set in a controller pod, any `plan.is_sweep` hard-
  fails on purpose. The sweep controller strips the parent `multiRun` block from
  each child so a child executes exactly one run instead of recursing.
- **The cardinality contract is one AIPerfJob and one controller pod per
  (variation, trial), and they run one at a time.** The sweep-controller awaits
  each child to reach a terminal phase before creating the next, so peak
  cluster demand is a single controller+worker set regardless of grid size.
  Size `--total-workers` for one child; a 32-way grid at 64 workers needs
  capacity for 64 workers, not 2048, and needs no Kueue serialization. What
  scales with the grid is wall-clock, not capacity.
- **The 200-variation cap is enforced only by the operator at admission; the
  10-trial cap is also a client-side schema bound.** `multiRun.numRuns` must be
  1..10, so `numRuns: 11` fails locally — `--dry-run` and `aiperf kube validate`
  both exit 1 with "Input should be less than or equal to 10". The
  200-variation cap raises `kopf.PermanentError` in the operator, so the CR is
  accepted by the apiserver and then goes `Failed` with the cap message in
  `status.error`, with no retry; `--dry-run` and `validate` both pass on an
  over-variation sweep. Count your grid before submitting: the product of every
  axis length must be <= 200.
- **The aggregate is inlined into CR status only up to ~600 KB**
  (`AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES`; the apiserver rejects
  patches over ~1 MiB with HTTP 413). Past that the sweep-controller drops
  `confidence` first, then omits `children` and adds a `childrenTruncated`
  marker if it is still over budget; the full document is served only from the
  results sidecar / PVC. Read big aggregates via `aiperf kube results`, never
  from `kubectl get -o json`.
- **Plotting runs once, in the sweep-controller, after cross-variation
  aggregation**, and only when `benchmark.artifacts.autoPlot` resolves true.
  With `benchmark.artifacts.plotRequired: true` a plot failure fails
  aggregation and withholds the ready marker (nothing is harvested); with
  `false` it is logged and the aggregate still becomes ready.
- **A partial sweep download fails `aiperf kube results` (exit 1)** even though
  the successfully downloaded children remain on disk.
- **CR object-map keys get alphabetized by the apiserver.** Anything order-
  sensitive in a sweep spec must be a list of named entries, not a mapping.

## Adaptive / Bayesian search

The sweep-controller instantiates the same Bayesian-optimization planner the
in-process path uses (`sweep.planner: bayesian`); the K8s executor creates one `AIPerfJob` per iteration, so
iterations are sequential by construction. Convergence flags
(`--convergence-metric`, `--convergence-threshold`, `--convergence-mode`,
`--convergence-stat`, `--confidence-level`, SLO flags) are accepted by
`aiperf kube sweep` exactly as by the local path. `--bo-constraint-mode` also
parses but is deprecated and has no effect on either path. See
`references/adaptive-search.md`.

## Cancelling and cleanup

```bash
aiperf kube cancel <sweep-name>            # whole sweep
aiperf kube cancel <sweep-name> -v 7       # one variation
aiperf kube cancel <name> --kind sweep     # when a job shares the name
aiperf kube delete <sweep-name> --kind sweep --force
```

Deleting the parent garbage-collects children through ownerReferences; results
already harvested to the operator PVC survive.
