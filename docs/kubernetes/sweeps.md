<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
sidebar-title: Parameter Sweeps and Multi-Run on Kubernetes

# Parameter Sweeps and Multi-Run on Kubernetes

`aiperf kube sweep` submits an `AIPerfSweep` CR that orchestrates parameter sweeps,
multi-run confidence trials, and adaptive convergence on a Kubernetes cluster.
The orchestration loop runs in a dedicated sweep-controller pod, not in the kopf
operator.

The local and Kubernetes paths share the same Config-v2 loader,
`build_benchmark_plan`, `MultiRunOrchestrator`, and search-planner factory. The
Kubernetes adapter preserves Jinja template leaves in the submitted CR, renders
them per variation from the retained raw envelope, attaches the operator's
`failurePolicy`, and executes each resulting `BenchmarkRun` through
`K8sChildJobExecutor`. Operator-owned child tracking, status rollup, aggregate
harvest, and result indexing remain outside the canonical planner.

## Quick start: grid sweep over concurrency × rate

`sweep.yaml`:

```yaml
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

Submit:

```bash
aiperf kube sweep --config sweep.yaml --image aiperf:latest --total-workers 64
```

Preview the submitted resource with `--dry-run`. The JSON includes the resolved
`metadata.namespace` (from `--namespace`, or `aiperf-benchmarks` by default), so
redirecting it to a file and applying it targets the same namespace as a live
submission.

Benchmark CLI flags override the matching Config-v2 YAML fields before the
`AIPerfSweep` is submitted. When the input is an `AIPerfJob` manifest, kube
deployment flags merge over its deployment settings; unrelated nested
`podTemplate` fields are preserved.

> Parameter keys are dotted paths rooted **inside** the `benchmark:` block,
> so `phases.profiling.concurrency` is correct and the redundant
> `benchmark.` prefix is rejected by the validator. A handful of bare names
> (`concurrency`, `rate`, `requests`, `duration`, ...) are sugar for
> `phases.profiling.<name>`. The one envelope-level escape is
> `variables.<name>`, which rewrites the top-level Jinja `variables:` block
> per variation.

### Credentials must remain fixed

Kubernetes sweeps reject credential-bearing parameter axes before the
sweep-controller or any child AIPerfJob is created. This includes API keys,
tokens, passwords, credential-bearing URLs, every `endpoint.headers.*` axis,
and credential-like `variables.*` names. The rule applies to grid, zip,
scenario, adaptive-search, Sobol, and Latin Hypercube sweeps.

Keep credentials fixed across every variation and inject them through the
Secret-backed endpoint credential environment variables described in
[RBAC and Security](./rbac-security.md). Safe endpoint URL axes remain
supported when their values contain no userinfo or sensitive query parameters.
Display redaction is a defense-in-depth boundary for legacy data, not a way to
make secret sweep parameters safe.

## What gets created

1. One `AIPerfSweep` CR (the parent — `kubectl get aiperfsweeps`).
2. One `JobSet` for the sweep-controller pod (which orchestrates the loop).
3. One `AIPerfJob` CR per `(variation, trial)` pair, deterministically named
   `<sweep>-v<idx:02d>[-t<trial:01d>]`. List them with
   `kubectl get aiperfjobs -l aiperf.nvidia.com/sweep=<sweep-name>`.
4. Per-child results under `<base>/<ns>/<sweep>-v07-t2/<child-epoch>/`.
5. Sweep aggregate under `<base>/<ns>/sweeps/<sweep>/<sweep-epoch>/`
   (`aggregate.json`, `children.json`, and a `sweep_aggregate/` directory).

The sweep-controller initially writes child `sweep.json` backlinks on its
ephemeral results volume. During aggregate commit, the operator recreates those
backlinks on its PVC from the canonical `children.json` manifest before it
publishes the aggregate or deletes the sweep-controller JobSet. Archived child
lookups therefore continue to resolve their parent, variation, trial, and both
run epochs after the child and parent CRs have been removed.

The sweep controller owns the `(variation, trial)` expansion. It omits the
parent `multiRun` block from each child AIPerfJob, so every child executes
exactly one benchmark run instead of recursively repeating all trials.

## Auto-plotting the parent aggregate

The Config-v2 `plot:` field has the same meaning for `AIPerfSweep` as it does
for an in-process sweep. Plotting runs once in the sweep-controller after
cross-variation aggregation, not once in every child AIPerfJob. Generated
plots and the materialized `.aiperf-plot-config.yaml` receipt are stored under
`<base>/<ns>/sweeps/<sweep>/<sweep-epoch>/` and are harvested to the operator
PVC with the other parent artifacts.

The sweep aggregate ready marker is written only after plotting returns. An
optional plot failure (`benchmark.artifacts.plotRequired: false`) is logged and
the aggregate still becomes ready. A required plot failure
(`plotRequired: true`) moves aggregation to failed and withholds the ready
marker, preventing the operator from harvesting a partial completion. An
explicit `benchmark.artifacts.autoPlot: false` suppresses plotting even when a
`plot:` envelope is present.

## Mode: independent vs repeated

`spec.sweep.iterationOrder` (or `--parameter-sweep-mode` for in-process runs)
selects the iteration order of variations and trials. Both produce the
same total runs and the same `sweep_aggregate/` output — only the artifact
path layout and submit order differ.

| Mode (default `repeated`) | Iteration order | Artifact tree |
|---|---|---|
| `repeated`    | trials outer, variations inner | `<base>/profile_runs/trial_NNNN/<variation>/` |
| `independent` | variations outer, trials inner | `<base>/<variation>/profile_runs/trial_NNNN/` |

`repeated` (the default) is the right choice when absolute wall-clock timing across variations matters and you
want to interleave variations within each trial — e.g. "run the whole
sweep, then run it again". `independent` groups all trials of a single
configuration in close temporal proximity, which gives tight confidence
intervals per variation.

Adaptive convergence (`spec.multiRun.convergence` / `--convergence-metric`) is
incompatible with `repeated` and rejected at submit time — convergence
needs to evaluate per-cell stability, which has no place to land in
trial-outer iteration. Use `independent` for adaptive sweeps.

## Multi-run confidence

```yaml
multiRun:
  numRuns: 5
  cooldownSeconds: 30
```

A parameter axis is optional. When `multiRun.numRuns > 1` is present without
`sweep:`, `aiperf kube sweep` and `aiperf kube generate --operator` create an
`AIPerfSweep` with one no-op `base` scenario. The canonical plan therefore has
one unchanged configuration and five trials; it never collapses the request to
one `AIPerfJob`. A config with no `sweep:` and the single-run default
`multiRun.numRuns: 1` still generates an `AIPerfJob`.
Hand-authored `AIPerfJob` resources must follow the same boundary: setting
`multiRun.numRuns > 1` or `multiRun.convergence` is rejected at admission so
the requested trials cannot be silently collapsed to one run.

## Adaptive convergence (composes with sweep)

```yaml
multiRun:
  numRuns: 10
  cooldownSeconds: 30
  convergence:
    metric: time_to_first_token
    stat: p99
    minRuns: 3
    threshold: 0.05
    mode: ci_width
```

When `multiRun.convergence` is set, trials early-stop once the criterion fires
(or `multiRun.numRuns` is hit, whichever comes first); the `numRuns` field
remains the hard ceiling. `sweep` and `multiRun.convergence` compose freely:
each cell of a grid sweep runs adaptive trials.

For convergence without a parameter axis, the generated one-cell scenario uses
independent iteration order so the controller can evaluate the completed cell
after each trial and stop before the hard ceiling.

## Adaptive search (Bayesian Optimization)

When the search space is too large to grid-enumerate (e.g. concurrency
1–1000) and a single scalar objective captures what you care about, use
`sweep.type: adaptive_search` instead of `sweep.type: grid`. The sweep-controller
pod instantiates the same `BayesianSearchPlanner` used in-process by
`aiperf profile --search-*` and drives the loop one `AIPerfJob` per
iteration; the kopf operator stays unaware of BO and just sees ordinary
child create/delete events. The completed `search_history.json` is archived
inside the parent sweep's run-epoch directory so histories from separate
sweeps cannot overwrite one another. `status.totalVariations` and
`status.maxTotalRuns` become upper bounds (early plateau convergence
may terminate sooner), and the BO trajectory is logged incrementally to
`search_history.json` so partial runs survive a crash or cancellation.
Sobol and Latin Hypercube sweeps similarly archive `sampling_design.json`
under the parent epoch's `sweep_aggregate/` directory; separate sweep runs
therefore retain separate design audits on the shared operator PVC. Before
the results-sidecar marks the bundle ready, the sweep controller removes the
canonical orchestrator's temporary top-level run and aggregate directories.
The operator harvests only `<namespace>/sweeps/<name>/<epoch>/`, so unrelated
sweeps cannot overwrite shared-PVC root paths.

See [Adaptive search on Kubernetes](./adaptive-search.md) for the full
walkthrough — CR examples, status semantics, mutual-exclusion rules,
operator-managed gate exception, cancellation behaviour, and on-disk
layout. For the algorithm reference, see
[Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md).

## Cancel

```bash
kubectl patch aiperfsweep <name> --type merge -p '{"spec":{"cancel":true}}'
```

The sweep-controller pod observes `spec.cancel`, propagates it to the current
child via `spec.cancel: true`, and after that child reaches a terminal phase,
skips remaining variations and runs aggregation over completed children only.
Child labels are discovery hints only: cancellation is authorized by the exact
parent owner reference, sweep UID and run epoch, then applied with an immutable
child-UID precondition so a same-named replacement cannot be cancelled.
If the controller restarts after cancellation was already recorded, it performs
a read-only recovery of terminal children before aggregation. Recovery requires
the exact parent name, UID, and sweep-run epoch, does not create new children,
and does not wait for non-terminal children. This preserves completed, failed,
and cancelled child outcomes even though the restarted controller has a fresh
`emptyDir`.

## Resume after sweep-controller crash

The sweep-controller pod is JobSet-managed: a crash restarts it, and idempotency
uses deterministic child names, `ownerReferences`, and a reserved hash of each
generated child execution contract. Terminal reads and result rollups recheck
the immutable child UID, parent owner identity, and sweep run epoch before a
child is credited. Grid sweeps therefore resume from the first non-existent
child without re-running terminal ones. For Sobol and Latin
hypercube sweeps, Kubernetes derives a stable sampling seed from the parent CR
UID when `sweep.seed` is omitted; an explicit seed is preserved. Adaptive
searches similarly derive a stable planner seed when `sweep.randomSeed` is
omitted, then replay terminal child metrics through `ask()` / `tell()` in order.
The first new proposal is made only after that planner history is reconstructed;
an owned child whose contract hash is missing or different is rejected rather
than silently reused. Once the epoch bundle has its ready marker, a controller
restart validates the raw parent aggregate and republishes terminal status
directly; it does not replay the planner or recreate temporary run artifacts.
If the operator has already published its PVC-backed result reference, a late
controller restart preserves that reference instead of replacing it with the
ephemeral sidecar address.
The controller's live status patch keeps `resultsAvailable: false`; the
operator flips it to true only after every advertised file is durable on the
results PVC. Parent TTL cleanup waits for that durable reference, including
when `ttlSecondsAfterFinished` is zero.
Sweep result retention does not depend on that parent CR remaining present.
The durable `aggregate.json` stores the sweep's `resultsTtlDays`, and an
operator background reconciliation removes expired epoch directories and
their index rows, then repoints or removes `latest.txt`. Archives without an
explicit value use the operator-level `AIPERF_RESULTS_TTL_DAYS` default.
Local execution keeps its ordinary unseeded behavior.

If the sweep-controller exhausts the JobSet retry budget before it can publish
its own terminal status, the operator marks the owning `AIPerfSweep` Failed
from the JobSet event. The fallback writes the same failure timestamps,
`aggregation.phase`, and `resultsAvailable: false` shape as the controller's
normal failure writer. Both the JobSet owner reference and the parent UID must
match, and the status update is resource-version fenced, so a delayed event
cannot fail a recreated same-named sweep or overwrite an already-terminal one.

## Failure policy

```yaml
failurePolicy:
  onChildFailure: continue       # default
  maxFailures: 3                 # 0 = unbounded (default)
```

A failed child becomes a `failedRuns` count entry on the parent and the sweep
advances to the next variation. Set `onChildFailure: abort` for stricter behavior.

`status.failedRuns` counts only `Failed` and `PartiallyFailed` children;
cancelled children are tallied separately under `status.runStates.cancelled`.
The full per-phase breakdown lives at `status.runStates`
(`pending` / `running` / `completed` / `failed` / `cancelled`). If you previously
relied on `failedRuns` as a "did anything not succeed" signal, read
`status.runStates` instead — `failedRuns` no longer includes cancelled
children.

## Schema invariants

The apiserver rejects these before any pod is scheduled (CEL rules on the
`AIPerfSweep` CRD):

- `spec.sweep` is required on `AIPerfSweep` (use `AIPerfJob` for a
  single benchmark).
- `spec.sweep` and `spec.multiRun` are immutable after creation
  (mutating them would corrupt the run-epoch ledger).

These are enforced by Pydantic instead — client-side by
`aiperf kube validate`, server-side by the operator on reconcile
(`status.phase=Failed`). They cannot move to CEL because `spec.benchmark`
is a preserve-unknown node:

- `multiRun.convergence.minRuns <= multiRun.numRuns` — the convergence
  criterion's minimum trials cannot exceed the multi-run hard ceiling.
- `spec.benchmark.sweep` / `spec.benchmark.multiRun` are rejected as
  unknown fields — the orchestration axes belong at the CR top level, not
  on the per-child benchmark stamp.

Full rule catalog: [CRD Validation Rules](crd-validation.md).

## Listing historical sweep runs

The sweep aggregate is served by the operator's results server under
`/api/v1/sweeps/<ns>/<sweep>/epochs/<epoch>/artifacts` (per-child results
stay on the `AIPerfJob` route, `/api/v1/results/<ns>/<sweep>-v00-t1/`):

```bash
aiperf kube results <sweep-name>
aiperf kube results <sweep-name> --run <epoch>
```
