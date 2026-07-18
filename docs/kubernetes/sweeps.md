<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
sidebar-title: Parameter Sweeps and Multi-Run on Kubernetes

# Parameter Sweeps and Multi-Run on Kubernetes

`aiperf kube sweep` submits an `AIPerfSweep` CR that orchestrates parameter sweeps,
multi-run confidence trials, and adaptive convergence on a Kubernetes cluster.
The orchestration loop runs in a dedicated sweep-controller pod, not in the kopf
operator.

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
  variables:
    benchmark.phases.profiling.concurrency: [4, 8, 16, 32]
```

Submit:

```bash
aiperf kube sweep --config sweep.yaml --image aiperf:latest --workers-max 64
```

> Note: dot-path traversal into `phases.<name>.<field>` is not currently
> supported by `_set_nested_value` after the `phases: list[PhaseConfig]`
> refactor. Sweep top-level config fields (e.g., `random_seed`) for v1.
> Phase-internal sweeping is tracked as a follow-up.

## What gets created

1. One `AIPerfSweep` CR (the parent — `kubectl get aiperfsweeps`).
2. One `JobSet` for the sweep-controller pod (which orchestrates the loop).
3. One `AIPerfJob` CR per `(variation, trial)` pair, deterministically named
   `<sweep>-v<idx:02d>[-t<trial:01d>]`. List them with
   `kubectl get aiperfjobs -l aiperf.nvidia.com/sweep=<sweep-name>`.
4. Per-child results under `<base>/<ns>/<sweep>-v07-t2/<child-epoch>/`.
5. Sweep aggregate under `<base>/<ns>/<sweep>/<sweep-epoch>/aggregate/`.

## Mode: independent vs repeated

`spec.sweep.iterationOrder` (or `--parameter-sweep-mode` for in-process runs)
selects the iteration order of variations and trials. Both produce the
same total runs and the same `sweep_aggregate/` output — only the artifact
path layout and submit order differ.

| Mode (default `repeated`) | Iteration order | Artifact tree |
|---|---|---|
| `repeated`    | trials outer, variations inner | `<base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_NNNN/` |
| `independent` | variations outer, trials inner | `<base>/<variation>/profile_runs/run_NNNN/` |

`repeated` (the default, matching `origin/main` / PR #699) is the right
choice when absolute wall-clock timing across variations matters and you
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
multi_run:
  num_runs: 5
  cooldown_seconds: 30
```

## Adaptive convergence (composes with sweep)

```yaml
multi_run:
  num_runs: 10
  cooldown_seconds: 30
  convergence:
    metric: ttft_p99
    min_runs: 3
    threshold: 0.05
    mode: ci_width
```

When `multi_run.convergence` is set, trials early-stop once the criterion fires
(or `multi_run.num_runs` is hit, whichever comes first); the `num_runs` field
remains the hard ceiling. `sweep` and `multi_run.convergence` compose freely:
each cell of a grid sweep runs adaptive trials.

## Adaptive search (Bayesian Optimization)

When the search space is too large to grid-enumerate (e.g. concurrency
1–1000) and a single scalar objective captures what you care about, use
`sweep.type: adaptive_search` instead of `sweep.type: grid`. The sweep-controller
pod instantiates the same `BayesianSearchPlanner` used in-process by
`aiperf profile --search-*` and drives the loop one `AIPerfJob` per
iteration; the kopf operator stays unaware of BO and just sees ordinary
child create/delete events. `status.totalVariations` and
`status.maxTotalRuns` become upper bounds (early plateau convergence
may terminate sooner), and the BO trajectory is logged incrementally to
`search_history.json` so partial runs survive a crash or cancellation.

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

## Resume after sweep-controller crash

The sweep-controller pod is JobSet-managed: a crash restarts it, and idempotency
based on deterministic child names + `ownerReferences` means the orchestration
loop resumes from the first non-existent child without re-running terminal ones.

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
relied on `failedRuns` as a "did anything not succeed" signal, see
Status field semantic change
in the migration tutorial for the consumer-side update.

## Schema invariants enforced at admission

The apiserver rejects malformed `AIPerfSweep` CRs before any pod is scheduled:

- `spec.sweep` is required on `AIPerfSweep` (use `AIPerfJob` for a
  single benchmark).
- `spec.multiRun.convergence.minRuns ≤ multiRun.numRuns` — the convergence
  criterion's minimum trials cannot exceed the multi-run hard ceiling.
- `spec.benchmark.sweep` and `.multiRun` are forbidden — the
  orchestration axes belong on the AIPerfSweep top level, not on the
  per-child stamp.
- `spec.sweep` and `spec.multiRun` are immutable after creation
  (mutating them would corrupt the run-epoch ledger).

Full rule catalog: [CRD Validation Rules](crd-validation.md).

## Listing historical sweep runs

The sweep aggregate is served at the operator's `/api/v1/results/<ns>/<sweep>/aggregate/`
URL using the same epoch-keyed results layout as `AIPerfJob`:

```bash
aiperf kube results <sweep-name>
aiperf kube results <sweep-name> --run <epoch>
```
