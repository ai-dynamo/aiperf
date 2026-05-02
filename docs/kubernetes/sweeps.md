<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
sidebar-title: Parameter Sweeps and Multi-Run on Kubernetes

# Parameter Sweeps and Multi-Run on Kubernetes

`aiperf kube sweep` submits an `AIPerfSweep` CR that orchestrates parameter sweeps,
multi-run confidence trials, and adaptive convergence on a Kubernetes cluster.
The orchestration loop runs in a dedicated sweep-controller pod, not in the kopf
operator — see `docs/superpowers/specs/2026-04-25-k8s-sweeps-design.md` for the
design rationale.

## Quick start: grid sweep over concurrency × rate

`sweep.yaml`:

```yaml
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
    random_seed: [1, 2, 3, 4]
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
   `<sweep>-v{idx:04d}[-t{trial:02d}]`. List them with
   `kubectl get aiperfjobs -l aiperf.nvidia.com/sweep=<sweep-name>`.
4. Per-child results under `<base>/<ns>/<sweep>-v0007-t02/<child-epoch>/`.
5. Sweep aggregate under `<base>/<ns>/<sweep>/<sweep-epoch>/aggregate/`.

## Mode: independent vs repeated

`spec.multiRun.mode` (or `--parameter-sweep-mode` for in-process runs)
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

Adaptive convergence (`spec.convergence` / `--convergence-metric`) is
incompatible with `repeated` and rejected at submit time — convergence
needs to evaluate per-cell stability, which has no place to land in
trial-outer iteration. Use `independent` for adaptive sweeps.

## Multi-run confidence

```yaml
multi_run:
  trials: 5
  cooldown_seconds: 30
```

## Adaptive convergence (composes with sweep)

```yaml
multi_run:
  cooldown_seconds: 30
convergence:
  metric: ttft_p99
  min_runs: 3
  max_runs: 10
  threshold: 0.05
```

When `convergence` is set, `multi_run.trials` must be unset; `convergence.maxRuns`
governs the per-cell trial cap. `sweep` and `convergence` compose freely: each
cell of a grid sweep runs adaptive trials.

## Adaptive search (Bayesian Optimization)

For large or continuous search spaces where grid enumeration is infeasible,
set `multi_run.adaptive_search` instead of `spec.sweep`. The controller pod
instantiates a `BayesianSearchPlanner` (the same `skopt`-backed planner used
in-process by `aiperf profile --search-*`) and drives the loop one
`AIPerfJob` per iteration. `multi_run.trials` continues to govern trials per
proposed point.

Use adaptive search when:

- The search space is too large to grid-enumerate (e.g., concurrency 1–1000).
- You only care about finding the best point, not characterizing the entire frontier.
- A single scalar objective captures what you care about.

Use grid sweeps (`spec.sweep`) when:

- You need a complete Pareto frontier across pre-agreed points.
- You want exact reproducibility of which variations were run.

### Example AIPerfSweep CR

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
metadata:
  name: bo-concurrency-search
spec:
  multiRun:
    trials: 3
    cooldownSeconds: 30
    adaptiveSearch:
      algorithm: bayes
      searchSpace:
        - path: phases.profiling.concurrency
          lo: 1
          hi: 1000
          kind: int
      objectiveMetric: output_token_throughput
      objectiveStat: avg
      objectiveDirection: maximize
      maxIterations: 30
      initialPoints: 5
      randomSeed: 42
  template:
    spec:
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
```

`multi_run.adaptive_search` is mutually exclusive with `spec.sweep` (grid
expansion); the controller pod hard-fails at submit time if both are set.

### Status fields are upper bounds

`status.totalVariations` and `status.maxTotalRuns` are **upper bounds** under
adaptive search — early plateau convergence (see
[Convergence detection](../sweeping/bayesian-optimization.md#convergence-detection))
may terminate the loop before `maxIterations` is reached, in which case the
actual run count will be lower than the projected `totalVariations`. This
mirrors the role `convergence.maxRuns` already plays for trial-level
convergence.

The controller pod writes its own terminal-phase status; the operator's
`Aggregating` rollup gate already accommodates short-circuited runs via the
JSON-patch test-op guard, so an early-stopped adaptive sweep aggregates over
the actually-completed children.

### Output

- `<base>/<ns>/<sweep>-i{iter:04d}[-t{trial:02d}]/<child-epoch>/` — per-iteration child artifacts.
- `<base>/<ns>/<sweep>/<sweep-epoch>/search_history.json` — incremental BO trajectory (rewritten after every iteration; partial trajectory survives a crash).
- `<base>/<ns>/<sweep>/<sweep-epoch>/aggregate/` — same `sweep_aggregate/` schema produced by grid sweeps; `aggregate_sweep_and_export` groups by stamped `variation_values` and is mode-agnostic.

For the in-process equivalent and full flag/grammar reference, see
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

## Schema invariants enforced at admission

The apiserver rejects malformed `AIPerfSweep` CRs before any pod is scheduled:

- `spec` requires at least one of `sweep`, `multiRun`, `convergence` (use
  `AIPerfJob` for a single benchmark).
- `spec.convergence` requires `spec.multiRun` (for cooldown/seed/warmup) and
  must leave `multiRun.trials` unset (`convergence.maxRuns` governs the
  per-cell cap).
- `spec.convergence.minRuns ≤ maxRuns`.
- `spec.template.spec.benchmark.sweep` and `.multiRun` are forbidden — the
  orchestration axes belong on the AIPerfSweep top level, not on the
  per-child stamp.
- `spec.sweep`, `spec.multiRun`, `spec.convergence` are immutable after
  creation (mutating them would corrupt the run-epoch ledger).

Full rule catalog: [CRD Validation Rules](crd-validation.md).

## Listing historical sweep runs

The sweep aggregate is served at the operator's `/api/v1/results/<ns>/<sweep>/aggregate/`
URL using the same epoch-keyed results layout as `AIPerfJob`:

```bash
aiperf kube results <sweep-name>
aiperf kube results <sweep-name> --run <epoch>
```
