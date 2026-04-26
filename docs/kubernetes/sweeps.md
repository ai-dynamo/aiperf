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
  main:
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

## Listing historical sweep runs

The sweep aggregate is served at the operator's `/api/v1/results/<ns>/<sweep>/aggregate/`
URL using the same epoch-keyed results layout as `AIPerfJob`:

```bash
aiperf kube results <sweep-name>
aiperf kube results <sweep-name> --run <epoch>
```
