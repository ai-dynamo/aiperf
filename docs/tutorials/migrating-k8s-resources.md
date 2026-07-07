<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migrating K8s Resources to the Unified Spec Shape

AIPerfJob and AIPerfSweep CRDs now share a single flat envelope spec shape and differ only by the `kind:` line and whether `spec.sweep` is set. This guide migrates CRs from the legacy shapes — embedded `benchmark.sweep` blocks, `template:`-wrapped AIPerfSweep specs, and `multi_run.adaptive_search` — onto the unified shape.

## Single benchmark — old vs. new

**Before** (any AIPerfJob with embedded sweep block — rejected):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
spec:
  benchmark:
    models: [llama]
    endpoint: {urls: [...]}
    sweep:                  # embedded; no longer permitted
      type: grid
      variables: {...}
```

**After** (move sweep to top, change kind):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep           # changed
spec:
  benchmark:
    models: [llama]
    endpoint: {urls: [...]}
  sweep:                    # moved out of benchmark
    type: grid
    variables: {...}
```

## AIPerfSweep — old vs. new

**Before** (template-wrapped):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
spec:
  sweep: {...}
  multi_run: {...}
  template:
    spec:
      benchmark: {...}
      image: ...
```

**After** (flat):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
spec:
  sweep: {...}
  multi_run: {...}
  benchmark: {...}
  image: ...
```

Shift every key under `template.spec.X` up to `spec.X` directly. The `template:` wrapper deletes.

## Adaptive search (BO) — old vs. new

**Before** (`adaptive_search` on `multi_run`):

```yaml
spec:
  multi_run:
    num_runs: 5
    adaptive_search:
      planner: bayesian
      metric: ttft_p99
      direction: minimize
```

**After** (`adaptive_search` is the third sweep type):

```yaml
spec:
  multi_run:
    num_runs: 5
  sweep:
    type: adaptive_search
    planner: bayesian
    objective:
      metric: ttft_p99
      direction: minimize
    search_space: [...]
    max_iterations: 30
```

## Status field semantic change: `AIPerfSweep.status.failedRuns`

`failedRuns` no longer counts cancelled children. If you have automation, dashboards, or watch loops that read `failedRuns` to detect "did anything not succeed," you must update them.

**Before** (legacy): `failedRuns` bucketed every non-success terminal child — `Failed`, `PartiallyFailed`, **and** `Cancelled` — into a single count.

**After** (unified): `failedRuns` only counts `Failed` and `PartiallyFailed` children. Cancelled children move to a new `runStates.cancelled` count. The full breakdown is exposed under `status.runStates`:

| `status.runStates` field | Source child phases (case-insensitive) |
|---|---|
| `pending` | `Pending`, missing, unknown |
| `running` | `Running`, `Profiling`, `Processing` |
| `completed` | `Succeeded`, `Completed` |
| `failed` | `Failed`, `PartiallyFailed` |
| `cancelled` | `Cancelled` |

The legacy `status.completedRuns` and `status.failedRuns` scalars are retained and now equal `runStates.completed` and `runStates.failed` respectively. Phase strings are matched case-insensitively against the bucketing table above, so both the canonical PascalCase form (`Cancelled`) and any lowercase variant the controller surfaces mid-write resolve to the same bucket.

### Updating consumer code

If you previously used `failedRuns` as a "did anything go wrong" signal, switch to summing the failed and cancelled buckets.

**`kubectl get` (jsonpath):**

```bash
# Before — single field, included cancelled.
kubectl get aiperfsweep my-sweep -o jsonpath='{.status.failedRuns}'

# After — read the two component fields separately.
kubectl get aiperfsweep my-sweep \
  -o jsonpath='failed={.status.runStates.failed} cancelled={.status.runStates.cancelled}{"\n"}'

# Or use jq for the arithmetic if you want a single number.
kubectl get aiperfsweep my-sweep -o json \
  | jq '.status.runStates | (.failed + .cancelled)'
```

**Python watch loop:**

```python
# Before
non_success = status.get("failedRuns", 0)

# After — pick the semantics you actually want:
run_states = status.get("runStates") or {}
hard_failures = run_states.get("failed", 0)                    # excludes cancellations
non_success = run_states.get("failed", 0) + run_states.get("cancelled", 0)
```

If your alerting cared about hard failures only (operator/job errors, not user cancellations), the new `runStates.failed` is strictly more accurate — a `kubectl delete aiperfjob` on a child will no longer trigger your alert.

## Migration tooling

There is no automated migration tool. Per the "zero backward compat" policy, regenerate CRs from your local `aiperf profile` config:

```bash
aiperf kube generate -f config.yaml > new-cr.yaml
```

The CLI emits the right `kind:` based on whether your local config has a sweep block, and writes the flat envelope.
