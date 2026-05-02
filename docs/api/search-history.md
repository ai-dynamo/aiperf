<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Search History API Reference

Schema reference for `search_history.json` — the on-disk trajectory log of an AIPerf adaptive Bayesian-Optimization (BO) run. The file is produced by [`src/aiperf/exporters/search_history.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/exporters/search_history.py) (`write_search_history`) and is rewritten in place after every BO iteration, so a partial trajectory survives a crash or cancellation. Each entry captures what the planner proposed, what the resulting benchmark measured, and (on terminal calls) why the loop stopped. For algorithm semantics see [Bayesian Optimization](../sweeping/bayesian-optimization.md); for cluster-side BO see [Adaptive Search on Kubernetes](../kubernetes/adaptive-search.md).

## Overview

`search_history.json` is the canonical artifact for post-run BO audit and dashboarding. It complements (it does not replace) `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}`, which carries the post-hoc grouping of all iterations by `variation_values`. The trajectory log is unique in that it preserves iteration order and convergence-reason metadata.

Use it to:

- Recover the order in which the planner proposed configurations.
- Identify the best observed point and how many iterations it took to find.
- Determine why the run terminated (budget exhaustion, no-improvement patience, or plateau).
- Reproduce the original search-space specification for a follow-up run.

## File Location

The exporter writes to `<base_dir>/search_history.json` where `base_dir` is the controlling artifact directory.

**In-process (`aiperf profile --search-space ...`):**

```text
artifacts/
  {benchmark_name}/
    search_history.json        # next to sweep_aggregate/, NOT inside it
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

**Cluster (`AIPerfSweep` CR with `multi_run.adaptive_search`):**

```text
<RESULTS.DIR>/
  <namespace>/
    <sweep-name>/
      <sweep-epoch>/
        search_history.json
        aggregate/
          profile_export_aiperf_sweep.json
          profile_export_aiperf_sweep.csv
```

The cluster path layout matches [`docs/kubernetes/sweeps.md`](../kubernetes/sweeps.md).

---

## JSON Schema

### Top-Level Structure

```json
{
  "config": { ... },
  "iterations": [ ... ],
  "best": { ... } | null,
  "convergence_reason": "max_iterations" | "improvement_patience" | "plateau_cv" | null
}
```

**Top-Level Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config` | object | yes | Frozen subset of the `AdaptiveSearchConfig` used to drive this run. |
| `iterations` | array&lt;object&gt; | yes | Per-iteration trajectory entries, in the order the planner proposed them. May be empty on the first write. |
| `best` | object \| null | yes | Argmax (MAXIMIZE) or argmin (MINIMIZE) over iterations whose `objective_value` is non-null. `null` until at least one iteration has produced a usable objective. |
| `convergence_reason` | string \| null | yes | Why the loop stopped, or `null` while the run is mid-flight or terminated abnormally. See [Convergence Reasons](#convergence-reasons). |

### `config` Section

A snapshot of the BO configuration sufficient to reproduce or audit the run. The writer records every field of `AdaptiveSearchConfig` that influences planner behavior — `random_seed`, `n_initial_points`, `improvement_patience`, `plateau_window`, and `plateau_threshold` are all serialized so the trajectory is fully reproducible from the file alone.

```json
{
  "config": {
    "algorithm": "bayes",
    "objective_metric": "output_token_throughput",
    "objective_stat": "avg",
    "objective_direction": "maximize",
    "max_iterations": 30,
    "n_initial_points": 5,
    "random_seed": 42,
    "improvement_patience": 10,
    "plateau_window": 5,
    "plateau_threshold": 0.01,
    "search_space": [
      {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
    ]
  }
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | yes | Search algorithm. v1 only emits `"bayes"`. |
| `objective_metric` | string | yes | Metric tag being optimized (e.g. `"output_token_throughput"`). Matches a key in `RunResult.summary_metrics`. |
| `objective_stat` | string | yes | Statistic on the metric: one of `"avg"`, `"p50"`, `"p90"`, `"p95"`, `"p99"`. |
| `objective_direction` | string | yes | Either `"maximize"` or `"minimize"` (lowercase, from `OptimizationDirection`). |
| `max_iterations` | int | yes | Iteration budget. The loop also stops earlier on convergence. |
| `n_initial_points` | int | yes | Sobol-random points before skopt fits the GP. Always `< max_iterations`. |
| `random_seed` | int \| null | yes | `random_state` passed to `skopt.Optimizer` for reproducibility. `null` when the run was unseeded. |
| `improvement_patience` | int | yes | Stop after this many consecutive iterations with no improvement over the running best objective. Drives the `"improvement_patience"` convergence reason. |
| `plateau_window` | int | yes | Number of recent iterations inspected for plateau detection. |
| `plateau_threshold` | float | yes | Coefficient-of-variation threshold (relative; scale-free) for the plateau test. Drives the `"plateau_cv"` convergence reason. |
| `search_space` | array&lt;object&gt; | yes | Original search-space spec, one entry per dimension. Min length 1. |

#### `search_space` Element Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Dotted path into `BenchmarkConfig` (e.g. `"phases.profiling.concurrency"`). |
| `lo` | float | yes | Inclusive lower bound. |
| `hi` | float | yes | Inclusive upper bound. Always `> lo`. |
| `kind` | string | yes | Either `"int"` (integer-valued; suggestions are coerced via `int()`) or `"real"` (float). |

### `iterations` Section

One entry per BO iteration, in submission order. `iteration_idx` is dense and zero-based. Mid-run writes leave the array open-ended; readers must tolerate any non-negative length, including zero.

```json
{
  "iterations": [
    {
      "iteration_idx": 0,
      "variation_values": {"phases.profiling.concurrency": 142},
      "objective_value": 8421.7
    },
    {
      "iteration_idx": 1,
      "variation_values": {"phases.profiling.concurrency": 256},
      "objective_value": 9512.3
    },
    {
      "iteration_idx": 2,
      "variation_values": {"phases.profiling.concurrency": 64},
      "objective_value": null
    }
  ]
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `iteration_idx` | int | yes | Zero-based, dense iteration counter. Matches `SweepVariation.index` for the iteration. |
| `variation_values` | object | yes | Map of dotted path to proposed value (one entry per `search_space` dimension). Values are plain Python `int` or `float` per dimension `kind`. |
| `objective_value` | float \| null | yes | Arithmetic mean of `objective_metric.objective_stat` across the iteration's trials. `null` when every trial failed or the configured metric/stat was missing — in that case the planner internally tells skopt a fallback loss to keep the optimizer's ask/tell pairing consistent, but the fallback is NOT persisted here. |

> **Note:** `objective_value` is the arithmetic mean across trials. The GP itself observes per-trial values (see the [Objective Semantics](../sweeping/bayesian-optimization.md) section of the BO guide); the trajectory log records the summary. The `SearchIteration.results` per-trial list held in memory by the planner is intentionally NOT serialized — read the per-trial `profile_export_aiperf.json` files under each iteration's variation directory if you need the spread.

### `best` Section

The argmax (when `objective_direction == "maximize"`) or argmin (when `"minimize"`) over iterations with a non-null `objective_value`. `null` until at least one iteration has produced a usable objective.

```json
{
  "best": {
    "iteration_idx": 1,
    "objective_value": 9512.3,
    "variation_values": {"phases.profiling.concurrency": 256}
  }
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `iteration_idx` | int | yes | Index of the winning iteration. |
| `objective_value` | float | yes | The best observed value (always non-null when `best` itself is non-null). |
| `variation_values` | object | yes | Proposed values that produced the best objective. Same shape as `iterations[i].variation_values`. |

> **Caveat:** `best` is "best of observed iterations," not "true argmax of the search space." Early termination (any `convergence_reason`) means the planner stopped before exhausting the budget; a higher (or lower) point may exist outside the explored region.

### Convergence Reasons

`convergence_reason` takes one of four values. The full set is defined by `BayesianSearchPlanner.convergence_reason()` in [`src/aiperf/orchestrator/search_planner/bayesian.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/orchestrator/search_planner/bayesian.py).

| Value | Meaning |
|-------|---------|
| `null` | Run is still in progress (mid-loop write), OR terminated abnormally (cancelled, crashed, or aborted cell — the orchestrator only records a non-null reason after `planner.ask()` returns `None`). |
| `"max_iterations"` | Budget exhausted: the loop ran `config.max_iterations` iterations. |
| `"improvement_patience"` | No improvement-over-best for `improvement_patience` consecutive iterations (skopt's `HollowIterationsStopper` / Hyperopt's `no_progress_loss` idiom). |
| `"plateau_cv"` | Coefficient of variation (sample stddev / abs(mean)) on the last `plateau_window` iterations fell below `plateau_threshold`. |

The first signal to fire wins; later iterations are not run. See the BO guide's [convergence section](../sweeping/bayesian-optimization.md) for tuning advice.

---

## Lifecycle and Consistency Guarantees

- **Rewritten after every iteration.** The orchestrator calls `write_search_history(...)` after each successful `tell()` AND once more on terminal exit (when `ask()` returns `None`). Readers MUST tolerate the partial state — the file is valid JSON at every observable instant only because each write is a single `Path.write_bytes(...)`.
- **NOT atomic.** The current writer issues one `Path.write_bytes` call without a temp-file-then-rename. Concurrent readers may observe a torn write (zero bytes, partial JSON) on a slow filesystem; in practice the payload is small (a few KB up to ~100 KB for a 200-iteration run) and the race window is short. Treat a parse failure as "retry in a moment," not as a corrupted run.
- **Iteration order is submission order.** `iterations[i].iteration_idx == i` (dense, zero-based). The planner-internal `_iter` counter increments on every `tell()`, regardless of trial success.
- **Final write carries `convergence_reason`.** All earlier writes carry `convergence_reason: null`. After a clean terminal exit, the file is rewritten one last time with the reason populated.
- **Crash semantics.** On controller-pod restart, cancellation, or a hard process kill, the last entry in `iterations` is the most recently-completed iteration, and `convergence_reason` will be `null`. The BO loop does NOT resume from the file in v1 — a restarted run begins with iteration 0.

---

## Programmatic Consumption

```python
from pathlib import Path

import orjson

artifact_dir = Path("artifacts/my_benchmark")
history = orjson.loads((artifact_dir / "search_history.json").read_bytes())

# Detect run state.
if history["convergence_reason"] is None:
    if history["iterations"]:
        last = history["iterations"][-1]
        print(f"Run in progress; last completed iter={last['iteration_idx']}")
    else:
        print("Run started but no iterations have completed yet")
else:
    print(f"Run terminated: {history['convergence_reason']}")

# Pull the best observed configuration.
if history["best"] is None:
    print("No successful iteration yet")
else:
    best = history["best"]
    best_concurrency = best["variation_values"]["phases.profiling.concurrency"]
    best_throughput = best["objective_value"]
    print(f"Best: concurrency={best_concurrency} -> {best_throughput:.1f} tokens/s "
          f"(iter {best['iteration_idx']} of {len(history['iterations'])})")
```

To compute the mean and stddev across the trajectory (e.g. to plot a learning curve), iterate `history["iterations"]` and skip entries where `objective_value is None`.

---

## Caveats

- **Schema is not yet stable across versions.** v1 emits the subset above; future releases may add fields (e.g. per-iteration timestamps, GP posterior summaries). Pin your `aiperf` version when building dashboards or downstream tooling against this artifact.
- **`objective_value` is the arithmetic mean across trials.** It is not the GP's observed loss (which sees per-trial values directly), and it is not a percentile of the pooled per-trial samples. If you need per-trial spread, read the per-trial `profile_export_aiperf.json` files at `<base_dir>/<variation>/profile_runs/trial_NNNN/` (in-process) or under each child `AIPerfJob`'s artifact path (cluster).
- **`convergence_reason: "plateau_cv"` can fire as early as iteration `plateau_window`.** When the random-Sobol initial points happen to land in a flat region of the objective, the coefficient-of-variation test trips immediately. This is correct, not a bug — increase `plateau_window` or tighten `plateau_threshold` if the run terminates too eagerly.
- **`config.search_space` is the original spec, not what skopt sampled.** Skopt's `Optimizer` may explore the dimension's range non-uniformly (Sobol initial points, then GP-driven exploitation). Use `iterations[i].variation_values` to see the actual samples; use `config.search_space` only to reproduce the original CLI/CRD invocation.

---

## See Also

- [Bayesian Optimization](../sweeping/bayesian-optimization.md) — algorithm semantics, convergence tuning, objective definition.
- [Adaptive Search on Kubernetes](../kubernetes/adaptive-search.md) — cluster-side BO, `AIPerfSweep` CR, child-job lifecycle.
- [Sweep Aggregate API Reference](sweep-aggregates.md) — the `sweep_aggregate/` companion artifact emitted alongside `search_history.json`.
- [Parameter Sweeping Tutorial](../tutorials/parameter-sweeping.md) — user guide for grid sweeps and adaptive search.
