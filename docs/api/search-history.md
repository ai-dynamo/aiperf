<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Search History API Reference

Schema reference for `search_history.json` — the on-disk trajectory log of an AIPerf adaptive Bayesian-Optimization (BO) run. The file is produced by [`src/aiperf/exporters/search_history.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/exporters/search_history.py) (`write_search_history`) and is rewritten in place after every BO iteration, so a partial trajectory survives a crash or cancellation. Each entry captures what the planner proposed, what the resulting benchmark measured, and (on terminal calls) why the loop stopped. For algorithm semantics see [Bayesian Optimization](../sweeping/bayesian-optimization.md).

## Overview

`search_history.json` is the canonical artifact for post-run BO audit and dashboarding. It complements (it does not replace) `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}`, which carries the post-hoc grouping of all iterations by `variation_values`. The trajectory log is unique in that it preserves iteration order and convergence-reason metadata.

Use it to:

- Recover the order in which the planner proposed configurations.
- Identify the best observed point(s) and how many iterations it took to find them. For multi-objective runs (`len(config.objectives) > 1`) `best_trials` is the Pareto front rather than a single argmax/argmin.
- Determine why the run terminated (budget exhaustion, no-improvement patience, plateau, or — for the Optuna terminator — posterior-regret bound).
- Reproduce the original search-space specification (including objectives and outcome constraints) for a follow-up run.

## File Location

The exporter writes to `<base_dir>/search_history.json` where `base_dir` is the controlling artifact directory. The companion sweep aggregate is under `<base_dir>/sweep_aggregate/` for single-trial and independent multi-run layouts, and under `<base_dir>/aggregate/sweep_aggregate/` for repeated multi-run layouts.

**In-process (`aiperf profile --search-space ...`):**

```text
artifacts/
  {benchmark_name}/
    search_history.json        # next to sweep_aggregate/, NOT inside it
    sweep_aggregate/
      profile_export_aiperf_sweep.json
      profile_export_aiperf_sweep.csv
```

---

## JSON Schema

### Top-Level Structure

```json
{
  "config": { ... },
  "iterations": [ ... ],
  "best_trials": [ ... ] | null,
  "boundary_summary": { ... } | null,
  "recipe": "max-concurrency-under-sla" | null,
  "convergence_reason": "max_iterations" | "improvement_patience" | "plateau_cv" | "posterior_regret_bound" | "emmr" | "unknown" | "smooth_isotonic_precision_reached" | "monotonic_precision_reached" | ... | null
}
```

**Top-Level Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config` | object | yes | Frozen subset of the BO configuration used to drive this run, including the full `objectives` list and any `outcome_constraints`. |
| `iterations` | array&lt;object&gt; | yes | Per-iteration trajectory entries, in the order the planner proposed them. May be empty on the first write. |
| `best_trials` | array&lt;object&gt; \| null | yes | Argmax/argmin (single-objective: `len(config.objectives) == 1`) or the Pareto front (multi-objective: `len(config.objectives) > 1`). `null` until at least one iteration has produced a usable objective. See [Interpreting `best_trials`](#interpreting-best_trials). |
| `boundary_summary` | object \| null | yes | Empirical SLA-feasibility boundary along the swept axis. `null` for multi-dim search spaces, and on empty history. See [`boundary_summary`](#boundary_summary). |
| `recipe` | string \| null | yes | Name of the search recipe (`AdaptiveSearchSweep.recipe_name`) that authored this configuration, e.g. `"max-concurrency-under-sla"`. `null` when the configuration was built ad-hoc rather than via a recipe. |
| `convergence_reason` | string \| null | yes | Why the loop stopped. `null` only on mid-loop writes or abnormal exit (cancellation, crash). A clean terminal exit always writes a non-null string — either the planner's own reason or the literal `"unknown"` fallback when the planner returned no reason. See [Convergence Reasons](#convergence-reasons). |

### `config` Section

A snapshot of the adaptive-search configuration fields that the v1 writer persists from `AdaptiveSearchSweep` (`src/aiperf/config/sweep/config.py`). It includes the planner name, objectives, outcome constraints, iteration budget, initial-point count, random seed, convergence knobs, search-space dimensions, and SLA filters. It does **not** serialize every planner knob yet (for example `optuna_sampler`, `optuna_acquisition`, `optuna_terminator`, `objective_pooling`, and smooth-isotonic replicate/warmup settings are omitted), so use it as an audit trail for the trajectory rather than a complete round-trip config. The optimization target is recorded as a list under `objectives` (length-1 for single-objective runs, length-N for Pareto BO); `outcome_constraints` is the parallel list of feasibility gates that BoTorch's acquisition masks against.

```json
{
  "config": {
    "planner": "optuna",
    "objectives": [
      {"metric": "output_token_throughput", "stat": "avg", "direction": "MAXIMIZE", "threshold": null},
      {"metric": "time_to_first_token", "stat": "p99", "direction": "MINIMIZE", "threshold": 250.0}
    ],
    "outcome_constraints": [
      {"metric": "request_error_rate", "op": "<=", "bound": 0.01}
    ],
    "max_iterations": 30,
    "n_initial_points": 5,
    "random_seed": 42,
    "improvement_patience": 10,
    "plateau_window": 8,
    "plateau_threshold": 0.01,
    "search_space": [
      {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
    ],
    "sla_filters": [
      {"metric_tag": "time_to_first_token", "stat": "p95", "op": "lt", "threshold": 200.0}
    ]
  }
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `planner` | string | yes | Search planner plugin name. One of `bayesian`, `optuna`, `monotonic_sla`, `smooth_isotonic`. Multi-objective is available through the curated `bayesian` planner or through `optuna` with `optuna_sampler=botorch` and a multi-objective acquisition such as `qlognehvi`. |
| `objectives` | array&lt;object&gt; | yes | Optimization targets. Length 1 = single-objective BO. Length > 1 = multi-objective Pareto BO. Min length 1. |
| `objectives[].metric` | string | yes | Metric tag (e.g. `"output_token_throughput"`). Matches a key in `RunResult.summary_metrics`. |
| `objectives[].stat` | string | yes | Statistic on the metric: one of `"avg"`, `"p50"`, `"p90"`, `"p95"`, `"p99"`. |
| `objectives[].direction` | string | yes | Either `"MAXIMIZE"` or `"MINIMIZE"` (uppercase; serialized from `OptimizationDirection.name`, locked by `tests/unit/exporters/test_search_history_multi_objective.py`). |
| `objectives[].threshold` | float \| null | yes | Pareto reference point for hypervolume computation (multi-objective only). Trials worse than this on this objective do not contribute to hypervolume. `null` = auto-derive from the worst observed value among Sobol initial points. Ignored for single-objective runs. |
| `outcome_constraints` | array&lt;object&gt; | yes | Feasibility gates on metrics the optimizer is **not** optimizing. Empty list = no constraints. Distinct from `objectives[].threshold` (Pareto reference point) and from `sla_filters` (post-hoc benchmark eligibility): outcome constraints down-weight infeasible candidates inside BoTorch's acquisition function. |
| `outcome_constraints[].metric` | string | yes | Metric tag to constrain. |
| `outcome_constraints[].op` | string | yes | Comparison operator. One of `"<="`, `">="`, `"=="`. **Distinct from `sla_filters[].op`**, which uses the lowercase mnemonics `lt`/`le`/`gt`/`ge` — outcome constraints feed BoTorch's acquisition mask, SLA filters feed post-hoc feasibility ranking. |
| `outcome_constraints[].bound` | float | yes | Threshold value. |
| `max_iterations` | int | yes | Iteration budget. The loop also stops earlier on convergence. |
| `n_initial_points` | int | yes | Sobol-random points before BoTorch fits the GP. Validator enforces `n_initial_points < max_iterations` for `bayesian` and `optuna` planners only; `monotonic_sla` / `smooth_isotonic` planners drive their own probe sequence and ignore this field. |
| `random_seed` | int \| null | yes | Reproducibility seed passed to the planner backend. `null` when the run was unseeded. |
| `improvement_patience` | int | yes | Stop after this many consecutive iterations with no improvement over the running best objective (single-objective) or no hypervolume gain (multi-objective). Drives the `"improvement_patience"` convergence reason. |
| `plateau_window` | int | yes | Number of recent iterations inspected for plateau detection. |
| `plateau_threshold` | float | yes | Coefficient-of-variation threshold (relative; scale-free) for the plateau test. Drives the `"plateau_cv"` convergence reason. |
| `search_space` | array&lt;object&gt; | yes | Original search-space spec, one entry per dimension. Min length 1. |
| `sla_filters` | array&lt;object&gt; | yes | Post-hoc feasibility gates applied to per-iteration verdicts. Empty list = no filters. Drives the `feasible` flag on `iterations[]` and `best_trials[]`, the 1D `boundary_summary` block, and feasibility-first lexicographic best-trial selection. Distinct from `outcome_constraints` (BO acquisition-mask gates fed into BoTorch). |
| `sla_filters[].metric_tag` | string | yes | Metric tag to filter on; matches a key in `RunResult.summary_metrics`. |
| `sla_filters[].stat` | string | yes | Statistic on the metric: one of `"avg"`, `"p50"`, `"p90"`, `"p95"`, `"p99"`. |
| `sla_filters[].op` | string | yes | Comparison operator. One of `"lt"`, `"le"`, `"gt"`, `"ge"` (lowercase mnemonics). **Distinct from `outcome_constraints[].op`**, which uses `"<="` / `">="` / `"=="`. |
| `sla_filters[].threshold` | float | yes | Numeric threshold the metric statistic is compared against. Finite (NaN/inf rejected at config time). |

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
      "objective_values": [8421.7],
      "feasible": true,
      "non_monotonic_warning": false
    },
    {
      "iteration_idx": 1,
      "variation_values": {"phases.profiling.concurrency": 256},
      "objective_values": [9512.3],
      "feasible": true,
      "non_monotonic_warning": false
    },
    {
      "iteration_idx": 2,
      "variation_values": {"phases.profiling.concurrency": 64},
      "objective_values": null,
      "feasible": true,
      "non_monotonic_warning": false
    }
  ]
}
```

A multi-objective iteration carries one entry per `config.objectives[i]`, in the same order:

```json
{
  "iteration_idx": 7,
  "variation_values": {"phases.profiling.concurrency": 256},
  "objective_values": [9512.3, 187.4],
  "feasible": true,
  "non_monotonic_warning": false
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `iteration_idx` | int | yes | Zero-based, dense iteration counter. Matches `SweepVariation.index` for the iteration. |
| `variation_values` | object | yes | Map of dotted path to proposed value (one entry per `search_space` dimension). Values are plain Python `int` or `float` per dimension `kind`. |
| `objective_values` | array&lt;float&gt; \| null | yes | One entry per `config.objectives[i]`, in the same order. Each entry is one aggregate value for the planner-proposed point: the mean of finite per-trial values for `objectives[i].metric`/`objectives[i].stat`, or the pooled percentile when percentile pooling is configured. The whole field is `null` (not a list of nulls) when every trial failed or any configured metric/stat was missing — in that case the planner internally tells BoTorch a fallback loss to keep ask/tell pairing consistent, but the fallback is NOT persisted here. For length-1 `objectives`, this is a length-1 list. |
| `feasible` | bool | yes | Whether at least one trial at this iteration satisfied every configured `sla_filters` entry. Computed by the planner's `tell()`. Defaults to `true` when no SLA filters are configured, so non-SLA runs degenerate to plain ranking unchanged. |
| `non_monotonic_warning` | bool | yes | `true` iff the verdict at this iteration violated the monotonicity assumption — a feasible point appeared at a swept value at-or-above the latched `infeasible_min`, or an infeasible point at-or-below `feasible_max`. Set only by `MonotonicSLASearchPlanner` and `SmoothIsotonicSLAPlanner`; always `false` for BO planners. |

> **Note:** `objective_values[i]` is one aggregate vector per search point/iteration: by default the mean of finite trial-level objective values, or the pooled percentile when percentile pooling is configured. The GP/Optuna planner observes that aggregate vector, not every per-trial value separately. The `SearchIteration.results` per-trial list held in memory by the planner is intentionally NOT serialized — read the per-trial `profile_export_aiperf.json` files under each iteration's variation directory if you need the spread.

### Interpreting `best_trials`

`best_trials` is the post-hoc winner set over iterations whose `objective_values` is non-null. The shape adapts to the number of objectives:

- **Single-objective (`len(config.objectives) == 1`).** `best_trials` is a length-1 list containing the global argmax (when `direction == "MAXIMIZE"`) or argmin (when `"MINIMIZE"`). Single-objective is treated as the length-1 special case of the multi-objective shape — there is no separate scalar-`best` block.
- **Multi-objective (`len(config.objectives) > 1`).** `best_trials` is the **Pareto front**: the set of iterations that are not dominated by any other iteration on every objective simultaneously. A trial *A* dominates *B* iff *A* is at least as good on every objective and strictly better on at least one. The front itself is unranked; if you want a tie-breaking order, sort by `pareto_rank` (always `0` for trials on the front) then by hypervolume contribution (not persisted here — recompute downstream if needed).

`best_trials` is `null` until at least one iteration has produced a usable objective. Readers MUST tolerate the `null` state during early-run reads (and any read where every scored iteration's `objective_values` is `None`).

```json
{
  "best_trials": [
    {
      "iteration_idx": 1,
      "objective_values": [9512.3],
      "variation_values": {"phases.profiling.concurrency": 256},
      "feasible": true,
      "feasible_count": 5,
      "pareto_rank": 0
    }
  ]
}
```

A multi-objective Pareto front:

```json
{
  "best_trials": [
    {
      "iteration_idx": 7,
      "objective_values": [9800.1, 215.4],
      "variation_values": {"phases.profiling.concurrency": 280},
      "feasible": true,
      "feasible_count": 18,
      "pareto_rank": 0
    },
    {
      "iteration_idx": 13,
      "objective_values": [9512.3, 187.4],
      "variation_values": {"phases.profiling.concurrency": 256},
      "feasible": true,
      "feasible_count": 18,
      "pareto_rank": 0
    },
    {
      "iteration_idx": 22,
      "objective_values": [8910.0, 162.7],
      "variation_values": {"phases.profiling.concurrency": 224},
      "feasible": true,
      "feasible_count": 18,
      "pareto_rank": 0
    }
  ]
}
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `iteration_idx` | int | yes | Index of the winning iteration. |
| `objective_values` | array&lt;float&gt; | yes | The objective tuple at the winner, one entry per `config.objectives[i]` (always non-null for entries in `best_trials`). For length-1 objectives, a length-1 list. |
| `variation_values` | object | yes | Proposed values that produced the winner. Same shape as `iterations[i].variation_values`. |
| `feasible` | bool | yes | Whether this iteration satisfied every configured `sla_filters` entry. Lexicographic feasibility-first selection means a feasible iteration is preferred over an infeasible one even if the latter has a better objective. |
| `feasible_count` | int | yes | Number of feasible iterations across the whole run among those with a non-null `objective_values` (an iteration with `feasible == true` but `objective_values is None` does NOT count). `0` flags the "no iteration was both feasible and scored — we fell back to ranking the full scored pool" case so the reader can distinguish it from the normal feasible-front case. |
| `pareto_rank` | int | yes | Always `0` in v1 — every entry of `best_trials` (single-objective argmax/argmin, or any member of the multi-objective non-dominated set) is emitted with `pareto_rank == 0`. Reserved for future non-dominated sorting (NSGA-II style) that would emit `0`, `1`, ... for successive fronts; until then, do not branch on this field. |

> **Caveat:** `best_trials` is "best of observed iterations," not "true Pareto front of the search space." Early termination (any `convergence_reason`) means the planner stopped before exhausting the budget; better trade-offs may exist outside the explored region.

### Convergence Reasons

`convergence_reason` takes one of the values below. The shared BO-set (everything except the `monotonic_*` and `smooth_isotonic_*` strings) is defined on `OptunaSearchPlanner.convergence_reason()` in [`src/aiperf/orchestrator/search_planner/optuna_planner.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/orchestrator/search_planner/optuna_planner.py); `BayesianSearchPlanner` inherits this implementation without override. The Optuna-terminator reasons (`posterior_regret_bound`, `emmr`) fire only when `--optuna-terminator` is set. The 1D-SLA planners (`MonotonicSLASearchPlanner`, `SmoothIsotonicSLAPlanner`) emit their own algorithm-specific strings — see the table below and the [Bayesian Optimization — 1D SLA saturation](../sweeping/bayesian-optimization.md) guide.

| Value | Meaning |
|-------|---------|
| `null` | Mid-loop write (run still in progress), OR terminated abnormally (cancelled, crashed, or aborted before the orchestrator's terminal `write_search_history` call). After a clean terminal exit this is never `null` — see `"unknown"`. |
| `"unknown"` | Clean terminal exit fallback: the orchestrator wrote `planner.convergence_reason() or "unknown"`, and the planner returned `None`. Indicates `planner.ask()` returned `None` but the planner did not record a structured reason. |
| `"max_iterations"` | Budget exhausted: the loop ran `config.max_iterations` iterations. Emitted by every planner family. |
| `"improvement_patience"` | No improvement-over-best for `improvement_patience` consecutive iterations (single-objective: improvement = better objective; multi-objective: improvement = positive hypervolume delta). BO planners only. |
| `"plateau_cv"` | Coefficient of variation (sample stddev / abs(mean)) on the last `plateau_window` iterations fell below `plateau_threshold`. Single-objective on the scalar objective; multi-objective on the hypervolume time series. BO planners only. |
| `"posterior_regret_bound"` | Optuna terminator: `RegretBoundEvaluator` (Makarova 2022) signalled that the high-probability bound on simple regret has fallen below the user-supplied threshold. Only fires under `--optuna-terminator regret`. |
| `"emmr"` | Optuna terminator: `EMMREvaluator` (Ishibashi 2023). Only fires under `--optuna-terminator emmr`. |
| `"monotonic_precision_reached"` | `MonotonicSLASearchPlanner`: bracket `(infeasible_min - feasible_max) / infeasible_min` fell below `SLA_PRECISION_DEFAULT`. |
| `"monotonic_no_pass_in_range"` | `MonotonicSLASearchPlanner`: even the lowest swept value violates SLA — no feasible point exists in the configured range. |
| `"monotonic_no_failure_in_range"` | `MonotonicSLASearchPlanner`: even the highest swept value satisfies SLA — no infeasible point exists in the configured range. |
| `"smooth_isotonic_precision_reached"` | `SmoothIsotonicSLAPlanner`: PCHIP-fitted boundary converged within the configured precision; `boundary_type == "smooth"`. |
| `"smooth_isotonic_cliff_precision_reached"` | `SmoothIsotonicSLAPlanner`: cliff guard tripped — PAVA residual exceeded `3·σ_local` AND bracket gap exceeded precision threshold. Planner emits an honest bracket; `boundary_type == "cliff"`. |
| `"smooth_isotonic_no_pass_in_range"` | `SmoothIsotonicSLAPlanner`: counterpart to `monotonic_no_pass_in_range`. |
| `"smooth_isotonic_no_failure_in_range"` | `SmoothIsotonicSLAPlanner`: counterpart to `monotonic_no_failure_in_range`. |
| `"smooth_isotonic_pchip_fallback_bisection"` | `SmoothIsotonicSLAPlanner`: PCHIP fit failed prerequisites (insufficient bracketing samples, monotonicity violations); planner fell back to monotonic bisection and reached its precision target there. |

The first signal to fire wins; later iterations are not run. See the BO guide's [convergence section](../sweeping/bayesian-optimization.md) for tuning advice and the [Bayesian Optimization — 1D SLA saturation](../sweeping/bayesian-optimization.md) guide for the SLA-planner termination semantics.

### `boundary_summary`

Top-level block. Emitted (non-null) when the search has exactly **one** dimension AND at least one iteration was recorded; `null` for multi-dim searches or empty history. Records the empirical feasibility boundary along the swept axis — most meaningful when at least one `SLAFilter` was configured (the [`max-concurrency-under-sla`](../sweeping/bayesian-optimization.md) recipe is the canonical user), but the exporter does NOT gate on filter presence: with no filters every iteration's `feasible` flag defaults to `true`, so `feasible_max` tracks the highest swept value and `infeasible_min` is `null`.

```json
{
  "boundary_summary": {
    "swept_dim_path": "phases.profiling.concurrency",
    "feasible_max": {"value": 256, "iteration_idx": 3, "objective_value": 4172.3},
    "infeasible_min": {
      "value": 320, "iteration_idx": 4,
      "first_breach": {
        "metric_tag": "time_to_first_token", "stat": "p95",
        "op": "lt", "threshold": 200.0, "observed": 213.4
      }
    },
    "boundary_type": "smooth",
    "binding_constraint": "time_to_first_token:p95",
    "boundary_ci": {"lo": 248.7, "hi": 264.2}
  }
}
```

**Base fields** (written by `MonotonicSLASearchPlanner`, `SmoothIsotonicSLAPlanner`, and the BO post-hoc derivation):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `swept_dim_path` | string | yes | Dotted path of the (single) swept dimension. Matches `config.search_space[0].path`. |
| `feasible_max` | object \| null | yes | Highest swept value observed to pass every SLA filter. `null` when no probe passed. |
| `feasible_max.value` | number | yes | The swept value (int when `kind=int`). |
| `feasible_max.iteration_idx` | int | yes | Index into `iterations[]` of the probe that observed this value. |
| `feasible_max.objective_value` | number \| null | yes | Objective at the same probe (when present), for context. |
| `infeasible_min` | object \| null | yes | Lowest swept value observed to violate at least one SLA filter. `null` when no probe failed. |
| `infeasible_min.value` | number | yes | The swept value. |
| `infeasible_min.iteration_idx` | int | yes | Index into `iterations[]` of the breaching probe. |
| `infeasible_min.first_breach` | object | yes | Identity of the SLA filter that triggered first at this point: `metric_tag`, `stat`, `op`, `threshold`, and the `observed` value. |

**Smooth-isotonic-only optional fields** (written by `SmoothIsotonicSLAPlanner` when applicable; absent — not `null` — when produced by other planners or when the relevant phase did not run):

| Field | Type | When present | Description |
|-------|------|--------------|-------------|
| `boundary_type` | `"smooth"` \| `"cliff"` | smooth_isotonic only | Cliff-guard verdict. `"smooth"` means the PAVA-residual at the most-recent probe was within `3·σ_local` and the planner is confident the curve is smooth at the boundary. `"cliff"` means the residual exceeded that threshold AND the bracket gap exceeded `precision · x_hi` — the planner is reporting an honest bracket `[feasible_max.value, infeasible_min.value]` instead of a single boundary point on a discontinuity. Catches the prefill-prioritizing-server pattern (Sarathi-Serve fig. 8). |
| `binding_constraint` | string | smooth_isotonic only, after at least one Phase-2 fit | The SLA filter key (`<metric_tag>:<stat>`) whose σ-normalized margin is tightest at termination — i.e. the constraint that defines the boundary in this run. When several SLAs are configured, only this one is replicated and CI'd in Phase 3, because it dominates the final boundary location. |
| `boundary_ci` | object | smooth_isotonic only, when Phase-3 replicates ran | Bootstrap CI on the binding margin at the candidate boundary `x*`, computed via `_replicate_budget.boundary_ci` over per-replicate margins. Object shape: `{"lo": float, "hi": float}`. When the CI brackets zero, the planner expands to nearby points and refits before terminating; a written CI that brackets zero therefore only appears when the planner exited via `--search-max-iterations`. |

For full algorithm context (when each phase runs, the cliff-detection threshold, how the binding constraint is selected) see [Bayesian Optimization — 1D SLA saturation (`smooth_isotonic`)](../sweeping/bayesian-optimization.md).

---

## Lifecycle and Consistency Guarantees

- **Rewritten after every iteration.** The orchestrator calls `write_search_history(...)` after each successful `tell()` AND once more on terminal exit (when `ask()` returns `None`). Readers MUST tolerate the partial state — the file is valid JSON at every observable instant only because each write is a single `Path.write_bytes(...)`.
- **NOT atomic.** The current writer issues one `Path.write_bytes` call without a temp-file-then-rename. Concurrent readers may observe a torn write (zero bytes, partial JSON) on a slow filesystem; in practice the payload is small (a few KB up to ~100 KB for a 200-iteration run) and the race window is short. Treat a parse failure as "retry in a moment," not as a corrupted run.
- **Iteration order is submission order.** `iterations[i].iteration_idx == i` (dense, zero-based). The planner-internal `_iter` counter increments on every `tell()`, regardless of trial success.
- **Final write carries `convergence_reason`.** All earlier (mid-loop) writes carry `convergence_reason: null`. After a clean terminal exit (i.e. `planner.ask()` returned `None`), the orchestrator rewrites the file with `planner.convergence_reason() or "unknown"` — so a clean terminal exit always lands a non-null string, even when the planner did not record a structured reason. `null` in a finalized-looking file therefore implies abnormal termination (cancellation, crash, or hard process kill).
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

# Pull the best observed configuration(s).
best_trials = history["best_trials"]  # list[dict] or None
objectives = history["config"]["objectives"]

if not best_trials:  # None or []
    print("No successful iteration yet")
elif len(objectives) == 1:
    # Single-objective: best_trials is length-1 by construction.
    best = best_trials[0]
    best_concurrency = best["variation_values"]["phases.profiling.concurrency"]
    best_throughput = best["objective_values"][0]
    print(f"Best: concurrency={best_concurrency} -> {best_throughput:.1f} tokens/s "
          f"(iter {best['iteration_idx']} of {len(history['iterations'])})")
else:
    # Multi-objective: best_trials is the Pareto front.
    print(f"Pareto front ({len(best_trials)} non-dominated trials):")
    for trial in best_trials:
        values = ", ".join(
            f"{obj['metric']}/{obj['stat']}={v:.2f}"
            for obj, v in zip(objectives, trial["objective_values"])
        )
        print(f"  iter={trial['iteration_idx']:>3}  {values}  "
              f"vars={trial['variation_values']}")
```

To compute summary statistics across the trajectory (e.g. learning curves), iterate `history["iterations"]` and skip entries where `objective_values is None`. For multi-objective hypervolume tracking, fold over `[ov[i] for ov in iter['objective_values'] if ov is not None]` paired with `config['objectives'][i].direction`.

---

## Caveats

- **Schema is not yet stable across versions.** v1 emits the subset above; future releases may add fields (e.g. per-iteration timestamps, GP posterior summaries, hypervolume time-series). Pin your `aiperf` version when building dashboards or downstream tooling against this artifact.
- **`objective_values[i]` is the arithmetic mean across trials.** It is the GP/Optuna planner's observed aggregate vector for the point: the mean of finite trial values by default, or a pooled percentile when percentile pooling is enabled. If you need per-trial spread, read the per-trial `profile_export_aiperf.json` files at `<base_dir>/search_iter_NNNN/profile_runs/run_NNNN/` — adaptive-search runs use a flat `search_iter_NNNN` per BO iteration (each holding `profile_runs/run_NNNN/` for that iteration's trials), distinct from grid sweeps' `{leaf}_{value}` layout. See [Sweep Aggregate API Reference](sweep-aggregates.md#artifact-directory-layout-reference) for the full layout table.
- **`convergence_reason: "plateau_cv"` can fire as early as iteration `plateau_window`.** When the random-Sobol initial points happen to land in a flat region of the (scalar or hypervolume) objective, the coefficient-of-variation test trips immediately. This is correct, not a bug — increase `plateau_window` or tighten `plateau_threshold` if the run terminates too eagerly.
- **`config.search_space` is the original spec, not what the planner sampled.** The planner may explore the dimension's range non-uniformly (Sobol initial points, then GP-driven exploitation). Use `iterations[i].variation_values` to see the actual samples; use `config.search_space` only to reproduce the original CLI/CRD invocation.
- **`best_trials` is orthogonal to `sweep_aggregate/`'s `best_configurations` and `pareto_optimal`.** Those belong to the `SweepAnalyzer` exporter, are computed across the whole `RunResult` set (including failed iterations), and may include points the BO planner never saw a finite objective for. Use `best_trials` for "what the BO loop converged on"; use `sweep_aggregate/profile_export_aiperf_sweep.json` for "what the post-hoc analyzer thinks is best across every cell."

---

## See Also

- [Bayesian Optimization](../sweeping/bayesian-optimization.md) — algorithm semantics, convergence tuning, objective definition.
- [Sweep Aggregate API Reference](sweep-aggregates.md) — the `sweep_aggregate/` companion artifact emitted alongside `search_history.json`.
- [Parameter Sweeping Tutorial](../tutorials/sweeps.md) — user guide for grid sweeps and adaptive search.
