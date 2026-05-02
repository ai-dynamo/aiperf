<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Bayesian-Optimization Outer Loop

`aiperf profile --search-space ... --search-metric ... --search-direction ... --search-max-iterations ...` runs an adaptive outer loop instead of a grid sweep. Each iteration the planner asks `skopt` for the next point in the search space, runs `--num-profile-runs` benchmarks at it, scores the configured objective, and feeds the result back to the optimizer.

## When to use it

Use BO when:
- The search space is too large to grid-enumerate (e.g. concurrency 1–1000 with no obvious step).
- You only care about finding the best point, not characterizing the whole frontier.
- A single scalar objective captures what you care about (throughput, p99 latency, etc.).

Use a grid sweep when:
- You need a complete Pareto frontier (use existing `--concurrency 10,20,50,100` magic-list flags + `sweep_aggregate/`).
- You want to compare specific points the team has agreed on.
- You need cluster-distributed sweeping (BO is in-process only in v1; cluster runs use `AIPerfSweep` CRDs).

## Quick start

```bash
aiperf profile \
    --models my-model \
    --endpoint http://infer.example.com \
    --search-space "phases.profiling.concurrency:1,1000:int" \
    --search-metric output_token_throughput \
    --search-direction maximize \
    --search-max-iterations 30 \
    --search-random-seed 42 \
    --num-profile-runs 3
```

This runs 30 search iterations × 3 trials each = 90 benchmarks. Output:
- `<artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/` — per-trial artifacts.
- `<artifact_dir>/search_history.json` — BO trajectory, written incrementally.
- `<artifact_dir>/sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` — same per-combination aggregate the grid path produces.

## Flag reference

| Flag | Required | Description |
|---|---|---|
| `--search-space PATH:LO,HI[:KIND]` | yes | Repeatable. Dotted-path into `BenchmarkConfig`. `KIND` is `int` or `real`, default `real`. |
| `--search-metric METRIC` | yes | Metric tag to optimize (e.g. `output_token_throughput`). The bare tag — NOT a flattened `_avg`/`_p99` aggregator-suffixed key. |
| `--search-stat STAT` | no | Statistic on the metric: `avg` / `p50` / `p90` / `p95` / `p99`. Default `avg`. |
| `--search-direction DIR` | yes | `maximize` (throughput, goodput) or `minimize` (latency, TTFT). |
| `--search-max-iterations N` | yes | Maximum search iterations. 2–200. |
| `--search-initial-points N` | no | Random Sobol points before GP fitting. Default 5. Must be < `--search-max-iterations`. |
| `--search-random-seed N` | no | Reproducibility seed for `skopt.Optimizer`. |

## Search space grammar

`PATH:LO,HI[:KIND]`

- `PATH` is a dotted path resolved by `_set_nested_value` (the same primitive grid sweeps use). For named-list segments like `phases.profiling.concurrency`, the segment matches against the `name` field. Typos error loudly with the available names listed.
- `LO` and `HI` are inclusive bounds parsed as floats. For `:int`, skopt rounds to integers.
- `:int` uses `skopt.space.Integer`, `:real` uses `skopt.space.Real`. Categorical dimensions are not supported in v1.

Multi-dim:

```bash
--search-space "phases.profiling.concurrency:1,500:int" \
--search-space "phases.profiling.request_rate:0.1,100.0:real"
```

## Objective semantics

The planner averages `--search-stat` across all successful trials at each search point. Failed trials are skipped; an iteration with zero successful trials gets a fallback "worst-seen" tell so skopt advances without crashing — a warning is logged.

`--search-metric` must match a key in `RunResult.summary_metrics` produced by the run — that is, the bare metric tag (`output_token_throughput`, `time_to_first_token`), not the flattened `_avg`/`_p99` aggregator-suffixed form.

## Convergence detection

The loop terminates when either:
1. `--search-max-iterations` iterations have been run.
2. The coefficient of variation (`stddev / |mean|`) of the last `plateau_window` (default 5) iterations' objective values falls below the plateau threshold (default 0.01 = 1% relative spread).

Plateau is **scale-free** — works for throughput (~1000) and latency (~50) without tuning. Convergence can fire as early as iteration `plateau_window` if the first 5 random-Sobol points happen to hit a flat region; this is correct behavior, not a bug.

## Mutual exclusion

`--search-*` is mutually exclusive with:
- Magic-list flags that produce sweeps (`--concurrency 10,20,30`).
- Explicit `sweep:` blocks in YAML.
- `--convergence-metric` (adaptive trial-level early stop). Reason: the trial-level convergence semantics are orthogonal to outer-loop convergence; they need separate thought before composition.
- The Kubernetes operator (`AIPERF_OPERATOR_MANAGED=1`). The operator's deterministic-cardinality contract is incompatible with adaptive variation choice. Submit cluster sweeps as `AIPerfSweep` CRDs (grid only) or run BO in-process.

## Output schema

`search_history.json`:

```json
{
  "config": {
    "algorithm": "bayes",
    "objective_metric": "output_token_throughput",
    "objective_stat": "avg",
    "objective_direction": "maximize",
    "max_iterations": 30,
    "search_space": [{"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}]
  },
  "iterations": [
    {"iteration_idx": 0, "variation_values": {"phases.profiling.concurrency": 503}, "objective_value": 1247.3},
    {"iteration_idx": 1, "variation_values": {"phases.profiling.concurrency": 178}, "objective_value": 942.1},
    "..."
  ],
  "best": {
    "iteration_idx": 23,
    "objective_value": 1822.5,
    "variation_values": {"phases.profiling.concurrency": 814}
  }
}
```

The file is rewritten after every iteration, so a crashed run still leaves the partial trajectory on disk.
