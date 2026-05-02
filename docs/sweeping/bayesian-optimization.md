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
- You need a complete characterization of every variation (BO converges on the best point and may stop early).

BO works in-process via `aiperf profile --search-*` AND under the operator via `AIPerfSweep` CRDs that include a `multi_run.adaptive_search` block. The controller pod owns the planner state; the kopf operator side is BO-agnostic. See [Adaptive search on Kubernetes](../kubernetes/sweeps.md#adaptive-search-bayesian-optimization) for the cluster-side wiring.

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
| `--search-stat STAT` | no | Statistic on the metric: `avg` / `p50` / `p90` / `p95` / `p99`. Default `avg`. See [Objective semantics](#objective-semantics) for the mean-vs-pooled trade-off when STAT is a percentile. |
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

`--search-metric` must match a key in `RunResult.summary_metrics` produced by the run — that is, the bare metric tag (`output_token_throughput`, `time_to_first_token`), not the flattened `_avg`/`_p99` aggregator-suffixed form.

**Per-trial observations to the GP.** When `--num-profile-runs` is N≥2, the planner tells skopt N observations at the same search point — one per successful trial — instead of pre-averaging to a single value. This lets the GP estimate the noise term (σ²ₙ) properly and shrink confidence at noisy regions of the search space. Following Letham et al. 2017, *Constrained Bayesian Optimization with Noisy Experiments* ([arXiv:1706.07094](https://arxiv.org/abs/1706.07094)). Pre-averaging would discard within-point variance the GP could have used.

The user-facing `objective_value` recorded in `search_history.json` per iteration is still the arithmetic mean across trials — the GP sees the per-trial spread; the trajectory log shows the summary.

**Failed trials.** Skipped when extracting the objective. An iteration with zero successful trials writes a fallback loss to skopt — `worst-seen-loss + max(10%, 1.0 absolute)` when prior successes exist, else a finite sentinel of `1e6` — and continues. A warning is logged.

**Mean of percentiles vs pooled percentiles.** When `--search-stat` is a percentile (`p50`/`p99`/...), the BO objective is the *expected per-trial percentile* (mean across trials), not the percentile of pooled samples across trials. These differ for skewed distributions: pooled-p99 over `N×requests` exposes more tail mass than `mean(per-trial p99)`. For BO finding the best config the choice doesn't change the optimum's location much; for SLO claims it does. Cite the bias/variance characterization in Nakayama 2014, *Confidence Intervals for Quantiles Using Sectioning* ([PDF](https://web.njit.edu/~marvin/papers/a19-nakayama.pdf)) and Glynn & Iglehart 1990 (DOI:[10.1287/moor.15.1.1](https://doi.org/10.1287/moor.15.1.1)) for the canonical results.

## Convergence detection

The loop terminates when any of:

1. `--search-max-iterations` iterations have been run.
2. **Improvement-over-best patience** (`improvement_patience`, default 10): no successful iteration has improved the running best for that many consecutive iterations. Idiom adopted from skopt's `HollowIterationsStopper` and Hyperopt's `no_progress_loss` — "we've stopped finding better points" is a stronger termination signal than "values stopped fluctuating."
3. **Coefficient-of-variation plateau** (`plateau_window`, `plateau_threshold`): on the last `plateau_window` (default 5) successful iterations, the sample CV (`stddev/|mean|`, Bessel's correction) falls below `plateau_threshold` (default 0.01 = 1% relative spread). Refused when `|mean|` is essentially zero — CV has no scale in that regime.

Whichever signal fires first wins; the reason is logged and recorded under `convergence_reason` in `search_history.json` so post-run audit can tell which terminated. `"max_iterations"`, `"improvement_patience"`, or `"plateau_cv"`.

Plateau detection is **scale-free** — works for throughput (~1000) and latency (~50) without tuning. Convergence can fire as early as iteration `plateau_window` if the first random-Sobol points happen to land in a flat region; this is correct behavior, not a bug.

## Mutual exclusion

`--search-*` is mutually exclusive with:
- Magic-list flags that produce sweeps (`--concurrency 10,20,30`).
- Explicit `sweep:` blocks in YAML.
- `--convergence-metric` (adaptive trial-level early stop). Reason: the trial-level convergence semantics are orthogonal to outer-loop convergence; their composition is undefined. Rejected at the v1→v2 boundary in `_converter_optionals.build_multi_run`.

`--search-*` is **not** mutually exclusive with the Kubernetes operator. Cluster-side adaptive search is supported via an `AIPerfSweep` CR with a `multi_run.adaptive_search` block — the controller pod instantiates the same `BayesianSearchPlanner` used in-process and drives the loop one `AIPerfJob` per iteration. See [Adaptive search on Kubernetes](../kubernetes/sweeps.md#adaptive-search-bayesian-optimization).

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
  },
  "convergence_reason": "improvement_patience"
}
```

`convergence_reason` is one of `"max_iterations"`, `"improvement_patience"`, `"plateau_cv"`, or `null` (still running / written mid-loop).

The file is rewritten after every iteration, so a crashed run still leaves the partial trajectory on disk.

## What this implementation isn't

The current planner is a single-objective skopt-backed BO with the conventional knobs. It is not — and we know it is not — the literature-state-of-the-art for noisy-objective HPO. Specific deferred upgrades, for context:

- **Noisy Expected Improvement (NEI)** as the acquisition function. Letham et al. 2017 ([arXiv:1706.07094](https://arxiv.org/abs/1706.07094)) show NEI is the principled EI generalization for noisy observations; we approximate it informally by feeding per-trial observations to skopt's stock EI. Proper NEI requires a custom acquisition that skopt's `Optimizer` doesn't expose; an Optuna or BoTorch backend would unlock it.
- **Posterior-regret stopping** (Wilson 2024, [arXiv:2402.16811](https://arxiv.org/abs/2402.16811)) — stops when the posterior probability that the incumbent is within ε of the optimum exceeds 1−δ. Requires Monte-Carlo over the GP posterior. Not exposed by skopt; would need BoTorch.
- **Multi-objective Pareto BO**. Single scalar only. Run two separate searches if you want a frontier.
- **Pooled-sample percentile aggregation**. We have summary stats per trial, not raw latency arrays; pooling would require extending `RunResult` to carry the underlying samples. Substantial cross-cutting refactor.
- **Heteroscedastic noise priors**. Implicit in the per-trial-observations approach (skopt's GP picks up the variance), but we don't pass explicit per-iteration noise estimates. Makarova et al. 2021, *Risk-averse Heteroscedastic Bayesian Optimization* ([arXiv:2111.03637](https://arxiv.org/abs/2111.03637)) is the relevant paper if/when this matters.
