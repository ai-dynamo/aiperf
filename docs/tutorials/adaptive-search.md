<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Search: Finding the Best Concurrency Without a Grid

This tutorial walks through using AIPerf's adaptive Bayesian-Optimization (BO) outer loop to find the concurrency that maximizes goodput on a real vLLM deployment, without enumerating a grid of points by hand.

For the full flag reference, search-space grammar, output schema, and the noise-handling theory, see [`docs/sweeping/bayesian-optimization.md`](../sweeping/bayesian-optimization.md). This page is the narrative companion: one scenario, one command, and how to read what comes back.

## The scenario

You are benchmarking a `meta-llama/Llama-3.1-8B-Instruct` deployment behind vLLM at `http://vllm.internal:8000`. You have already profiled at a fixed `--concurrency 64` and noticed that output token throughput plateaus somewhere past that — but you don't know where. You also don't want to pay for `--concurrency 8,16,32,64,128,256,512,1024` followed by a "now sweep around the best one" second pass.

Your goal is operational: find the single concurrency value that maximizes `output_token_throughput` on this deployment, in the bounded range `[1, 1000]`, treating the inference server as a black box. You do not need a Pareto frontier; you need a number you can put into the production deployment manifest.

## The first run

```bash
aiperf profile \
    --models meta-llama/Llama-3.1-8B-Instruct \
    --endpoint http://vllm.internal:8000 \
    --search-space "phases.profiling.concurrency:1,1000:int" \
    --search-metric output_token_throughput \
    --search-direction maximize \
    --search-max-iterations 25 \
    --search-initial-points 5 \
    --search-random-seed 42 \
    --num-profile-runs 3 \
    --warmup-request-count 50
```

Flag-by-flag for this scenario (general semantics live in the [BO reference](../sweeping/bayesian-optimization.md#flag-reference)):

- `--search-space "phases.profiling.concurrency:1,1000:int"` — the dotted path is the same one a YAML grid sweep would use; `:int` makes skopt round to integers so we never propose `concurrency=472.6`.
- `--search-max-iterations 25` — upper bound on outer iterations. Convergence may stop earlier (improvement-patience or plateau-CV; see [Convergence detection](../sweeping/bayesian-optimization.md#convergence-detection)).
- `--search-initial-points 5` — the first 5 iterations are random Sobol draws (no GP yet); iterations 6–25 are GP-driven. With a one-dimensional search 5 is plenty; raise it for higher-dimensional spaces.
- `--search-random-seed 42` — same seed, same trajectory. Drop it for production search; keep it while you are tuning the *recipe* itself.
- `--num-profile-runs 3` — three benchmarks per proposed point. The planner feeds all three observations to the GP, not the average — see [Objective semantics](../sweeping/bayesian-optimization.md#objective-semantics).
- `--warmup-request-count 50` — 50 warmup requests before each timed run, so cold-cache effects don't poison early observations the GP is fitting on.

The total timed work here is `25 iterations × 3 trials = 75` benchmarks (capped — the loop may exit earlier on improvement-patience).

You did not specify `--search-stat`, so the converter defaults it to `avg`. You did not specify a goodput SLO yet — see [Common follow-ups](#common-follow-ups) below for the percentile-objective variant.

## What you'll see during the run

The orchestrator logs an opening line on entry to the adaptive loop, then one log line per iteration as it proposes a point, then a single termination line on exit. Roughly:

- Startup, from `execute_adaptive_search`: `Starting adaptive outer-loop benchmark (bayes, max_iterations=25, trials per point=3)`.
- Per iteration, before the cell runs: `[BO iter <N>] proposing {'phases.profiling.concurrency': <value>}`.
- Per iteration, after the trials: the standard per-run profile-export logs from each of the 3 trials — same output you would see from a non-adaptive `aiperf profile`.
- On exit (whichever convergence signal fired): `Adaptive outer loop terminated after <N> iterations (reason=<convergence_reason>)`. The reason string is one of `max_iterations`, `improvement_patience`, `plateau_cv`, or — only in the cancelled-mid-run case — `unknown`.

The first 5 iterations sample the space coarsely (Sobol). After that the GP starts steering toward the high-throughput region; do not be alarmed if iterations 6–10 cluster within a narrow concurrency band.

## Reading the artifacts

The artifact tree under `<artifact_dir>/` has three things to look at, in roughly this order:

### `search_history.json` — the trajectory

This is the file that tells you *what BO actually did*. It is rewritten after every iteration, so even a crashed or cancelled run leaves the partial trajectory:

```json
{
  "config": {
    "algorithm": "bayes",
    "objective_metric": "output_token_throughput",
    "objective_stat": "avg",
    "objective_direction": "maximize",
    "max_iterations": 25,
    "search_space": [
      {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
    ]
  },
  "iterations": [
    {"iteration_idx": 0, "variation_values": {"phases.profiling.concurrency": 503}, "objective_value": 1247.3},
    {"iteration_idx": 1, "variation_values": {"phases.profiling.concurrency": 178}, "objective_value":  942.1},
    {"iteration_idx": 2, "variation_values": {"phases.profiling.concurrency": 814}, "objective_value": 1801.4},
    {"iteration_idx": 3, "variation_values": {"phases.profiling.concurrency":  62}, "objective_value":  611.7},
    {"iteration_idx": 4, "variation_values": {"phases.profiling.concurrency": 327}, "objective_value": 1455.0}
  ],
  "best": {
    "iteration_idx":  2,
    "objective_value": 1801.4,
    "variation_values": {"phases.profiling.concurrency": 814}
  },
  "convergence_reason": "improvement_patience"
}
```

The `objective_value` per iteration is the arithmetic mean of `output_token_throughput` across the 3 trials at that point. That is the value displayed in the trajectory; the GP itself sees the per-trial spread, not just the mean.

To parse this in Python:

```python
import orjson

with open("search_history.json", "rb") as fp:
    history = orjson.loads(fp.read())

best = history["best"]
print(f"argmax concurrency: {best['variation_values']['phases.profiling.concurrency']}")
print(f"objective at best:  {best['objective_value']:.1f} tok/s")
print(f"converged by:        {history['convergence_reason']}")
```

Schema reference: [Output schema](../sweeping/bayesian-optimization.md#output-schema).

### `search_iter_NNNN/profile_runs/run_NNNN/profile_export_aiperf.json` — per-trial detail

Each iteration writes its 3 trials under `search_iter_NNNN/profile_runs/run_NNNN/`. These are *the same per-run JSONs that a normal `aiperf profile` produces* — you can open `profile_export_aiperf.json` for any single trial and inspect the full metric table, percentiles, error counts, etc.

You will want these when an iteration looks like a noise spike (a clearly out-of-trend point) and you want to confirm whether one of the three trials had elevated errors or a tail-latency event. The GP feeds on all 3 observations precisely so a single noisy trial does not pin the surrogate's belief.

### `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` — combination summary

The same per-combination aggregate the grid sweep path emits. One row per `(concurrency)` value visited, with the four sections (per-combination / best / pareto / metadata). Read this when you want a tabular CSV of "what concurrency values did BO actually visit, and what was the throughput at each." The `best` section is consistent with `search_history.json["best"]`, modulo CSV formatting.

## Interpreting "best"

`history["best"]` is the iteration whose `objective_value` was highest *among iterations actually run*. Two caveats worth being honest about:

1. The loop may have early-stopped on `improvement_patience` (no improvement over the running best for 10 consecutive iterations) or `plateau_cv` (objective values plateaued in CV terms). When that happens, the true argmax could be at an unvisited point — but improvement-patience as a stopping rule is calibrated against the same idea behind skopt's `HollowIterationsStopper` and Hyperopt's `no_progress_loss`: when you have stopped finding better points, further iterations have diminishing returns. See [Convergence detection](../sweeping/bayesian-optimization.md#convergence-detection).
2. `objective_value` is a noisy estimate of `output_token_throughput` at the proposed point, with `--num-profile-runs` samples behind it. The GP knows this and shrinks confidence in noisy regions; you should treat `best.objective_value` as a point estimate, not a guarantee of a future production throughput.

A practical sanity check: re-run with `--search-random-seed` *unset* (or a different seed). If the chosen `concurrency` is consistent within +/- 10% across seeds, the optimum is robust. If seeds disagree wildly, your search space is probably too wide or your objective is too noisy for `--num-profile-runs 3` — bump it to 5.

## When to use a grid sweep instead

| Use BO when... | Use a grid sweep when... |
|---|---|
| You want one optimal point and don't care about the shape between points. | You want a complete characterization of the frontier (Pareto). |
| The search space is too large to enumerate (concurrency 1–1000, no obvious step). | The team has agreed on specific points to compare. |
| A single scalar objective captures what you care about. | You need every combination's results for a downstream report. |
| Early-stop is acceptable (and desirable). | You need every variation to actually run. |

If you want both — find the best, then characterize around it — run BO first to get the argmax, then run a tight grid sweep over the neighborhood. See [`docs/tutorials/sweeps.md`](sweeps.md) for the grid path and [`docs/sweeping/bayesian-optimization.md`](../sweeping/bayesian-optimization.md) for the comparison in more depth.

## Common follow-ups

- **Refining the range.** First BO run pointed at `concurrency=814`. Re-run with a tighter band: `--search-space "phases.profiling.concurrency:600,1000:int" --search-max-iterations 15`. Same theory, narrower prior, faster convergence.
- **Targeting a percentile.** SLO chasing instead of throughput maximizing: `--search-metric time_to_first_token --search-stat p99 --search-direction minimize`. Read the [mean-of-percentiles vs pooled-percentiles caveat](../sweeping/bayesian-optimization.md#objective-semantics) before publishing the resulting number — for SLO claims the distinction matters.
- **Multi-dimensional search.** Pass `--search-space` more than once: `--search-space "phases.profiling.concurrency:1,500:int" --search-space "phases.profiling.request_rate:0.1,100.0:real"`. Increase `--search-initial-points` (10+) and `--search-max-iterations` (50+) accordingly; the sample budget scales with dimensionality.
- **Reproducibility.** Same `--search-random-seed` + same code revision + same target deployment = same trajectory. Drop the seed once you trust the recipe.
- **Cluster execution.** The same flags are exposed via `AIPerfSweep` CRs with a `multi_run.adaptive_search` block — one `AIPerfJob` per iteration, planner state owned by the controller pod. See [Adaptive search on Kubernetes](../kubernetes/adaptive-search.md).

## Limits

- **Single-objective only.** No multi-objective Pareto BO. If you want a frontier, run two scalar searches or use a grid sweep.
- **Numeric dimensions only.** `:int` and `:real`. Categorical dimensions (e.g. swap between two model variants) are not supported.
- **Optional `[bo]` extra.** The planner is behind an extras gate so the base install stays slim. Install with `uv pip install -e ".[bo]"` (or the equivalent for your packaging path) — the relevant extra is `bo` and pulls in `scikit-optimize>=0.10`.

For the explicit list of "what this implementation isn't" (NEI acquisition, posterior-regret stopping, multi-objective, pooled percentiles, heteroscedastic noise priors), see [What this implementation isn't](../sweeping/bayesian-optimization.md#what-this-implementation-isnt).
