<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Adaptive Search: Finding the Best Concurrency Without a Grid

This tutorial walks through using AIPerf's adaptive Bayesian-Optimization (BO) outer loop to find the concurrency that maximizes goodput on a real vLLM deployment, without enumerating a grid of points by hand.

For the full flag reference, search-space grammar, output schema, and the noise-handling theory, see [`docs/sweeping/bayesian-optimization.md`](../sweeping/bayesian-optimization.md). This page is the narrative companion: one scenario, one command, and how to read what comes back.

> **Kubernetes execution — *coming soon*.** This walkthrough uses
> `aiperf profile` (the local CLI). Cluster execution via `aiperf kube
> sweep` + the `AIPerfSweep` CRD is designed and implemented on the
> upcoming K8s integration branch but not yet on `main` — once shipped,
> the same `--search-*` flags and YAML schema run unmodified, with the
> in-cluster `sweep-controller` pod replacing the local subprocess
> executor. The artifacts (`search_history.json`, `sweep_aggregate/...`)
> are byte-for-byte identical between execution modes.

## The scenario

You are benchmarking a `meta-llama/Llama-3.1-8B-Instruct` deployment behind vLLM at `http://vllm.internal:8000`. You have already profiled at a fixed `--concurrency 64` and noticed that output token throughput plateaus somewhere past that — but you don't know where. You also don't want to pay for `--concurrency 8,16,32,64,128,256,512,1024` followed by a "now sweep around the best one" second pass.

Your goal is operational: find the single concurrency value that maximizes `output_token_throughput` on this deployment, in the bounded range `[1, 1000]`, treating the inference server as a black box. You do not need a Pareto frontier; you need a number you can put into the production deployment manifest. (If you *do* need a frontier — e.g. throughput vs. p99 TTFT — see [Going multi-objective](#going-multi-objective) below.)

## The first run

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --url http://vllm.internal:8000 \
    --search-space "concurrency:1,1000:int" \
    --search-metric output_token_throughput \
    --search-direction maximize \
    --search-max-iterations 25 \
    --search-initial-points 5 \
    --search-random-seed 42 \
    --num-profile-runs 3 \
    --request-count 1000 \
    --warmup-request-count 50
```

Flag-by-flag for this scenario (general semantics live in the [BO reference](../sweeping/bayesian-optimization.md#flag-reference)):

- `--search-space "concurrency:1,1000:int"` — `concurrency` is bare-name sugar for `phases.profiling.concurrency` (the same dotted path a YAML grid sweep would use; see [Bare-Name Aliases](sweeps.md#bare-name-aliases-for-common-phase-fields)); `:int` makes the planner round to integers so we never propose `concurrency=472.6`.
- `--search-max-iterations 25` — upper bound on outer iterations. Convergence may stop earlier (improvement-patience or plateau-CV; see [Convergence detection](../sweeping/bayesian-optimization.md#convergence-detection)).
- `--search-initial-points 5` — the first 5 iterations are random Sobol draws (no GP yet); iterations 6–25 are GP-driven. With a one-dimensional search 5 is plenty; raise it for higher-dimensional spaces.
- `--search-random-seed 42` — same seed, same trajectory. Drop it for production search; keep it while you are tuning the *recipe* itself.
- `--num-profile-runs 3` — three benchmarks per proposed point. The planner records one aggregate vector per point: by default each objective is the mean of finite trial values, or the pooled percentile when percentile pooling is configured. The GP/Optuna planner sees one observation per point, not three separate per-trial observations. See [Objective semantics](../sweeping/bayesian-optimization.md#objective-semantics).
- `--warmup-request-count 50` — 50 warmup requests before each timed run, so cold-cache effects don't poison early observations the GP is fitting on.

The total timed work here is `25 iterations × 3 trials = 75` benchmarks (capped — the loop may exit earlier on improvement-patience).

You did not specify `--search-stat`, so the converter defaults it to `avg`. You did not specify a goodput SLO yet — see [Common follow-ups](#common-follow-ups) below for the percentile-objective variant.

## What you'll see during the run

The orchestrator logs an opening line on entry to the adaptive loop, then one log line per iteration as it proposes a point, then a single termination line on exit. Roughly:

- Startup, from `execute_adaptive_search`: `Starting adaptive outer-loop benchmark (bayes, max_iterations=25, trials per point=3)`.
- Per iteration, before the cell runs: `[search iter <N>] proposing {'phases.profiling.concurrency': <value>}`. (The prefix is planner-agnostic — the same line shape is emitted regardless of whether the active planner is the bayesian preset, the optuna expert mode, smooth-isotonic, or monotonic-SLA.)
- Per iteration, after the trials: the standard per-run profile-export logs from each of the 3 trials — same output you would see from a non-adaptive `aiperf profile`.
- On exit (whichever convergence signal fired): `Adaptive outer loop terminated after <N> iterations (reason=<convergence_reason>)`. The reason string is one of `max_iterations`, `improvement_patience`, `plateau_cv` (and, for the Optuna planner, additionally `posterior_regret_bound` or `emmr`); `unknown` appears only when the planner ran out of proposals without recording a reason. Cancelled-mid-run leaves whatever `convergence_reason` the last completed iteration wrote (typically `null`), since the cancel branch returns before the terminal `search_history.json` write.

The first 5 iterations sample the space coarsely (Sobol). After that the GP starts steering toward the high-throughput region; do not be alarmed if iterations 6–10 cluster within a narrow concurrency band.

## Reading the artifacts

The artifact tree under `<artifact_dir>/` has three things to look at, in roughly this order. For single-trial and independent layouts the sweep aggregate is `<artifact_dir>/sweep_aggregate/`; for repeated multi-run layouts it is `<artifact_dir>/aggregate/sweep_aggregate/`.

### `search_history.json` — the trajectory

This is the file that tells you *what BO actually did*. It is rewritten after every iteration, so even a crashed or cancelled run leaves the partial trajectory:

```json
{
  "config": {
    "planner": "bayesian",
    "objectives": [
      {"metric": "output_token_throughput", "stat": "avg", "direction": "MAXIMIZE", "threshold": null}
    ],
    "outcome_constraints": [],
    "max_iterations": 25,
    "search_space": [
      {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
    ]
  },
  "iterations": [
    {"iteration_idx": 0, "variation_values": {"phases.profiling.concurrency": 503}, "objective_values": [1247.3]},
    {"iteration_idx": 1, "variation_values": {"phases.profiling.concurrency": 178}, "objective_values": [ 942.1]},
    {"iteration_idx": 2, "variation_values": {"phases.profiling.concurrency": 814}, "objective_values": [1801.4]},
    {"iteration_idx": 3, "variation_values": {"phases.profiling.concurrency":  62}, "objective_values": [ 611.7]},
    {"iteration_idx": 4, "variation_values": {"phases.profiling.concurrency": 327}, "objective_values": [1455.0]}
  ],
  "best_trials": [
    {
      "iteration_idx": 2,
      "objective_values": [1801.4],
      "variation_values": {"phases.profiling.concurrency": 814},
      "feasible": true,
      "feasible_count": 5,
      "pareto_rank": 0
    }
  ],
  "convergence_reason": "improvement_patience"
}
```

`objectives` is a length-1 list — single-objective is the length-1 special case of the multi-objective shape, not a separate format. `objective_values` per iteration is therefore also a length-1 list; `objective_values[0]` is the aggregate objective for that point: by default the mean of finite trial values, or the pooled percentile when percentile pooling is configured. This aggregate vector is what Optuna's GP sees as the observation for the point.

To parse this in Python:

```python
import orjson

with open("search_history.json", "rb") as fp:
    history = orjson.loads(fp.read())

# Single-objective: best_trials is length-1 by construction.
best = history["best_trials"][0]
print(f"argmax concurrency: {best['variation_values']['phases.profiling.concurrency']}")
print(f"objective at best:  {best['objective_values'][0]:.1f} tok/s")
print(f"converged by:        {history['convergence_reason']}")
```

Schema reference: [Output schema](../sweeping/bayesian-optimization.md#output-schema).

### `search_iter_NNNN/profile_runs/run_NNNN/profile_export_aiperf.json` — per-trial detail

Each iteration writes its 3 trials under `search_iter_NNNN/profile_runs/run_NNNN/`. These are *the same per-run JSONs that a normal `aiperf profile` produces* — you can open `profile_export_aiperf.json` for any single trial and inspect the full metric table, percentiles, error counts, etc.

You will want these when an iteration looks like a noise spike (a clearly out-of-trend point) and you want to confirm whether one of the three trials had elevated errors or a tail-latency event. The per-trial files let you inspect the spread behind the single aggregate objective that the planner observed for this point.

### `sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` — combination summary

The same per-combination aggregate the grid sweep path emits. One row per `(concurrency)` value visited, with the four sections (per-combination / best / pareto / metadata). Read this when you want a tabular CSV of "what concurrency values did BO actually visit, and what was the throughput at each." The `best_configurations` and `pareto_optimal` sections here are computed by `SweepAnalyzer` across the whole `RunResult` set; they are **orthogonal** to `search_history.json["best_trials"]` (which is what the BO planner converged on). For a single-objective run with no failed iterations the two usually agree on the winning concurrency, but they can disagree — see [Sweep Orchestrator — Stage 7 aggregate](../dev/sweep-orchestrator.md).

## Interpreting `best_trials`

`history["best_trials"]` is the iteration list selected post-hoc from the trajectory. For single-objective runs (`len(config.objectives) == 1`, the default) it is always a length-1 list — single-objective is the length-1 special case of the multi-objective shape. For multi-objective runs (`len(config.objectives) > 1`, opt-in via `objectives:` in YAML or via a multi-objective recipe), it is the Pareto front: the set of trials not dominated on every objective by another trial. See [Going multi-objective](#going-multi-objective) below for the full multi-objective walkthrough.

Two caveats worth being honest about for both shapes:

1. The loop may have early-stopped on `improvement_patience` (no improvement over the running best for 10 consecutive iterations; for multi-objective, no hypervolume gain) or `plateau_cv` (objective values plateaued in CV terms). When that happens, the true argmax — or a better Pareto trade-off — could be at an unvisited point. Improvement-patience as a stopping rule embodies a simple intuition: when you have stopped finding better points, further iterations have diminishing returns. See [Convergence detection](../sweeping/bayesian-optimization.md#convergence-detection).
2. Each `objective_values[i]` is a noisy estimate at the proposed point, with `--num-profile-runs` samples behind it. The GP knows this and shrinks confidence in noisy regions; treat the per-objective values in `best_trials` as point estimates, not as guarantees of future production performance.

A practical sanity check: re-run with `--search-random-seed` *unset* (or a different seed). If the chosen `concurrency` (or, in multi-objective, the front shape) is consistent within +/- 10% across seeds, the result is robust. If seeds disagree wildly, your search space is probably too wide or your objective is too noisy for `--num-profile-runs 3` — bump it to 5.

## When to use a grid sweep instead

| Use BO when... | Use a grid sweep when... |
|---|---|
| You want one optimal point (single-objective) or the trade-off frontier between metrics (multi-objective) and don't care about the shape between points. | The team has agreed on specific points to compare. |
| The search space is too large to enumerate (concurrency 1–1000, no obvious step). | You need every combination's results for a downstream report. |
| A single scalar objective captures what you care about — or you want the Pareto front between two-or-more metrics (see [Going multi-objective](#going-multi-objective)). | You need every variation to actually run. |
| Early-stop is acceptable (and desirable). |  |

If you want both — find the best, then characterize around it — run BO first to get the argmax, then run a tight grid sweep over the neighborhood. See [`docs/tutorials/sweeps.md`](sweeps.md) for the grid path and [`docs/sweeping/bayesian-optimization.md`](../sweeping/bayesian-optimization.md) for the comparison in more depth.

## Common follow-ups

- **Refining the range.** First BO run pointed at `concurrency=814`. Re-run with a tighter band: `--search-space "concurrency:600,1000:int" --search-max-iterations 15`. Same theory, narrower prior, faster convergence.
- **Targeting a percentile.** SLO chasing instead of throughput maximizing: `--search-metric time_to_first_token --search-stat p99 --search-direction minimize`. Read the [mean-of-percentiles vs pooled-percentiles caveat](../sweeping/bayesian-optimization.md#objective-semantics) before publishing the resulting number — for SLO claims the distinction matters.
- **Multi-dimensional search.** Pass `--search-space` more than once: `--search-space "concurrency:1,500:int" --search-space "phases.profiling.request_rate:0.1,100.0:real"`. Increase `--search-initial-points` (10+) and `--search-max-iterations` (50+) accordingly; the sample budget scales with dimensionality.
- **Reproducibility.** Same `--search-random-seed` + same code revision + same target deployment = same trajectory. Drop the seed once you trust the recipe.

## Limits

- **Single-objective is the default; multi-objective is opt-in.** The `--search-metric` / `--search-direction` CLI shape produces a length-1 `objectives` list. To opt into Pareto BO across two-or-more metrics, supply an explicit `objectives:` list in YAML (or use a multi-objective recipe) and pick a multi-objective acquisition (`--optuna-acquisition qlognehvi`); see [Going multi-objective](#going-multi-objective) below.
- **Numeric dimensions only.** `:int` and `:real`. Categorical dimensions (e.g. swap between two model variants) are not supported.
- **Optional BoTorch extra.** Optuna is installed by default and the planners prefer BoTorch when it is available. Install `uv pip install -e ".[botorch]"` for qLogNEI / qLogNEHVI behavior; if the implicit BoTorch default is unavailable, AIPerf warns and falls back to TPE. Explicit `--optuna-sampler botorch` still raises when the optional stack is missing.

For the explicit list of remaining limitations (including MORBO-style high-dimensional Pareto BO and heteroscedastic noise priors), see [What this implementation isn't](../sweeping/bayesian-optimization.md#what-this-implementation-isnt).

## Going multi-objective

The single-objective run above gave you one number: the concurrency that maximizes throughput. Same deployment, same vLLM URL, same model — but suppose the question changes. The capacity-planning team comes back and wants:

1. **Throughput numbers** so they can size hardware.
2. **p99 TTFT numbers** so they can compare against the user-facing latency target.
3. **The trade-off** between (1) and (2). Operations is willing to accept higher TTFT for more throughput up to a point, but they do not want to pre-commit to a scalar weighting like `0.7*throughput - 0.3*ttft` before they have seen the curve.

Your new goal: explore concurrency in `[1, 1000]` and produce a Pareto frontier of `(concurrency, throughput, p99_ttft)` points. The deployment team picks the operating point off the frontier afterward, possibly by saying "give me the highest throughput where p99 TTFT stays below 200 ms," or "give me the lowest TTFT where we still hit 60% of the throughput ceiling." Either question is answerable from the same artifact.

This is **not** the [`pareto-sweep` recipe](../sweeping/search-recipes.md#pareto-sweep). That recipe characterizes paired ISL/OSL across a discrete concurrency grid for capacity-planning charts; this scenario is BO over a single concurrency axis with two objectives, with the optimizer steering toward the front rather than enumerating cells.

### A note on current wiring

Multi-objective Pareto BO is **wired end-to-end** through YAML config: the `AdaptiveSearchSweep` model accepts a multi-element `objectives:` list, the `OptunaSearchPlanner` installs a qNEHVI candidates function via BoTorch (`_optuna_helpers.py::build_qnehvi_candidates_func`), and `search_history.json` writes the Pareto front under `best_trials` with `pareto_rank: 0` per entry. The component-integration smoke `tests/component_integration/test_multi_objective_e2e.py::test_qlognehvi_two_objective_search` exercises the full schema → planner → exporter path against a synthetic two-objective surface.

The CLI shorthand (`--search-metric` / `--search-direction`) only emits a length-1 `objectives` list — there is no `--search-metric` repetition syntax. Multi-objective therefore requires either:

- A YAML config with an explicit `objectives:` list (this section), or
- A multi-objective search recipe registered under the `search_recipe` plugin category (no built-in multi-objective recipe ships yet; the existing recipes — `max-throughput-ttft-sla`, `pareto-sweep`, etc. — are single-objective with optional `sla_filters`).

The `botorch` install extra is required for qLogNEHVI: `uv pip install -e ".[botorch]"` pulls in `optuna-integration`, `botorch>=0.10`, `gpytorch`, and `torch`.

### The config

Save this as `multi-objective.yaml`:

```yaml
benchmark:
  models: [meta-llama/Llama-3.1-8B-Instruct]
  endpoint:
    urls: [http://vllm.internal:8000]
    type: chat
    streaming: true
  datasets:
    - name: profiling
      type: synthetic
  phases:
    - name: profiling
      type: concurrency
      concurrency: 1   # placeholder; the search_space below overrides it per iteration
      requests: 200

multi_run:
  num_runs: 3
  cooldown_seconds: 5.0

sweep:
  type: adaptive_search
  planner: optuna
  optuna_sampler: botorch
  optuna_acquisition: qlognehvi
  search_space:
    - {path: concurrency, lo: 1, hi: 1000, kind: int}
  objectives:
    - {metric: output_token_throughput, stat: avg, direction: maximize}
    - {metric: time_to_first_token,     stat: p99, direction: minimize, threshold: 250.0}
  outcome_constraints:
    - {metric: error_request_count, op: "<=", bound: 1.0}
  max_iterations: 40
  n_initial_points: 10
  random_seed: 42
```

Then run:

```bash
aiperf profile \
    --config multi-objective.yaml \
    --warmup-request-count 50
```

Field-by-field, with reasoning specific to this scenario (the canonical reference is [Bayesian Optimization → Schema](../sweeping/bayesian-optimization.md#three-knobs-that-look-similar)):

- `planner: bayesian` (or, equivalently, `planner: optuna` + `optuna_sampler: botorch` + `optuna_acquisition: qlognehvi`). The bayesian preset auto-selects `qlognehvi` for `len(objectives) > 1`. The 1D-saturation planners `monotonic_sla` and `smooth_isotonic` reject `len(objectives) > 1` at config-time. The cross-field validator on `AdaptiveSearchSweep` rejects `qlognehvi` (and `qehvi` / `qnehvi`) with a single-objective configuration, and rejects single-objective acquisitions like `qlognei` here — see `_check_acquisition_matches_objective_count` in `src/aiperf/config/sweep/config.py`.
- Two `objectives` entries. The first maximizes mean `output_token_throughput`; the second minimizes p99 `time_to_first_token`. The `threshold: 250.0` on the TTFT objective is the **Pareto reference point** for hypervolume: trials with p99 TTFT > 250 ms contribute zero hypervolume and are effectively ignored by the convergence test. Leave it `null` to let the planner auto-derive a reference from the worst Sobol initial point, or set it explicitly when you have an operational floor (here: "TTFT past 250 ms is unacceptable for this workload"). `Objective.threshold` does **not** filter trials — they still flow into the GP. To actively avoid a region, use `outcome_constraints`.
- `outcome_constraints: error_request_count <= 1.0`. The optimizer is not optimizing error count, but Optuna's `constraints_func` (consumed by the BoTorch sampler) will downweight candidates predicted to violate this (Letham et al. 2019, [arXiv:1706.07094](https://arxiv.org/abs/1706.07094)). Distinct from `Objective.threshold` (Pareto reference point) and from `sla_filters` (post-hoc benchmark eligibility). Useful when the BO loop occasionally proposes a concurrency that drives the server into errors and you want it to back off.
- `max_iterations: 40`, `n_initial_points: 10`. Pareto fronts plateau later than scalar objectives — there is more "frontier" to explore. Doubling the single-objective defaults (25 / 5) is the standard recommendation in [Bayesian Optimization → Stopping criteria](../sweeping/bayesian-optimization.md#convergence). With `num_runs: 3` and 40 iterations, the timed-work upper bound is `40 × 3 = 120` benchmarks.
- `random_seed: 42`. Same trajectory across runs; drop it for production once you trust the recipe.

### What you'll see during the run

The orchestrator log shape is the same as single-objective:

- Startup: `Starting adaptive outer-loop benchmark (optuna, max_iterations=40, trials per point=3)`.
- Per iteration: `[search iter <N>] proposing {'phases.profiling.concurrency': <value>}`. The prefix is planner-agnostic; the line shape is identical to single-objective.
- After each iteration's three trials, the standard per-run `profile_export_aiperf.json` logs.
- On exit: `Adaptive outer loop terminated after <N> iterations (reason=<convergence_reason>)`. Convergence reasons are the same set as single-objective; in multi-objective mode `improvement_patience` and `plateau_cv` operate on the **hypervolume time series** (a non-dominated point either expands the front or doesn't, so hypervolume is monotone non-decreasing across iterations). See [Search History API → Convergence Reasons](../api/search-history.md#convergence-reasons).

The first 10 iterations are Sobol draws covering the concurrency range coarsely. After that the qLogNEHVI acquisition starts steering toward expected-hypervolume-gain — typically a mix of "exploit the current front" (push the throughput end further) and "explore a low-TTFT region" (find non-dominated points the front does not yet contain).

### How `best_trials` becomes the Pareto front

The artifact tree under `<artifact_dir>/` mirrors the single-objective case (per-iteration trial directories under `search_iter_NNNN/`, plus `sweep_aggregate/`), with two important differences in `search_history.json`:

1. `iterations[i].objective_values` is a **length-2 list** (one entry per `objectives[i]`, in the order declared). For length-N objectives it is length-N; the field shape generalizes.
2. `best_trials` is the **Pareto front** rather than a single argmax. A trial *A* dominates *B* iff *A* is at least as good on every objective and strictly better on at least one. The Pareto front is the set of trials not dominated by any other trial. Each entry carries `pareto_rank: 0` (all front members are non-dominated by definition); the front itself is unranked.

Excerpt of what a converged run produces:

```json
{
  "config": {
    "planner": "optuna",
    "objectives": [
      {"metric": "output_token_throughput", "stat": "avg", "direction": "maximize", "threshold": null},
      {"metric": "time_to_first_token",     "stat": "p99", "direction": "minimize", "threshold": 250.0}
    ],
    "outcome_constraints": [
      {"metric": "error_request_count", "op": "<=", "bound": 1.0}
    ],
    "search_space": [
      {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
    ]
  },
  "iterations": [
    {"iteration_idx": 7,  "variation_values": {"phases.profiling.concurrency": 280}, "objective_values": [9800.1, 215.4]},
    {"iteration_idx": 13, "variation_values": {"phases.profiling.concurrency": 256}, "objective_values": [9512.3, 187.4]},
    {"iteration_idx": 22, "variation_values": {"phases.profiling.concurrency": 224}, "objective_values": [8910.0, 162.7]},
    {"iteration_idx": 27, "variation_values": {"phases.profiling.concurrency": 312}, "objective_values": [9905.4, 248.9]}
  ],
  "best_trials": [
    {"iteration_idx": 7,  "objective_values": [9800.1, 215.4], "variation_values": {"phases.profiling.concurrency": 280}, "feasible": true, "feasible_count": 36, "pareto_rank": 0},
    {"iteration_idx": 13, "objective_values": [9512.3, 187.4], "variation_values": {"phases.profiling.concurrency": 256}, "feasible": true, "feasible_count": 36, "pareto_rank": 0},
    {"iteration_idx": 22, "objective_values": [8910.0, 162.7], "variation_values": {"phases.profiling.concurrency": 224}, "feasible": true, "feasible_count": 36, "pareto_rank": 0}
  ],
  "convergence_reason": "improvement_patience"
}
```

Read this top-down. Three iterations made the front: at concurrency 224 the run had the lowest p99 TTFT (162.7 ms) but the lowest throughput (8910 tok/s); at concurrency 280 the run had the highest throughput on the front (9800 tok/s) at a p99 TTFT of 215.4 ms; concurrency 256 sits between them. Iteration 27's `(9905.4, 248.9)` is **dominated** by no front member on throughput, but its p99 TTFT exceeds the `threshold: 250.0` reference point only by ~1 ms; depending on whether the planner's actual threshold-based dominance check kept it, it may or may not appear on the front in your run. Trials worse than the threshold contribute zero hypervolume but are not removed from the GP.

For the full schema — including how `feasible` interacts with `sla_filters`, what `feasible_count: 0` means, and the multi-objective hypervolume time-series caveats — see [Search History API → Interpreting `best_trials`](../api/search-history.md#interpreting-best_trials).

### Picking a deployment point off the frontier

The defining property: **for any pair of trials on the front, neither dominates the other**. Pick a trial, and there is no other front trial that beats it on both throughput and p99 TTFT simultaneously.

To pick a deployment point off the frontier, apply *your* scalar criterion **after** the run:

- "Highest throughput where p99 TTFT stays below 200 ms" → filter `best_trials` to entries where `objective_values[1] < 200.0`, then pick the max of `objective_values[0]`. From the example: `(concurrency=256, throughput=9512.3, ttft=187.4)`.
- "Lowest p99 TTFT where throughput is at least 90% of the maximum on the front" → compute `0.9 * max(throughput)`, filter, pick min TTFT. From the example: max throughput on the front is 9800.1, 90% is 8820.0; both 256 and 224 qualify; pick 224 with TTFT 162.7.
- "Knee point" (the visual elbow of the curve) → no closed-form pick, but it is usually obvious by eye when the front is plotted with p99 TTFT on the x-axis and throughput on the y-axis.

A small Python snippet to load and filter:

```python
import orjson

with open("artifacts/<benchmark_name>/search_history.json", "rb") as fp:
    history = orjson.loads(fp.read())

objectives = history["config"]["objectives"]
front = history["best_trials"]

# Pareto-aware: every entry of `front` carries one objective_values entry
# per objectives[i], in the same order.
print(f"Pareto front: {len(front)} non-dominated trials")
for trial in front:
    desc = ", ".join(
        f"{obj['metric']}/{obj['stat']}={v:.1f}"
        for obj, v in zip(objectives, trial["objective_values"])
    )
    print(f"  iter={trial['iteration_idx']:>3}  vars={trial['variation_values']}  {desc}")

# "Highest throughput where p99 TTFT < 200 ms"
ttft_idx = next(i for i, o in enumerate(objectives) if o["metric"] == "time_to_first_token")
tput_idx = next(i for i, o in enumerate(objectives) if o["metric"] == "output_token_throughput")
under_sla = [t for t in front if t["objective_values"][ttft_idx] < 200.0]
if under_sla:
    pick = max(under_sla, key=lambda t: t["objective_values"][tput_idx])
    print(f"\nDeployment pick: {pick['variation_values']} -> {pick['objective_values']}")
else:
    print("\nNo front trial is under 200 ms p99 TTFT; widen the threshold or re-run.")
```

This is the payoff: you defined "feasible" once (`p99 TTFT < 200`), applied it post-hoc to a Pareto front the optimizer produced without seeing your weighting, and got a single deployment point. Re-running with a different post-hoc criterion (`< 250 ms` instead of `< 200 ms`, or `> 9000 tok/s` instead of throughput-max-under-SLA) is a single Python read of the same file. No second BO run needed.

### A practical sanity check (multi-objective)

Re-run with `random_seed` unset (or a different value). If the **shape** of the front is consistent across seeds — same approximate concurrency range, same approximate trade-off curvature — the result is robust. If two seeds disagree on whether the front contains a trial near concurrency 280, your search is probably under-budgeted; raise `max_iterations` to 60+ and `n_initial_points` to 15.

You can also check `feasible_count` in `best_trials`. With `outcome_constraints` set, this is the number of trials across the run that satisfied the constraint. `feasible_count: 0` flags the "no iteration was feasible" case and means BO fell back to ranking the full pool — treat the front as suspect and tighten `lo`/`hi` to avoid the failure region.

### When NOT to use multi-objective BO

- **You can articulate a defensible scalar.** If `0.7*throughput - 0.3*ttft` or a goodput metric (which already encodes the SLA) captures what the team cares about, single-objective BO is faster, has tighter convergence guarantees, and produces one number directly. Go back to [The first run](#the-first-run) above.
- **You want paired ISL/OSL × concurrency characterization for a capacity-planning chart.** That is the [`pareto-sweep` recipe](../sweeping/search-recipes.md), not multi-objective BO. It runs a discrete grid (no BO loop), tags each cell `pareto_optimal: true | false`, and writes a plot-ready JSON. Different artifact, different question.
- **You want a hard SLA cutoff** ("p99 TTFT must NEVER exceed 250 ms — drop any trial that breaches"). `Objective.threshold` does not filter; `outcome_constraints` are soft (acquisition mask, not rejection); `sla_filters` are the right tool for hard eligibility. See [Bayesian Optimization → Schema](../sweeping/bayesian-optimization.md#three-knobs-that-look-similar) for the full three-way distinction.
- **You only have one metric.** `len(objectives) == 1` is single-objective BO; the cross-field validator will reject `qlognehvi` here.

## Further reading

- [Bayesian Optimization](../sweeping/bayesian-optimization.md) — full reference for both single- and multi-objective BO: flag reference, acquisition / sampler / reference-point semantics, the `Objective.threshold` vs `OutcomeConstraint` vs `sla_filters` distinction, reference-point auto-derivation, hypervolume-based stopping, and the qLogNEHVI acquisition.
- [Search Recipes](../sweeping/search-recipes.md) — registered single-objective recipes (`max-throughput-ttft-sla`, `pareto-sweep`, etc.) and when to reach for one instead of writing a YAML config from scratch.
- [Sweep Troubleshooting](../troubleshooting/sweeps.md) — common configuration errors (acquisition / objective-count mismatches, `n_initial_points >= max_iterations`, missing optional BoTorch extra) and how to fix them.
- [Search History API](../api/search-history.md) — full `search_history.json` schema, including the `best_trials` Pareto-front shape, `feasible_count` semantics, and convergence reasons.
