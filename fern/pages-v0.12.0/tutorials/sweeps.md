---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Parameter Sweeps and Multi-Run Statistics
---

> **Kubernetes execution — *coming soon*.** Native cluster sweeps via the
> `AIPerfSweep` CRD and an `aiperf kube sweep` CLI are designed and
> implemented on the upcoming K8s integration branch but not yet on
> `main`. The YAML/sweep semantics on this page are the same in both
> execution modes (local subprocess today; an in-cluster `sweep-controller`
> pod creating child `AIPerfJob` CRs once shipped). Until the K8s path
> lands, `aiperf kube profile` rejects `sweep:` and `multi_run:` keys and
> hands you off to `aiperf profile` for the local CLI.

# Parameter Sweeps and Multi-Run Statistics

Finding the optimal operating point for an inference server requires exploring a multi-dimensional space of concurrency, request rate, input lengths, and batch sizes. Rather than hand-tuning one variable at a time, parameter sweeps let you define the search space declaratively and let AIPerf run every combination, collecting statistically rigorous results for each.

## Choosing a sweep mode

A **sweep** is one benchmark configuration that produces many benchmark runs. Instead of running `aiperf profile` ten times by hand, each time editing the YAML, you write the YAML once with a `sweep:` block that says "vary these values and run each one." AIPerf takes care of running them in sequence and putting the results in side-by-side folders so you can compare them. The mode you pick decides **which combinations of values get run** — that's the whole question.

Pick the row that matches your situation:

| Your situation | Mode to use | Why |
|---|---|---|
| "I want to try concurrency 8, 32, 64, and 128 — and also rate 10 and 50 — at every combination." | [**Grid**](#grid-sweep) | Cartesian product is exactly what you described. |
| "I want three runs: small/short, medium/medium, large/long. Each one sets ISL and OSL together." | [**Zip**](#zip-sweep) | Pairs values lockstep instead of cross-producting them. |
| "I want to compare three named workloads: chatbot, summarization, long-context QA. Each one tweaks several settings at once." | [**Scenarios**](#scenario-sweep) | Each scenario is a labelled patch on top of a base config. |
| "I want broad characterization across 4-D space (concurrency × ISL × OSL × rate) without 625 runs." | [**Sobol** or **Latin Hypercube**](../sweeping/space-filling.md) | Even coverage on a fixed sample budget. |
| "I want to find the single best concurrency value, but I don't know the right range and don't want to enumerate." | [**Adaptive Search (BO)**](./adaptive-search.md) | Bayesian optimization steers the search toward the optimum. |
| "I want the trade-off frontier between two metrics (throughput vs. p99 TTFT) without picking weights up front." | [**Multi-Objective BO**](./adaptive-search.md#going-multi-objective) | Pareto BO produces a frontier you reason over after the run. |
| "I want a chart showing the throughput-vs-latency trade-off across realistic workload shapes." | [**Pareto Sweep recipe**](../sweeping/search-recipes.md#pareto-sweep) | Built-in recipe that emits a frontier-ready artifact. |
| "I want the highest concurrency that still passes my p95 TTFT SLA." | [**Search recipe**](../sweeping/search-recipes.md) | `max-throughput-ttft-sla` does this in one flag. |
| "I want a confidence interval on my numbers, not just one run." | [**Multi-Run**](#multi-run-statistics) | Repeats every variation N times and reports CIs. Combine with any sweep mode above. |

If your answer is two of these at once, that's fine — pick the one that captures the **search structure**, then read the section to see how it composes with the others.

### Grid: every combination

**Mental model:** a multiplication table. Two axes with N and M values produce `N × M` runs. Three axes produce `N × M × K`. Add a fourth and you'll be sorry.

Reach for grid when you have **two or three** independent axes and you genuinely want every combination — concurrency doesn't care what rate you picked, and vice versa — and you want a tidy table at the end where every cell is filled in.

Grid is the wrong answer when you have four or more axes (the combination count explodes — `5 × 5 × 5 × 5 = 625` runs at, say, two minutes apiece is a 21-hour benchmark; look at Sobol instead), when the values are **coupled** (you want ISL and OSL to move together as a pair, not cross-product them — grid will run nonsense combinations like `isl=2048, osl=64`; use zip or scenarios), or when you don't actually know what range of values is interesting yet (use adaptive search to find the interesting region first, then come back to grid for a tight characterization sweep). Full reference: [Grid Sweep](#grid-sweep) below.

### Zip: pair things up

**Mental model:** zipping two lists together. The first values pair, the second values pair, the third values pair. No cross-product.

Reach for zip when two or more parameters need to **move together**. The classic case is paired ISL/OSL: small prompts have short outputs, big prompts have long outputs, and benchmarking `isl=2048, osl=64` (a huge prompt with a one-token reply) tells you nothing useful. Use zip when you want the runs to be **anonymous** — just numbered variations, no human-readable label per run.

Zip is the wrong answer when the lists have different lengths (zip rejects this at config-load time — either pad the lists or split into multiple sweeps), when you want each pairing to carry a **descriptive name** in the output directory (use scenarios), or when the combinations you want aren't all the same shape (zip can only set the same set of fields on every run; if scenario A also tweaks `phases.profiling.duration` while scenario B leaves it alone, you need scenarios). Full reference: [Zip Sweep](#zip-sweep) below.

### Scenarios: named, hand-picked configs

**Mental model:** a list of `git diff` patches against a base config. Each scenario has a name and only specifies the fields it overrides; everything else is inherited.

Reach for scenarios when you're comparing a **small set of qualitatively different** configurations — "three workload archetypes" or "four candidate model serving setups," not "every combination of two axes" — or when each scenario tweaks **multiple fields at once** in ways that don't follow a regular pattern (grid and zip can only vary one field per axis; scenarios let you change ISL, OSL, rate, and phase duration simultaneously per run), or when you want the result folders **named** after what they represent (`summarization/` instead of `variation_0001_isl_2048_osl_512/`).

Scenarios are the wrong answer when your variations follow a regular pattern (every value of A crossed with every value of B — use grid, much less typing) or when you have more than ~10 scenarios (the YAML gets unwieldy — either generate it programmatically or step up to a search recipe). Full reference: [Scenario Sweep](#scenario-sweep) below.

### Sobol / Latin Hypercube: broad coverage on a budget

**Mental model:** a grid sweep would put a point at every grid intersection, which gets expensive fast in 3-D and 4-D. Sobol and Latin Hypercube instead drop a fixed number of points (say, 64) **scattered evenly** across the same space — fewer cells, but every region of the space gets representative coverage.

Reach for space-filling sweeps when you have **3+ axes** to explore and a fixed time budget ("I have time for 60 runs total. Cover the space well."), when you want to **plot a perf surface** across realistic workload variation (Sobol gives you points in every region, ready for a scatter or a fitted surface), when you want **A/B build comparisons** (same `seed` produces identical points on build A and build B, giving paired comparisons much tighter than independent random sweeps), or when all your dimensions are **discrete and small** (model choice, batch size in `[1,2,4,8,16]`) — pick **Latin Hypercube**, which guarantees each option appears the same number of times.

Space-filling is the wrong answer when you only have one or two axes (use grid — the math is the same and the YAML is simpler), when you want **the optimum** rather than the surface (use adaptive search — it spends its budget zeroing in on the best point instead of covering the space evenly), or when you want every run to have a human-readable label (Sobol and Latin Hypercube produce numbered variations). Full reference: [Space-filling Sweeps](../sweeping/space-filling.md). Default to Sobol unless your dimensions are all small and discrete.

### Adaptive Search: let the tool find the best

**Mental model:** instead of you picking the values, AIPerf picks them for you, one at a time, learning from each run. After a few random pokes to get oriented, it fits a model of "where is the good region likely to be?" and proposes the next concurrency value to try. By iteration 25 it's converged on the best concurrency in the range.

Reach for adaptive search when you want **the single best value** for one parameter (often concurrency) under a single objective (often `output_token_throughput`), when the range is **wide and you don't know the answer** ("concurrency between 1 and 1000, somewhere" is a perfect fit), when you're willing to trade "every cell of a grid filled in" for "fewer total runs and a better answer," or when you want the loop to **stop itself** when it has converged instead of running every cell of a grid that you know is wasteful past iteration 10.

Adaptive search is the wrong answer when you need every grid cell's results for a downstream report or chart, when your objective isn't a single scalar (you want to see the **trade-off** between two metrics — use multi-objective BO for the BO-driven Pareto frontier, or Pareto sweep for the recipe-driven paired-ISL/OSL × concurrency variant), when you want to compare a **named set** of configurations rather than search a continuous range (use scenarios), or when the dimension you want to vary is **categorical** (model variant A vs B — BO supports `:int` and `:real`, not categories). Full walkthrough: [Adaptive Search tutorial](./adaptive-search.md). Optuna ships by default; BoTorch-backed acquisitions require the optional `botorch` extra.

### Multi-Objective BO: Pareto frontier without picking weights

**Mental model:** adaptive search finds the single best value for one scalar metric. Multi-objective BO instead produces a **Pareto frontier** between two-or-more metrics — the set of operating points where you cannot improve one metric without hurting another. The optimizer steers the search toward the frontier; you pick a deployment point off the frontier afterward, applying your scalar criterion ("highest throughput where p99 TTFT < 200 ms") only at the end.

The CLI shorthand (`--search-metric` / `--search-direction`) is single-objective only — multi-objective requires YAML with an explicit `objectives:` list. qLogNEHVI requires the optional `botorch` extra.

Reach for multi-objective BO when you need the **trade-off shape** between two metrics rather than a single argmax ("throughput vs. p99 TTFT" or "throughput vs. error rate" are the canonical pairs), when you **do not want to commit to a scalar weighting** up front (with single-objective + scalarization `0.7*tput - 0.3*ttft` you have to pick the weights before the search; multi-objective BO defers that decision until you've seen the curve), or when your axes are **continuous** (concurrency in `[1, 1000]`) and you want the optimizer to steer rather than enumerate.

Multi-objective BO is the wrong answer when you can articulate a defensible scalar (a goodput metric that already encodes the SLA, or a weighting the team has agreed on — use adaptive search: faster, tighter convergence, one number out), when you want **paired ISL/OSL × concurrency characterization** for a capacity-planning chart (that is the `pareto-sweep` recipe, not multi-objective BO — different artifact, different question), or when you want a **hard SLA cutoff** ("p99 TTFT must NEVER exceed 250 ms": `Objective.threshold` is a Pareto reference point, not a filter; `outcome_constraints` are soft (acquisition mask); for hard eligibility use `sla_filters` — see [Bayesian Optimization → Multi-objective Pareto BO](../sweeping/bayesian-optimization.md#multi-objective-pareto-bo)). Full walkthrough: [Adaptive Search → Going multi-objective](./adaptive-search.md#going-multi-objective).

### Pareto Sweep: the throughput-vs-latency frontier

**Mental model:** you don't have a single best answer because two things matter at once — throughput and tail latency. Higher concurrency gets you more throughput, but the tail latency gets worse. The "Pareto frontier" is the set of operating points where you can't improve one without hurting the other. Pareto sweep is a one-flag recipe that runs the cells, computes the frontier, and writes a plot-ready JSON.

Reach for Pareto sweep when you want a **chart for a capacity-planning doc** showing how throughput trades off against latency across realistic workload shapes, when the shapes are **paired ISL/OSL** (the recipe's specialty) and you want to characterize each shape across a range of concurrency, or when one curve per workload shape plus a global frontier across all shapes is exactly the picture you'd draw.

Pareto sweep is the wrong answer when you want a single **best** concurrency rather than a frontier (use adaptive search), when your axes aren't `(isl, osl, concurrency)` (the recipe is hard-wired for that shape — for arbitrary axes write a scenarios sweep and post-process yourself, or use multi-objective BO), or when you're profiling a non-streaming endpoint (the recipe rejects this: `output_token_throughput` requires streaming). Full walkthrough: [Search Recipes → `pareto-sweep`](../sweeping/search-recipes.md#pareto-sweep).

### Search recipes: the shortcut for common questions

**Mental model:** the underlying knobs (`--search-space`, `--search-metric`, `--search-direction`, `--search-max-iterations`, post-process configuration) are powerful but it's easy to get them wrong. A "search recipe" is a named bundle of those knobs designed for a specific real-world question. You ask the question, the recipe sets the knobs.

| You want to | Recipe |
|---|---|
| Maximize throughput under a TTFT SLA | `max-throughput-ttft-sla` |
| Maximize throughput under an ITL SLA | `max-throughput-itl-sla` |
| Find the highest concurrency that still passes one or more SLAs | `max-concurrency-under-sla` |
| Maximize goodput under per-request TTFT/TPOT/E2E SLOs | `max-goodput-under-slo` |
| Find the concurrency knee where p99 latency degrades sharply | `concurrency-ramp` |
| Characterize TTFT vs ISL for capacity planning | `prefill-ttft-curve` |
| Characterize ITL across concurrency × OSL | `decode-itl-curve` |
| Pareto frontier across paired ISL/OSL workload shapes | `pareto-sweep` |

```bash
aiperf profile --model my-model --url http://infer.example.com --streaming \
  --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200
```

Reach for a recipe when your question is in the table above. Skip the manual `--search-*` flag stack and let the recipe pick the right metric, direction, and termination conditions. Recipes are the wrong answer when your question isn't in the table, or when you need to tweak something the recipe doesn't expose — drop down to the explicit `--search-*` flags or to a YAML sweep block; the underlying machinery is the same. Full catalog: [Search Recipes](../sweeping/search-recipes.md).

### Multi-Run: this is not a sweep, but it pairs with one

**Mental model:** a single benchmark run gives you one number. That number has noise. Multi-run repeats the run N times and gives you a mean and a confidence interval, so you can tell whether two configurations are actually different or just within the noise floor.

Use multi-run any time you care about whether a difference is real. The run-to-run coefficient of variation (CV) on a server under load is rarely zero; without multi-run you can't tell a 3% throughput improvement from random jitter.

Multi-run multiplies with sweeps: `3 sweep variations × 5 runs each = 15 total benchmarks`. Every sweep mode above composes with `multi_run:` — the sweep decides what to vary, multi-run decides how many times to repeat each variation. See [Multi-Run Statistics](#multi-run-statistics) below for the field reference, and [Multi-Run Confidence Reporting](./multi-run-confidence.md) for the statistical methodology.

### Worked example: throughput optimization + capacity chart

You're tuning a vLLM deployment of `meta-llama/Llama-3.1-8B-Instruct`. Your boss wants three things:

1. **One** concurrency value to put in the production manifest.
2. A chart showing **how throughput and tail latency trade off** across three workload shapes.
3. **Tight numbers** — the boss will ask "is that 1247 tok/s repeatable?"

The right play is two sweeps plus multi-run, not one giant grid:

- For (1), an [adaptive search](./adaptive-search.md) over concurrency `[1, 1000]` maximizing `output_token_throughput`. ~25 iterations × 3 trials each. Drops out a single number.
- For (2), a [Pareto sweep](../sweeping/search-recipes.md#pareto-sweep) with three `--isl-osl-pairs` and a concrete `--concurrency` list. Drops out a frontier JSON ready to plot.
- For (3), keep `--num-profile-runs 3` on both. The variance and CI come along for free.

A single grid sweep over `(concurrency × isl × osl)` would have been hundreds of runs and still wouldn't have given you the convergence guarantee adaptive search does.

### Common mistakes

- **Using grid when you wanted zip.** If your runs include `isl=2048, osl=64`, the grid is testing nonsense. Switch to zip or scenarios.
- **Using a giant grid when you wanted Sobol.** A 4-axis grid with 5 values per axis is 625 runs. A 64-sample Sobol sweep covers the same space with comparable resolution and 10× less wall time.
- **Using grid when you wanted adaptive search.** If you started with `--concurrency 8,16,32,64,128,256,512,1024` and immediately did a "now sweep around the best one" second pass, you wanted BO from the start.
- **Forgetting multi-run.** A single run's number is suggestive, not statistical. If your benchmark is informing a real decision, repeat it.
- **Mixing recipes with explicit `--search-*` flags.** The CLI rejects this with a clear error — drop one or the other, don't try to override a recipe in flight.

## Sweep Strategies

AIPerf supports five enumeration / sampling sweep strategies:

| Strategy | How it works | Best for | Variations generated |
|---|---|---|---|
| **Grid** | Cartesian product of variable lists | Systematic exploration of 2-3 variables | `len(v1) * len(v2) * ...` |
| **Zip** | Element-wise (lockstep) pairing of variable lists | Coordinated tuples (e.g. paired ISL/OSL) without N x M blow-up | `len(v1)` (all lists must match length) |
| **Scenarios** | Named configs deep-merged onto base | Comparing hand-picked workload profiles | One per scenario |
| **Sobol** | Quasi-Monte-Carlo low-discrepancy samples | Even joint coverage at fixed budget; characterization plots | `samples` |
| **Latin Hypercube** | Stratified sampling, one bin per axis | Discrete-dim sweeps; perfect marginal balance | `samples` |

For Sobol and Latin Hypercube, see [Space-filling sweeps (Sobol, Latin Hypercube)](../sweeping/space-filling.md). For adaptive (Bayesian) search, which closes the loop on prior results to choose the next sample, see [Bayesian Optimization](../sweeping/bayesian-optimization.md).

## UI in Sweep Mode

Sweep mode rejects `--ui dashboard`. Use `--ui simple` (progress bars per variation) or `--ui none` (minimal output, ideal for CI). With no explicit `--ui`, AIPerf falls back to the standard auto-selection rules.

```text
Dashboard UI is not supported with sweep/multi-run mode.
Please use '--ui simple' or '--ui none' instead.
```

## Grid Sweep

A grid sweep takes one or more variables, each with a list of values, and runs every combination (Cartesian product). Variables use dot-notation paths that map to fields in the YAML config tree.

### Example: Sweep Concurrency x Rate to Find Saturation

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

  phases:
    - name: profiling
      type: poisson
      duration: 120
      rate: 10     # overridden by sweep
      concurrency: 8 # overridden by sweep
      grace_period: 30

  artifacts:
    dir: ./artifacts/saturation_sweep
    summary: [json]
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [8, 32, 64, 128]
    phases.profiling.rate: [10, 50, 100]
```

This produces `4 * 3 = 12` benchmark runs. Each variation overrides the dot-path fields on a deep copy of the base config. Because `phases:` is a list of named entries, the second segment of the dot-path (`profiling`) is matched against each phase's `name` field — so `phases.profiling.concurrency: 32` sets the `concurrency` field inside the phase whose `name` is `profiling`. Phases not mentioned in the override are inherited from the base unchanged.

The results directory will contain one subdirectory per variation, making it straightforward to compare throughput and latency across the concurrency-rate surface.

### Bare-Name Aliases for Common Phase Fields

The most-swept phase fields have bare-name shortcuts that expand to the full `phases.profiling.<field>` path. The two snippets below are equivalent:

```yaml
sweep:
  type: grid
  parameters:
    concurrency: [8, 32, 64, 128]      # sugar
    rate: [10, 50, 100]                # sugar
```

```yaml
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [8, 32, 64, 128]
    phases.profiling.rate: [10, 50, 100]
```

Aliases (each expands to `phases.profiling.<name>`):

`concurrency`, `prefill_concurrency`, `rate`, `requests`, `duration`, `sessions`, `users`, `smoothness`, `grace_period`, `concurrency_ramp`, `prefill_ramp`, `rate_ramp`.

Sugar is opt-in by spelling: only a bare token equal to one of these names is rewritten. `concurrency.value` (compound) or `phases.warmup.requests` (already-canonical) are left untouched. Sweep aggregates, audit files, and result-directory labels always use the full canonical path regardless of which form you wrote — the sugar is purely an input convenience. Mixing both spellings for the same parameter is rejected.

### CLI Magic-List Sugar

Several CLI flags accept a comma-separated list and auto-promote to a sweep on the corresponding phase or dataset path — no YAML needed.

**Phase-rooted (`phases.profiling.<field>`):**

```bash
aiperf profile --model X --url Y --concurrency 1,2,4,8,16
aiperf profile --model X --url Y --prefill-concurrency 1,2,4 --streaming
aiperf profile --model X --url Y --request-rate 10,20,50
aiperf profile --model X --url Y --request-count 100,500,1000
aiperf profile --model X --url Y --benchmark-duration 30,60,120
aiperf profile --model X --url Y --num-conversations 50,100,200    # sweeps phase.sessions; dataset pool sized to max
aiperf profile --model X --url Y --user-centric-rate 10 --num-users 4,8,16   # user-centric only
```

**Dataset-rooted (synthetic prompts):**

```bash
aiperf profile --model X --url Y --isl 128,512,2048              # datasets.main.prompts.isl.mean
aiperf profile --model X --url Y --osl 64,128,256                # datasets.main.prompts.osl.mean
aiperf profile --model X --url Y --isl-stddev 10,50,200          # datasets.main.prompts.isl.stddev
aiperf profile --model X --url Y --osl-stddev 5,25,100           # datasets.main.prompts.osl.stddev
aiperf profile --model X --url Y --conversation-turn-mean 1,3,8  # datasets.main.turns.mean
```

Pass multiple flags together to cross-product (e.g. `--isl 128,512 --concurrency 4,8` yields a 4-cell grid). Scalar values pass through as plain phase/dataset fields and do not create a sweep. Mutually exclusive with `--variant` and grid `--search-recipe`; both raise a clear error.

#### Pairing magic-lists with `--sweep-type zip`

By default multiple magic-list flags form a Cartesian product. Pass `--sweep-type zip` to switch to element-wise pairing — equivalent to the YAML `sweep: {type: zip}` block. All lists must have equal length; mismatches are rejected at expand time.

```bash
# 3 paired cells: (isl=128,osl=128,conc=4) (isl=512,osl=256,conc=16) (isl=2048,osl=512,conc=64)
aiperf profile --model X --url Y --sweep-type zip \
  --isl 128,512,2048 --osl 128,256,512 --concurrency 4,16,64
```

`--sweep-type` only affects CLI-driven sweeps. If a YAML `sweep:` block is loaded, its own `type:` wins.

The dataset-rooted stddev and turn-mean flags are designed to be paired with their corresponding `--isl` / `--osl` / `--num-conversations` flags in zip mode to model realistic traffic shapes:

```bash
# Realistic small/medium/large request distributions: each tier co-varies mean and stddev
aiperf profile --model X --url Y --sweep-type zip \
  --isl 128,512,2048  --isl-stddev 10,50,200 \
  --osl 64,256,1024   --osl-stddev 5,25,100

# Multi-turn realism curve: 1-turn single-shot, 3-turn dialog, 8-turn extended
aiperf profile --model X --url Y --sweep-type zip \
  --num-conversations 10,50,200 --conversation-turn-mean 1,3,8 \
  --concurrency 4,16,64
```

## Zip Sweep

A zip sweep pairs parameter lists element-wise (lockstep) instead of taking their Cartesian product. All parameter lists must have identical length; the i-th run sets each path to its i-th value. Use this when you want N coordinated runs each setting a tuple of fields together — without the N x M blow-up of a grid sweep. The canonical use case is paired input-sequence-length / output-sequence-length (ISL/OSL) benchmarking, where each run should set both lengths to a coordinated pair (small/short, medium/medium, large/long) rather than test every cross-product. Path semantics are identical to grid: bare paths target fields under `benchmark:`, and `variables.<name>` writes the envelope-level Jinja block.

### Example: Paired ISL/OSL

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000

  phases:
    - name: profiling
      type: concurrency
      duration: 120
      concurrency: 32
      grace_period: 30

  artifacts:
    dir: ./artifacts/isl_osl_pairs
    summary: [json]
sweep:
  type: zip
  parameters:
    dataset.prompts.isl: [128, 512, 2048]
    dataset.prompts.osl: [128, 256, 512]
```

This produces exactly **3** runs: `(isl=128, osl=128)`, `(isl=512, osl=256)`, `(isl=2048, osl=512)` — not the 9 a grid sweep would produce. Mismatched list lengths are rejected at config-load time. The base-class knobs `iteration_order` and `same_seed` apply identically to grid (zip inherits the same `_GridSweepBase`).

## Scenario Sweep

A scenario sweep defines named configurations that are deep-merged onto the base config. Each scenario overrides only the fields it specifies; everything else inherits from the base. This is ideal when comparing qualitatively different workload profiles that touch multiple config sections.

### Example: Compare Workload Profiles

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

  phases:
    - name: profiling
      type: poisson
      duration: 120
      rate: 20
      concurrency: 32
      grace_period: 30

  artifacts:
    dir: ./artifacts/workload_comparison
    summary: [json]
sweep:
  type: scenarios
  runs:
    - name: short_chatbot
      benchmark:
        dataset:
          prompts:
            isl: {type: normal, mean: 64, stddev: 10}
            osl: {type: normal, mean: 32, stddev: 8}
        phases:
          - name: profiling
            rate: 100

    - name: summarization
      benchmark:
        dataset:
          prompts:
            isl: {type: normal, mean: 2048, stddev: 200}
            osl: {type: normal, mean: 256, stddev: 50}
        phases:
          - name: profiling
            concurrency: 16
            rate: 10

    - name: long_context_qa
      benchmark:
        dataset:
          prompts:
            isl: {type: normal, mean: 8192, stddev: 500}
            osl: {type: normal, mean: 512, stddev: 100}
        phases:
          - name: profiling
            concurrency: 8
            rate: 5
```

Deep-merge means nested dicts are merged recursively, and `phases:` overrides are matched by `name` against the base's phase list — only fields you set on a named override are changed; everything else is inherited. In the `short_chatbot` scenario, `dataset.prompts` is replaced entirely because it is the leaf being overridden, while `dataset.type` and `dataset.entries` remain inherited from the base, and the `profiling` phase keeps its base `type`, `duration`, and `grace_period` while picking up the new `rate`. Each scenario's `name` field becomes its label in the output directory.

## Sweep + Distributions

Distribution parameters are just nested fields in the config tree, so they can be sweep parameters like any other field. This lets you study how sequence length affects latency and throughput.

### Example: Sweep ISL Across Fixed Values

Use a grid sweep to test three different input sequence lengths:

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000
    prompts:
      isl: 128 # overridden by sweep
      osl: {type: normal, mean: 128, stddev: 25}

  phases:
    - name: profiling
      type: poisson
      duration: 120
      rate: 30
      concurrency: 32
      grace_period: 30

  artifacts:
    dir: ./artifacts/isl_sweep
    summary: [json]
sweep:
  type: grid
  parameters:
    dataset.prompts.isl: [128, 512, 2048]
```

This produces 3 runs, one per ISL value. Since ISL accepts both fixed integers and distribution objects, each value is set as a fixed distribution (no variance).

### Example: Sweep Distribution Type via Scenarios

To compare different distribution shapes, use a scenario sweep that replaces the entire distribution object:

```yaml
sweep:
  type: scenarios
  runs:
    - name: fixed_512
      benchmark:
        dataset:
          prompts:
            isl: 512

    - name: normal_512_wide
      benchmark:
        dataset:
          prompts:
            isl: {type: normal, mean: 512, stddev: 100}

    - name: normal_512_narrow
      benchmark:
        dataset:
          prompts:
            isl: {type: normal, mean: 512, stddev: 20}
```

### Paired ISL/OSL via Scenarios

When you want to compare hand-picked input/output length pairings — 128/128 for chatbot-style turns, 256/256 for short Q&A, 512/1024 for summarization — a grid sweep is the wrong tool (it produces a Cartesian product, not paired combinations). The [zip sweep](#zip-sweep) shown above is the most compact way to express paired ISL/OSL when you don't need per-run names; scenarios add value when you want each pair to carry its own human-readable label in the output directory.

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000

  phases:
    - name: profiling
      type: poisson
      duration: 120
      rate: 30
      concurrency: 32
      grace_period: 30

  artifacts:
    dir: ./artifacts/isl_osl_pairs
    summary: [json]
sweep:
  type: scenarios
  runs:
    - name: short
      benchmark:
        dataset: {prompts: {isl: 128, osl: 128}}
    - name: medium
      benchmark:
        dataset: {prompts: {isl: 256, osl: 256}}
    - name: long
      benchmark:
        dataset: {prompts: {isl: 512, osl: 1024}}
```

This produces three variations with paired (`isl`, `osl`) values. Mechanically, the scenario's `benchmark.dataset:` block deep-merges into the base's dataset; the base has only one dataset (auto-named `default` after normalization), so the merge target is unambiguous and the scenario's `dataset:` override does not need to repeat a `name:` field.

> Multiple datasets per config are not currently supported. `BenchmarkConfig.datasets` is constrained to a single entry — the list shape only exists to share the schema between YAML and the AIPerfSweep CRD. If you need to compare different datasets, run separate sweeps and compare their aggregates.

## Multi-Run Statistics

When a single benchmark run is insufficient to account for system jitter, multi-run mode repeats each benchmark multiple times and computes aggregate statistics with confidence intervals.

### Configuration

```yaml
multi_run:
  num_runs: 5
  cooldown_seconds: 10.0
  confidence_level: 0.95
  set_consistent_seed: true
  disable_warmup_after_first: true
```

### Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `num_runs` | int (1-10) | 1 | Number of benchmark executions. Set >1 to enable statistical reporting. Cap matches `BenchmarkPlan.trials`. |
| `cooldown_seconds` | float (0-86400) | 0.0 | Seconds to wait between runs. Allows GPU thermals and server state to stabilize. Capped at 24h to catch typos like `1e18` at config-load time. |
| `confidence_level` | float (0-1) | 0.95 | Confidence level for interval computation. Common values: 0.90, 0.95, 0.99. |
| `set_consistent_seed` | bool | true | Auto-set `random_seed: 42` if no seed is specified. Ensures identical workloads across runs so variance reflects system noise, not workload differences. |
| `disable_warmup_after_first` | bool | true | Skip warmup phases on runs 2-N. The server is already warm after the first run, so re-running warmup wastes time and can introduce variance. |

### Sample Output with Confidence Intervals

With `num_runs: 5` and `confidence_level: 0.95`, the aggregate report includes:

```json
{
  "metadata": {
    "aggregation_type": "confidence",
    "num_profile_runs": 5,
    "num_successful_runs": 5,
    "confidence_level": 0.95
  },
  "metrics": {
    "request_throughput_avg": {
      "mean": 47.2,
      "std": 1.8,
      "min": 44.9,
      "max": 49.6,
      "cv": 0.038,
      "se": 0.80,
      "ci_low": 44.9,
      "ci_high": 49.4,
      "t_critical": 2.776,
      "unit": "requests/sec"
    },
    "time_to_first_token_p99": {
      "mean": 85.3,
      "std": 4.1,
      "min": 79.8,
      "max": 91.2,
      "cv": 0.048,
      "se": 1.83,
      "ci_low": 80.2,
      "ci_high": 90.4,
      "t_critical": 2.776,
      "unit": "ms"
    }
  }
}
```

A CV below 0.05 (5%) indicates excellent repeatability. The confidence interval tells you the range likely containing the true mean -- if two configurations have non-overlapping intervals, the performance difference is statistically meaningful.

## Sweep + Multi-Run

Sweeps and multi-run combine naturally: each sweep variation is executed `num_runs` times. The total number of benchmark executions is:

```
total_runs = sweep_variations * num_runs
```

### Example: 3 Concurrency Levels x 3 Runs = 9 Total

```yaml
benchmark:
  models:
    - meta-llama/Llama-3.1-8B-Instruct

  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  dataset:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

  phases:
    - name: warmup
      type: concurrency
      exclude_from_results: true
      requests: 100
      concurrency: 8

    - name: profiling
      type: poisson
      duration: 120
      rate: 30
      concurrency: 16 # overridden by sweep
      seamless: true
      grace_period: 30

  artifacts:
    dir: ./artifacts/concurrency_confidence
    summary: [json]
sweep:
  type: grid
  parameters:
    concurrency: [16, 64, 128]

multi_run:
  num_runs: 3
  cooldown_seconds: 5.0
  confidence_level: 0.95
  disable_warmup_after_first: true

random_seed: 42
```

This produces `3 * 3 = 9` total benchmark executions. For each of the 3 concurrency levels, AIPerf runs the benchmark 3 times and computes aggregate statistics. The `disable_warmup_after_first` setting means warmup runs once per variation, not once per repetition.

The output directory structure (default `iteration_order: repeated`, which interleaves trials across all cells) looks like:

```
artifacts/concurrency_confidence/
  profile_runs/
    trial_0001/
      concurrency_16/
      concurrency_64/
      concurrency_128/
    trial_0002/
      concurrency_16/
      concurrency_64/
      concurrency_128/
    trial_0003/
      concurrency_16/
      concurrency_64/
      concurrency_128/
  aggregate/
    concurrency_16/profile_export_aiperf_aggregate.json
    concurrency_64/profile_export_aiperf_aggregate.json
    concurrency_128/profile_export_aiperf_aggregate.json
    sweep_aggregate/profile_export_aiperf_sweep.json
```

Cell directory names come from the swept parameter's leaf segment plus its value (`concurrency_16`, `concurrency_64`, `concurrency_128`). The per-trial inner directory is `trial_NNNN` for sweep + multi-run; the no-sweep multi-run case uses `run_NNNN` instead. If you set `sweep.iteration_order: independent`, the layout flips so each cell is a top-level directory containing its own `profile_runs/trial_NNNN/` and `aggregate/` subtrees.

### Repeated vs Independent — Choosing an Iteration Order

`sweep.iteration_order` controls how trials and variations interleave. Both modes execute the same total runs; they differ in which loop is outer and how artifacts are laid out.

| Aspect | `repeated` (default) | `independent` |
|---|---|---|
| Execution | Trial 1: [v1 -> v2 -> v3], Trial 2: [v1 -> v2 -> v3], ... | All trials at v1, then all trials at v2, ... |
| Dynamic load behavior | Captured | Not captured |
| Isolation | Possible correlation between consecutive variations | Each variation isolated |
| Best for | Real-world dynamic-batching/scaling characterization | Steady-state per-variation comparison |
| Layout | `profile_runs/trial_NNNN/<cell>/` shared parent | `<cell>/profile_runs/trial_NNNN/` per cell |

For a longer treatment with worked decision examples, see [Choosing a sweep mode](#choosing-a-sweep-mode) above.

## Random Seeds and Workload Consistency

Each sweep variation needs a random seed for prompt selection and request ordering. The default behavior derives a unique seed per variation so that different variations don't share artificial correlation:

- Base seed comes from the envelope (`random_seed:` at the top level, or auto-set to 42 by `multi_run.set_consistent_seed`).
- Per-variation seed: `base_seed + variation.index`. With `random_seed: 42` and four variations, seeds are 42, 43, 44, 45.

To force every variation to draw the **same** workload (identical prompts, ordering, and timing pattern across cells), set `sweep.same_seed: true`:

```yaml
random_seed: 42
sweep:
  type: grid
  same_seed: true
  parameters:
    concurrency: [10, 20, 30, 40]
```

Use `same_seed` when you want to isolate the effect of the swept parameter against an identical workload — for example, when debugging why one concurrency level behaves differently. Avoid it for general performance characterization, since correlated workloads make consecutive variations look more similar than they really are.

`sweep.same_seed: true` reuses the envelope's `random_seed` across variations. If `random_seed` is unset, `multi_run.set_consistent_seed` (default True) auto-fills 42, so the practical default is "all variations share seed 42." Set `random_seed` explicitly if you want a different shared seed.

The CLI equivalents for ad-hoc invocations are `--random-seed N` and `--parameter-sweep-same-seed` / `--no-parameter-sweep-same-seed`.

## Cooldown Between Sweep Variations

`sweep.cooldown_seconds` introduces an idle delay between variations, letting GPU thermals, server caches, and KV-cache state settle before the next variation starts. It is independent of `multi_run.cooldown_seconds`, which is the inter-trial cooldown within a single variation.

```yaml
sweep:
  type: grid
  cooldown_seconds: 30.0    # between variations
  parameters:
    concurrency: [10, 20, 30, 40]

multi_run:
  num_runs: 5
  cooldown_seconds: 10.0    # between trials within a variation
```

Typical values: `0` (default — no cooldown, fastest), `10-30s` for basic stabilization, `60s+` for systems with long-memory effects (large KV caches, GPU thermal throttling under sustained load).

In `repeated` mode `sweep.cooldown_seconds` falls between variations within a trial; `multi_run.cooldown_seconds` falls between full sweeps. In `independent` mode they swap roles: `multi_run.cooldown_seconds` separates trials at the same variation; `sweep.cooldown_seconds` separates variations.

## Pareto-Frontier Analysis of Sweep Aggregates

The sweep aggregate JSON includes a post-hoc `pareto_optimal` field that flags which variations are non-dominated on the (throughput-up, p99-TTFT-down) plane. This is **post-hoc analysis of an already-completed sweep** — it does not change which variations were run.

> Distinct from the [Pareto Sweep recipe](../sweeping/search-recipes.md#pareto-sweep), which pre-flattens paired `(isl, osl, concurrency)` cells into a scenarios sweep and post-processes the per-combination metrics into a frontier JSON. The post-hoc analysis below operates on whatever variations the sweep already ran.

A configuration is Pareto optimal if no other variation in the sweep dominates it — that is, no other variation is better or equal on **both** throughput and p99 TTFT. With four concurrency levels (10, 20, 30, 40), it is common for all four to be Pareto optimal because each represents a different point on the throughput-vs-latency trade-off curve.

```json
{
  "best_configurations": {
    "best_throughput": {"parameters": {"concurrency": 40}, "metric": 255.1, "unit": "requests/sec"},
    "best_latency_p99": {"parameters": {"concurrency": 10}, "metric": 125.4, "unit": "ms"}
  },
  "pareto_optimal": [
    {"concurrency": 10},
    {"concurrency": 20},
    {"concurrency": 30},
    {"concurrency": 40}
  ]
}
```

Choose from the frontier based on your service-level objectives: latency-sensitive workloads pick the lowest-latency Pareto point; batch-style workloads pick the highest-throughput Pareto point; balanced services pick a middle point.

For the full sweep-aggregate JSON schema (including `per_combination_metrics`, `failed_runs`, and metadata fields), see the [Sweep Aggregates API Reference](../api/sweep-aggregates.md).

### Interpreting Per-Variation Metrics

For each variation, the aggregate reports `mean`, `std`, `cv`, `min`, `max`, and `ci_low` / `ci_high`. Quick rules of thumb when reading these:

- **CV < 0.10**: results are trustworthy at this variation.
- **CV > 0.20**: high variability — increase `multi_run.num_runs`, add cooldown, or investigate the system at that load.
- **Narrow CI**: high confidence in the reported mean.
- **Wide CI**: more trials needed.

## Environment Variables in Sweeps

YAML configs support `${VAR}` and `${VAR:default}` syntax for environment variable substitution. This is useful for CI pipelines that override sweep base values without editing the YAML file. The example below uses literal defaults so it round-trips against `AIPerfConfig`; in production, replace any of the values with `${VAR:default}` and substitute at deploy time.

```yaml
benchmark:
  endpoint:
    urls:
      - http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true

  models:
    - meta-llama/Llama-3.1-8B-Instruct

  dataset:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

  phases:
    - name: profiling
      type: poisson
      duration: 120
      rate: 30
      concurrency: 32
      grace_period: 30

sweep:
  type: grid
  parameters:
    concurrency: [16, 32, 64, 128]

multi_run:
  num_runs: 3
  cooldown_seconds: 5.0
```

A CI job can then override any default:

```bash
INFERENCE_URL=http://gpu-server:8000/v1/chat/completions \
MODEL_NAME=nvidia/Llama-3.1-Nemotron-70B-Instruct \
NUM_RUNS=5 \
DURATION=300 \
aiperf profile --config sweep_ci.yaml
```

`${VAR}` (without a default) is a required variable -- AIPerf will error if it is not set. `${VAR:default}` falls back to the default value when the variable is unset.

## Best Practices

**Start coarse, then refine.** Begin with a wide grid sweep over 2-3 values per variable (e.g., `concurrency: [5, 10, 20, 40, 80]`) to map the performance envelope. Then define a scenario sweep with hand-picked configurations around the interesting region for detailed comparison.

**Always pair production sweeps with multi-run.** `multi_run.num_runs: 5` quantifies variance and gives you confidence intervals; without it, a single noisy run can mislead capacity-planning decisions.

**Check CV before drawing conclusions.** A variation with CV > 0.20 has too much noise to trust on its own — increase `num_runs`, add cooldown, or investigate the system at that load.

**Use warmup exclusion and `disable_warmup_after_first`.** Define a warmup phase with `exclude_from_results: true` and enable `multi_run.disable_warmup_after_first` (default). The server is then warm without re-warming on every trial.

**Set `random_seed` for reproducibility.** A fixed seed ensures identical prompt selection and request ordering. When `multi_run.set_consistent_seed` is enabled (default), seed 42 is auto-set if you don't supply one.

**Use cooldown between runs.** Even a few seconds of cooldown (`multi_run.cooldown_seconds: 5.0`, `sweep.cooldown_seconds: 5.0`) lets GPU thermals settle and server-side caches reach steady state, reducing correlation between consecutive runs.

**Keep sweep dimensions small.** Two to three variables with three to five values each keeps total runtime manageable. A `3 * 4 * 5 = 60` variation grid with `num_runs: 3` produces 180 benchmark executions — plan your time budget accordingly.

**Choose the right strategy.** Use grid when variables are independent (concurrency vs ISL). Use zip when variables must move together but you don't need named labels (paired ISL/OSL). Use scenarios when variables are coupled and you want hand-labeled comparisons (e.g., chatbot / summarization / long-context profiles).

**Compare apples to apples.** When comparing two infrastructure variants (e.g., two model deployments), use the same sweep values, the same `num_runs`, and the same seed strategy across both runs.

## Troubleshooting

For schema validation errors and config-load failures, see [Sweep & Adaptive Search Errors](../troubleshooting/sweeps.md). At runtime, the most common issues are:

- **High CV at one variation, low elsewhere.** Usually a system-threshold effect — that load level is near a saturation point or hits resource contention. Increase `multi_run.num_runs`, add `sweep.cooldown_seconds`, and inspect server-side metrics at that load.
- **Pareto frontier looks wrong.** If a variation you expected to be dominated appears as Pareto optimal, check its CV: high variance can flip dominance. Lower variance (more trials, more cooldown) and re-check.
- **No clear inflection in the throughput curve.** The sweep range probably doesn't cover saturation. Extend to higher values (e.g., `concurrency: [10, 20, 40, 80, 160, 320]`) until throughput stops scaling.
- **Sweep takes too long.** Reduce `num_runs` to 3, drop `multi_run.cooldown_seconds` and `sweep.cooldown_seconds` to 0, shrink the dataset (`dataset.entries`), or test fewer values initially.
- **Some variations fail.** AIPerf continues with the remaining variations and excludes failed cells from the aggregate. The failure entries appear in `failed_runs` of the sweep aggregate JSON. Investigate whether the failing load level exceeds the server's capacity and adjust `phases.profiling.duration` / endpoint timeouts as needed.

## Programmatic Analysis of Sweep Aggregates

The sweep-aggregate JSON is a stable consumption surface — load it in Python or any other language to drive custom dashboards, regression checks, or visualisations. A minimal example:

```python
import json
import pandas as pd

with open("artifacts/.../aggregate/sweep_aggregate/profile_export_aiperf_sweep.json") as f:
    sweep = json.load(f)

rows = []
for combo in sweep["per_combination_metrics"]:
    rows.append({
        "concurrency": combo["parameters"]["concurrency"],
        "throughput": combo["metrics"]["request_throughput_avg"]["mean"],
        "ttft_p99":    combo["metrics"]["time_to_first_token_p99"]["mean"],
        "throughput_cv": combo["metrics"]["request_throughput_avg"].get("cv", 0.0),
    })

df = pd.DataFrame(rows).sort_values("concurrency")
pareto = {tuple(sorted(p.items())) for p in sweep["pareto_optimal"]}
```

The full schema (every field, every metric stat, the `failed_runs` shape) is documented at [Sweep Aggregates API Reference](../api/sweep-aggregates.md).

## Related Documentation

- [Multi-Run Confidence Reporting](./multi-run-confidence.md) -- Statistical methodology and aggregate output format
- [Sweep Aggregates API Reference](../api/sweep-aggregates.md) -- complete sweep-aggregate JSON schema
- [Pareto Sweep recipe](../sweeping/search-recipes.md#pareto-sweep) -- paired ISL/OSL × concurrency scenarios sweep with a post-process frontier export (distinct from post-hoc Pareto analysis above, which operates on any sweep's output)
- [Warmup Phase Configuration](./warmup.md) -- Warmup phase setup and best practices
- [Sequence Length Distributions](./sequence-distributions.md) -- ISL/OSL distribution configuration
- [Arrival Patterns](./arrival-patterns.md) -- Rate-controlled arrival distributions
- [Sweep & Adaptive Search Errors](../troubleshooting/sweeps.md) -- schema validation and config-load failures
