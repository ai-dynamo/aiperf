<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweeps Error Troubleshooting Guide

This page covers configuration and runtime errors for both grid-style parameter sweeps and adaptive (Bayesian) search. For algorithm semantics, see [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md). For the YAML reference, see [Parameter Sweeps](../tutorials/sweeps.md).

Each entry quotes the literal error/warning string raised by the code today, with a source-file pointer so you can verify against `main`.

---

## Grid / Zip / Scenarios Errors

### 1. Invalid Concurrency Value

**Error Message (Pydantic, from CLI parse):**
```text
Input should be a valid integer, unable to parse string as an integer
[type=int_parsing, input_value='abc', input_type=str]
```

**Cause:** You provided a non-numeric value for `--concurrency`. `parse_int_or_int_list` calls `int(s)` directly, so the stdlib `ValueError` propagates and Pydantic wraps it as the `int_parsing` error above on the `concurrency` field.

**Where it's raised:** `src/aiperf/config/loader/parsing.py` (parser), `src/aiperf/config/flags/cli_config.py` (field).

**Solution:**
```bash
# Wrong
aiperf --concurrency abc ...

# Correct
aiperf --concurrency 10 ...
```

---

### 2. Invalid Concurrency List

**Error Message (stdlib, surfaced through Pydantic):**
```text
invalid literal for int() with base 10: 'abc'
```

**Cause:** One element of a comma-separated `--concurrency` list is not a valid integer. The list parser does `[int(p) for p in parts]` and the stdlib `ValueError` is raised on the first bad token, with no list context or position information.

**Where it's raised:** `src/aiperf/config/loader/parsing.py`.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,abc,30 ...

# Correct
aiperf --concurrency 10,20,30 ...
```

---

### 3. Negative or Zero Concurrency Values

**Error Message (Pydantic):**
```text
Input should be greater than or equal to 1
[type=greater_than_equal, input_value=-5, input_type=int]
```

**Cause:** A concurrency value is zero or negative. `PhaseConfig.concurrency` is constrained to `ge=1`, so each value is rejected individually with the standard Pydantic `greater_than_equal` error — there is no aggregated, position-aware message.

**Where it's raised:** `src/aiperf/config/phases.py`.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,-5,30 ...
aiperf --concurrency 0,10,20 ...

# Correct
aiperf --concurrency 10,5,30 ...
aiperf --concurrency 1,10,20 ...
```

**Why:** Concurrency represents the number of in-flight requests. Zero or negative is meaningless.

---

### 4. Dashboard UI with Parameter Sweeps or Multi-Run

**Error Message (late-stage, plan validation — covers both sweep and multi-run):**
```text
Dashboard UI is not supported with sweep/multi-run mode.
Please use '--ui simple' or '--ui none' instead.
```

**Where it's raised:** `src/aiperf/cli_runner.py` (`_validate_multi_benchmark_plan`).

**Earlier sweep-only message (fires first when `--ui dashboard` is explicitly set on a sweep config):**
```text
Dashboard UI is incompatible with parameter sweeps; sweep results would
overwrite each other in the live console. Use --ui simple or --ui none
with --concurrency <list> / any sweep configuration.
```

**Where it's raised:** `src/aiperf/config/config.py` (`validate_sweep_no_dashboard_ui`, model-validator). Only triggers when `runtime.ui` is explicitly set by the user and a sweep is configured; multi-run alone does not trip this early check.

**Cause:** The dashboard UI requires exclusive terminal control and would overwrite itself between sequential runs.

**Solution:**
```bash
# Wrong - sweep
aiperf --concurrency 10,20,30 --ui dashboard ...

# Wrong - multi-run
aiperf --num-profile-runs 5 --ui dashboard ...

# Correct
aiperf --concurrency 10,20,30 --ui simple ...
aiperf --num-profile-runs 5    --ui none   ...
```

---

### 5. Invalid Cooldown Duration

**CLI path (Pydantic, fires first):**
```text
Input should be greater than or equal to 0
[type=greater_than_equal, input_value=-10.0, input_type=float]
```

`--parameter-sweep-cooldown-seconds` has `Field(ge=0)`, so any negative value is rejected at config-parse time before the strategy ever sees it.

**Where it's raised:** `src/aiperf/config/flags/cli_config.py`.

**Programmatic path (`FixedTrialsStrategy` direct construction):**
```text
Invalid cooldown_seconds: -10. Must be non-negative.
```

**Where it's raised:** `src/aiperf/orchestrator/strategies.py`.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds -10 ...

# Correct - no cooldown
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 0 ...

# Or - positive cooldown
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 10 ...
```

---

### 6. Empty Sweep-Block Value List

**Error Message (grid sweep):**
```text
grid sweep parameter '<path>': value list must be non-empty.
```

**Error Message (zip sweep):**
```text
zip sweep parameter '<path>': value list must be non-empty.
```

**Cause:** A sweep block (in a YAML config) declared a parameter with an empty `values:` list. This applies to YAML-defined sweeps only; the magic-list CLI path (e.g. `--concurrency 10,20,30`) collapses `--concurrency ""` to `None` and never enters this sweep-block code, so there is no CLI-side trigger for these messages.

**Where it's raised:** `src/aiperf/config/sweep/expand.py` (grid), `src/aiperf/config/sweep/expand.py` (zip).

---

### 7. Insufficient Successful Runs for Aggregation

**Warning Message (sweep mode, per-variation):**
```text
Skipping per-variation aggregate for '<variation_label>': 0 successful runs.
```

**Where it's raised:** `src/aiperf/cli_runner/_sweep_aggregate.py`.

**Note:** Sweep mode does **not** require at least 2 successful runs. `ConfidenceAggregation` has a documented single-run degraded mode (std=0, CI collapsed to mean, `single_run: True` in metadata), and per-variation aggregation explicitly lets single-success cells through — see the comment at `src/aiperf/cli_runner/_sweep_aggregate.py`. Only cells with **zero** successful runs are skipped.

**Related sweep-level warnings:**
- `Skipping per-variation aggregate for '<label>': ConfidenceAggregation raised <exc>` — aggregation crashed for that cell (`cli_runner/_sweep_aggregate.py`).
- `Sweep aggregate skipped: no successful runs across all variations.` — the whole-sweep summary is skipped only when every variation had zero successes (`cli_runner/_sweep_aggregate.py`).

**Warning Message (non-sweep multi-run path):**
```text
Only 1 successful run - cannot compute confidence statistics.
At least 2 successful runs are required.
```

**Where it's raised:** `src/aiperf/cli_runner.py`. This message applies to plain `--num-profile-runs` runs (no sweep), where the "need at least 2" rule does hold.

**Solution:**
```bash
# Increase number of runs
aiperf --concurrency 10,20,30 --num-profile-runs 5 ...

# Or investigate why runs are failing
# Check logs for error messages at the failing variation
```

---

### Silently-Ignored Flag Combinations

Some flag combinations that *look* incorrect do not currently raise. Listing them here so users searching for an error message don't waste time looking:

- **Sweep-only flags used without a sweep.** `--parameter-sweep-mode`, `--parameter-sweep-cooldown-seconds`, and `--parameter-sweep-same-seed` are silently no-ops when no sweep is configured. The sweep-override pathway in `src/aiperf/config/flags/converter.py` only consults these fields when a sweep block is present. No validator exists today.
- **Multi-run-only flags used in single-run mode.** `--confidence-level`, `--profile-run-cooldown-seconds`, and `--profile-run-disable-warmup-after-first` are silently ignored when `--num-profile-runs` is 1. The CLI help text for `--confidence-level` says "Only applies when --num-profile-runs > 1" but this is informational, not enforced (`src/aiperf/config/flags/cli_config.py`). `--set-consistent-seed` also applies in sweep-without-multi-run mode (`src/aiperf/config/config.py`), so it is not strictly multi-run-only.

If you hit one of these and were expecting an error, please file an issue — these are good UX targets for future validators.

---

### Quick Reference: Common Patterns

#### Single Concurrency (No Sweep)
```bash
# Basic
aiperf --concurrency 10 ...

# With multi-run confidence reporting
aiperf --concurrency 10 --num-profile-runs 5 ...
```

#### Parameter Sweep (No Confidence)
```bash
# Basic sweep
aiperf --concurrency 10,20,30 ...

# With cooldown between values
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 10 ...

# With same seed across all values
aiperf --concurrency 10,20,30 --parameter-sweep-same-seed ...
```

#### Parameter Sweep + Confidence Reporting
```bash
# Repeated mode (default) - full sweep N times
aiperf --concurrency 10,20,30 --num-profile-runs 5 ...

# Independent mode - N trials at each value
aiperf --concurrency 10,20,30 --num-profile-runs 5 --parameter-sweep-mode independent ...

# With cooldowns at both levels
aiperf --concurrency 10,20,30 --num-profile-runs 5 \
  --parameter-sweep-cooldown-seconds 10 \
  --profile-run-cooldown-seconds 5 ...
```

---

## Adaptive Search Errors

This section resolves errors and warnings from AIPerf's adaptive-search feature — `aiperf profile --search-space ... --search-metric ... --search-direction ... --search-max-iterations ...`. AIPerf wraps Optuna+BoTorch to drive a Bayesian-Optimization (BO) outer loop; most errors come from input validation and a small set of mutual-exclusion guards.

For the deeper "why does BO behave this way," see [../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).

---

### 1. Missing Optional BoTorch Dependency

**Error message:**

```text
BoTorch sampler requires the optional `botorch` extra. Install via `uv pip install -e '.[botorch]'`.
```

**Cause:**

`OptunaSearchPlanner` uses Optuna core by default, but its implicit preferred sampler is BoTorch. Explicit `--optuna-sampler botorch` or BoTorch-only acquisitions require `optuna-integration`, `botorch>=0.10`, `gpytorch`, and `torch`. When BoTorch is only the implicit default, AIPerf falls back to TPE with a warning if this optional stack is unavailable; explicit BoTorch requests fail instead of silently changing semantics.

**Fix:**

```bash
uv pip install -e ".[botorch]"     # editable / dev install
pip install "aiperf[botorch]"      # from PyPI
```

---

### 2. Malformed `--search-space` String

**Error message:**

```text
--search-space '<raw>': expected 'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'.
```

Other shapes from the same parser:

```text
--search-space '<raw>': kind must be 'int' or 'real', got '<kind>'.
--search-space '<raw>': hi (<hi>) must be > lo (<lo>).
--search-space '<raw>': could not parse bound as float (<error>).
```

**Cause:**

`parse_search_space` in `src/aiperf/orchestrator/search_planner/parsing.py` implements the grammar `PATH:LO,HI[:KIND]` with `KIND` in `{int, real}` (default `real`). Common bugs: missing the `:` separator, swapping HI/LO, non-numeric bound, or a kind outside `int|real`.

**Fix:**

```bash
# Wrong — no separator
aiperf profile --search-space "phases.profiling.concurrency 1 1000 int" ...
# Wrong — hi <= lo
aiperf profile --search-space "phases.profiling.concurrency:1000,1:int" ...
# Wrong — 'integer' instead of 'int'
aiperf profile --search-space "phases.profiling.concurrency:1,1000:integer" ...

# Correct
aiperf profile --search-space "concurrency:1,1000:int" ...
aiperf profile --search-space "phases.profiling.request_rate:0.5,50.0" ...
```

`--search-space` is repeatable; pass it once per dimension.

---

### 3. Search Path Doesn't Resolve

**Error message:**

```text
sweep path '<path>': no entry named '<segment>' found (existing: [...]).
Add the entry first or fix the typo.
```

**Cause:**

The dotted path is resolved by `_set_nested_value` in `src/aiperf/config/sweep/expand.py` against the dict form of `BenchmarkConfig`. Named-list segments (e.g. `phases.profiling.*`) match on the entry's `name` field. Typos like `phase.profiling.concurrency` (no `s`) or `phases.profilling.concurrency` (extra `l`) error loudly rather than silently creating a phantom phase.

**Fix:**

Common top-level segments: `phases.<name>.<field>` (typically `profiling` or `warmup`; `<field>` is a `BasePhaseConfig` scalar like `concurrency`, `request_rate`, `request_count`), `endpoint.<field>`, `runtime.<field>`.

```bash
# Wrong — typo in 'phases'
aiperf profile --search-space "phase.profiling.concurrency:1,1000:int" ...
# Correct
aiperf profile --search-space "concurrency:1,1000:int" ...
```

---

### 4. `--search-metric` Uses an Aggregator-Suffixed Key

**Cause:**

The BO objective is the **bare metric tag** (e.g. `output_token_throughput`, `time_to_first_token`) — not the flattened `_avg` / `_p99` form that appears in CSV/JSON exports. The statistic is selected separately via `--search-stat` (one of `avg`, `p50`, `p90`, `p95`, `p99`; default `avg`). See `_extract_objective_vector` in `src/aiperf/orchestrator/search_planner/optuna_planner.py` and `AdaptiveSearchSweep.objectives[0].metric` in `src/aiperf/config/sweep/config.py`.

**Fix:**

```bash
# Wrong — _avg suffix is an aggregator key, not a metric tag
aiperf profile --search-metric output_token_throughput_avg ...

# Correct — bare tag, stat is its own flag
aiperf profile --search-metric output_token_throughput --search-stat avg ...
```

See "Objective Semantics" in [../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md) for which metric tags are produced and how stats map to JSON fields.

---

### 5. `--search-metric` Names a Metric the Run Doesn't Produce

**Warning message:**

```text
Search iteration <N> at <values> produced no usable objective;
telling Optuna fallback objective=<sentinel-vector> and continuing.
```

**Cause:**

`_extract_objective_vector` in `src/aiperf/orchestrator/search_planner/optuna_planner.py` keeps trials only if `r.summary_metrics[self._cfg.objectives[0].metric]` is present. If the metric never appears (e.g. `time_to_first_token` against a non-streaming endpoint, or `inter_token_latency` for a single-token completion), every trial is filtered out, the iteration produces no usable objective, and the planner feeds Optuna a per-objective sentinel vector — see entry 6 for the mechanics.

**Fix:**

Confirm the metric is produced before driving a long BO run:

```bash
aiperf profile --model meta-llama/Llama-3.1-8B-Instruct --concurrency 10 \
  --artifact-dir /tmp/aiperf-probe ...
# Inspect the records' metric keys (the on-disk export has no top-level
# `summary_metrics` key — that field lives on the planner-side `RunResult`).
cat /tmp/aiperf-probe/profile_export_aiperf.json | jq '.records[0].metrics | keys'
```

If the desired metric is missing, pick one that is produced or adjust the run to produce it (e.g. enable streaming for time-to-first-token).

---

### 6. All Trials in an Iteration Failed

**Warning message:**

Same as entry 5. The corresponding entry in `search_history.json` has `objective_values: null`.

**Cause:**

When every trial fails, the planner builds a per-objective sentinel via `_failure_sentinel_vector` (see `src/aiperf/orchestrator/search_planner/optuna_planner.py`) and feeds it to `study.tell(trial, ...)` so the ask/tell pairing stays consistent. Each sentinel is the worst-of-prior value for that objective plus a 10%-or-1.0 margin in the worse direction; if no prior history exists for that objective, it falls back to `+/- NO_DATA_SENTINEL_LOSS`. The sentinel value IS observed by Optuna's surrogate (the GP sees a strictly-worse-than-anything-seen point so it deprioritizes that region), but the fallback value is NOT persisted to `search_history.json` — `objective_values` is set to `null` for that iteration, matching what [../api/search-history.md](../api/search-history.md) describes.

This keeps the ask/tell loop consistent and lets the loop continue rather than aborting.

**Fix:**

The fallback is a *degraded* mode, not a clean signal — investigate the failures rather than letting them accumulate:

```bash
ls <artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/
less <artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/aiperf.log
```

Common causes: server timeouts, OOM at high concurrency, endpoint refusing streaming, metric-collection error. Tighten server availability or narrow the search-space bounds before re-running. See [../api/search-history.md](../api/search-history.md) for the `search_history.json` schema and how to filter sentinel iterations.

---

### 7. Mutual Exclusion: `--search-*` + Magic-List Flag

**Error message:**

```text
ValidationError: 1 validation error for AIPerfConfig
sweep.adaptive_search.parameters
  Extra inputs are not permitted [type=extra_forbidden, input_value={'phases.profiling.concurrency': [10, 20, 30]}, input_type=dict]
```

**Cause:**

Magic-list flags (`--concurrency 10,20,30`) are promoted to a top-level `sweep:` block by `_promote_magic_lists_to_sweep_block` in `src/aiperf/config/flags/converter.py`. The converter's Pydantic validation of `AdaptiveSearchSweep` (declared with `extra="forbid"` in `src/aiperf/config/sweep/config.py`) then rejects the combination — BO chooses iterations adaptively from continuous ranges, while a magic-list expects you to enumerate the discrete points up front.

**Fix:**

```bash
# Wrong — magic-list AND --search-space
aiperf profile --concurrency 10,20,30 \
  --search-space "concurrency:1,1000:int" ...

# Correct — BO over a continuous range
aiperf profile --search-space "concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Correct — explicit grid sweep
aiperf profile --concurrency 10,20,30 ...
```

See the "grid vs BO" decision matrix in [../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).

---

### 8. Mutual Exclusion: `--search-*` + Explicit `sweep:` YAML Block

**Error message:**

```text
ValidationError: 1 validation error for AIPerfConfig
sweep.adaptive_search.parameters
  Extra inputs are not permitted [type=extra_forbidden, input_value={...}, input_type=dict]
```

**Cause:**

Same guard as entry 7: `AdaptiveSearchSweep`'s `extra="forbid"` validator in `src/aiperf/config/sweep/config.py` rejects the merged dict. Triggered when an `aiperf-config.yaml` contains a top-level `sweep:` block AND the CLI invocation passes `--search-*` flags.

**Fix:**

Drop one or the other. If your config carries a leftover `sweep:` block from an earlier experiment, remove it before adding `--search-*`:

```yaml
# aiperf-config.yaml — drop this block when using BO
sweep:
  type: grid
  parameters:
    concurrency: [10, 20, 30]
```

---

### 9. Mutual Exclusion: `--search-*` + `--convergence-metric`

**Error message:**

```text
--search-* (Bayesian Optimization) is mutually exclusive with --convergence-metric (trial-level adaptive early-stop). The two operate at different levels (outer-loop vs. inner-trial) and their composition is undefined. Drop one of them.
```

Raised as `TypeError` from `_reject_search_plus_convergence` in `src/aiperf/config/flags/_converter_optionals.py` when both `--search-space` (with its companion `--search-*` flags) and `--convergence-metric` are set on the same `aiperf profile` invocation.

**Cause:**

`--convergence-metric` is a **trial-level** adaptive stop (stop trials at a single benchmark point once the metric stabilizes); `--search-*` is an **outer-loop** adaptive search (choose the next benchmark point). The two are conceptually orthogonal but their composition is not yet well-defined: which value to report to the planner under early-stop, and whether to count convergence-stopped trials toward the per-iteration trial budget, both need explicit semantics.

**Fix:**

Pick one until composition is supported:

```bash
# Outer-loop only
aiperf profile --search-space "concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Trial-level only
aiperf profile --concurrency 100 --convergence-metric output_token_throughput ...
```

---

### 10. `--search-initial-points` >= `--search-max-iterations`

**Error message:**

```text
n_initial_points (<n>) must be < max_iterations (<m>); otherwise the GP never fits.
```

**Cause:**

`AdaptiveSearchSweep._check_initial_points_below_max_iterations` in `src/aiperf/config/sweep/config.py` rejects the configuration. BO needs at least one iteration **after** the random Sobol-seeded initial points so the GP can fit and the sampler can propose informed points. Default for `--search-initial-points` is `5`; `--search-max-iterations` has no default and is required whenever `--search-space` is set.

**Fix:**

```bash
# Wrong — 10 initial points but only 10 iterations total
aiperf profile --search-max-iterations 10 --search-initial-points 10 ...
# Correct
aiperf profile --search-max-iterations 30 --search-initial-points 5 ...
```

**Why this rule exists:**

The Sobol-random phase exists to seed the GP with diverse points before it can fit a meaningful posterior. If the entire iteration budget is consumed by the random phase, the run is just expensive uniform sampling — there's no BO-shaped value left to extract. The strict `<` ensures at least one GP-driven iteration runs.

---

## Getting Help

If you encounter an error not covered in this guide:

1. **Check the error message carefully** - Pydantic errors include the field path, the constraint that failed, and the offending input value.

2. **Review the documentation**:
   - [Parameter Sweeping Tutorial](../tutorials/sweeps.md)
   - [Adaptive Search Tutorial](../tutorials/adaptive-search.md)
   - [CLI Options Reference](../cli-options.md)

3. **Report a bug** if:
   - The error message is unclear or unhelpful
   - You believe the error is incorrect
   - The suggested fix doesn't work

Include in your bug report:
- Full command line you ran
- Complete error message
- AIPerf version (`aiperf --version`)
- What you expected to happen

---

## See also

- [Bayesian-Optimization Outer Loop](../sweeping/bayesian-optimization.md) — Canonical BO reference: algorithm choice, objective semantics, convergence criteria, grid-vs-BO decision matrix.
- [Parameter Sweeps](../tutorials/sweeps.md) — Parameter sweeping tutorial and YAML reference.
- [Adaptive Search](../tutorials/adaptive-search.md) — Adaptive search tutorial.
- [Search History API Reference](../api/search-history.md) — `search_history.json` schema and how to inspect per-iteration objective values.
