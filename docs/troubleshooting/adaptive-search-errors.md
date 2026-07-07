# Adaptive Search (Bayesian Optimization) Error Troubleshooting Guide

This guide resolves errors and warnings from AIPerf's adaptive-search
feature — `aiperf profile --search-space ... --search-metric ...
--search-direction ... --search-max-iterations ...` and the Kubernetes
equivalent (`AIPerfSweep` CR with `spec.sweep.type: adaptive_search`). AIPerf
drives an Optuna-based Bayesian-Optimization (BO) outer loop:
`BayesianSearchPlanner` (in
`src/aiperf/orchestrator/search_planner/bayesian.py`) subclasses
`OptunaSearchPlanner` and defaults to BoTorch's Gaussian-process
qLogNEI / qLogNEHVI acquisition. Most errors come from input validation
and a small set of mutual-exclusion guards.

For the deeper "why does BO behave this way," see
[../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).
For cluster-side specifics, see
[../kubernetes/adaptive-search.md](../kubernetes/adaptive-search.md).

---

## 1. Missing Optional Dependency (`botorch`)

### Error message

```text
BoTorch sampler requires the optional `botorch` extra. Install via `uv pip install -e '.[botorch]'`.
```

If you set `--optuna-acquisition` explicitly you may instead see:

```text
--optuna-acquisition requires the optional `botorch` extra.Install via `uv pip install -e '.[botorch]'`.
```

### Cause

`BayesianSearchPlanner.__init__` selects `optuna_sampler="botorch"` and an
appropriate acquisition (`qlognei` single-objective / `qlognehvi`
multi-objective), then delegates to `OptunaSearchPlanner`. The BoTorch
sampler needs the optional `botorch` extra (`optuna-integration`,
`botorch`, `gpytorch`, `torch`), which is NOT pulled in by default — the
optimization stack is heavy, so BO is opt-in to keep the base wheel small.

Note: there is no `bo` extra. The Optuna/BoTorch planner does not use
`scikit-optimize`; use `[botorch]` (or its alias `[optuna]`).

### Fix

```bash
uv pip install -e ".[botorch]"    # editable / dev install
pip install "aiperf[botorch]"     # from PyPI
```

When the `botorch` default is implicit and the optional stack is
unavailable, the planner warns and falls back to the dependency-light
`tpe` (Tree-Parzen) sampler rather than failing hard. Install `[botorch]`
to get the preferred GP path.

---

## 2. Malformed `--search-space` String

### Error message

```text
--search-space '<raw>': expected 'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'.
```

Other shapes from the same parser:

```text
--search-space '<raw>': kind must be 'int' or 'real', got '<kind>'.
--search-space '<raw>': hi (<hi>) must be > lo (<lo>).
--search-space '<raw>': could not parse bound as float (<error>).
```

### Cause

`parse_search_space` in `src/aiperf/orchestrator/search_planner/parsing.py`
implements the grammar `PATH:LO,HI[:KIND]` with `KIND` in `{int, real}`
(default `real`). Common bugs: missing the `:` separator, swapping HI/LO,
non-numeric bound, or a kind outside `int|real`.

### Fix

```bash
# Wrong — no separator
aiperf profile --search-space "phases.profiling.concurrency 1 1000 int" ...
# Wrong — hi <= lo
aiperf profile --search-space "phases.profiling.concurrency:1000,1:int" ...
# Wrong — 'integer' instead of 'int'
aiperf profile --search-space "phases.profiling.concurrency:1,1000:integer" ...

# Correct
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" ...
aiperf profile --search-space "phases.profiling.request_rate:0.5,50.0" ...
```

`--search-space` is repeatable; pass it once per dimension.

---

## 3. Search Path Doesn't Resolve

### Error message

```text
sweep path '<path>': no entry named '<segment>' found (existing: [...]). Add the entry first or fix the typo.
```

### Cause

The dotted path is resolved by `_set_nested_value` in
`src/aiperf/config/sweep/expand.py` (re-exported from
`src/aiperf/config/sweep/config.py`) against the dict form of
`BenchmarkConfig`. Named-list segments (e.g. `phases.profiling.*`) match on
the entry's `name` field. Typos like `phase.profiling.concurrency` (no `s`)
or `phases.profilling.concurrency` (extra `l`) error loudly rather than
silently creating a phantom phase.

### Fix

Common top-level segments: `phases.<name>.<field>` (typically `profiling`
or `warmup`; `<field>` is a `PhaseConfig` scalar like `concurrency`,
`request_rate`, `request_count`), `endpoint.<field>`, `runtime.<field>`.

```bash
# Wrong — typo in 'phases'
aiperf profile --search-space "phase.profiling.concurrency:1,1000:int" ...
# Correct
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" ...
```

---

## 4. `--search-metric` Uses an Aggregator-Suffixed Key

### Cause

The BO objective is the **bare metric tag** (e.g. `output_token_throughput`,
`time_to_first_token`) — not the flattened `_avg` / `_p99` form that
appears in CSV/JSON exports. The statistic is selected separately via
`--search-stat` (one of `avg`, `p50`, `p90`, `p95`, `p99`; default `avg`).
See `_extract_objective_vector` in
`src/aiperf/orchestrator/search_planner/optuna_planner.py` and each
`objectives[*].metric` on `AdaptiveSearchSweep` in
`src/aiperf/config/sweep/config.py` (`objectives` is a list — length-1 for
single-objective BO, length-N for Pareto BO).

### Fix

```bash
# Wrong — _avg suffix is an aggregator key, not a metric tag
aiperf profile --search-metric output_token_throughput_avg ...

# Correct — bare tag, stat is its own flag
aiperf profile --search-metric output_token_throughput --search-stat avg ...
```

See "Objective Semantics" in
[../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md)
for which metric tags are produced and how stats map to JSON fields.

---

## 5. `--search-metric` Names a Metric the Run Doesn't Produce

### Warning message

```text
Search iteration <N> at <values> produced no usable objective; telling Optuna fallback objective=<vec> and continuing.
```

### Cause

`_extract_objective_vector` keeps an objective only if the configured
metric/stat is present and finite across the iteration's trials. If the
metric never appears (e.g. `time_to_first_token` against a non-streaming
endpoint, or `inter_token_latency` for a single-token completion), the
projection returns `None`, the iteration produces no usable objective, and
a per-direction sentinel vector is fed to Optuna — see entry 6 for sentinel
mechanics.

### Fix

Confirm the metric is produced before driving a long BO run:

```bash
aiperf profile --model meta-llama/Llama-3.1-8B-Instruct --concurrency 10 \
  --artifact-dir /tmp/aiperf-probe ...
cat /tmp/aiperf-probe/profile_export_aiperf.json | jq '.summary_metrics | keys'
```

If the desired metric is missing, pick one that is produced or adjust the
run to produce it (e.g. enable streaming for time-to-first-token).

---

## 6. All Trials in an Iteration Failed

### Warning message

Same as entry 5. The corresponding entry in `search_history.json` has
`objective_value: null`.

### Cause

When an iteration yields no usable objective, `_failure_sentinel_vector`
in `src/aiperf/orchestrator/search_planner/optuna_planner.py` synthesizes a
per-objective sentinel vector to `study.tell`:

- With prior successful iterations for that objective: worst-seen value
  plus a 10%-or-1.0-absolute margin, in that objective's *worse* direction.
  Keeps the GP posterior well-posed while telling Optuna this point is
  unambiguously worse than anywhere it has succeeded.
- With no prior data for that objective: `+/- NO_DATA_SENTINEL_LOSS`
  (module constant `NO_DATA_SENTINEL_LOSS = 1.0e6`), sign chosen by the
  objective's direction. BO is essentially random until the first
  successful iteration.

One sentinel per configured objective — multi-objective `tell` expects a
vector, and each direction has its own sense of "worse". This keeps the
ask/tell loop consistent (Optuna cannot accept `None`) and lets the loop
continue rather than aborting.

### Fix

The fallback is a *degraded* mode, not a clean signal — investigate the
failures rather than letting them accumulate:

```bash
ls <artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/
less <artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/aiperf.log
```

Common causes: server timeouts, OOM at high concurrency, endpoint
refusing streaming, metric-collection error. Tighten server availability
or narrow the search-space bounds before re-running. See
[../api/search-history.md](../api/search-history.md) for the
`search_history.json` schema and how to filter sentinel iterations.

---

## 7. Mutual Exclusion: Adaptive Search + an Enumerated Sweep

### What you'll see

A Pydantic validation error, e.g. `Extra inputs are not permitted` on
`sweep.adaptive_search.variables` (or a discriminator error on `sweep.type`)
when a single config would carry both an adaptive-search sweep and an
enumerated (grid/zip) sweep.

### Cause

`sweep.type` is a single **discriminated union**
(`grid | zip | scenarios | adaptive_search | sobol | latin_hypercube`), so a
config holds exactly **one** sweep block. Three inputs all resolve to that
one block:

- `--search-*` flags synthesize `sweep.type: adaptive_search`.
- Magic-list flags (`--concurrency 10,20,30`) synthesize `sweep.type: grid`
  with a `variables:` map (via `_promote_magic_lists_to_sweep_block` in
  `src/aiperf/config/flags/converter.py`).
- An explicit `sweep:` block in a YAML config file.

BO chooses iterations adaptively from continuous ranges, while a grid/zip
sweep enumerates discrete points up front — the two cannot coexist. The
plan-builder (`build_benchmark_plan` in
`src/aiperf/config/loader/plan.py`) documents this: by the time it runs,
`config.sweep` is either absent, a grid/scenario sweep, or an
adaptive_search sweep — never both.

### Fix

Pick one strategy:

```bash
# BO over a continuous range (no magic-lists, no sweep: YAML block)
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Explicit grid sweep instead
aiperf profile --concurrency 10,20,30 ...
```

If your config file carries a leftover `sweep:` block from an earlier
experiment, remove it before adding `--search-*`:

```yaml
# aiperf-config.yaml — drop this block when using BO
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [10, 20, 30]
```

See the "grid vs BO" decision matrix in
[../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).

---

## 8. Mutual Exclusion: `--search-*` + `--convergence-metric`

### Error message

```text
--search-* (Bayesian Optimization) is mutually exclusive with --convergence-metric (trial-level adaptive early-stop). The two operate at different levels (outer-loop vs. inner-trial) and their composition is undefined. Drop one of them.
```

Raised as `TypeError` from the CLI assembly pipeline in
`src/aiperf/config/flags/_converter_optionals.py` when both `--search-space`
(with its companion `--search-*` flags) and `--convergence-metric` are set
on the same `aiperf profile` invocation.

### Cause

`--convergence-metric` is a **trial-level** adaptive stop (stop trials at
a single benchmark point once the metric stabilizes); `--search-*` is an
**outer-loop** adaptive search (choose the next benchmark point). The two
are conceptually orthogonal but their composition is not designed: the BO
orchestrator path silently ignores `convergence_metric`, so the guard
rejects the combination explicitly rather than letting the user believe
trial-level convergence is doing anything during a BO run.

### Fix

Pick one:

```bash
# Outer-loop only
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Trial-level only
aiperf profile --concurrency 100 --convergence-metric output_token_throughput ...
```

---

## 9. `--search-initial-points` >= `--search-max-iterations`

### Error message

```text
n_initial_points (<n>) must be < max_iterations (<m>); otherwise the GP never fits.
```

### Cause

`AdaptiveSearchSweep._check_initial_points_below_max_iterations` in
`src/aiperf/config/sweep/config.py` rejects the configuration for the
`bayesian` / `optuna` planners. BO needs at least one iteration **after**
the Sobol-seeded initial points so the GP can fit and propose informed
points. Defaults: `5` initial points, `30` max iterations.

### Fix

```bash
# Wrong — 10 initial points but only 10 iterations total
aiperf profile --search-max-iterations 10 --search-initial-points 10 ...
# Correct
aiperf profile --search-max-iterations 30 --search-initial-points 5 ...
```

### Why this rule exists

The Sobol-random phase exists to seed the GP with diverse points before
it can fit a meaningful posterior. If the entire iteration budget is
consumed by the random phase, the run is just expensive uniform sampling —
there's no BO-shaped value left to extract. The strict `<` ensures at
least one GP-driven iteration runs. (The gate only applies to `bayesian`
and `optuna` planners; 1-D SLA planners drive their own probe sequence and
ignore `n_initial_points`.)

---

## 10. "In-Process Sweep Rejected Under Operator" — False-Positive Concern

### Concern

You see `_reject_in_process_sweep_under_operator` cited in the docs and
worry that running adaptive search inside an operator-managed pod
(`AIPERF_OPERATOR_MANAGED=1`) will hit:

```text
In-process parameter sweep (<N> variations across <params>) is not supported
in operator-managed runs (AIPERF_OPERATOR_MANAGED=1). Use the AIPerfSweep CRD
(cluster-scope) for cross-job sweeps — see docs/kubernetes/sweeps.md — or
submit one AIPerfJob per variation. To run as a single point benchmark,
drop the comma in --concurrency / other magic-list flags.
```

### Why this is not a problem for adaptive search

`_reject_in_process_sweep_under_operator` (in `src/aiperf/cli_runner/_multi_run.py`)
only fires when `plan.is_sweep` is true. Adaptive-search plans set
`plan.is_adaptive_search` and have a single placeholder variation in
`plan.configs` — `plan.is_sweep` is **false**. Adaptive outer loops run
inside the controller pod itself via `BayesianSearchPlanner`: the
controller proposes each variation one at a time, so the in-process
adaptive path is allowed under the operator and is not blocked here.

The cardinality contract is preserved: one `AIPerfJob` (one controller
pod) per `AIPerfSweep` invocation, with the controller pod synthesizing
each next iteration's config in-process.

### Fix

None needed. If the error fires on what you believe is an adaptive-search
run, the plan was probably built with both magic-list flags AND
`--search-*` — see entry 7.

---


## See also

- [../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md) — Canonical BO reference: algorithm choice, objective semantics, convergence criteria, grid-vs-BO decision matrix.
- [../kubernetes/adaptive-search.md](../kubernetes/adaptive-search.md) — Cluster-side adaptive search via `AIPerfSweep` CR.
- [../api/search-history.md](../api/search-history.md) — `search_history.json` schema and how to inspect per-iteration objective values.
- [./parameter-sweeping-errors.md](./parameter-sweeping-errors.md) — Sibling guide for grid-style parameter sweeps and magic-list flags.
