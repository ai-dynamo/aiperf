# Adaptive Search (Bayesian Optimization) Error Troubleshooting Guide

This guide resolves errors and warnings from AIPerf's adaptive-search
feature — `aiperf profile --search-space ... --search-metric ...
--search-direction ... --search-max-iterations ...` and the Kubernetes
equivalent (`AIPerfSweep` CR with `multi_run.adaptive_search`). AIPerf
wraps `scikit-optimize` to drive a Bayesian-Optimization (BO) outer loop;
most errors come from input validation and a small set of mutual-exclusion
guards.

For the deeper "why does BO behave this way," see
[../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).
For cluster-side specifics, see
[../kubernetes/adaptive-search.md](../kubernetes/adaptive-search.md).

---

## 1. Missing Optional Dependency (`scikit-optimize` / `skopt`)

### Error message

```text
Bayesian Optimization requires the 'bo' extra: `uv pip install -e '.[bo]'`
(or add scikit-optimize to your env). Underlying import error: <ImportError ...>
```

### Cause

`BayesianSearchPlanner.__init__` lazy-imports `skopt`. The dependency
lives in the `[bo]` optional extra (`pyproject.toml`:
`bo = ["scikit-optimize>=0.10"]`) and is NOT pulled in by default — BO is
opt-in to keep the base wheel small.

### Fix

```bash
uv pip install -e ".[bo]"     # editable / dev install
pip install "aiperf[bo]"      # from PyPI
uv add scikit-optimize        # add to an existing env
```

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
sweep path '<path>': no entry named '<segment>' found (existing: [...]).
Add the entry first or fix the typo.
```

### Cause

The dotted path is resolved by `_set_nested_value` in
`src/aiperf/config/sweep.py` against the dict form of `BenchmarkConfig`.
Named-list segments (e.g. `phases.profiling.*`) match on the entry's
`name` field. Typos like `phase.profiling.concurrency` (no `s`) or
`phases.profilling.concurrency` (extra `l`) error loudly rather than
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
See `_extract_trial_objectives` in
`src/aiperf/orchestrator/search_planner/bayesian.py` and
`AdaptiveSearchConfig.objective_metric` in
`src/aiperf/config/adaptive_search.py`.

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
Search iteration <N> at <values> produced no usable objective;
telling skopt fallback loss=<loss> and continuing.
```

### Cause

`_extract_trial_objectives` keeps trials only if
`r.summary_metrics[self._cfg.objective_metric]` is present. If the metric
never appears (e.g. `time_to_first_token` against a non-streaming
endpoint, or `inter_token_latency` for a single-token completion), every
trial is filtered out, the iteration produces no usable objective, and a
sentinel loss is fed to skopt — see entry 6 for sentinel mechanics.

### Fix

Confirm the metric is produced before driving a long BO run:

```bash
aiperf profile --model meta-llama/Llama-3.1-8B-Instruct --concurrency 10 \
  --output-dir /tmp/aiperf-probe ...
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

When every trial fails, `_failed_iteration_loss` in
`src/aiperf/orchestrator/search_planner/bayesian.py` synthesizes a
sentinel loss in skopt's loss space:

- With prior successful iterations: worst-seen loss plus a
  10%-or-1.0-absolute margin. Keeps the GP kernel matrix well-posed
  while telling skopt this point is unambiguously worse than anywhere it
  has succeeded.
- With no prior data: constant `_NO_DATA_SENTINEL_LOSS = 1.0e6`. BO is
  essentially random until the first successful iteration.

This keeps the ask/tell loop consistent with skopt (which cannot accept
`None`) and lets the loop continue rather than aborting.

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

## 7. Mutual Exclusion: `--search-*` + Magic-List Flag

### Error message

```text
sweep block and --search-* flags are mutually exclusive: BO drives
variation choice adaptively, while sweep enumerates them up-front.
Drop the sweep block to use BO, or drop the --search-* flags.
```

### Cause

Magic-list flags (`--concurrency 10,20,30`) are promoted to a top-level
`sweep:` block by `_promote_magic_lists_to_sweep_block` in
`src/aiperf/config/v1/converter.py`. The plan-builder (`build_benchmark_plan`
in `src/aiperf/config/loader/plan.py`) then rejects the combination — BO
chooses iterations adaptively from continuous ranges, while a magic-list
expects you to enumerate the discrete points up front.

### Fix

```bash
# Wrong — magic-list AND --search-space
aiperf profile --concurrency 10,20,30 \
  --search-space "phases.profiling.concurrency:1,1000:int" ...

# Correct — BO over a continuous range
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Correct — explicit grid sweep
aiperf profile --concurrency 10,20,30 ...
```

See the "grid vs BO" decision matrix in
[../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md).

---

## 8. Mutual Exclusion: `--search-*` + Explicit `sweep:` YAML Block

### Error message

```text
sweep block and --search-* flags are mutually exclusive: BO drives
variation choice adaptively, while sweep enumerates them up-front.
Drop the sweep block to use BO, or drop the --search-* flags.
```

### Cause

Same guard as entry 7 (`build_benchmark_plan` in
`src/aiperf/config/loader/plan.py`). Triggered when an `aiperf-config.yaml`
contains a top-level `sweep:` block AND the CLI invocation passes
`--search-*` flags.

### Fix

Drop one or the other. If your config carries a leftover `sweep:` block
from an earlier experiment, remove it before adding `--search-*`:

```yaml
# aiperf-config.yaml — drop this block when using BO
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [10, 20, 30]
```

---

## 9. Mutual Exclusion: `--search-*` + `--convergence-metric`

### Error message

```text
--search-* (Bayesian Optimization) is mutually exclusive with --convergence-metric (trial-level adaptive early-stop). The two operate at different levels (outer-loop vs. inner-trial) and their composition is undefined. Drop one of them.
```

Raised as `TypeError` from the v1->v2 converter in
`src/aiperf/config/v1/_converter_optionals.py::build_multi_run` when both
`--search-space` (with its companion `--search-*` flags) and
`--convergence-metric` are set on the same `aiperf profile` invocation.

### Cause

`--convergence-metric` is a **trial-level** adaptive stop (stop trials at
a single benchmark point once the metric stabilizes); `--search-*` is an
**outer-loop** adaptive search (choose the next benchmark point). The two
are conceptually orthogonal but their composition is not yet well-defined:
which loss to feed skopt under early-stop, and whether to count
convergence-stopped trials toward the per-iteration trial budget, both
need explicit semantics.

### Fix

Pick one until composition is supported:

```bash
# Outer-loop only
aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize --search-max-iterations 30 ...

# Trial-level only
aiperf profile --concurrency 100 --convergence-metric output_token_throughput ...
```

---

## 10. `--search-initial-points` >= `--search-max-iterations`

### Error message

```text
n_initial_points (<n>) must be < max_iterations (<m>); otherwise the GP never fits.
```

### Cause

`AdaptiveSearchConfig._check_initial_points_below_max_iterations` in
`src/aiperf/config/adaptive_search.py` rejects the configuration. BO needs
at least one iteration **after** the random Sobol-seeded initial points so
skopt can fit the GP and propose informed points. Defaults: `5` and `30`.

### Fix

```bash
# Wrong — 10 initial points but only 10 iterations total
aiperf profile --search-max-iterations 10 --search-initial-points 10 ...
# Correct
aiperf profile --search-max-iterations 30 --search-initial-points 5 ...
```

### Why this rule exists

skopt's Sobol-random phase exists to seed the GP with diverse points
before it can fit a meaningful posterior. If the entire iteration budget
is consumed by the random phase, the run is just expensive uniform
sampling — there's no BO-shaped value left to extract. The strict `<`
ensures at least one GP-driven iteration runs.

---

## 11. "In-Process Sweep Rejected Under Operator" — False-Positive Concern

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

`_reject_in_process_sweep_under_operator` (in `src/aiperf/cli_runner.py`)
only fires when `plan.is_sweep` is true. Adaptive-search plans set
`plan.is_adaptive_search` and have a single placeholder variation in
`plan.configs` — `plan.is_sweep` is **false**. The guard's docstring calls
this out:

> Adaptive outer loops, in contrast, run inside the controller pod itself
> via `BayesianSearchPlanner` — the controller proposes each variation
> one at a time, so the in-process adaptive path is allowed under the
> operator and is not blocked here.

The cardinality contract is preserved: one `AIPerfJob` (one controller
pod) per `AIPerfSweep` invocation, with the controller pod synthesizing
each next iteration's config in-process.

### Fix

None needed. If the error fires on what you believe is an adaptive-search
run, the plan was probably built with both magic-list flags AND
`--search-*` — see entry 7.

---

## 12. AIPerfSweep CR with Both `spec.sweep` and `multi_run.adaptive_search`

### Error message

```text
sweep block and --search-* flags are mutually exclusive: BO drives
variation choice adaptively, while sweep enumerates them up-front.
Drop the sweep block to use BO, or drop the --search-* flags.
```

### Cause

The cluster-side controller pod runs `build_plan_from_sweep` in
`src/aiperf/sweep_controller/plan_builder.py`, which constructs a
`BenchmarkPlan` from the AIPerfSweep CR. Plan construction goes through
the same validation as the in-process path, and the mutual-exclusion
guard from `build_benchmark_plan` (`src/aiperf/config/loader/plan.py`)
applies equally: if the CR sets both `spec.sweep` and
`spec.multiRun.adaptiveSearch`, the controller pod fails fast at startup.

### Fix

Drop one in your CR:

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
spec:
  multiRun:
    adaptiveSearch:
      searchSpace:
        - path: phases.profiling.concurrency
          lo: 1
          hi: 1000
          kind: int
      objectiveMetric: output_token_throughput
      objectiveDirection: maximize
      maxIterations: 30
  template:
    spec:
      benchmark: { ... }
```

See [../kubernetes/adaptive-search.md](../kubernetes/adaptive-search.md)
for the full cluster-side guide and CR examples.

---

## See also

- [../sweeping/bayesian-optimization.md](../sweeping/bayesian-optimization.md) — Canonical BO reference: algorithm choice, objective semantics, convergence criteria, grid-vs-BO decision matrix.
- [../kubernetes/adaptive-search.md](../kubernetes/adaptive-search.md) — Cluster-side adaptive search via `AIPerfSweep` CR.
- [../api/search-history.md](../api/search-history.md) — `search_history.json` schema and how to inspect per-iteration objective values.
- [./parameter-sweeping-errors.md](./parameter-sweeping-errors.md) — Sibling guide for grid-style parameter sweeps and magic-list flags.
