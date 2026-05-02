<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Search Recipes — Design

Named, plugin-registered presets for adaptive search and curve-characterization sweeps. Lifts the user-facing surface from "write `--search-space` / `--search-metric` / `--search-direction` / `--search-max-iterations` and pick the right combination" to `--search-recipe <name>`. Inspired by NVIDIA Dynamo's `SearchStrategy = rapid|thorough` profiler split (`/tmp/dynamo/components/src/dynamo/profiler/{thorough,rapid}.py`), but generalized so any AIPerf user — not just Dynamo's profiler — can declare a named optimization or characterization workflow.

## Goal

Make AIPerf's adaptive search **discoverable and reusable**:

1. **Named recipes** that bundle search-space + objective + termination + (optional) constraint filters + (optional) post-process step into a single CLI selector.
2. **Plugin-registered** in `plugins.yaml` so external authors can ship recipes without touching AIPerf core, on the same mechanism as endpoints / composers / exporters.
3. **Curve recipes** as first-class output — recipes that sweep a dimension and emit fitted curves (`prefill_curve.json`, `decode_itl_surface.json`) instead of just picking a winner.
4. **SLA-constrained optimization** — "maximize throughput where p95 TTFT < 200ms" expressible declaratively, not with hand-written post-filters.

The shipping surface becomes:

```bash
aiperf profile --model X --url Y --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200
aiperf profile --model X --url Y --search-recipe prefill-ttft-curve --isl-min 256 --isl-max 32768
```

Every recipe expands at the v1→v2 boundary into the same `AdaptiveSearchConfig` + sweep machinery that exists today (`src/aiperf/config/adaptive_search.py`); no orchestrator changes for the BO loop itself.

## Non-goals

- **Multi-objective Bayesian optimization** (true Pareto BO with EHVI/ParEGO). v1 supports single-objective with optional SLA constraints — multi-objective stays on grid + post-process Pareto, which AIPerf already does in `SweepAnalyzer`.
- **Replacing the explicit `--search-*` flags.** Recipes coexist; power users keep the raw flags. `--search-recipe` is mutually exclusive with the explicit flags at the converter, mirroring the existing `--convergence-metric` ↔ adaptive-search rejection in `src/aiperf/config/v1/converter.py::_converter_optionals.build_multi_run`.
- **Inventing a new sweep engine.** Recipes compile to existing `AdaptiveSearchConfig` (BO recipes) or `sweep.variables` blocks (grid recipes). The runtime path is unchanged.
- **Cluster-shape decisions.** Operator-managed sweeps already handle BO under `AIPERF_OPERATOR_MANAGED=1` per the converter rules. Recipes inherit that without modification.
- **Cross-run optimization** (a single recipe scheduling sweeps across multiple AIPerfJobs in series). Each recipe expands into one plan that the existing engine runs end-to-end.

## Motivation

Today's adaptive-search surface (`src/aiperf/config/adaptive_search.py`) is the lowest level the user can target:

```bash
aiperf profile --model X --url Y \
  --search-space "phases.profiling.concurrency:1,1000:int" \
  --search-metric output_token_throughput \
  --search-direction maximize \
  --search-max-iterations 30
```

Three barriers:

1. **Knob discovery.** Users must know which `BenchmarkConfig` dotted-path to sweep. `phases.profiling.concurrency` is grep-able if you already have it; not if you're starting cold.
2. **Metric discovery.** `--search-metric` accepts any `RunResult.summary_metrics` key — no curated list of "metrics worth optimizing for typical workloads."
3. **Workflow discovery.** "I want max throughput under TTFT SLA" is one of ~5 common questions. Each forces the user to construct the BO config from scratch.

Dynamo's profiler shows the value of named recipes: `dynamo.profiler.thorough.run_thorough_sweep()` is a hard-coded "sweep TP × GPU × concurrency, fit TTFT/ITL curves, pick SLA winner" recipe that ships as a first-class strategy. AIPerf has no such layer; every consumer rolls its own. `--search-recipe` closes that gap.

## Design

### Recipe interface

A recipe is a Pydantic-validated plugin class that, given the user's CLI inputs, produces a fully-populated `AdaptiveSearchConfig` (or sweep block) plus optional post-process work. It does **not** run benchmarks itself — it configures them.

New module: `src/aiperf/search_recipes/_base.py`

```python
class SearchRecipeContext(BaseConfig):
    """Inputs available to a recipe at expansion time. Populated from CLI flags."""

    model_config = ConfigDict(extra="forbid")

    user_config: UserConfig = Field(description="Full v1 UserConfig as parsed from CLI.")
    sla_targets: dict[str, float] = Field(
        default_factory=dict,
        description="SLA values from --ttft-sla-ms, --itl-sla-ms, etc. Keyed by metric tag.",
    )
    sweep_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional bound overrides keyed by recipe-defined names "
            "(e.g. {'isl_min': '256', 'isl_max': '32768'} for prefill-ttft-curve)."
        ),
    )


class SearchRecipeOutput(BaseConfig):
    """What a recipe produces. Exactly one of adaptive_search / sweep_variables is set."""

    model_config = ConfigDict(extra="forbid")

    # BO recipes: a fully-built AdaptiveSearchConfig. Reuses the existing
    # SearchSpaceDimension type from src/aiperf/config/adaptive_search.py
    # for its inner search_space list — no new sweep-variable type is introduced.
    adaptive_search: AdaptiveSearchConfig | None = Field(default=None)

    # Grid recipes: a dict matching the existing AIPerfConfig sweep.variables
    # schema (path -> list of values). Inserted directly into the v1->v2 sweep
    # block by the converter; no new sweep-variable type is introduced.
    sweep_variables: dict[str, list[Any]] | None = Field(default=None)

    sla_filters: list[SLAFilter] = Field(default_factory=list)
    post_process: PostProcessSpec | None = Field(default=None)


class SearchRecipe(Protocol):
    """All recipes implement this. Stateless; one instance is reused across runs."""

    name: ClassVar[str]
    description: ClassVar[str]

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput: ...
```

`SLAFilter` and `PostProcessSpec` are new lightweight types (see [SLA filters](#sla-filters) and [Post-process hook](#post-process-hook) below).

### Plugin registration

New `PluginType` enum entry: `SEARCH_RECIPE = "search_recipe"` (added to `tools/generate_plugin_artifacts.py` source; regenerated into `src/aiperf/plugin/enums.pyi` via `make generate-all-plugin-files`).

New section in `src/aiperf/plugin/plugins.yaml`:

```yaml
search_recipe:
  max-throughput-ttft-sla:
    class: aiperf.search_recipes.builtins:MaxThroughputUnderTTFTSLA
    description: |
      Maximize output_token_throughput at the highest concurrency where p95 TTFT
      stays under --ttft-sla-ms. Bayesian-optimized over concurrency.
    metadata:
      algorithm: bayes
      sweep_path: phases.profiling.concurrency

  max-throughput-itl-sla:
    class: aiperf.search_recipes.builtins:MaxThroughputUnderITLSLA
    description: |
      Maximize output_token_throughput where p95 ITL stays under --itl-sla-ms.
      Bayesian-optimized over concurrency.
    metadata:
      algorithm: bayes
      sweep_path: phases.profiling.concurrency

  concurrency-ramp:
    class: aiperf.search_recipes.builtins:ConcurrencyRamp
    description: |
      Linear concurrency ramp; post-process detects the knee where p99 latency
      degrades >--degradation-threshold vs the lowest-concurrency baseline.
      Full grid sweep with post-hoc knee detection — no sweep-level early-stop.
    metadata:
      algorithm: grid
      post_process: degradation_knee_detect

  prefill-ttft-curve:
    class: aiperf.search_recipes.builtins:PrefillTTFTCurve
    description: |
      Sweep ISL across [--isl-min, --isl-max] at fixed concurrency=1, fit
      TTFT(ISL) curve, emit prefill_curve.json. Capacity-planning artifact.
    metadata:
      algorithm: grid
      post_process: ttft_curve_fit

  decode-itl-curve:
    class: aiperf.search_recipes.builtins:DecodeITLCurve
    description: |
      Sweep concurrency × OSL grid, fit ITL surface, emit decode_itl_surface.json.
    metadata:
      algorithm: grid
      post_process: itl_surface_fit
```

Loaded via the existing plugin registry (`plugins.get_class(PluginType.SEARCH_RECIPE, name)`). The four builtins live in `src/aiperf/search_recipes/builtins.py`.

### CLI surface

New flag added to `src/aiperf/cli_commands/profile.py` and threaded through `LoadGeneratorConfig` in `src/aiperf/config/v1/_loadgen.py` — the existing v1 nested class that already houses the `search_*` family (`search_space`, `search_metric`, `search_direction`, `search_max_iterations`, `convergence_metric`):

```
--search-recipe NAME            Named recipe; mutually exclusive with --search-space.
--ttft-sla-ms FLOAT             SLA target consumed by *-ttft-sla recipes.
--itl-sla-ms FLOAT              SLA target consumed by *-itl-sla recipes.
--degradation-threshold FLOAT   Used by concurrency-ramp; default 0.20 (20%).
--isl-min INT --isl-max INT     Used by prefill-ttft-curve.
```

The new flags land on `LoadGeneratorConfig` alongside the existing search flags — per the v1 hard rule "fits an existing nested class? Add it there" in CLAUDE.md. No new top-level `UserConfig` fields, no new nested classes. Recipes declare which SLA / bound flags they consume in their `metadata.consumes:` list (documentation only); the flags are silently ignored when a recipe that doesn't consume them is selected.

`aiperf profile --search-recipe foo` is mutually exclusive with `--search-space` / `--search-metric` / `--search-direction` / `--search-max-iterations`. Rejected at the converter, with an error message naming the recipe and pointing at the explicit-flag escape hatch.

### Converter expansion

New helper in `src/aiperf/config/v1/converter.py`:

```python
def _expand_search_recipe(
    user_config: UserConfig,
) -> tuple[AdaptiveSearchConfig | None, dict[str, list[Any]] | None,
           list[SLAFilter], PostProcessSpec | None]:
    """Resolve --search-recipe NAME to its expanded config artifacts.

    Called from build_multi_run() before _build_adaptive_search runs (see
    "Converter ordering" below). Returns (adaptive_search, sweep_variables,
    sla_filters, post_process); exactly one of adaptive_search / sweep_variables
    is non-None. All four entries are None when no recipe was selected.
    """
```

The converter:

1. Checks `user_config.search_recipe` — None means "no recipe path, behave as today."
2. Looks up the recipe class via the plugin registry.
3. Builds `SearchRecipeContext` from `user_config` + extracted SLA flags.
4. Calls `recipe.expand(ctx)`.
5. Writes the result into the same fields the explicit flags would write to:
   - BO recipes → `multi_run.adaptive_search` (consumed by `MultiRunOrchestrator.execute_adaptive_search`)
   - Grid recipes → `sweep.variables` block (consumed by `expand_sweep` + `MultiRunOrchestrator.execute`)
6. Stores `sla_filters` on the new field `BenchmarkPlan.sla_filters` (described next).
7. Stores `post_process` on `BenchmarkPlan.post_process` (described next).

Mutual-exclusion rules are validated in the converter, not in `UserConfig` — keeps the v1 layer validator-free per the existing rules in CLAUDE.md.

#### Converter ordering

Critical sequencing inside `_converter_optionals.build_multi_run`:

```
1. _expand_search_recipe(user_config)
     ├─ if no recipe selected: returns (None, None, [], None) → fall through to step 3 unchanged
     └─ if recipe selected:
         ├─ writes the expanded BO config into lg.search_space / lg.search_metric
         │   / lg.search_direction / lg.search_max_iterations (BO recipes), OR
         └─ stores the grid dict for later sweep-block injection (grid recipes)
2. Mutual-exclusion check: if both --search-recipe AND any of the explicit
   --search-* flags were user-supplied (tracked via cyclopts' "set by user"
   marker, not just non-default value — recipes legitimately populate the
   same fields), reject with a message naming both.
3. _build_adaptive_search(lg) — existing code path, now sees the recipe-populated
   fields as if the user had typed --search-* themselves.
4. Existing --convergence-metric ↔ adaptive-search rejection runs as today.
5. Sweep-variables injection (grid recipes only) before AIPerfConfig validation.
```

Step 1 must precede step 3 because both write to the same `lg.search_*` fields. Step 2 must precede step 3 to catch user-vs-recipe collisions before the existing path normalizes them away. The cyclopts "set by user" marker (rather than "non-default value") is what makes step 2 honest — a recipe writing the default `max_iterations=30` shouldn't trip mutual exclusion against a user who didn't pass `--search-max-iterations` at all.

### SLA filters

`SLAFilter` is a typed constraint applied during BO scoring and after grid aggregation:

```python
class SLAFilter(BaseConfig):
    model_config = ConfigDict(extra="forbid")

    metric_tag: str = Field(description="RunResult.summary_metrics key, e.g. 'time_to_first_token'.")
    stat: Literal["avg", "p50", "p90", "p95", "p99"] = "p95"
    op: Literal["lt", "le", "gt", "ge"] = Field(description="Constraint direction.")
    threshold: float = Field(description="Boundary value in metric's native units (e.g. ms for TTFT).")
```

**BO integration** (`src/aiperf/orchestrator/search_planner/bayesian.py`):

skopt's `Optimizer.tell(x, y)` only accepts a finite scalar `y`, so SLA constraints can't be expressed as a hard infeasibility flag inside the GP. The honest two-layer approach:

1. **GP scoring (what skopt sees).** Each iteration's score is `base + penalty` where `base` is the objective metric value (sign-flipped for `MAXIMIZE`) and `penalty` is the soft-constraint contribution: `Σ_i w_i * max(0, violation_i)` over each `SLAFilter` (where `violation_i` is the per-constraint shortfall in the metric's native units, normalized by `threshold`). `w_i` is finite — large enough to dominate typical objective values, small enough not to wreck GP variance estimates. Use `100 * abs(threshold)` as the default weight; tune in implementation.
2. **Best-result selection (what users see).** `best_configurations` filters by **strict feasibility first**, then ranks feasible points by `objective_metric` per `objective_direction`. Lexicographic `(feasible_flag, base_value)` ordering — feasible always beats infeasible regardless of base value.

```python
def _score_for_skopt(
    run: RunResult,
    cfg: AdaptiveSearchConfig,
    filters: list[SLAFilter],
) -> float:
    base = _extract_metric(run, cfg.objective_metric, cfg.objective_stat)
    base = -base if cfg.objective_direction is OptimizationDirection.MAXIMIZE else base
    penalty = sum(_soft_violation(run, f) for f in filters)
    return base + penalty


def _is_feasible(run: RunResult, filters: list[SLAFilter]) -> bool:
    return all(_satisfies(run, f) for f in filters)
```

Why not `±inf` / `-1e18`: feeding skopt a constant infeasibility sentinel poisons its GP variance and degenerates acquisition. The soft-penalty approach lets the GP still learn local structure near the constraint boundary while strict filtering at output time guarantees no infeasible "winner" leaks into `best_configurations`. (Multi-objective Pareto BO with proper hard constraints is a separate design — out of scope here.)

**Grid integration** (`src/aiperf/orchestrator/aggregation/sweep.py::SweepAnalyzer.compute()`):

Filters apply to `best_configurations` (already filtered before "best" selection) and to `pareto_optimal` (infeasible points excluded from the frontier). The existing JSON/CSV exporters get a new `sla_constraints` metadata block listing the active filters and how many configurations were rejected.

**`search_history.json`:** BO recipes write the same `search_history.json` the existing adaptive-search path emits — no new file. Its top-level metadata block gains a `recipe` key (`{"recipe": "max-throughput-ttft-sla", "sla_filters": [...], ...}`) so post-hoc readers can tell a recipe-driven run from a hand-rolled `--search-*` run.

### CRD pass-through

`--search-recipe` is a v1-only CLI input. The converter expands it before any v2 / `AIPerfConfig` field is populated — the recipe name itself never reaches `AIPerfConfig`, only the expanded `MultiRunConfig.adaptive_search` (BO recipes) or `sweep.variables` block (grid recipes). Both are existing `AIPerfConfig` fields the CRD generator (`tools/generate_crd.py`) already understands, so **no CRD generator changes are required**.

Consequence: K8s users submitting `AIPerfJob` / `AIPerfSweep` CRs cannot say `spec.benchmark.search_recipe: foo` — the CR schema only knows the expanded forms. They must either (a) use the explicit `--search-*` equivalents in the CR's benchmark block, or (b) use `aiperf kube generate` to expand the recipe CLI-side and submit the resulting CR. Native CR-side recipe expansion (operator pre-expands at submission) is a follow-on design — explicitly out of scope for v1.

### Post-process hook

Curve recipes need to run user-defined Python on the aggregated results. Existing `aggregate_sweep_and_export` in `src/aiperf/_cli_runner_sweep_helpers.py:411` ends after writing standard artifacts; we add one post-process call.

```python
class PostProcessSpec(BaseConfig):
    model_config = ConfigDict(extra="forbid")

    handler: str = Field(description="Plugin name within the post_process plugin category.")
    params: dict[str, Any] = Field(default_factory=dict)
    output_filename: str = Field(description="Filename written under sweep_aggregate/.")
```

A second new plugin type — `SEARCH_RECIPE_POST_PROCESS = "search_recipe_post_process"` — registers post-process callables. Two builtins ship with v1:

- `ttft_curve_fit` — accepts grid output where ISL was swept, fits `TTFT = a * ISL + b` (or quadratic if `r²` is bad), writes `prefill_curve.json` with coefficients + raw points + r².
- `itl_surface_fit` — accepts 2D grid (concurrency × OSL), writes `decode_itl_surface.json` with bilinear-interpolation grid + raw points.

The hook fires after `SweepAnalyzer.compute()` returns and before writing succeeds, and is wired in `aggregate_sweep_and_export`. Failures in post-process are logged + recorded in the sweep_aggregate metadata but do not fail the sweep — the standard artifacts are already written.

## Implementation plan

Phased so each phase ships independently and has its own tests. Plan ceremony stays minimal per repo conventions: one `pytest -n auto tests/unit/` per phase, no subfolder splits.

### Phase 1 — Plugin scaffolding + one recipe (smallest viable)

- Add `SearchRecipe` Protocol + base classes in `src/aiperf/search_recipes/_base.py`.
- Add `SEARCH_RECIPE` to `PluginType` enum source; regenerate `enums.pyi`.
- Wire registry lookup in `aiperf.plugin` (no new code — existing registry handles new types).
- Implement `MaxThroughputUnderTTFTSLA` in `src/aiperf/search_recipes/builtins.py`.
- Add `--search-recipe` + `--ttft-sla-ms` to `LoadGeneratorConfig` in `_loadgen.py` (no validators).
- Implement `_expand_search_recipe` in `converter.py`.
- Add mutual-exclusion check (recipe vs. explicit `--search-*` flags).
- `plugins.yaml` entry for the one recipe.
- Tests:
  - Unit: recipe expansion produces expected `AdaptiveSearchConfig` for a synthetic `UserConfig`.
  - Unit: mutual-exclusion error names the recipe and the conflicting flag.
  - Component-integration: end-to-end `aiperf profile --search-recipe max-throughput-ttft-sla --ttft-sla-ms 200` against the mock server, asserting BO ran and a feasible best emerged.

### Phase 2 — SLA filters in BO scoring

- Implement `SLAFilter` type.
- Extend `AdaptiveSearchConfig` with `sla_filters: list[SLAFilter]` field.
- Wire filter penalty into `BayesianSearchPlanner` (or wherever the objective is computed).
- Update recipe Phase-1 builtin to actually emit a filter (not just pick a metric).
- Tests:
  - Unit: scoring penalty for infeasible point.
  - Component-integration: BO run where mock server returns SLA-violating values for low concurrency — best should land at higher concurrency, not just the highest-throughput infeasible point.

### Phase 3 — Grid recipes + post-process hook

- Add `SEARCH_RECIPE_POST_PROCESS` plugin type.
- Implement `PostProcessSpec`, hook into `aggregate_sweep_and_export`.
- Implement `concurrency-ramp` (full grid sweep; post-process handler computes the degradation knee from aggregated results and emits it as part of the recipe's output). True sweep-level early-stop infrastructure does not exist today (`AdaptiveStrategy` in `orchestrator/strategies.py` only operates intra-variation, not across grid points) — adding it is a separate design and explicitly out of scope.
- Implement `prefill-ttft-curve` + `ttft_curve_fit` post-process.
- Tests:
  - Unit: curve-fit handler produces expected JSON for synthetic grid points.
  - Unit: post-process failure leaves standard artifacts intact and records the error in metadata.
  - Component-integration: ramp recipe emits the correct knee point for a synthetic latency-degradation curve.

### Phase 4 — Remaining builtins + docs

- Implement `MaxThroughputUnderITLSLA`, `DecodeITLCurve` + `itl_surface_fit`.
- New doc: `docs/sweeping/search-recipes.md` (catalog + how to write a recipe).
- Update `docs/sweeping/bayesian-optimization.md` to cross-link.
- Update `llms.txt` and `docs/index.yml` for the new doc.
- Update CLI auto-generated docs via `make generate-all-docs`.

### Phase 5 — Dynamo profiler integration (out-of-tree, optional)

Not in this branch's scope; tracked here for context. Once Phases 1–4 ship, Dynamo's `dynamo.profiler` can replace `get_prefill_aiperf_cmd` / `get_decode_aiperf_cmd` (`/tmp/dynamo/components/src/dynamo/profiler/utils/aiperf.py`) with `aiperf profile --search-recipe {prefill-ttft-curve, decode-itl-curve}` invocations and consume the emitted curve JSON instead of fitting curves themselves.

## Failure modes

| Failure | Behavior |
|---|---|
| Recipe name not in registry | Converter raises with the full list of available recipe names (per existing plugin lookup pattern). |
| Recipe produces invalid `AdaptiveSearchConfig` | `AdaptiveSearchConfig`'s existing Pydantic validators reject; converter wraps with "recipe `<name>` produced invalid config: <pydantic error>". |
| User passes both `--search-recipe` and `--search-space` | Converter rejects with a message naming both flags and the existing escape hatch. |
| Recipe needs streaming but `--streaming` not set | Recipe's `expand()` raises `ValueError` with a message naming the recipe and the missing flag; converter wraps and surfaces. No metadata-driven enforcement (the converter does not introspect plugin metadata for validation in v1). |
| SLA target flag missing for a recipe that needs it | Converter rejects naming the missing flag and the recipe; no silent default. |
| Post-process handler raises | Standard sweep artifacts are still written; error captured in `sweep_aggregate/post_process_errors.json`; sweep exits 0 (post-process is informational, not load-bearing). |
| BO with active SLA filters finds zero feasible points | Best result reported as "no feasible point found"; `best_configurations` array empty in output JSON; sweep exits with non-zero status (caller can decide to treat as failure). |

## Out of scope (explicit)

- True multi-objective BO (separate design — needs a different optimizer than skopt).
- Recipes that span multiple sequential AIPerfJobs / multiple `aiperf profile` invocations.
- Recipe versioning / migration. v1 ships unversioned; once external authors register recipes, a v2 will need it.
- Native CR-side recipe expansion. v1 keeps `--search-recipe` as a CLI-only input; CR users must use the explicit `--search-*` equivalents or pre-expand via `aiperf kube generate` (see [CRD pass-through](#crd-pass-through)).
- UI surfacing of recipes in the dashboard. Possible future phase; not blocked by this design.

## Documentation updates

Per the documentation table in CLAUDE.md:

- `docs/sweeping/search-recipes.md` — new (catalog + author's guide).
- `docs/sweeping/bayesian-optimization.md` — update to cross-link recipes as the recommended entry point.
- `docs/cli-options.md` — auto-regenerated via `make generate-cli-docs`.
- `docs/dev/patterns.md` — add a "Search Recipes" subsection with a recipe-author example.
- `llms.txt` — add the new doc.
- `docs/index.yml` — add the new doc.

No four-file sync (AGENTS.md / CLAUDE.md / `.github/copilot-instructions.md` / `.cursor/rules/python.mdc`) is needed. Those files document coding standards and patterns; they don't carry a per-plugin-type listing today, and a new plugin type doesn't introduce a new pattern that the four mirrors document. Skip unless this design later introduces a recipe-author convention worth codifying there.

## Reference catalog (target shipping set)

| Recipe | Algorithm | What it answers | Inputs | Output |
|---|---|---|---|---|
| `max-throughput-ttft-sla` | BO | "Highest tokens/s where p95 TTFT < X ms" | `--ttft-sla-ms` | `best_configurations` filtered to feasible |
| `max-throughput-itl-sla` | BO | "Highest tokens/s where p95 ITL < X ms" | `--itl-sla-ms` | `best_configurations` filtered to feasible |
| `concurrency-ramp` | Grid + post-process knee detect | "Where does p99 degrade by >N%?" | `--degradation-threshold` | `sweep_aggregate/degradation_knee.json` |
| `prefill-ttft-curve` | Grid + post-process | "TTFT(ISL) curve" | `--isl-min`, `--isl-max` | `sweep_aggregate/prefill_curve.json` |
| `decode-itl-curve` | Grid + post-process | "ITL(concurrency, OSL) surface" | optional bounds | `sweep_aggregate/decode_itl_surface.json` |

Last two replace the equivalent hand-rolled logic in Dynamo's `BuildingCurves` profiling phase.
