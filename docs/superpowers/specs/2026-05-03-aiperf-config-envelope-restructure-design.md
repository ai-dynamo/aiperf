<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf config envelope restructure (Plan A)

## Problem

`AIPerfConfig` is the YAML/CLI loader's primary type. Today it's a flat container that inherits from `BenchmarkConfig` and adds two fields:

```python
class AIPerfConfig(BenchmarkConfig):
    sweep: SweepConfig | None = None
    multi_run: MultiRunConfig = ...
```

`BenchmarkConfig` itself owns everything else: `models`, `endpoint`, `datasets`, `phases`, `artifacts`, `slos`, `tokenizer`, `gpu_telemetry`, `server_metrics`, `runtime`, `logging`, `metrics`, `accuracy`, `random_seed`, `variables`. The result is a single flat shape where sweep machinery (`sweep`, `multi_run`, `variables`, `random_seed`) sits beside benchmark body fields (`endpoint`, `phases`, …) on the same level.

This produces a few accumulating frictions:

- **Sweep expansion has to dance around envelope keys.** `expand_sweep` operates on the flat config dict; for each variation it has to `pop("sweep", None)` and `pop("multi_run", None)` before per-variation `BenchmarkConfig.model_validate`, and the deferred-Jinja path has to validate `AIPerfConfig` separately on rendered variation 0 to recover global cross-field invariants. Both gymnastics exist solely because envelope and body share the same level.
- **Scenario merge logic has to know which keys are sweep machinery vs. swept body.** A scenario `runs[i]` deep-merging into the base must avoid touching `sweep` / `multi_run` etc. Today this is a list of side conditions baked into helpers; with a structural split it becomes "scenario merges into `benchmark:` subtree, never touches envelope."
- **K8s asymmetry.** `AIPerfSweepSpec` already hoists `sweep` / `multi_run` / `convergence` / `failure_policy` to envelope and nests the body under `template.spec.benchmark`. The YAML loader doesn't, so users editing both surfaces switch mental models.
- **No clean answer for "is this field swept across variations or constant across them?"** Every new config field today lives on `BenchmarkConfig` regardless of whether it makes sense to vary per-variation. Adding `variables:` and `random_seed:` to BenchmarkConfig was a workaround.

## Goal

Restructure `AIPerfConfig` into a thin envelope around `BenchmarkConfig`:

```python
class AIPerfConfig(BaseConfig):
    benchmark: BenchmarkConfig
    sweep: SweepConfig | None = None
    multi_run: MultiRunConfig = Field(default_factory=MultiRunConfig)
    variables: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
```

`BenchmarkConfig` keeps all body fields except `variables` and `random_seed`, which move up to envelope (they apply across variations, not within one).

The YAML wire format mirrors the class shape:

```yaml
sweep:
  type: scenarios
  runs: [...]
multi_run:
  num_runs: 5
variables:
  isl: 128
random_seed: 42

benchmark:
  models: [llama]
  endpoint: {urls: [...]}
  datasets: [{name: main, type: synthetic, entries: 200}]
  phases: [...]
  artifacts: {...}
  runtime: {...}
  # ...all body fields
```

Sweep expansion only ever merges into the `benchmark:` subtree (for body overrides) and the `variables:` block (for Jinja overlay). The envelope's other fields are constant across variations, period.

## Non-goals

- **Backward compatibility.** Hard cut. Old flat YAML produces a clear `ConfigurationError` at load time naming the fields to re-indent. No dual-shape acceptance, no deprecation window.
- **Plan B (deferred-Jinja simplification).** That work falls out naturally once the envelope is real, but it's a separate plan executed after this one.
- **Plan C (CRD unification — `AIPerfJob` becomes envelope, stamps `AIPerfRun` children).** Brainstormed separately. Plan A only retypes `AIPerfJob.spec.benchmark: BenchmarkConfig` (mechanical alignment with the new model). The bigger CRD restructure is its own design pass.
- **No new YAML syntax surface beyond the envelope/body split.** Scenario `runs[i]` is a partial envelope (allows `name`, `variables`, `benchmark`); no additional shorthand.
- **No magic `__getattr__` passthrough on `AIPerfConfig`.** Per CLAUDE.md ("conventions are explicit, not tacit"), call sites that read body fields explicitly go through `.benchmark.*`.

## Non-trivial design choices

### Where each field lives, settled (target state)

```
envelope (top-level YAML keys, AIPerfConfig fields):
  benchmark        BenchmarkConfig  — the swept body
  sweep            SweepConfig | None
  multi_run        MultiRunConfig
  variables        dict[str, Any]   — Jinja context
  random_seed      int | None       — base seed; per-variation derived

benchmark body (BenchmarkConfig fields, all under `benchmark:` in YAML):
  models, endpoint, datasets, phases,
  artifacts, slos, tokenizer, gpu_telemetry, server_metrics,
  runtime, logging, metrics, accuracy
```

Today's location: `variables` and `random_seed` live on `BenchmarkConfig` (`src/aiperf/config/config.py:324-345`); under the new shape they move up to envelope because they're cross-variation by nature: `variables` is the Jinja context that scenario `runs[i]` overlay into, and `random_seed` is the base from which per-variation seeds derive (`_apply_sweep_seed_derivation`).

Everything else stays under `benchmark:` regardless of whether it's "commonly swept." Splitting based on access frequency or sweep frequency was considered and rejected — it produces an arbitrary line that has no clean conceptual story for scenario overlay (does a scenario `runs[i].runtime: {workers: 10}` overlay envelope-level `runtime`? Probably not, but the asymmetry is impossible to explain). The whole point of the restructure is mental clarity: envelope = sweep machinery and overlays; body = the swept thing. Anything else at envelope dilutes that contract.

### `MultiRunConfig` ambiguity (in-process vs k8s)

Two `MultiRunConfig` classes exist today, with different field surfaces:

- **In-process** (`src/aiperf/config/_models_benchmark.py:27`) — full surface: `num_runs`, `cooldown_seconds`, `confidence_level`, `set_consistent_seed`, `disable_warmup_after_first`, `convergence_*`, `parameter_sweep_*`, `mode`, `adaptive_search`, `post_process`, `sla_filters`.
- **K8s CRD** (`src/aiperf/kubernetes/sweep_models.py:52`) — leaner surface: `trials`, `cooldown_seconds`, `auto_set_seed`, `disable_warmup_after_first`, `mode`, `adaptive_search`.

Plan A's envelope `multi_run` field uses the **in-process** `MultiRunConfig` (full surface), as today's `AIPerfConfig.multi_run` already does. The K8s `AIPerfSweepSpec.multi_run` keeps its leaner CRD-side type — they're different surfaces serving different consumers (CRD has a curated subset for cluster-side wire shape; in-process has the full surface for local config). Plan A does NOT unify them. If Plan C (the `AIPerfJob`/`AIPerfRun` CRD restructure) chooses to harmonize, it does so as part of that work.

This restructure adds `.benchmark.` to ~335 call sites that read body fields (`config.endpoint` → `config.benchmark.endpoint`, etc.). Empirical breakdown via `grep -rE "(\b[a-z_]*config\b|\bcfg\b)\.(models|endpoint|datasets|phases)\b"`: ~125 in `src/aiperf/`, ~210 in `tests/`. The longest reads (e.g. `config.benchmark.endpoint.streaming` inside hot service paths) get the standard local-alias pattern:

```python
def setup(self, config: AIPerfConfig) -> None:
    bench = config.benchmark
    if bench.endpoint.streaming and bench.phases[0].prefill_concurrency:
        ...
```

Same length as today. The K8s side already lives this way (`spec.benchmark.endpoint`); production code on that side already does `body = spec.benchmark` once. Documenting the alias pattern in `docs/dev/patterns.md` and CLAUDE.md is enough.

### Scenario `runs[i]` shape

Each run dict is a partial envelope. Allowed top-level keys: `name` (variation label, stripped before merge), `variables` (overlay into envelope `variables`), `benchmark` (deep-merge into envelope `benchmark`). Anything else inside a run dict raises:

```
sweep run [i]: unknown field 'X'; allowed: name, variables, benchmark
```

This is more verbose than the pre-restructure `runs: [{phases: {...}}]`, but it removes ambiguity (a run carrying `phases:` today is implicitly merged into the body; explicit `benchmark.phases:` makes the merge target obvious). The `dataset:` shorthand from the prior spec (`docs/superpowers/specs/2026-05-02-scenario-sweep-singular-dataset-design.md`) continues to work, scoped to `runs[i].benchmark.dataset:` / `runs[i].benchmark.datasets:`.

### Grid `sweep.variables` keys

Envelope-rooted dot paths. Allowed prefixes: `benchmark.*`, `variables.*`. Anything else raises:

```
grid sweep variable 'runtime.workers' targets a non-sweepable subtree; allowed prefixes: benchmark.*, variables.*
```

Magic-list flag promotion (`--concurrency 10,20,30`) in the v1→v2 converter emits `benchmark.phases.<name>.concurrency: [...]` — just the prefix changes from today's `phases.<name>.concurrency`.

### Jinja context

`build_template_context` flattens both the envelope `variables:` block AND the benchmark body. Templates reference:

- `{{ isl }}` — envelope variable (top-level alias, as today)
- `{{ variables.isl }}` — explicit envelope path
- `{{ benchmark.endpoint.urls[0] }}` — body path with explicit `benchmark.` prefix
- `{{ phases.profiling.rate }}` — body path WITHOUT prefix (top-level alias, matching today's flat-shape ergonomics)

The "no prefix" alias for body keys preserves user templates from gaining typing burden. Variables and benchmark live in different namespaces; aliases never collide because envelope-level field names (`sweep`, `multi_run`, `variables`, `random_seed`) don't appear inside `benchmark`. If a future field name collides, the explicit `benchmark.X` form always works.

### Loader pipeline

```
parse YAML → env-var sub
  → reject flat shape with migration error if any body key at top level
  → branch on sweep presence:
      no sweep:
        build Jinja context (envelope variables + benchmark body, both visible)
        render Jinja
        validate AIPerfConfig
        plan = single-config plan from config.benchmark
      with sweep:
        validate sweep block via TypeAdapter[SweepConfig]
        for each variation:
          deep-merge runs[i].benchmark into envelope.benchmark
          deep-merge runs[i].variables into envelope.variables
          (or, for grid, _set_nested_value at envelope-rooted paths)
          build Jinja context from merged variation envelope
          render Jinja on the merged variation
          validate BenchmarkConfig on rendered variation.benchmark
        validate envelope (AIPerfConfig minus benchmark) once for global cross-field checks
        plan = N-variation plan
```

The "strip multi_run / sweep from variation_dict" dance disappears entirely. `expand_sweep` only ever sees the body subtree for body merges; `variables:` overlay is a separate dict-merge at envelope level. The "validate AIPerfConfig on rendered variation 0 to recover globals" recovery in the prior deferred-Jinja design becomes unnecessary because envelope-level fields are validated once on the parsed envelope, period.

This simplification IS Plan B's content — it falls out of Plan A naturally. Plan A keeps the existing logic shape (parallels of the prior deferred-render path) so reviewers can see the structural change in isolation. Plan B then deletes the now-vestigial recovery code.

## Migration

### Hard cut, error message at load time

Pre-restructure flat configs trigger:

```
benchmark.yaml: This config uses the pre-restructure flat shape (got top-level
keys: ['models', 'endpoint', 'phases']). Body fields must be nested under a
top-level `benchmark:` key, alongside envelope keys (`sweep`, `multi_run`,
`variables`, `random_seed`). To migrate, indent body fields under `benchmark:`:

  benchmark:
    models: [...]
    endpoint:
      urls: [...]
    phases: [...]
  # sweep / multi_run / variables stay at top level

See docs/tutorials/migrating-config.md for examples, or run:
  uv run python tools/migrate_config_yaml.py path/to/config.yaml --in-place
```

The detector triggers when any of `BODY_KEYS = {models, endpoint, datasets, phases, artifacts, slos, tokenizer, gpu_telemetry, server_metrics, runtime, logging, metrics, accuracy}` appears at the top level. Loader fails fast before any other validation so the error is unambiguous.

`variables` and `random_seed` are intentionally NOT in `BODY_KEYS`: they're envelope-level in the new shape, so a top-level `variables:` or `random_seed:` is valid envelope syntax (no migration needed for those keys specifically). A user with a pre-restructure flat config that has only top-level `variables`/`random_seed` and no body keys could load on the new shape without re-indenting — but in practice every real config has body keys, so they hit the migration error first.

### Migration script

`tools/migrate_config_yaml.py` (single-purpose, ~150 lines):

- **Input:** path to YAML file (or `-` for stdin), `--in-place` to overwrite.
- **Behavior:** parse YAML. Partition top-level keys: envelope (`sweep`, `multi_run`, `variables`, `random_seed`) stay at top, body keys re-indent under `benchmark:`. Preserve key order, preserve comments via `ruamel.yaml`. Inside `sweep.runs[i]`, also rewrite any of BODY_KEYS into `runs[i].benchmark.X`. Inside `sweep.variables` (grid), prefix path keys: `phases.X` → `benchmark.phases.X`, etc.
- **Output:** new YAML on stdout (or in-place edit).
- **Idempotent:** running on an already-migrated config is a no-op.
- **Used twice:** (a) at landing time to bulk-migrate test fixtures and tutorial examples; (b) shipped to users as a one-time guidance tool.

### Test fixtures

~200 YAML strings (rough estimate; counts vary depending on whether nested fixtures and dataset/checkpoint YAML are included) + a smaller number of programmatic `AIPerfConfig(...)` constructions across `tests/`. The migration script handles YAML strings (find triple-quoted YAML literals in `*.py`, parse, rewrite, splice back). Programmatic constructions are manual edits — they need the new constructor shape `AIPerfConfig(benchmark=BenchmarkConfig(...), ...)`.

### Tutorials

All YAML examples in `docs/tutorials/*.md` re-indented via the migration script. New `docs/tutorials/migrating-config.md` with a single end-to-end before/after example.

### K8s CRs on running clusters

Same hard-cut policy. Operator startup detects old-shape AIPerfJob CRs (those with sweep blocks inside `spec.benchmark`, or AIPerfSweep CRs with the old envelope shape) and emits a clear warning + skips them. Cluster-side migration: delete + recreate from regenerated `aiperf kube generate` output.

## Code touchpoints

### Models

- `src/aiperf/config/config.py`
  - `AIPerfConfig` rewritten to envelope shape (drop `BenchmarkConfig` inheritance, gain `benchmark`, `variables`, `random_seed` fields; keep `sweep`, `multi_run`).
  - `BenchmarkConfig` loses `variables` and `random_seed` fields (move to envelope).
  - The class-level docstrings explain the split.
  - **Existing `AIPerfConfig` `model_validator(mode="after")` validators** (`validate_sweep_no_dashboard_ui`, `validate_sweep_same_seed_requires_seed`, `validate_sweep_cooldown_nonneg`, `validate_sweep_flags_require_sweep` — `config.py:503-609`) read body fields like `self.runtime.ui` and `self.multi_run.parameter_sweep_*`. After the split, the body reads become `self.benchmark.runtime.ui`. Validators stay on `AIPerfConfig`.

- `src/aiperf/config/_benchmark_normalizers.py`
  - **No move needed.** The `model_validator(mode="before")` that calls `normalize_benchmark_input` already lives on `BenchmarkConfig` (`src/aiperf/config/config.py:351-359`), not on `AIPerfConfig`. Singular→plural normalizers (`dataset:` → `datasets:`, `model:` → `models:`, flat `phases:` → list) and mutual-exclusivity validators stay where they are. Once `AIPerfConfig` drops the `BenchmarkConfig` inheritance, the normalizer simply runs against `BenchmarkConfig` instances directly (which is its current behavior).

### Loader

- `src/aiperf/config/loader/core.py`
  - `load_config_from_string` adds the flat-shape detector + migration error before any other parsing.
  - Jinja context-builder updated to flatten envelope `variables:` AND benchmark body. Body fields stay aliased at the top level for template ergonomics — this is a **behavior change** in `build_template_context` (`src/aiperf/config/loader/jinja.py:66-96`): today the recursion would produce `benchmark.endpoint.urls[0]` only; under the new shape we also alias body keys at top level (`endpoint.urls[0]`) to preserve user templates from gaining typing burden. The `_flatten_into_context` helper grows a "lift body keys to top level" pass for the `benchmark.*` subtree.

- `src/aiperf/config/loader/plan.py`
  - `build_benchmark_plan(config: AIPerfConfig)` simplifies: `configs = [config.benchmark]` for non-sweep, expand-sweep+per-variation render for sweep. The "strip multi_run from variation_dict" code path goes away.
  - The `_assemble_plan_from_aiperf_config` helper from prior work simplifies — `multi_run` already sits on the typed envelope; no need to model_dump and re-extract.

- `src/aiperf/config/sweep.py`
  - `expand_sweep` operates on the envelope dict. Body merges land in `envelope["benchmark"]`; variable overlays land in `envelope["variables"]`. The dispatch on sweep type stays the same.
  - `_set_nested_value` walks envelope-rooted paths (no behavior change; the path strings just include the `benchmark.` prefix).
  - Grid path-prefix validator added (rejects non-`benchmark.*` / non-`variables.*` keys).
  - Scenario-run validator added (allowed top-level keys: `name`, `variables`, `benchmark`).
  - `_normalize_scenario_dataset_form` from `2026-05-02-scenario-sweep-singular-dataset-design.md` rebases onto `runs[i].benchmark.dataset:` instead of `runs[i].dataset:`.

### CLI converter (v1 → v2)

- `src/aiperf/config/v1/converter.py`
  - Final assembly point wraps body keys under `benchmark:` before `AIPerfConfig.model_validate`.
  - Magic-list promotion (`_promote_magic_lists_to_sweep_block`) emits `benchmark.phases.<name>.<field>` paths.
  - Recipe sweep-variable promotion (`_apply_recipe_sweep_variables`) similarly prefixes.

- `src/aiperf/config/v1/_converter_*.py` sub-converters
  - Internal dict shapes don't change; only the final assembly wraps the body.

### K8s

- `src/aiperf/operator/models.py`
  - `AIPerfJobSpec.benchmark: BenchmarkConfig` (was `AIPerfConfig`). AIPerfJob no longer carries sweep / multi_run / variables / random_seed inside its benchmark — those live at the AIPerfSweep envelope (or in the in-process YAML envelope for local runs). This is a CRD breaking change.
  - The `from_crd_spec` / `model_validate` validators reject sweep blocks inside `spec.benchmark` automatically because `BenchmarkConfig` doesn't have a `sweep` field.

- `src/aiperf/kubernetes/sweep_models.py`
  - `AIPerfSweepSpec.template.spec.benchmark: BenchmarkConfig` (auto-aligns once `AIPerfJobSpec.benchmark` retypes).
  - Add `variables: dict[str, Any] = Field(default_factory=dict)` to `AIPerfSweepSpec` envelope (cluster sweeps drive Jinja vars per-variation).
  - Add `random_seed: int | None = None` to `AIPerfSweepSpec` envelope (base seed for per-variation derivation).
  - The "no sweep allowed inside template.spec.benchmark" CEL rule and Pydantic validator both removed — type system enforces it.

- `src/aiperf/sweep_controller/plan_builder.py`
  - Build envelope dict from CRD spec → `AIPerfConfig.model_validate` → expand sweep over benchmark subtree. Mirrors the in-process YAML loader.

- `src/aiperf/operator/handlers/{create,monitor,...}.py`
  - Mechanical: callers reading `body["spec"]["benchmark"][...]` continue to work because the shape inside `benchmark` is still `BenchmarkConfig`. Field paths inside benchmark don't change.

- `tools/generate_crd.py` regenerates both CRDs from new Pydantic models. CEL rules attached by shape-detector decorators (`_decorate_aiperf_config_node`, `_decorate_endpoint_node`) keep firing — they were structural, not name-bound.

- `src/aiperf/cli_commands/kube/generate.py`
  - AIPerfJob output: stamp body under `spec.benchmark` (no envelope keys).
  - AIPerfSweep output: envelope keys at `spec.{sweep, multi_run, variables, random_seed, ...}`, body under `spec.template.spec.benchmark`.

### Service-side call sites

- ~335 sites across `src/aiperf/` and `tests/` reading `config.X` for body fields gain `.benchmark.` prefix. Mechanical grep+replace.
- Long reads inside hot paths use the local-alias pattern (`bench = config.benchmark`).
- Production count: ~125 sites. Test count: ~210 sites. The migration script handles YAML literals in tests; programmatic constructions are manual edits.

### Tests

- All existing unit / integration / component tests must pass post-restructure (with their fixtures migrated). No new behavior, just shape change.
- New unit tests on the loader:
  - `test_loader_rejects_flat_shape_with_clear_error`
  - `test_envelope_validates_independently_of_sweep_block`
  - `test_grid_sweep_rejects_non_sweepable_path_prefix`
  - `test_scenario_run_rejects_unknown_top_level_key`
  - `test_migration_script_idempotent_on_already_migrated_yaml`
  - `test_migration_script_rewrites_grid_sweep_paths`

### Docs

- `docs/tutorials/migrating-config.md` — new doc, single before/after example, links to error message.
- All `docs/tutorials/*.md` YAML examples re-indented via the migration script.
- `docs/architecture.md` — config plane subsection adds "envelope vs benchmark body" framing.
- `docs/dev/patterns.md` — examples use the new shape; reference local-alias pattern (`bench = config.benchmark`).
- `CLAUDE.md` + `AGENTS.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` — four-file sync. Update "Adding a New Config Field" rules: new fields decide envelope-vs-benchmark based on "does this vary per sweep variation?" Update YAML coding standards examples.
- `docs/cli-options.md`, `docs/environment-variables.md` — auto-regenerated; pick up shape from model docstrings.
- `docs/index.yml` — add `migrating-config.md`.
- `llms.txt` — add the migration doc; update one-line summary if it mentions shape.
- `README.md` — tutorial index gets the migration doc.

## Risk and rollback

- **Mechanical churn risk.** ~344 call-site rewrites + ~200 fixture migrations + tutorial regen is a lot of files touched. Mitigated by: (a) the migration script handles 90% of YAML automatically; (b) typed `AIPerfConfig.benchmark` makes call-site errors caught at type-check / first-test-run time, not at runtime; (c) merging in one branch (no half-migrated state in main).
- **K8s CRD breaking change.** Existing CRs on running clusters become invalid. Mitigated by: documented in release notes; operator startup logs which CRs were rejected and why; `aiperf kube generate` regenerates them from local YAML configs.
- **Verbosity at call sites.** Mitigated by local-alias pattern documented in `docs/dev/patterns.md` and CLAUDE.md.
- **Rollback.** Plan A is one cohesive restructure. Rollback = revert the merge. Pre-restructure tooling is fully recoverable via `git revert`. Post-Plan-A consumers (Plan B, Plan C) build on this; if Plan A reverts, Plan B/C revert with it.
- **Pre-merge verification.** All existing tests pass after migration. New validation tests cover the migration error path and the new sweep-path constraints. The migration script's idempotency test ensures running it twice is safe. `aiperf kube generate` round-trips a YAML envelope through to a CR and back without loss.

## Out of scope (future plans)

- **Plan B — Simplify deferred-Jinja path.** Once the envelope is real, the "validate AIPerfConfig on rendered variation 0" recovery and the strip-multi_run dance in `expand_sweep` become vestigial. Plan B deletes them. Small follow-up; only makes sense after Plan A.
- **Plan C — Unify CRDs as `AIPerfJob` (envelope) + `AIPerfRun` (per-variation child).** Brainstormed separately. Replaces the current AIPerfJob/AIPerfSweep split with a single user-facing `AIPerfJob` parent CRD that always stamps 1+ `AIPerfRun` children. Mirrors Tekton's `PipelineRun → TaskRuns`, Argo's `Workflow → WorkflowSteps`. Plan A only retypes `AIPerfJob.spec.benchmark: BenchmarkConfig` (mechanical alignment); the bigger restructure is deferred.
- **Per-field magic shortcuts at envelope level.** No `__getattr__` passthrough on `AIPerfConfig`. Explicit `.benchmark.*` access only.
- **Backward-compat shim accepting both shapes.** Hard cut, no dual-accept window.
