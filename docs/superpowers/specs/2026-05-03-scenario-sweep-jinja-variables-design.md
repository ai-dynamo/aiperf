<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sweep over Jinja `variables:` — defer global render until after sweep expansion

## Problem

A user wants every sweep variation to drive a different value into a Jinja-templated field. Today this fails because the YAML loader resolves Jinja **before** sweep expansion, so by the time `expand_sweep` runs the template strings are already collapsed to single base-time values. Two concrete shapes the user wants to write:

```yaml
# (1) Scenario sweep with per-run variables overlay
variables:
  isl: 128
  osl: 128
datasets:
  - name: main
    type: synthetic
    entries: 2000
    prompts:
      isl: "{{ isl }}"
      osl: "{{ osl }}"
sweep:
  type: scenarios
  runs:
    - variables: {isl: 128, osl: 128}
    - variables: {isl: 256, osl: 256}
    - variables: {isl: 512, osl: 1024}
```

```yaml
# (2) Grid sweep keyed at variables.* paths
variables:
  rps: 100
phases:
  profiling:
    type: rate
    rate: "{{ rps }}"
sweep:
  type: grid
  variables:
    "variables.rps": [100, 250, 500]
```

The current pipeline in `load_config_from_string` (`src/aiperf/config/loader/core.py:178-188`) renders Jinja and validates `AIPerfConfig` *before* the sweep block is expanded. `AIPerfConfig` validation strict-types templated fields (e.g. `prompts.isl: SamplingDistribution`), so even if we tried to thread a raw `"{{ isl }}"` through, validation rejects it. By the time `build_benchmark_plan` calls `config.model_dump()` (`loader/plan.py:34`), every `{{ }}` reference has already been resolved with the base value — there is nothing for the per-variation re-render at `loader/plan.py:109-110` to operate on.

## Goal

Reorder the YAML-loader pipeline so Jinja rendering happens **after** sweep expansion, with per-variation context (incorporating that variation's overrides into the `variables:` block). Both shapes above should produce N variations whose templated fields hold the variation-specific values.

Out-of-scope follow-ups (deferred to subsequent specs):

- Programmatic construction of `AIPerfConfig` with un-rendered template strings (rejected at field validation by design — users construct values, not templates).
- The Kubernetes `AIPerfSweep` CRD path (`src/aiperf/sweep_controller/plan_builder.py`). The CRD validates strict types via Pydantic before our code runs; deferred Jinja for that path needs separate work on the CRD shape.
- Deferred Jinja inside `model:` ↔ `models:` merge paths (different shape than `dataset:` per the prior scenario-shorthand spec).

## Non-goals

- **No new sweep `type:`.** Reuses existing `grid` / `scenarios`.
- **No new YAML syntax for "deferred" fields** (no `{{! ... !}}` sigil, no per-field opt-in). The reorder is universal — the only observable effect is that templates now resolve later, which is a strict superset of today's behavior.
- **No new top-level config keys.** Per-scenario `variables:` already validates as a free-form dict on `ScenarioSweep.runs[i]` (typed `dict[str, Any]` on `sweep.py:52`). Grid `variables.<name>` paths already work via `_set_nested_value`'s dot-notation walk.
- **No changes to magic-list semantics.** Magic-list still triggers via `MAGIC_LIST_FIELDS`; if a user wants Jinja-driven sweep over a non-magic-list field, they use the `sweep:` block.
- **No changes to the `AIPerfConfig` field surface.** Field validators stay strict.

## Approach

### Pipeline reorder

Before:

```
parse YAML
  → env-var sub
  → render Jinja (base variables only)
  → AIPerfConfig.model_validate           ← rejects raw template strings
  → model_dump
  → expand_sweep (on rendered dict)        ← {{ }} already gone
  → per-variation re-render                ← no-op in practice
  → BenchmarkConfig.model_validate
```

After:

```
parse YAML
  → env-var sub
  → branch:
      no `sweep:` block → today's path (render → validate AIPerfConfig → build_benchmark_plan)
      has `sweep:`      → deferred path:
          expand_sweep on raw dict          ← raw `{{ }}` survives
          for each variation_dict:
            build_template_context(variation_dict)   ← uses variation's merged variables:
            render_jinja2_templates(variation_dict, context)
            BenchmarkConfig.model_validate(rendered)
          AIPerfConfig.model_validate on rendered variation 0    ← global cross-field checks once
          assemble BenchmarkPlan
```

The deferred path lives in `load_benchmark_plan` (`src/aiperf/config/loader/plan.py:137`). The non-sweep path keeps today's `load_config → build_benchmark_plan(AIPerfConfig)` flow, untouched.

### Why universal reorder rather than scoped opt-in

The reorder is observably equivalent for non-templated fields: `expand_sweep` still receives the same dict shape, `_set_nested_value` writes the same numeric values, per-variation Jinja render is a no-op when there are no `{{ }}` strings, and `BenchmarkConfig.model_validate` sees the same input. The only behavioral change is that template strings now resolve with per-variation context — which is the feature. There is no path where the reorder causes a config that worked before to stop working, so a scoped opt-in adds two-paths-forever maintenance cost with no benefit.

### Per-scenario `variables:` overlay

A scenario run carrying `variables: {isl: 128}` deep-merges into the base `variables:` block via the existing `_deep_merge` (both sides are dicts, so the dict branch fires). After expansion, `build_template_context(variation_dict)` reads the merged `variables:` and exposes the new values to Jinja. No new merge logic is needed; the shape already works.

The variation's `SweepVariation.values` (used for run-labels, seeds, exporter columns) already records the scenario delta; for scenario sweeps this becomes `{"variables": {"isl": 128, ...}}`. Aggregation tooling (`SweepAnalyzer`, `sweep_aggregate/profile_export_aiperf_sweep.json`) reads `values` as opaque dict, so the new shape is forward-compatible.

### Grid sweep at `variables.*` paths

`sweep.variables: { "variables.isl": [128, 256, 512] }` already works syntactically via `_set_nested_value(variant, "variables.isl", value)` — the existing dot-notation walk writes to `variant["variables"]["isl"]`. Today this has no effect because Jinja already ran with the base `variables.isl` value. With the reorder, per-variation Jinja render picks up the new value.

No code change needed for this surface beyond the pipeline reorder. Tutorial doc adds an example under "Sweep + Jinja variables".

(Optional sugar — *not* in this spec, listed under "Future work": shorthand `sweep.variables: { isl: [128, 256, 512] }` that auto-routes to `variables.isl` when the key matches a base `variables:` block name. Adds two-source-of-truth resolution; defer until users actually ask for it.)

### Validation surface

Per option 2 from brainstorm: keep `AIPerfConfig.model_validate` running once for global cross-field invariants (SLO/streaming consistency on `slos`, prefill_concurrency/streaming consistency on phases, etc. — see `validate_config_file` for the documented set), but run it on the **rendered variation 0 dict** rather than on the pre-expand dict.

Rationale:
- Per-variation `BenchmarkConfig.model_validate` already covers field-level checks for every variation.
- Global checks (SLOs, multi_run cross-fields) are not variation-specific in any way the existing validators express; running them once on a rendered representative variation catches the same user mistakes as running them on the un-swept base.
- Choosing variation 0 (rather than re-rendering the base) keeps it simple and avoids an extra render path.

`multi_run`, `slos`, `random_seed`, and other AIPerfConfig-level top-level fields propagate through both paths identically — `expand_sweep` strips `sweep:` per-variation but leaves the rest of the dict intact.

### Code touchpoints

**Modified:**

- `src/aiperf/config/loader/plan.py`
  - Add `_load_plan_with_deferred_jinja(raw_dict, file_path)` implementing the deferred path. Returns `BenchmarkPlan`.
  - Add `_expand_then_render(raw_dict)` helper: drives `expand_sweep` then per-variation `build_template_context` + `render_jinja2_templates`, returning `list[(rendered_dict, SweepVariation)]`.
  - `load_benchmark_plan`: parse YAML + env-var sub → branch on `sweep` presence. Sweep branch calls deferred path. Non-sweep branch keeps today's `load_config → build_benchmark_plan` flow.
  - `_apply_sweep_seed_derivation` continues to apply post-construction (operates on `BenchmarkPlan.configs`, agnostic to render order).

- `src/aiperf/config/loader/core.py`
  - No behavioral change. `load_config` / `load_config_from_string` keep the today-shape pipeline (render → validate AIPerfConfig). Used by callers that need an `AIPerfConfig` directly (CLI introspection, validators, kube-server runtime context).
  - Optionally extract a `_parse_yaml_and_env_sub(content, file_path) -> dict` helper if `load_benchmark_plan` and `load_config_from_string` benefit from sharing pre-Jinja parse logic. Lightweight refactor.

- `tests/unit/config/test_sweep.py` and `tests/unit/config/test_benchmark_plan.py`
  - New cases listed in §Test plan.

- `docs/tutorials/sweeps.md`
  - New subsection "Sweeping over Jinja variables" with both shapes (scenarios + grid). Validate end-to-end via `load_benchmark_plan` before merging — same discipline as the dataset-shorthand spec.

- `docs/tutorials/parameter-sweeping.md`
  - Cross-reference to the new subsection.

**Unchanged:**

- `src/aiperf/config/sweep.py` — `expand_sweep`, `_deep_merge`, `_set_nested_value`, `_normalize_scenario_dataset_form`. All operate on dicts; tolerate `{{ }}` strings as opaque values.
- `src/aiperf/config/_benchmark_normalizers.py` — `normalize_benchmark_input` is shape-only (singular→plural key renames, `_hoist_synthetic_prompt_fields`); value-agnostic, tolerates template strings. Keeps running inside `BenchmarkConfig.model_validate` via the existing `model_validator(mode="before")`.
- `src/aiperf/config/loader/jinja.py` — `build_template_context`, `render_jinja2_templates`, `_resolve_variables_block`. Same code, called later in the pipeline.
- `src/aiperf/sweep_controller/plan_builder.py` — K8s sweep path. Out of scope.
- `build_benchmark_plan(AIPerfConfig)` signature. Continues to handle no-sweep (single variation) and code-path callers that already hold an `AIPerfConfig`.

## Worked example (validates the pipeline)

Input YAML:

```yaml
variables:
  isl: 128
  osl: 128
datasets:
  - name: main
    type: synthetic
    entries: 2000
    prompts:
      isl: "{{ isl }}"
      osl: "{{ osl }}"
sweep:
  type: scenarios
  runs:
    - variables: {isl: 128, osl: 128}
    - variables: {isl: 256, osl: 256}
    - variables: {isl: 512, osl: 1024}
```

After parse + env-var sub, `expand_sweep` operates on the raw dict (templates intact). For variation 0 the merged `variables:` block resolves to `{isl: 128, osl: 128}` (base + scenario, identical). For variation 1 it resolves to `{isl: 256, osl: 256}` because scenario `variables:` deep-merges over base. Per-variation `build_template_context` exposes those values; `render_jinja2_templates` substitutes `{{ isl }}` / `{{ osl }}` to the variation-specific ints; `BenchmarkConfig.model_validate` accepts the now-numeric `prompts.isl` / `prompts.osl`. AIPerfConfig validation runs once on rendered variation 0 to catch any global cross-field issues.

Result: 3 `BenchmarkConfig` variations with `datasets[0].prompts.isl` ∈ {128, 256, 512} and `datasets[0].prompts.osl` ∈ {128, 256, 1024}.

## Test plan

All tests run the full `load_benchmark_plan` (or `load_config_from_string` + `build_benchmark_plan`) path against a YAML string. Assertions hit `plan.configs[i]` directly — integration-level, mirroring the discipline from the dataset-shorthand spec.

In `tests/unit/config/test_sweep.py` (extending the existing test suite):

1. **`test_scenario_variables_overlay_drives_jinja_per_variation`** — Worked example above. Asserts `plan.configs[i].datasets[0].prompts.isl/osl` per variation matches the scenario's `variables:` overlay. Main-path coverage.

2. **`test_scenario_variables_partial_overlay_inherits_base`** — Base `variables: {isl: 128, osl: 64}`, scenario runs only override `isl`. Expects `osl` inherits 64 across all variations; `isl` varies per scenario.

3. **`test_grid_sweep_at_variables_path_drives_jinja`** — Grid form (shape 2 in problem statement). Asserts three variations with `phases.profiling.rate` ∈ {100, 250, 500}.

4. **`test_grid_sweep_mixed_variables_and_field_paths`** — Grid sweep with both `"variables.isl": [128, 256]` AND `"phases.profiling.concurrency": [10, 20]`. Cartesian product → 4 variations, each with the right combination of templated `isl` and direct `concurrency`.

5. **`test_no_sweep_jinja_unchanged`** — YAML with `variables:` and `{{ var }}` references but no `sweep:` block. Goes through the non-sweep path (today's behavior). Assert resolved values match the base `variables:` block. Regression guard for the non-sweep path.

6. **`test_scenario_variables_with_dataset_shorthand_compose`** — Scenario run carrying both `variables:` overlay AND singular `dataset:` shorthand from the prior spec. Both rewrites compose; expects the variation's prompts come from rendered Jinja with the overlaid variables. Composition guard between the two scenario-shorthand features.

7. **`test_aiperfconfig_global_validation_runs_once_on_variation_0`** — YAML with an SLO/streaming inconsistency (e.g. `slos.time_to_first_token` set with `endpoint.streaming: false`). Sweep over `variables.rps`. Expects the AIPerfConfig validation error to surface (regardless of which variation triggers it, since the inconsistency is global). Validates option-2 semantics.

8. **`test_template_string_in_swept_field_renders_with_overlaid_variable`** — `phases.profiling.concurrency: "{{ rate * 2 }}"` with `sweep.variables: { "variables.rate": [10, 20, 30] }`. Expects per-variation `concurrency` ∈ {20, 40, 60}, exercising Jinja arithmetic against overlaid variables.

9. **`test_circular_variables_reference_in_scenario_overlay_errors`** — Scenario overlay introduces a cycle (`variables: {a: "{{ b }}", b: "{{ a }}"}`). Expects `ConfigurationError` from `_resolve_variables_block` with the cycle members listed. Edge-case for the variables-block resolver under per-variation render.

10. **`test_scenario_variables_jinja_strict_undefined_per_variation`** — Scenario references a name not present in either base or scenario `variables:`. Expects `ConfigurationError` (StrictUndefined fires at per-variation render). Verifies error path produces a useful message naming the variation.

`tests/unit/config/test_benchmark_plan.py` gets one regression test confirming `build_benchmark_plan(AIPerfConfig)` still works for the (programmatic, no-template) path.

## Risk and rollback

- **Behavioral equivalence for non-templated configs.** No change in observable behavior because `expand_sweep` and per-variation Jinja render were already running in sequence (just on a dict that had been pre-rendered). Test 5 covers this as a regression guard.
- **Validation order shift.** Global cross-field invariants now surface as a validation error against rendered variation 0 rather than the pre-expand dict. Error messages name field paths; the variation context is invisible to the error since variation-specific overrides at the swept paths don't affect global checks. If a user hits this and wants the pre-expand error, they can drop the sweep and re-run.
- **`AIPerfConfig` callers unaffected.** `load_config` returns `AIPerfConfig` exactly as today, on a rendered base dict. Any caller that wants the pre-expand validation outcome can keep using `load_config`.
- **Rollback.** Revert the deferred branch in `load_benchmark_plan`; non-sweep callers and `build_benchmark_plan(AIPerfConfig)` are untouched. The change is contained to one function plus a helper, fully behind the sweep-block presence check.
- **Pre-merge verification.** Reuse the validation-script discipline from the prior spec: write a small script that loads each tutorial example through `load_benchmark_plan` and asserts the resolved field values per variation. Run before and after — every example should pass post-change.

## Future work (not in this spec)

- **Shorthand grid keys for variables.** `sweep.variables: { isl: [...] }` auto-routing to `variables.isl` when `isl` is a key in the base `variables:` block. Adds resolution-rule complexity; defer until users ask.
- **K8s `AIPerfSweep` deferred Jinja.** Requires changes to the CRD shape and `sweep_controller/plan_builder.py`. The CRD currently strict-validates fields via Pydantic, so template strings get rejected by the apiserver-side validation. Tracked as a separate spec.
- **`model:` ↔ `models:` shorthand parity inside scenarios.** Different shape than `dataset:`; per the prior spec, needs its own rewrite branch.
- **Pareto frontier visualization across sweep variables.** Data already lands in `sweep_aggregate/profile_export_aiperf_sweep.json`; UI is a separate scope.
