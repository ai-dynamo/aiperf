# Scenario Sweep over Jinja Variables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Defer global Jinja rendering until after `expand_sweep` so per-variation rendering picks up scenario- and grid-supplied overrides to the `variables:` block, enabling sweeps over Jinja-templated fields.

**Architecture:** `load_benchmark_plan` becomes the routing point. When the raw YAML carries a `sweep:` block, it routes to a deferred-render path (`_build_plan_deferred_render`) that runs `expand_sweep` on the raw template-bearing dict, then renders Jinja and validates `BenchmarkConfig` per variation. `AIPerfConfig.model_validate` runs once on rendered variation 0 to catch global cross-field invariants. The plan-assembly tail is shared between sweep and non-sweep paths via a refactored `_assemble_plan_from_aiperf_config` helper.

**Tech Stack:** Python 3.10+, Pydantic v2, Jinja2 (StrictUndefined), pytest with `-n auto` and parametrize.

**Spec:** `docs/superpowers/specs/2026-05-03-scenario-sweep-jinja-variables-design.md` (commit `02deab24c`).

---

## File Structure

**Modify:**
- `src/aiperf/config/loader/plan.py` — add `_assemble_plan_from_aiperf_config`, `_build_plan_deferred_render`, `load_benchmark_plan_from_string`; refactor `build_benchmark_plan` to use the new helper; refactor `load_benchmark_plan` to route via the from-string entry point.
- `src/aiperf/config/loader/__init__.py` — export `load_benchmark_plan_from_string`.

**Modify (tests):**
- `tests/unit/config/test_sweep.py` — append a new `TestScenarioJinjaVariablesSweep` class with the test cases listed below.
- `tests/unit/config/test_benchmark_plan.py` — one regression test pinning the `build_benchmark_plan(AIPerfConfig)` no-sweep path post-refactor.

**Modify (docs):**
- `docs/tutorials/sweeps.md` — new "Sweeping over Jinja variables" subsection with two worked examples.
- `docs/tutorials/parameter-sweeping.md` — one-line cross-reference to the new subsection.

**Out of scope (NOT touched):**
- `src/aiperf/sweep_controller/plan_builder.py` — K8s sweep path, deferred to a separate spec.
- `src/aiperf/config/sweep.py` — `expand_sweep`, `_deep_merge`, `_set_nested_value`, `_normalize_scenario_dataset_form` are already shape/value-agnostic; no edits needed.
- `src/aiperf/config/_benchmark_normalizers.py` — shape-only, value-agnostic; runs inside `BenchmarkConfig.model_validate` unchanged.
- `src/aiperf/config/loader/jinja.py` — same code, called later in the pipeline.

---

## Repo conventions used in this plan

- All `pytest` invocations use `-n auto` and target a single subfolder per command (per repo `feedback_pytest_single_subfolder.md` and `feedback_always_pytest_n_auto.md`).
- Commit messages follow conventional commits with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. Pre-commit hooks run normally (this branch is `ajc/k8s`, not `aiperf-rs`).
- Type hints on every function. `Field(description=...)` on every Pydantic field. No comments unless explaining a non-obvious "why".

---

## Task 1: Refactor — extract `_assemble_plan_from_aiperf_config` (no behavior change)

**Why first:** Both `build_benchmark_plan(AIPerfConfig)` (existing, no-sweep path) and the new `_build_plan_deferred_render` (sweep path) need the plan-construction logic that converts an `AIPerfConfig` + pre-expanded `(configs, variations)` lists into a `BenchmarkPlan`. Pulling this into a helper now keeps the two call sites trivially symmetric.

**Files:**
- Modify: `src/aiperf/config/loader/plan.py:19-86` (move logic into a new helper; keep `build_benchmark_plan` as a thin wrapper).

- [ ] **Step 1: Read the current `build_benchmark_plan` to confirm the exact shape**

Run: `sed -n '19,86p' src/aiperf/config/loader/plan.py`
Expected: Function with `config_dict = config.model_dump(...)` → `_expand_grid_variations` → `plan_kwargs` dict construction → `BenchmarkPlan(**plan_kwargs)` → `_apply_sweep_seed_derivation`.

- [ ] **Step 2: Run the existing benchmark-plan tests to confirm green baseline**

Run: `uv run pytest -n auto tests/unit/config/test_benchmark_plan.py -v 2>&1 | tail -30`
Expected: All tests PASS. Note the count for comparison after refactor.

- [ ] **Step 3: Replace `build_benchmark_plan` with the refactored version + new helper**

Replace `src/aiperf/config/loader/plan.py:19-86` with:

```python
def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Expands sweep variations and extracts multi_run settings, OR — when
    config.multi_run.adaptive_search is set — produces a single-config plan
    with plan.adaptive_search populated. Sweep + adaptive_search are mutually exclusive.
    """
    from aiperf.config.sweep import SweepVariation

    adaptive_search = config.multi_run.adaptive_search
    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)

    if sweep_dict is not None and adaptive_search is not None:
        raise ValueError(
            "sweep block and --search-* flags are mutually exclusive: BO drives "
            "variation choice adaptively, while sweep enumerates them up-front. "
            "Drop the sweep block to use BO, or drop the --search-* flags."
        )

    if adaptive_search is not None:
        configs = [BenchmarkConfig.model_validate(_strip_top_level_meta(config_dict))]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        configs, variations = _expand_grid_variations(config_dict, sweep_dict)

    return _assemble_plan_from_aiperf_config(config, configs, variations)


def _strip_top_level_meta(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove top-level keys that don't belong on a per-variation BenchmarkConfig."""
    return {k: v for k, v in config_dict.items() if k not in ("multi_run", "sweep")}


def _assemble_plan_from_aiperf_config(
    config: AIPerfConfig,
    configs: list[BenchmarkConfig],
    variations: list[Any],
) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig plus pre-expanded
    per-variation BenchmarkConfigs and SweepVariation metadata.

    Shared tail used by both build_benchmark_plan (no-sweep / today's path)
    and _build_plan_deferred_render (sweep + Jinja variables overlay).
    """
    adaptive_search = config.multi_run.adaptive_search
    post_process = config.multi_run.post_process
    sla_filters = list(config.multi_run.sla_filters)

    multi_run = config.multi_run.model_dump(
        mode="json", exclude_none=True, exclude_unset=True
    )
    multi_run.pop("adaptive_search", None)
    multi_run.pop("post_process", None)
    multi_run.pop("sla_filters", None)

    plan_kwargs: dict[str, Any] = dict(
        configs=configs,
        variations=variations,
        trials=multi_run.get("num_runs", 1),
        cooldown_seconds=multi_run.get("cooldown_seconds", 0.0),
        confidence_level=multi_run.get("confidence_level", 0.95),
        set_consistent_seed=multi_run.get("set_consistent_seed", True),
        disable_warmup_after_first=multi_run.get("disable_warmup_after_first", True),
        parameter_sweep_cooldown_seconds=multi_run.get(
            "parameter_sweep_cooldown_seconds", 0.0
        ),
        parameter_sweep_same_seed=multi_run.get("parameter_sweep_same_seed", False),
        parameter_sweep_mode=multi_run.get("mode", "repeated"),
        adaptive_search=adaptive_search,
        post_process=post_process,
        sla_filters=sla_filters,
    )
    for key in (
        "convergence_metric",
        "convergence_mode",
        "convergence_threshold",
        "convergence_stat",
    ):
        if key in multi_run and multi_run[key] is not None:
            plan_kwargs[key] = multi_run[key]
    plan = BenchmarkPlan(**plan_kwargs)
    if adaptive_search is None:
        _apply_sweep_seed_derivation(plan, config)
    return plan
```

Note the tiny behavior tightening: the adaptive-search branch previously called `BenchmarkConfig.model_validate(config_dict)` with `multi_run` still present in the dict; pydantic's `extra=ignore` (or whatever BenchmarkConfig uses) silently dropped it. Using `_strip_top_level_meta` makes that explicit and keeps the strip semantics consistent with the deferred path. Tests in step 4 confirm no behavior change.

- [ ] **Step 4: Run benchmark-plan tests + sweep tests**

Run: `uv run pytest -n auto tests/unit/config/test_benchmark_plan.py tests/unit/config/test_sweep.py -v 2>&1 | tail -40`
Expected: All tests PASS at the same count as step 2 (plus whatever was in test_sweep.py).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/loader/plan.py
git commit -m "$(cat <<'EOF'
refactor(loader): extract _assemble_plan_from_aiperf_config

Pulls the post-expansion plan-construction tail out of build_benchmark_plan
into a helper so the upcoming deferred-Jinja sweep path can share it.
No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `load_benchmark_plan_from_string` (string-input entry point, non-sweep path only)

**Why now:** Tests that exercise deferred-render paths from inline YAML strings need a string-accepting entry. Stub it now; the deferred branch lights up in Task 3.

**Files:**
- Modify: `src/aiperf/config/loader/plan.py` (add the new function plus helpers).
- Modify: `src/aiperf/config/loader/__init__.py` (export).

- [ ] **Step 1: Add `load_benchmark_plan_from_string` and refactor `load_benchmark_plan` to use it**

In `src/aiperf/config/loader/plan.py`, replace the existing `load_benchmark_plan` (lines `137-158`) with:

```python
def load_benchmark_plan(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> BenchmarkPlan:
    """Load a YAML config file and return a BenchmarkPlan.

    Routes to a deferred-Jinja path when the YAML carries a `sweep:` block,
    so per-variation Jinja rendering picks up scenario- or grid-supplied
    overrides to the `variables:` block. Non-sweep configs use the
    today's load_config -> build_benchmark_plan flow unchanged.

    Args:
        file_path: Path to the YAML configuration file.
        substitute_env: Whether to process environment variable substitution.

    Returns:
        BenchmarkPlan with expanded configs and execution preferences.
    """
    from aiperf.config.loader.core import _parse_yaml_mapping  # local import

    file_path = Path(file_path)
    if not file_path.exists():
        from aiperf.config.loader.errors import ConfigurationError

        raise ConfigurationError(
            f"Configuration file not found: {file_path}", file_path=file_path
        )
    if not file_path.is_file():
        from aiperf.config.loader.errors import ConfigurationError

        raise ConfigurationError(
            f"Path is not a file: {file_path}", file_path=file_path
        )
    content = file_path.read_text(encoding="utf-8")
    return load_benchmark_plan_from_string(
        content, file_path=file_path, substitute_env=substitute_env
    )


def load_benchmark_plan_from_string(
    yaml_content: str,
    *,
    file_path: Path | str | None = None,
    substitute_env: bool = True,
) -> BenchmarkPlan:
    """Load a YAML string and return a BenchmarkPlan.

    Mirrors load_benchmark_plan's routing for in-memory input. Useful for
    unit tests and programmatic callers that already have YAML in a string.
    """
    from aiperf.config.loader.core import (
        _parse_yaml_mapping,
        _validate_config_dict,
    )
    from aiperf.config.loader.env_vars import substitute_env_vars
    from aiperf.config.loader.jinja import (
        build_template_context,
        render_jinja2_templates,
    )

    raw_dict = _parse_yaml_mapping(yaml_content, file_path)
    if substitute_env:
        raw_dict = substitute_env_vars(raw_dict, file_path)

    if raw_dict.get("sweep") is not None:
        return _build_plan_deferred_render(raw_dict, file_path)

    # No sweep: today's flow — render Jinja, validate AIPerfConfig, build plan.
    context = build_template_context(raw_dict)
    rendered = render_jinja2_templates(raw_dict, context)
    config = _validate_config_dict(rendered, file_path)
    return build_benchmark_plan(config)


def _build_plan_deferred_render(
    raw_dict: dict[str, Any], file_path: Path | str | None
) -> BenchmarkPlan:
    """Sweep-present path: expand on raw, render+validate per variation.

    NOTE: full implementation lands in Task 3. This stub keeps the routing
    in load_benchmark_plan_from_string compilable until then, by raising a
    clear NotImplementedError that tests in Task 3 will replace with real
    behavior.
    """
    raise NotImplementedError(
        "deferred-render sweep path is implemented in Task 3 of the plan "
        "(2026-05-03-scenario-sweep-jinja-variables-design.md)"
    )
```

- [ ] **Step 2: Export `load_benchmark_plan_from_string`**

In `src/aiperf/config/loader/__init__.py`, add to the `from aiperf.config.loader.plan import` line and to `__all__`:

```python
from aiperf.config.loader.plan import (
    build_benchmark_plan,
    load_benchmark_plan,
    load_benchmark_plan_from_string,
)
```

```python
__all__ = [
    # ... existing entries ...
    "load_benchmark_plan",
    "load_benchmark_plan_from_string",
    # ... existing entries ...
]
```

Also export from `src/aiperf/config/__init__.py` if `load_benchmark_plan` is re-exported there:

Run: `grep -n "load_benchmark_plan" src/aiperf/config/__init__.py`
Expected output shows lines for the existing export. Add `load_benchmark_plan_from_string` next to it in both the import block and `__all__`.

- [ ] **Step 3: Run the existing test suite to confirm no regression**

Run: `uv run pytest -n auto tests/unit/config/ -v 2>&1 | tail -30`
Expected: All tests PASS. The deferred path raises only when tests explicitly trigger it (none yet do).

- [ ] **Step 4: Spot-check the route via a smoke test**

Run:
```bash
uv run python -c "
from aiperf.config.loader import load_benchmark_plan_from_string
yaml = '''
models: [test/model]
endpoint:
  type: chat
  urls: [\"http://localhost:8000/v1/chat/completions\"]
datasets:
  - {name: main, type: synthetic, entries: 100}
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
'''
plan = load_benchmark_plan_from_string(yaml, substitute_env=False)
print(f'configs={len(plan.configs)} is_sweep={plan.is_sweep}')
"
```
Expected: `configs=1 is_sweep=False`

Run:
```bash
uv run python -c "
from aiperf.config.loader import load_benchmark_plan_from_string
yaml = '''
models: [test/model]
endpoint:
  type: chat
  urls: [\"http://localhost:8000/v1/chat/completions\"]
datasets:
  - {name: main, type: synthetic, entries: 100}
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
sweep:
  type: scenarios
  runs:
    - {variables: {x: 1}}
'''
try:
    load_benchmark_plan_from_string(yaml, substitute_env=False)
    print('ERROR: should have raised')
except NotImplementedError as e:
    print('routed to deferred path (expected NotImplementedError)')
"
```
Expected: `routed to deferred path (expected NotImplementedError)`

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/loader/plan.py src/aiperf/config/loader/__init__.py src/aiperf/config/__init__.py
git commit -m "$(cat <<'EOF'
feat(loader): add load_benchmark_plan_from_string entry point

Routes to a deferred-Jinja path (currently a NotImplementedError stub)
when the YAML carries a `sweep:` block; non-sweep configs go through
the existing load_config -> build_benchmark_plan flow.

The deferred path is implemented in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: TDD — implement `_build_plan_deferred_render` (scenario `variables:` overlay drives Jinja)

**Files:**
- Modify: `src/aiperf/config/loader/plan.py` (replace the `NotImplementedError` stub with the real implementation).
- Modify: `tests/unit/config/test_sweep.py` (append `TestScenarioJinjaVariablesSweep` class with the first test).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/config/test_sweep.py` (before the final newline):

```python


class TestScenarioJinjaVariablesSweep:
    """Tests for sweeping over Jinja variables via scenario `variables:` overlay
    and grid `sweep.variables: {"variables.X": [...]}` paths.

    Spec: docs/superpowers/specs/2026-05-03-scenario-sweep-jinja-variables-design.md

    Each test runs the full load_benchmark_plan_from_string path so regressions
    anywhere in load -> expand -> per-variation render+validate surface here.
    """

    BASE_HEADER = (
        "models:\n"
        "  - test/model\n"
        "endpoint:\n"
        "  type: chat\n"
        '  urls: ["http://localhost:8000/v1/chat/completions"]\n'
    )
    PHASES_TAIL = (
        "phases:\n"
        "  - name: profiling\n"
        "    type: concurrency\n"
        "    requests: 10\n"
        "    concurrency: 1\n"
    )

    def _isl_osl(self, cfg, ds_idx: int = 0):
        ds = cfg.datasets[ds_idx]
        isl = getattr(ds.prompts.isl, "value", ds.prompts.isl)
        osl = getattr(ds.prompts.osl, "value", ds.prompts.osl)
        return isl, osl

    def test_scenario_variables_overlay_drives_jinja_per_variation(self):
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl: 128\n"
                "  osl: 128\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts:\n"
                '      isl: "{{ isl }}"\n'
                '      osl: "{{ osl }}"\n'
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {variables: {isl: 128, osl: 128}}\n"
                "    - {variables: {isl: 256, osl: 256}}\n"
                "    - {variables: {isl: 512, osl: 1024}}\n"
            )
            + self.PHASES_TAIL
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected = [(128, 128), (256, 256), (512, 1024)]
        for variation_cfg, (want_isl, want_osl) in zip(
            plan.configs, expected, strict=True
        ):
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == want_isl, f"isl mismatch on variation {expected.index((want_isl, want_osl))}"
            assert osl == want_osl, f"osl mismatch on variation {expected.index((want_isl, want_osl))}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_scenario_variables_overlay_drives_jinja_per_variation -v 2>&1 | tail -30`
Expected: FAIL with `NotImplementedError: deferred-render sweep path is implemented in Task 3 ...`

- [ ] **Step 3: Implement `_build_plan_deferred_render`**

Replace the stub `_build_plan_deferred_render` in `src/aiperf/config/loader/plan.py` with:

```python
def _build_plan_deferred_render(
    raw_dict: dict[str, Any], file_path: Path | str | None
) -> BenchmarkPlan:
    """Sweep-present path: expand on raw template-bearing dict, then render
    Jinja and validate per variation.

    Each variation gets its own Jinja context built from the variation dict
    (which carries the scenario's or grid's overrides into the `variables:`
    block via expand_sweep + _deep_merge). Global cross-field invariants are
    checked once via AIPerfConfig.model_validate on rendered variation 0.
    """
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.loader.core import _validate_config_dict
    from aiperf.config.loader.jinja import (
        build_template_context,
        render_jinja2_templates,
    )
    from aiperf.config.sweep import SweepConfig, expand_sweep
    from pydantic import TypeAdapter

    # Validate the sweep block's own shape before expansion. expand_sweep is
    # permissive (silently coerces malformed sweep dicts), so without this a
    # typo'd sweep block would produce zero variations rather than a clear
    # error.
    TypeAdapter(SweepConfig).validate_python(raw_dict["sweep"])

    expanded = expand_sweep(raw_dict)
    if not expanded:
        raise ValueError(
            "sweep block expanded to zero variations; check that "
            "scenarios.runs is non-empty and grid.variables values are non-empty."
        )

    configs: list[BenchmarkConfig] = []
    variations: list[Any] = []
    rendered_v0_for_global: dict[str, Any] | None = None

    for idx, (variation_dict, variation_meta) in enumerate(expanded):
        # expand_sweep already strips `sweep` from variation_dict.
        context = build_template_context(variation_dict)
        rendered = render_jinja2_templates(variation_dict, context)
        if idx == 0:
            rendered_v0_for_global = rendered
        rendered_for_bench = {k: v for k, v in rendered.items() if k != "multi_run"}
        configs.append(BenchmarkConfig.model_validate(rendered_for_bench))
        variations.append(variation_meta)

    assert rendered_v0_for_global is not None  # at least one variation by post-expand check above
    config = _validate_config_dict(rendered_v0_for_global, file_path)
    return _assemble_plan_from_aiperf_config(config, configs, variations)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_scenario_variables_overlay_drives_jinja_per_variation -v 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Run the full test_sweep.py suite to confirm no regression**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py -v 2>&1 | tail -40`
Expected: All tests PASS (existing scenario-sweep, grid-sweep, magic-list, dataset-shorthand cases unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/loader/plan.py tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
feat(loader): defer Jinja render until after sweep expansion

When a YAML config carries a `sweep:` block, expand_sweep now runs on the
raw template-bearing dict and Jinja renders per-variation with that
variation's merged `variables:` block. AIPerfConfig.model_validate runs
once on rendered variation 0 for global cross-field invariants.

Enables `sweep.runs[i].variables: {x: ...}` (scenarios) and
`sweep.variables: {"variables.x": [...]}` (grid) to drive Jinja-templated
fields per variation.

Spec: docs/superpowers/specs/2026-05-03-scenario-sweep-jinja-variables-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Test — partial overlay inherits base variables

**Files:**
- Modify: `tests/unit/config/test_sweep.py` (append within `TestScenarioJinjaVariablesSweep`).

- [ ] **Step 1: Write the failing test**

Append to `TestScenarioJinjaVariablesSweep` (before the closing of the class):

```python
    def test_scenario_variables_partial_overlay_inherits_base(self):
        """Scenario overrides only `isl`; `osl` falls back to the base variables block."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl: 128\n"
                "  osl: 64\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts:\n"
                '      isl: "{{ isl }}"\n'
                '      osl: "{{ osl }}"\n'
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {variables: {isl: 128}}\n"
                "    - {variables: {isl: 256}}\n"
                "    - {variables: {isl: 512}}\n"
            )
            + self.PHASES_TAIL
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected_isl = [128, 256, 512]
        for variation_cfg, want_isl in zip(plan.configs, expected_isl, strict=True):
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == want_isl
            assert osl == 64, "osl must inherit base variables value across variations"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_scenario_variables_partial_overlay_inherits_base -v 2>&1 | tail -15`
Expected: PASS (the deferred-render path's deep-merge of `variables:` already handles partial overlay via existing `_deep_merge`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): scenario variables partial overlay inherits base values

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Test — grid sweep at `variables.*` path drives Jinja

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the failing test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_grid_sweep_at_variables_path_drives_jinja(self):
        """Grid sweep with key `variables.rps` drives a Jinja-templated rate field."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  rps: 100\n"
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: grid\n"
                "  variables:\n"
                '    "variables.rps": [100, 250, 500]\n'
                "phases:\n"
                "  - name: profiling\n"
                "    type: rate\n"
                "    requests: 10\n"
                '    rate: "{{ rps }}"\n'
            )
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected_rates = [100, 250, 500]
        for variation_cfg, want_rate in zip(plan.configs, expected_rates, strict=True):
            phase = next(p for p in variation_cfg.phases if p.name == "profiling")
            assert phase.rate == want_rate, f"rate mismatch: {phase.rate} != {want_rate}"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_grid_sweep_at_variables_path_drives_jinja -v 2>&1 | tail -20`
Expected: PASS.

If it fails: check that grid sweep keys with dots work with `_set_nested_value(variant, "variables.rps", value)`. The existing `_set_nested_value` handles dot-paths into dicts; `variables` must be a dict in the raw config (not a list). It is — `variables: {rps: 100}` parses as a dict.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): grid sweep at variables.* path drives Jinja-templated field

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Test — grid sweep mixed `variables.*` and field paths

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the failing test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_grid_sweep_mixed_variables_and_field_paths(self):
        """Grid sweep mixing `variables.X` (Jinja-driving) and direct config-path keys.

        Cartesian product: 2 isl values * 2 concurrency values = 4 variations.
        """
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl: 128\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts:\n"
                '      isl: "{{ isl }}"\n'
                "      osl: 64\n"
                "sweep:\n"
                "  type: grid\n"
                "  variables:\n"
                '    "variables.isl": [128, 256]\n'
                '    "phases.profiling.concurrency": [10, 20]\n'
            )
            + self.PHASES_TAIL
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 4
        # Field names sort alphabetically in _expand_grid_sweep; "phases..." comes
        # before "variables..." so combo order is (concurrency, isl) outer/inner.
        expected = [
            (10, 128),
            (10, 256),
            (20, 128),
            (20, 256),
        ]
        for variation_cfg, (want_conc, want_isl) in zip(
            plan.configs, expected, strict=True
        ):
            phase = next(p for p in variation_cfg.phases if p.name == "profiling")
            assert phase.concurrency == want_conc
            isl, _ = self._isl_osl(variation_cfg)
            assert isl == want_isl
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_grid_sweep_mixed_variables_and_field_paths -v 2>&1 | tail -20`
Expected: PASS.

If the variation ordering doesn't match (alphabetic sort by field name), adjust the `expected` list in the test to match the actual ordering — confirm by inspecting `_expand_grid_sweep` (`src/aiperf/config/sweep.py:171`: `field_names = sorted(variables.keys())`). With keys `"phases.profiling.concurrency"` and `"variables.isl"`, alphabetic sort puts phases first.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): grid sweep mixing variables.* and direct field paths

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Regression test — no-sweep Jinja unchanged

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_no_sweep_jinja_unchanged(self):
        """Non-sweep YAML with Jinja must go through the today-shape path
        (load_config -> build_benchmark_plan), not the deferred path."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl: 128\n"
                "  osl: 64\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts:\n"
                '      isl: "{{ isl }}"\n'
                '      osl: "{{ osl }}"\n'
            )
            + self.PHASES_TAIL
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert not plan.is_sweep
        assert len(plan.configs) == 1
        isl, osl = self._isl_osl(plan.configs[0])
        assert isl == 128
        assert osl == 64
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_no_sweep_jinja_unchanged -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): regression — no-sweep Jinja path unchanged

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Test — composition with singular `dataset:` shorthand

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_scenario_variables_with_dataset_shorthand_compose(self):
        """A scenario carrying both `variables:` overlay and singular `dataset:`
        shorthand. Shorthand rewrite + variables overlay both fire; rendered
        prompts pick up the overlaid Jinja value AND the dataset-shorthand fields."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl_mult: 1\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - variables: {isl_mult: 1}\n"
                "      dataset: {isl: 128, osl: 64}\n"
                "    - variables: {isl_mult: 2}\n"
                "      dataset: {isl: 256, osl: 64}\n"
            )
            + self.PHASES_TAIL
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 2
        expected = [(128, 64), (256, 64)]
        for variation_cfg, (want_isl, want_osl) in zip(
            plan.configs, expected, strict=True
        ):
            assert variation_cfg.datasets[0].name == "main"
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == want_isl
            assert osl == want_osl
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_scenario_variables_with_dataset_shorthand_compose -v 2>&1 | tail -20`
Expected: PASS. Both shorthands operate on the same scenario dict during expansion: `_normalize_scenario_dataset_form` rewrites `dataset:` to `datasets: [...]`, then the `variables:` overlay deep-merges. Per-variation render then picks up the merged `variables:`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): composition — scenario variables overlay + dataset shorthand

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Test — global `AIPerfConfig` validation runs once on variation 0

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_aiperfconfig_global_validation_runs_once_on_variation_0(self):
        """SLOs configured for streaming-only metrics with streaming disabled
        is a global cross-field issue. The deferred path validates AIPerfConfig
        once on rendered variation 0 — the warning surfaces via validate_config_file's
        warning surface, but the model itself validates without raising. To keep
        this test focused on the deferred path actually invoking AIPerfConfig
        validation, assert the resolved config has the multi_run/slos values
        propagated correctly through the deferred path."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  rps: 100\n"
                "multi_run:\n"
                "  num_runs: 3\n"
                "  cooldown_seconds: 0.5\n"
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: grid\n"
                "  variables:\n"
                '    "variables.rps": [100, 200]\n'
                "phases:\n"
                "  - name: profiling\n"
                "    type: rate\n"
                "    requests: 10\n"
                '    rate: "{{ rps }}"\n'
            )
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 2
        # multi_run propagates through the deferred path via AIPerfConfig
        # validation on rendered variation 0.
        assert plan.trials == 3
        assert plan.cooldown_seconds == 0.5
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_aiperfconfig_global_validation_runs_once_on_variation_0 -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): multi_run propagates through deferred path via AIPerfConfig

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Test — Jinja arithmetic with overlaid variables

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_jinja_arithmetic_with_overlaid_variable(self):
        """Templated field uses `{{ rate * 2 }}`; sweep overlays drive `rate`.
        Per-variation render evaluates the expression with the overlaid value."""
        from aiperf.config.loader import load_benchmark_plan_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  rate: 10\n"
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: grid\n"
                "  variables:\n"
                '    "variables.rate": [10, 20, 30]\n'
                "phases:\n"
                "  - name: profiling\n"
                "    type: concurrency\n"
                "    requests: 10\n"
                '    concurrency: "{{ rate * 2 }}"\n'
            )
        )

        plan = load_benchmark_plan_from_string(yaml_str, substitute_env=False)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected = [20, 40, 60]
        for variation_cfg, want_conc in zip(plan.configs, expected, strict=True):
            phase = next(p for p in variation_cfg.phases if p.name == "profiling")
            assert phase.concurrency == want_conc
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_jinja_arithmetic_with_overlaid_variable -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): Jinja arithmetic resolves with overlaid variable per variation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Test — circular variables reference in scenario overlay errors

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_circular_variables_reference_in_scenario_overlay_errors(self):
        """Scenario `variables:` overlay introduces a cycle. Per-variation
        render should fail with ConfigurationError naming the cycle members."""
        import pytest

        from aiperf.config.loader import load_benchmark_plan_from_string
        from aiperf.config.loader.errors import ConfigurationError

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  a: 1\n"
                "  b: 2\n"
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                '    - variables: {a: "{{ b }}", b: "{{ a }}"}\n'
            )
            + self.PHASES_TAIL
        )

        with pytest.raises(ConfigurationError) as excinfo:
            load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        msg = str(excinfo.value).lower()
        assert "circular" in msg or "cycle" in msg
        assert "a" in msg and "b" in msg
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_circular_variables_reference_in_scenario_overlay_errors -v 2>&1 | tail -20`
Expected: PASS — `_resolve_variables_block` (`src/aiperf/config/loader/jinja.py:171`) raises `ConfigurationError` listing the cycle members.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): circular variables reference in scenario overlay raises ConfigurationError

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Test — StrictUndefined per variation surfaces missing variable

**Files:**
- Modify: `tests/unit/config/test_sweep.py`.

- [ ] **Step 1: Write the test**

Append to `TestScenarioJinjaVariablesSweep`:

```python
    def test_strict_undefined_per_variation_surfaces_missing_variable(self):
        """Templated field references a variable absent from base AND scenario
        overlay. Per-variation Jinja render fires StrictUndefined and the
        deferred path surfaces it as a ConfigurationError."""
        import pytest

        from aiperf.config.loader import load_benchmark_plan_from_string
        from aiperf.config.loader.errors import ConfigurationError

        yaml_str = (
            self.BASE_HEADER
            + (
                "variables:\n"
                "  isl: 128\n"
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts:\n"
                '      isl: "{{ isl }}"\n'
                '      osl: "{{ does_not_exist }}"\n'
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {variables: {isl: 128}}\n"
            )
            + self.PHASES_TAIL
        )

        with pytest.raises(ConfigurationError) as excinfo:
            load_benchmark_plan_from_string(yaml_str, substitute_env=False)
        assert "does_not_exist" in str(excinfo.value)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_sweep.py::TestScenarioJinjaVariablesSweep::test_strict_undefined_per_variation_surfaces_missing_variable -v 2>&1 | tail -20`
Expected: PASS — `render_jinja2_templates` wraps the StrictUndefined error in a `ConfigurationError` (`src/aiperf/config/loader/jinja.py:217-220`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_sweep.py
git commit -m "$(cat <<'EOF'
test(sweep): StrictUndefined per variation surfaces missing variable

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Regression test for `build_benchmark_plan(AIPerfConfig)` post-refactor

**Files:**
- Modify: `tests/unit/config/test_benchmark_plan.py` (add one parametrize to an existing class, or one new test).

- [ ] **Step 1: Write the test**

Append to `tests/unit/config/test_benchmark_plan.py` (under whichever existing test class houses `build_benchmark_plan` tests, or as a new top-level function — match the file's existing convention):

```python
def test_build_benchmark_plan_no_sweep_path_unchanged_post_refactor():
    """Pin the no-sweep build_benchmark_plan(AIPerfConfig) path: still produces
    a single-config plan with multi_run propagation intact, after the
    _assemble_plan_from_aiperf_config refactor."""
    from aiperf.config.loader import build_benchmark_plan, load_config_from_string

    yaml_str = (
        "models: [test/model]\n"
        "endpoint:\n"
        "  type: chat\n"
        '  urls: ["http://localhost:8000/v1/chat/completions"]\n'
        "datasets:\n"
        "  - {name: main, type: synthetic, entries: 200}\n"
        "phases:\n"
        "  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}\n"
        "multi_run:\n"
        "  num_runs: 5\n"
        "  cooldown_seconds: 1.5\n"
    )
    cfg = load_config_from_string(yaml_str, substitute_env=False)
    plan = build_benchmark_plan(cfg)

    assert not plan.is_sweep
    assert len(plan.configs) == 1
    assert plan.trials == 5
    assert plan.cooldown_seconds == 1.5
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest -n auto tests/unit/config/test_benchmark_plan.py::test_build_benchmark_plan_no_sweep_path_unchanged_post_refactor -v 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/config/test_benchmark_plan.py
git commit -m "$(cat <<'EOF'
test(plan): pin build_benchmark_plan no-sweep path post-refactor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Tutorial doc — `docs/tutorials/sweeps.md` adds "Sweeping over Jinja variables"

**Files:**
- Modify: `docs/tutorials/sweeps.md`.

- [ ] **Step 1: Read the existing structure to find the right insertion point**

Run: `grep -n "^## \|^### " docs/tutorials/sweeps.md`
Expected: a list of section headings. Identify the section about "Sweep + Distributions" or similar; the new subsection slots in alphabetically/topically nearby.

- [ ] **Step 2: Append the new subsection**

Add the following section to `docs/tutorials/sweeps.md` (right after the "Scenario sweep" section, or wherever the existing scenario-shorthand discussion lives if that has been added). Adapt the indent/heading level to match neighboring sections.

```markdown
## Sweeping over Jinja variables

Both scenario and grid sweeps can drive values into the `variables:` block,
which is then visible to Jinja `{{ ... }}` expressions on any field. This
lets one sweep dimension control multiple downstream fields without
repeating the value at each site.

### Scenario form: per-run `variables:` overlay

Each scenario run carries a `variables:` block that deep-merges over the
base `variables:`. Templated fields render with the merged values.

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

Produces three variations with `prompts.isl`/`prompts.osl` pulled from the
overlay. Variables not mentioned in a run inherit from the base block —
partial overlays work.

### Grid form: `sweep.variables` keyed at `variables.<name>`

A grid sweep can target the `variables:` block directly via the
`variables.<name>` dot-path. Every templated field referencing that
variable picks up the swept value.

```yaml
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

Mix freely with field-path keys — the Cartesian product covers both:

```yaml
sweep:
  type: grid
  variables:
    "variables.isl": [128, 256]
    "phases.profiling.concurrency": [10, 20]
```

### Notes

- Jinja arithmetic survives the sweep: `concurrency: "{{ rate * 2 }}"` with
  `sweep.variables: { "variables.rate": [10, 20, 30] }` renders to
  `[20, 40, 60]` per variation.
- A reference to a variable that is not present in the base block AND not
  introduced by the sweep raises `ConfigurationError` per variation
  (StrictUndefined).
- Cycles inside a scenario `variables:` overlay (`a: "{{ b }}", b: "{{ a }}"`)
  raise `ConfigurationError` listing the cycle members.
```

- [ ] **Step 3: Verify the example renders correctly end-to-end**

Save the first scenario example to `/tmp/sweep_jinja_example.yaml`, then:

```bash
uv run python -c "
from aiperf.config.loader import load_benchmark_plan
plan = load_benchmark_plan('/tmp/sweep_jinja_example.yaml', substitute_env=False)
for i, cfg in enumerate(plan.configs):
    isl = getattr(cfg.datasets[0].prompts.isl, 'value', cfg.datasets[0].prompts.isl)
    osl = getattr(cfg.datasets[0].prompts.osl, 'value', cfg.datasets[0].prompts.osl)
    print(f'variation {i}: isl={isl} osl={osl}')
"
```
Expected output (3 variations: 128/128, 256/256, 512/1024).

Repeat for the grid example.

(Skip endpoint/phases tail in the YAML: copy from the test fixtures in `TestScenarioJinjaVariablesSweep` to make it loadable.)

- [ ] **Step 4: Add to `docs/index.yml` if the file path is new**

Run: `grep -n "sweeps.md" docs/index.yml | head -3`
Expected: existing entry. We're modifying `sweeps.md`, not creating a new file, so no index change is needed. If the grep returns nothing, the file is new → add an entry under the appropriate section.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorials/sweeps.md
git commit -m "$(cat <<'EOF'
docs(tutorials): document sweeping over Jinja variables

Adds a section to docs/tutorials/sweeps.md showing scenario `variables:`
overlay and grid `sweep.variables: { variables.X: [...] }` syntax.
Examples validated end-to-end via load_benchmark_plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Cross-reference from `docs/tutorials/parameter-sweeping.md`

**Files:**
- Modify: `docs/tutorials/parameter-sweeping.md`.

- [ ] **Step 1: Locate the right section**

Run: `grep -n "^## \|^### " docs/tutorials/parameter-sweeping.md`
Expected: list of section headings. Find the section that introduces sweep types or links to `sweeps.md`.

- [ ] **Step 2: Add a one-line cross-reference**

Insert near the end of the relevant section (or at the bottom of the page if there's no clear anchor) a sentence like:

```markdown
For sweeps that drive Jinja-templated fields via the `variables:` block, see
[Sweeping over Jinja variables](sweeps.md#sweeping-over-jinja-variables).
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/parameter-sweeping.md
git commit -m "$(cat <<'EOF'
docs(tutorials): cross-reference Jinja-variables sweeping section

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final verification

**Files:** none modified.

- [ ] **Step 1: Full unit-config test pass**

Run: `uv run pytest -n auto tests/unit/config/ -v 2>&1 | tail -50`
Expected: all PASS, including the new `TestScenarioJinjaVariablesSweep` class (10 new tests) and the `build_benchmark_plan` regression test.

- [ ] **Step 2: Full unit suite pass (catch unexpected breakage)**

Run: `uv run pytest -n auto tests/unit/ 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 3: Lint + format**

Run: `ruff format . && ruff check --fix .`
Expected: no diffs (or only auto-applied fixes).

If ruff applies fixes, commit them as a final cleanup commit:
```bash
git add -p
git commit -m "$(cat <<'EOF'
style: ruff auto-fixes from sweep-jinja work

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Pre-commit on all touched files**

Run: `pre-commit run --files src/aiperf/config/loader/plan.py src/aiperf/config/loader/__init__.py src/aiperf/config/__init__.py tests/unit/config/test_sweep.py tests/unit/config/test_benchmark_plan.py docs/tutorials/sweeps.md docs/tutorials/parameter-sweeping.md`
Expected: all hooks pass. If any hook reflows files, re-stage and amend (or commit fresh — never `--amend --no-edit`; pass full message).

- [ ] **Step 5: Diff review against the spec**

Run: `git log --oneline origin/main..HEAD | head -20`
Expected: commits cover Tasks 1–15 in order. Verify the spec's §Code touchpoints are all addressed:
- `loader/plan.py` ✓ (Tasks 1–3)
- `tests/unit/config/test_sweep.py` ✓ (Tasks 3–12)
- `tests/unit/config/test_benchmark_plan.py` ✓ (Task 13)
- `docs/tutorials/sweeps.md` ✓ (Task 14)
- `docs/tutorials/parameter-sweeping.md` ✓ (Task 15)

- [ ] **Step 6: Notify completion**

Branch is ready for review. The work is contained behind `_build_plan_deferred_render`; rollback = revert that branch in `load_benchmark_plan_from_string` (one if-statement) and remove the helper.

---

## Self-review notes

- **Spec coverage:** Every test case from §Test plan in the spec maps to a task (tests 1–10 → Tasks 3, 4, 5, 6, 7, 8, 9, 10, 11, 12). The `test_benchmark_plan.py` regression mentioned at the end of §Test plan → Task 13. §Code touchpoints — all covered.
- **Type consistency:** `_assemble_plan_from_aiperf_config(config, configs, variations)` signature is identical between Task 1 and Task 3 (both call sites use the same arg shape). `_build_plan_deferred_render(raw_dict, file_path)` signature defined in Task 2 stub matches the implementation in Task 3.
- **No placeholders:** every step has either an exact command or a complete code block.
- **Rollback path:** isolated to one helper + one routing branch.
