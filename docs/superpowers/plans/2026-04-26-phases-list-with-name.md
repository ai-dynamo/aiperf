# `phases` dict → list with `name` field — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `BenchmarkConfig.phases` from `dict[str, PhaseConfig]` to `list[PhaseConfig]` where each entry has a required `name: str` field. Eliminate Kubernetes CRD alphabetical-keying bug that silently inverts phase order (e.g. warmup → profiling becomes profiling → warmup).

**Architecture:** Hard breaking change — no backcompat for the dict shape (this branch hasn't shipped to main). The user-facing YAML migrates from `phases: {warmup: {...}, profiling: {...}}` to `phases: [{name: warmup, ...}, {name: profiling, ...}]`. Top-level `warmup:` / `profiling:` shorthand still works (operator-side normalizer still builds the list in correct order). Sweep dot-path overrides (`phases.warmup.rate`) become index-based or name-targeted — see Task 14.

**Tech Stack:** Pydantic v2 (`field_validator` / `model_validator`), kopf operator, kubernetes_asyncio, ZMQ message bus, pytest with xdist (`-n auto`).

---

## File Structure

**Core schema (Pydantic):**
- `src/aiperf/config/phases.py` — `BasePhaseConfig` gains `name: str` field; drop `_name` PrivateAttr.
- `src/aiperf/config/config.py` — `BenchmarkConfig.phases: list[PhaseConfig]`; replace `parse_phases`/`inject_phase_names` validators with `validate_phase_names_unique` (after-mode).
- `src/aiperf/config/_benchmark_normalizers.py` — `_normalize_warmup_profiling_to_phases` builds list; `_normalize_dataset_and_phases` builds single-entry list when `phases` is a flat config dict.
- `src/aiperf/config/cli_converter.py` — build list, not dict.

**Consumers (iterate phases):**
- `src/aiperf/timing/config.py` — `TimingConfig.from_config` already does `for name, phase in config.phases.items()`; switch to iterating list.
- `src/aiperf/orchestrator/strategies.py` — `del config.phases[key]` becomes list comprehension filter.
- `src/aiperf/config/resolvers.py` — phase iteration.
- `src/aiperf/config/_benchmark_helpers.py` — phase iteration.
- `src/aiperf/dataset/loader/base_hf_dataset.py`, `base_trace_loader.py` — `.phases.values()` → just iterate list.
- `src/aiperf/cli_commands/kube/{profile,generate}.py`, `cli_commands/kube/profile_deploy_direct.py`, `cli_commands/config_cli.py` — same.
- `src/aiperf/kubernetes/_memory_estimator/params.py`, `src/aiperf/workers/{worker,scaling}.py` — same.
- `src/aiperf/config/kube.py`, `src/aiperf/config/loader/core.py` — same.

**Templates (14 YAML files):**
All under `src/aiperf/config/templates/`: `composed_dataset.yaml`, `embeddings.yaml`, `audio_multimodal.yaml`, `kv_cache_test.yaml`, `jinja2_variables.yaml`, `scenario_workload_profiles.yaml`, `trace_replay.yaml`, `request_cancellation.yaml`, `sweep_distributions.yaml`, `long_context.yaml`, `goodput_slo.yaml`, `minimal.yaml`, `multimodal_vision.yaml`, `public_dataset.yaml`. (`latency_test.yaml` and `warmup_profiling.yaml` use top-level shorthand — no change needed.)

**Tests (TDD — write failing first):**
- `tests/unit/config/test_phases_list_shape.py` (new) — list shape, name uniqueness, name-required, validation errors.
- `tests/unit/config/test_config_normalization.py` — update existing parametrized tests for shorthand → list.
- `tests/unit/config/test_config_validation.py` — update consumers.
- `tests/unit/timing/test_timing_config.py` — verify list-iteration ordering.
- `tests/unit/orchestrator/test_strategies.py` — verify `_remove_warmup_phases` filters list.
- `tests/component_integration/`, `tests/integration/`, `tests/kubernetes/`, `tests/operator/` — sweep for any fixture / parametrize using dict shape and update.

**Docs (search & update):**
- `docs/tutorials/yaml-config.md`, `docs/tutorials/sweeps.md`, `docs/tutorials/template-endpoint.md`, `docs/tutorials/distributions.md`
- `src/aiperf/config/schema/README.md`, `src/aiperf/config/templates/README.md`
- `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` (three-file sync)

**Generated / regen-on-commit:**
- `src/aiperf/config/schema/aiperf-config.schema.json` — auto-regen via `tools/generate_config_schema.py` (pre-commit hook). The schema's `phases` shape will switch to JSON-Schema array.
- CRD: `spec.benchmark` is opaque (`x-kubernetes-preserve-unknown-fields: true`), so no CRD schema change needed; only the operator-side normalizer matters.

---

## Task Decomposition

Per-task discipline:
- **One** `uv run pytest -n auto tests/unit/ -x` invocation max per task (do not split by subfolder).
- Pre-commit on **staged files only** between tasks; run `pre-commit run --all-files` only at the end-of-plan integration check.
- Commits via `git commit -s` with `--no-verify` if the pre-commit fmt drift bites — see CLAUDE.md.

---

### Task 1: Lock the new shape with a contract test

**Files:**
- Create: `tests/unit/config/test_phases_list_shape.py`

- [ ] **Step 1: Write the failing contract tests**

```python
# tests/unit/config/test_phases_list_shape.py
"""Contract tests for the list-of-named-phases schema (post-refactor)."""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig


_BASE: dict = {
    "models": "mock",
    "endpoint": {"urls": ["http://x:8000/v1/chat/completions"], "streaming": True},
    "datasets": {"main": {"type": "synthetic"}},
}


def _cfg(phases):
    return BenchmarkConfig.model_validate({**_BASE, "phases": phases})


def test_phases_accepts_list_with_name_field():
    cfg = _cfg([
        {"name": "warmup", "type": "concurrency", "requests": 10, "concurrency": 2,
         "exclude_from_results": True},
        {"name": "profiling", "type": "concurrency", "requests": 100, "concurrency": 4},
    ])
    assert isinstance(cfg.phases, list)
    assert [p.name for p in cfg.phases] == ["warmup", "profiling"]


def test_phases_preserves_input_order_warmup_first():
    cfg = _cfg([
        {"name": "warmup", "type": "concurrency", "requests": 1, "concurrency": 1,
         "exclude_from_results": True},
        {"name": "profiling", "type": "concurrency", "requests": 1, "concurrency": 1},
    ])
    assert cfg.phases[0].name == "warmup"
    assert cfg.phases[1].name == "profiling"


def test_phases_preserves_input_order_profiling_first():
    cfg = _cfg([
        {"name": "profiling", "type": "concurrency", "requests": 1, "concurrency": 1},
        {"name": "warmup", "type": "concurrency", "requests": 1, "concurrency": 1,
         "exclude_from_results": True},
    ])
    assert cfg.phases[0].name == "profiling"
    assert cfg.phases[1].name == "warmup"


def test_phases_rejects_dict_shape():
    with pytest.raises(ValueError, match="phases must be a list"):
        _cfg({"warmup": {"type": "concurrency", "requests": 1, "concurrency": 1}})


def test_phases_rejects_missing_name():
    with pytest.raises(ValueError, match="name"):
        _cfg([{"type": "concurrency", "requests": 1, "concurrency": 1}])


def test_phases_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate phase name"):
        _cfg([
            {"name": "p", "type": "concurrency", "requests": 1, "concurrency": 1},
            {"name": "p", "type": "concurrency", "requests": 2, "concurrency": 2},
        ])


def test_phases_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 1 item"):
        _cfg([])
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest -n auto tests/unit/config/test_phases_list_shape.py -x
```

Expected: all 7 tests FAIL (current schema is `dict[str, PhaseConfig]`).

- [ ] **Step 3: Commit the contract**

```bash
git add tests/unit/config/test_phases_list_shape.py
git commit -s -m "test(config): contract for phases as list with name field"
```

---

### Task 2: Migrate `BasePhaseConfig` to carry `name`

**Files:**
- Modify: `src/aiperf/config/phases.py:85` — replace `_name: str | None = PrivateAttr(default=None)` with public `name: str` field.

- [ ] **Step 1: Update `BasePhaseConfig`**

In `src/aiperf/config/phases.py`, find the `class BasePhaseConfig(BaseConfig):` block. Replace:

```python
    _name: str | None = PrivateAttr(default=None)
```

with (placed right after `model_config = ConfigDict(extra="forbid")`):

```python
    name: Annotated[
        str,
        Field(
            min_length=1,
            description="Phase identifier — unique within the benchmark's phases list. "
            "Used in logs, status, sweep targeting, and result file naming. "
            "Common names: 'warmup', 'profiling'. Must be 1+ chars; allowed everywhere "
            "an identifier is allowed (no shell meta-chars).",
        ),
    ]
```

Remove the `from pydantic import PrivateAttr` import if it's no longer used elsewhere in the file (check via grep).

- [ ] **Step 2: Find and replace all `phase._name` accesses**

```bash
grep -rn "_name\b" src/aiperf/config/ src/aiperf/timing/ src/aiperf/orchestrator/ \
  src/aiperf/dataset/ src/aiperf/cli_commands/ src/aiperf/operator/ src/aiperf/kubernetes/
```

For each `phase._name` or `config._name` reference, replace with `phase.name`. Note: skip results that are unrelated `_name` attributes (e.g. `model._name`); only update PhaseConfig accesses.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -s -m "refactor(config): move PhaseConfig name from PrivateAttr to required field"
```

(Tests still red — that's fine; we'll go green after Task 3.)

---

### Task 3: Switch `BenchmarkConfig.phases` to `list[PhaseConfig]`

**Files:**
- Modify: `src/aiperf/config/config.py:158` (field type), `298-318` (parse_phases — DELETE), `329-334` (inject_phase_names — DELETE), `336-348` (validate_dataset_references — UPDATE), `350-361` (validate_seamless_not_on_first_phase — UPDATE), `363-371` (validate_prefill_requires_streaming — UPDATE), `373-389` (validate_phase_dataset_compatibility — UPDATE).

- [ ] **Step 1: Change the field type and validators**

In `src/aiperf/config/config.py`, replace the `phases:` field definition:

```python
    phases: Annotated[
        list[PhaseConfig],
        Field(
            min_length=1,
            description="Ordered benchmark phases. Each entry must have a unique 'name' "
            "(e.g. 'warmup', 'profiling'). Order in the list IS the execution order; "
            "the first phase runs first. Single-config shorthand "
            "({'type': 'concurrency', ...}) is normalized to a list of one. "
            "Top-level 'warmup:'/'profiling:' shorthand is normalized to a "
            "[warmup, profiling] list pre-validation.",
        ),
    ]
```

Delete the entire `parse_phases` validator (lines ~298–318) and the `inject_phase_names` validator (lines ~329–334).

Add a new `model_validator(mode="after")` to enforce uniqueness:

```python
    @model_validator(mode="after")
    def validate_phase_names_unique(self) -> Self:
        """Reject duplicate phase names — they must be unique within the list."""
        seen: set[str] = set()
        for phase in self.phases:
            if phase.name in seen:
                raise ValueError(
                    f"duplicate phase name '{phase.name}' — names must be unique. "
                    f"Found names: {[p.name for p in self.phases]}"
                )
            seen.add(phase.name)
        return self
```

Update `validate_dataset_references`:

```python
    @model_validator(mode="after")
    def validate_dataset_references(self) -> Self:
        """Validate that all dataset references in phase configs exist."""
        dataset_names = set(self.datasets.keys())
        for phase in self.phases:
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Phase config '{phase.name}' references undefined dataset "
                    f"'{phase.dataset}'. Available datasets: {sorted(dataset_names)}"
                )
        return self
```

Update `validate_seamless_not_on_first_phase`:

```python
    @model_validator(mode="after")
    def validate_seamless_not_on_first_phase(self) -> Self:
        """Ensure seamless is not enabled on the first phase config."""
        if self.phases and self.phases[0].seamless:
            raise ValueError(
                f"Phase config '{self.phases[0].name}' cannot have seamless=True "
                "because it is first. Seamless transitions only apply to "
                "subsequent phase configs."
            )
        return self
```

Update `validate_prefill_requires_streaming`:

```python
    @model_validator(mode="after")
    def validate_prefill_requires_streaming(self) -> Self:
        """Prefill concurrency requires streaming to measure TTFT boundaries."""
        for phase in self.phases:
            if phase.prefill_concurrency is not None and not self.endpoint.streaming:
                raise ValueError(
                    f"Phase '{phase.name}': prefill_concurrency requires "
                    "endpoint.streaming=true"
                )
        return self
```

Update `validate_phase_dataset_compatibility`:

```python
    @model_validator(mode="after")
    def validate_phase_dataset_compatibility(self) -> Self:
        from aiperf.config.resolved import check_phase_dataset_compatibility
        for phase in self.phases:
            dataset_name = phase.dataset or self.get_default_dataset_name()
            ds = self.datasets.get(dataset_name)
            if ds is None:
                continue
            errors = check_phase_dataset_compatibility(
                phase, ds, phase.name, dataset_name
            )
            if errors:
                raise ValueError("\n".join(errors))
        return self
```

- [ ] **Step 2: Update normalizers in `_benchmark_normalizers.py`**

Modify `_normalize_warmup_profiling_to_phases` (lines 51–65):

```python
def _normalize_warmup_profiling_to_phases(data: dict[str, Any]) -> None:
    has_warmup = "warmup" in data
    has_profiling = "profiling" in data
    if not (has_warmup or has_profiling):
        return

    phases: list[dict[str, Any]] = []
    if has_warmup:
        warmup = data.pop("warmup")
        if isinstance(warmup, dict):
            warmup = {"name": "warmup", **warmup}
            warmup.setdefault("exclude_from_results", True)
        phases.append(warmup)
    if has_profiling:
        prof = data.pop("profiling")
        if isinstance(prof, dict):
            prof = {"name": "profiling", **prof}
        phases.append(prof)
    data["phases"] = phases
```

Modify `_normalize_dataset_and_phases` (lines 82–89):

```python
def _normalize_dataset_and_phases(data: dict[str, Any]) -> None:
    if "dataset" in data and "datasets" not in data:
        data["datasets"] = {"default": data.pop("dataset")}

    if "phases" in data:
        phases = data["phases"]
        # Single flat-dict shorthand: phases: {type: concurrency, ...}
        if isinstance(phases, dict) and "type" in phases:
            data["phases"] = [{"name": "default", **phases}]
```

- [ ] **Step 3: Update `cli_converter.py`**

In `src/aiperf/config/cli_converter.py`, find the block that builds `phases: dict[str, Any]` (around line 71) and convert to a list. Locate the surrounding code:

```bash
sed -n '60,95p' src/aiperf/config/cli_converter.py
```

Replace the dict construction (the `phases: dict[str, Any] = {}` plus its `phases["warmup"] = ...` / `phases["profiling"] = ...` follow-ups) with:

```python
    phases: list[dict[str, Any]] = []
    if (warmup := build_warmup(cli, s)) is not None:
        phases.append({"name": "warmup", **warmup})
    if (prof := build_profiling(cli, s)) is not None:
        phases.append({"name": "profiling", **prof})
    if phases:
        nested["phases"] = phases
```

(Read the actual surrounding code first; the `build_warmup` / `build_profiling` symbol names are correct per the existing imports.)

- [ ] **Step 4: Run the contract tests + the affected unit suite**

```bash
uv run pytest -n auto tests/unit/config/ tests/unit/timing/ -x 2>&1 | tail -50
```

Expected: contract tests from Task 1 PASS. **Other tests in `tests/unit/config/test_config_normalization.py`, `test_config_validation.py`, etc., will FAIL — that's the breaking change surface; Task 4 fixes them.**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -s -m "refactor(config): phases is now list[PhaseConfig] with required 'name' field"
```

---

### Task 4: Fix consumers — TimingConfig + orchestrator strategy

**Files:**
- Modify: `src/aiperf/timing/config.py:71` — iterate list.
- Modify: `src/aiperf/orchestrator/strategies.py:349-353` — filter list instead of `del`.

- [ ] **Step 1: Update `TimingConfig.from_config`**

In `src/aiperf/timing/config.py`, find `from_config` (around line 62). Replace:

```python
        for name, phase in config.phases.items():
            phase_config = _build_credit_phase_config(
                phase, phase_name=name, exclude_from_results=phase.exclude_from_results
            )
            phase_configs.append(phase_config)
```

with:

```python
        for phase in config.phases:
            phase_config = _build_credit_phase_config(
                phase, phase_name=phase.name, exclude_from_results=phase.exclude_from_results
            )
            phase_configs.append(phase_config)
```

(Adjust the cancellation-detection branch the same way — `phase.cancellation` access already works on the model object.)

- [ ] **Step 2: Update `_remove_warmup_phases` in strategies.py**

In `src/aiperf/orchestrator/strategies.py`, replace lines ~348–353:

```python
        config = config.model_copy(deep=True)
        config.phases = [p for p in config.phases if not p.exclude_from_results]
        return config
```

- [ ] **Step 3: Run the affected unit suite**

```bash
uv run pytest -n auto tests/unit/timing/ tests/unit/orchestrator/ -x 2>&1 | tail -30
```

Expected: green for these subsystems.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "refactor(timing,orchestrator): consume phases as ordered list"
```

---

### Task 5: Fix remaining consumers — bulk update

**Files:**
- Modify: `src/aiperf/config/resolvers.py:150,326` — phase iteration.
- Modify: `src/aiperf/config/_benchmark_helpers.py:72,80` — phase iteration.
- Modify: `src/aiperf/dataset/loader/base_hf_dataset.py:125` — `phases.values()` → `phases`.
- Modify: `src/aiperf/dataset/loader/base_trace_loader.py:28` — same.
- Modify: `src/aiperf/cli_commands/kube/profile.py:93,119`, `cli_commands/kube/generate.py:70`, `cli_commands/kube/profile_deploy_direct.py:98`, `cli_commands/config_cli.py:62`.
- Modify: `src/aiperf/kubernetes/_memory_estimator/params.py:188`.
- Modify: `src/aiperf/workers/{worker.py:211,scaling.py:33}`.
- Modify: `src/aiperf/config/kube.py:251`, `src/aiperf/config/loader/core.py:309`.

- [ ] **Step 1: Sweep every `.phases.items()` / `.phases.values()` / `.phases.keys()`**

```bash
grep -rn "\.phases\.\(items\|values\|keys\)\|\.phases\[" src/aiperf/ \
  | grep -v "JobProgress\|\.phases.*CombinedPhaseStats\|operator/progress_models\|operator/handlers/monitor"
```

(Exclude the operator's `JobProgress.phases` — that's a *separate* dict for runtime stats keyed by `CreditPhase` enum, not config phases. Confirm by reading `src/aiperf/operator/progress_models.py:74`.)

- [ ] **Step 2: For each match, apply the mechanical translation**

| Old | New |
|---|---|
| `for name, phase in config.phases.items():` | `for phase in config.phases:` (use `phase.name` inside) |
| `for phase in config.phases.values():` | `for phase in config.phases:` |
| `for name in config.phases.keys():` | `for phase in config.phases:` (use `phase.name` inside) |
| `config.phases[name]` | `next(p for p in config.phases if p.name == name)` (or refactor to iterate) |
| `name in config.phases` | `any(p.name == name for p in config.phases)` |
| `config.phases[name] = ...` | replace by index, or rebuild list |
| `del config.phases[name]` | `config.phases = [p for p in config.phases if p.name != name]` |

Read each callsite before editing — many of them only need the variable name change (`for phase in ...`) since they don't use `name`.

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -50
```

Expected: most `tests/unit/` green. Some test files still red (they construct `phases` as dict literals) — fixed in Task 6.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "refactor: update all config.phases consumers to iterate list"
```

---

### Task 6: Update unit tests that build `phases` literals

**Files:**
- Modify: `tests/unit/config/test_config_normalization.py`, `test_config_validation.py`.
- Modify: `tests/unit/timing/test_timing_config.py`.
- Modify: any other test under `tests/unit/` that grep finds.

- [ ] **Step 1: Find every dict-shaped phases literal in tests**

```bash
grep -rn "phases.*=.*{\|\"phases\":\s*{" tests/unit/ | head -50
```

- [ ] **Step 2: Translate each fixture / parametrize entry**

Mechanical translation:
```python
# old
{"phases": {"warmup": {"type": "concurrency", "requests": 10, "concurrency": 2}}}
# new
{"phases": [{"name": "warmup", "type": "concurrency", "requests": 10, "concurrency": 2}]}
```

Tests that explicitly assert on dict-shape behavior (e.g. "rejects unknown keys", "preserves order") should either be deleted (if Task 1 already covers the equivalent) or rewritten to assert list behavior.

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -30
```

Expected: **fully green**.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "test(config): migrate phases dict literals to list-with-name shape"
```

---

### Task 7: Update YAML templates

**Files:** all 14 templates listed in File Structure above.

- [ ] **Step 1: List every template that needs editing**

```bash
grep -l "^phases:" src/aiperf/config/templates/*.yaml
```

- [ ] **Step 2: For each file, transform the `phases:` block**

Example — `src/aiperf/config/templates/goodput_slo.yaml`:

```yaml
# OLD
phases:
  warmup:
    type: concurrency
    excludeFromResults: true
    requests: 100
    concurrency: 16
  low_load:
    type: poisson
    duration: 120
    rate: 10.0
    concurrency: 50
    seamless: true
    gracePeriod: 60

# NEW
phases:
  - name: warmup
    type: concurrency
    excludeFromResults: true
    requests: 100
    concurrency: 16
  - name: low_load
    type: poisson
    duration: 120
    rate: 10.0
    concurrency: 50
    seamless: true
    gracePeriod: 60
```

Apply the same transformation to all other templates with `phases:` blocks. **Special case — `scenario_workload_profiles.yaml`:** has nested phase overrides under `sweep.runs[].phases.test`. Those are sweep override targets, not full phase configs — they remain dot-path or change to list-of-overrides. See Task 14 for the sweep-override migration; in this task, leave the nested sweep `phases.test` blocks alone (Task 14 will revisit).

- [ ] **Step 3: Validate the templates parse**

```bash
uv run aiperf plugins --validate
make validate-plugin-schemas
```

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/config/templates/
git commit -s -m "templates: migrate phases dict to list-with-name across all examples"
```

---

### Task 8: Regenerate JSON config schema and CRD

**Files:**
- Regenerate: `src/aiperf/config/schema/aiperf-config.schema.json` (auto-generated from Pydantic model).
- Confirm CRD: `helm/` or wherever `tools/generate_crd.py` writes — `spec.benchmark` should remain opaque.

- [ ] **Step 1: Regenerate**

```bash
make generate-config-schema
make generate-crd
```

- [ ] **Step 2: Diff the schema for sanity**

```bash
git diff src/aiperf/config/schema/aiperf-config.schema.json | head -100
```

Expected: `"phases"` definition switches from `"type": "object"` (with `additionalProperties` referencing PhaseConfig) to `"type": "array"` (with `items` referencing PhaseConfig). The PhaseConfig schema gains a required `name` property.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/config/schema/aiperf-config.schema.json
# add CRD if changed
git commit -s -m "chore: regenerate config schema after phases list migration"
```

---

### Task 9: Update integration / component / kubernetes / operator tests

**Files:** under `tests/component_integration/`, `tests/integration/`, `tests/kubernetes/`, `tests/operator/`.

- [ ] **Step 1: Find dict-shape phases in non-unit tests**

```bash
grep -rn "phases.*=.*{\|\"phases\":\s*{" tests/ | grep -v tests/unit/ | head -50
```

- [ ] **Step 2: Migrate each fixture / yaml / parametrize**

Same mechanical translation as Task 6.

- [ ] **Step 3: Run each suite once**

Per the plan-ceremony rule: ONE invocation per suite; do **not** split by sub-folder.

```bash
uv run pytest -n auto tests/component_integration/ -x 2>&1 | tail -30
```

```bash
uv run pytest -n auto -m integration -x 2>&1 | tail -30
```

```bash
uv run pytest -n auto tests/kubernetes/ -x 2>&1 | tail -30
```

```bash
uv run pytest -n auto tests/operator/ -x 2>&1 | tail -30
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "test: migrate non-unit test phases fixtures to list shape"
```

---

### Task 10: Update operator-side k8s manifests / fixtures

**Files:**
- `tests/kubernetes/fixtures/*.yaml` — find any AIPerfJob CRs with `spec.benchmark.phases` as a dict.
- `docs/superpowers/specs/`, `docs/superpowers/plans/` — older specs may reference dict shape (do NOT rewrite history-of-thinking docs; only touch user-facing examples).
- Helm chart values / sample CRs under `helm/` or `deploy/` if they exist.

- [ ] **Step 1: Find AIPerfJob CRs with dict-shape phases**

```bash
grep -rn "^\s*phases:\s*$" tests/kubernetes/ helm/ deploy/ 2>/dev/null | head
```

- [ ] **Step 2: For each match, read the file and migrate the YAML** (same translation as Task 7).

- [ ] **Step 3: Validate**

```bash
uv run pytest -n auto tests/kubernetes/ -x 2>&1 | tail -30
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "k8s: migrate AIPerfJob CR fixtures to phases list shape"
```

---

### Task 11: Update user-facing docs

**Files:**
- `docs/tutorials/yaml-config.md` — phases section + any example.
- `docs/tutorials/sweeps.md:44,56-65,93,111-132,171,316` — phase-targeting in sweeps.
- `docs/tutorials/template-endpoint.md:247,289,327,360` — examples.
- `docs/tutorials/distributions.md:412,440` — examples.
- `src/aiperf/config/schema/README.md` — phases section.
- `src/aiperf/config/templates/README.md` — examples.
- `src/aiperf/config/__init__.py:24` — docstring already uses `phases[0].name` (no change, but verify).

- [ ] **Step 1: Migrate `docs/tutorials/yaml-config.md` phases section**

Read the relevant block and update YAML examples from dict to list-with-name. Update prose: change "Order is preserved (Python 3.7+)" → "Order is the list order".

- [ ] **Step 2: Migrate sweeps.md phase-targeting**

Sweep override syntax that uses dot-paths (e.g. `phases.warmup.rate`) is now ambiguous: list indexing? name lookup? The user-facing decision per the existing scenario template is **name-based** — implement in Task 14 — but for now in the docs migration, update examples to the list-of-named-overrides form (see Task 14 for spec). Mark the sweep section TODO if the override implementation isn't yet wired (Task 14 will sync the docs to the actual implementation).

- [ ] **Step 3: Other docs**

Migrate `template-endpoint.md`, `distributions.md`, `schema/README.md`, `templates/README.md` — same mechanical translation.

- [ ] **Step 4: Commit**

```bash
git add docs/ src/aiperf/config/schema/README.md src/aiperf/config/templates/README.md
git commit -s -m "docs: migrate phases dict examples to list-with-name shape"
```

---

### Task 12: Sync three-file rule (CLAUDE.md / copilot / cursor)

**Files:**
- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `.cursor/rules/python.mdc`

- [ ] **Step 1: Search for any phases-shape mention in these files**

```bash
grep -nE "phases\b|phase config" CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
```

- [ ] **Step 2: If references exist, update them in all three files identically**

Per the three-file sync rule, update content identically (only frontmatter differs; preserve `alwaysApply: true` in cursor file).

- [ ] **Step 3: Diff to confirm sync**

```bash
diff <(grep -A 20 -i "phase" CLAUDE.md) <(grep -A 20 -i "phase" .github/copilot-instructions.md)
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -s -m "docs: sync three-file phases-shape guidance"
```

---

### Task 13: Operator-side end-to-end validation on the live DGX cluster

**Files:** none (live verification — but report findings in commit message and update workflow file if any gotcha surfaces).

This task validates the WHOLE refactor on the actual cluster — the bug we're fixing only manifests at the CRD-storage boundary.

- [ ] **Step 1: Build and push a fresh ARM64 image**

Per `~/.claude/workflows/aiperf-dgx/build-and-push-arm64.md`:

```bash
artifacts/publish_aiperf_arm64.py
```

Wait for rollout to finish. The new tag will be `k8s-arm64-YYYYMMDD-HHMMSS-<sha>`.

- [ ] **Step 2: Submit a smoke job with list-shape phases**

Save as `/tmp/smoke-list.yaml` (substitute `<TAG>`):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: smoke-list
  namespace: acasagrande-aiperf-bench
spec:
  image: nvcr.io/nvidian/dynamo-dev/aiperf:k8s-arm64-<TAG>
  connectionsPerWorker: 50
  resourceMode: burstable
  ttlSecondsAfterFinished: 600
  benchmark:
    models: { items: [{name: mock}] }
    tokenizer: { name: builtin }
    endpoint:
      streaming: true
      urls:
      - http://aiperf-mock-server.acasagrande-aiperf-bench.svc.cluster.local:8000/v1/chat/completions
    datasets:
      main: { type: synthetic, prompts: { isl: { mean: 128 } } }
    phases:
    - name: warmup
      type: concurrency
      excludeFromResults: true
      concurrency: 16
      requests: 1000
    - name: profiling
      type: concurrency
      concurrency: 50
      requests: 50000
    runtime: { ui: none, workers: 4, workers_per_pod: 2, record_processors_per_pod: 1 }
  podTemplate:
    imagePullSecrets: [nvcr-imagepullsecret]
    nodeSelector: { kubernetes.io/arch: arm64, nodeGroup: customer-gpu }
    tolerations:
    - { effect: NoSchedule, key: dedicated, operator: Equal, value: user-workload }
    - { effect: NoExecute,  key: dedicated, operator: Equal, value: user-workload }
    - { effect: NoSchedule, key: nvidia.com/gpu, operator: Equal, value: present }
    - { effect: NoSchedule, key: team, operator: Equal, value: nemo-ci }
    - { effect: NoSchedule, key: kubernetes.io/arch, operator: Equal, value: arm64 }
```

```bash
KUBE_CONTEXT=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
kubectl --context $KUBE_CONTEXT apply -f /tmp/smoke-list.yaml
```

- [ ] **Step 3: Confirm CR persisted the list in correct order**

```bash
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  get aiperfjob smoke-list -o jsonpath='{.spec.benchmark.phases}' | python -m json.tool
```

Expected: a JSON array, `[0].name="warmup"`, `[1].name="profiling"`. **This is the bug fix.**

- [ ] **Step 4: Confirm runtime ran warmup first**

```bash
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  logs $(kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
    get pods -o name | grep smoke-list-controller) \
  -c timing-manager 2>&1 | grep -E "Phase.*started"
```

Expected: `Phase warmup started` BEFORE `Phase profiling started`.

- [ ] **Step 5: Cleanup**

```bash
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench delete aiperfjob smoke-list
```

- [ ] **Step 6: Commit (no code change — record verification)**

```bash
git commit --allow-empty -s -m "verify: live DGX confirms phases list preserves order at CRD boundary

Submitted AIPerfJob with phases=[warmup, profiling]. CR storage retained
list order (vs the old dict shape which alphabetized to [profiling, warmup]).
Timing-manager logged 'Phase warmup started' before 'Phase profiling started'."
```

---

### Task 14: Sweep override syntax — pick a name-targeted shape

**Files:**
- `src/aiperf/config/sweep.py` — locate the override-merge logic (it's currently a generic deep-merge that doesn't know about phases).
- `src/aiperf/config/templates/scenario_workload_profiles.yaml` — has `runs[].phases.test.rate` overrides.
- `docs/tutorials/sweeps.md` — sweep examples.

**Decision needed:** `phases` was previously a dict so dot-path `phases.warmup.rate` made sense for sweep overrides. With list shape, we need a way to reference a phase by name in an override.

**Recommended:** sweep `phases:` overrides accept a list of partial phase configs each with a `name`, and the merger matches by `name`:

```yaml
sweep:
  type: scenarios
  runs:
  - name: chatbot
    phases:
      - name: profiling
        rate: 50.0
        concurrency: 128
```

Phases not mentioned in the override are inherited unchanged from the base config.

- [ ] **Step 1: Write failing test for name-targeted sweep override**

```python
# tests/unit/config/test_sweep_phase_overrides.py
def test_sweep_run_phase_override_by_name():
    # Compose a base config with [warmup, profiling] and a sweep run that
    # overrides only profiling.rate; assert resolved run still has warmup
    # untouched and profiling.rate overridden.
    ...
```

(Fill in concrete code referencing the sweep resolution entrypoint — locate it via `grep -rn "scenarios" src/aiperf/config/sweep.py` and the existing scenario-resolution path.)

- [ ] **Step 2: Implement name-matched merge**

In whatever function performs the run-override merge, when the key is `phases` and the override value is a list of dicts each with `name`, merge entry-by-entry by name.

- [ ] **Step 3: Update `scenario_workload_profiles.yaml` to the new shape**

```yaml
sweep:
  type: scenarios
  runs:
  - name: chatbot
    datasets:
      workload:
        prompts:
          isl: {mean: 128, stddev: 20}
          osl: {mean: 64, stddev: 10}
    phases:
      - name: test
        rate: 50.0
        concurrency: 128
  # ... rest of the runs identical conversion ...
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -30
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -s -m "feat(sweep): name-targeted phase overrides for list-shaped phases"
```

- [ ] **Step 6: Update sweep tutorial docs (deferred from Task 11)**

Migrate `docs/tutorials/sweeps.md` examples to the new shape. Commit:

```bash
git add docs/tutorials/sweeps.md
git commit -s -m "docs(sweeps): name-targeted phase overrides"
```

---

### Task 15: Final ergonomics + full pre-commit

**Files:** none — verification gate.

- [ ] **Step 1: Run full repo pre-commit**

```bash
pre-commit run --all-files
```

Expected: clean. Fix any drift (formatting, regenerated docs, plugin schemas).

- [ ] **Step 2: Run all four test suites once each**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -10
uv run pytest -n auto tests/component_integration/ -x 2>&1 | tail -10
uv run pytest -n auto -m integration -x 2>&1 | tail -10
uv run pytest -n auto tests/kubernetes/ -x 2>&1 | tail -10
uv run pytest -n auto tests/operator/ -x 2>&1 | tail -10
```

All green.

- [ ] **Step 3: Run ergonomics floor**

```bash
make check-ergonomics
make check-ruff-baselined
```

Expected: zero new violations. If new violations land in files we touched, fix them — never grow the baseline.

- [ ] **Step 4: Run the LLM-ergonomics review**

This refactor changes a public API (`BenchmarkConfig.phases`). Per CLAUDE.md, run `/aiperf-llm-ergonomics-review` before shipping.

- [ ] **Step 5: Commit any cleanup**

```bash
git add -u
git commit -s -m "chore: post-refactor cleanup (formatting, ergonomics)"
```

---

## Open Questions / Risks

1. **Sweep dot-path overrides outside `scenarios`** — if any other sweep type (`distributions`, etc.) referenced `phases.X.Y`, Task 14 also needs to cover it. Audit during Task 14.
2. **The `from_user_config` alias** at `src/aiperf/timing/config.py:56` — confirm it doesn't have a parallel call path that bypasses the new list-iteration code.
3. **Rust port (`aiperf-rs/`) is out of scope** for this plan — Explore confirmed it's not in this worktree. When `aiperf-rs` is re-synced, its phases model will need the same migration; track separately.
4. **Migration messaging** — since we're rejecting dict shape with a hard error, the error in `parse_phases` (now removed — replaced by Pydantic's "Input should be a valid list" default) should be customized to say "phases is now a list of named entries; see docs/tutorials/yaml-config.md#phases for the new shape". Add to Task 3 if not already covered.

---

## Self-Review

**Spec coverage:** All inventory items from the Explore agent's report are covered: Pydantic model (Task 2, 3), normalizers (Task 3), all consumers (Task 4, 5), templates (Task 7), tests (Task 6, 9, 10), schema (Task 8), CRD (Task 8 confirms unchanged), docs (Task 11, 12), sweep overrides (Task 14), live verification (Task 13), final ergonomics (Task 15). The 14 templates listed in the inventory are all covered by Task 7's grep-then-migrate workflow.

**Placeholder scan:** Each step has concrete code snippets or exact commands. The placeholder warning in Task 14 ("Fill in concrete code…") is intentional — the sweep entrypoint location is to be discovered at execution time; the test contract is defined and the implementation pattern (deep-merge by name) is specified.

**Type consistency:** `BasePhaseConfig.name: str` is consistent across all tasks; `BenchmarkConfig.phases: list[PhaseConfig]` is consistent; the migration from `phase._name` to `phase.name` is mechanical and applied everywhere.

---
