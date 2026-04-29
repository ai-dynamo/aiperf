# `datasets` dict → list with `name` field — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `BenchmarkConfig.datasets` from `dict[str, DatasetConfig]` to `list[DatasetConfig]` where each entry carries a required `name: str` field. Eliminates the same Kubernetes CRD alphabetical-keying bug as the phases refactor — `get_default_dataset_name()` currently returns "first by insertion order" which K8s alphabetizes.

**Architecture:** Hard breaking change — no backcompat for the dict shape (this branch hasn't shipped to main, no users depend on the schema yet). The user-facing YAML migrates from `datasets: {main: {...}, eval: {...}}` to `datasets: [{name: main, ...}, {name: eval, ...}]`. Top-level `dataset:` (singular) shorthand still normalizes to a one-entry list. Mirror the phases-list refactor's helper/normalizer shape exactly — much smaller scope (~5 src files, ~10 templates+tests, ~30 fixture sites) since `datasets` has fewer consumers than phases.

**Tech Stack:** Pydantic v2, Pydantic discriminated unions (DatasetConfig is `SyntheticDataset | FileDataset | PublicDataset | ComposedDataset`), `uv run pytest -n auto`, ruff, kopf operator + kubernetes_asyncio.

---

## Naming decision

The existing `PublicDataset.name: PublicDatasetType` field collides with the new top-level `name: str` identifier. Rename the existing field:

| Before | After |
|---|---|
| `PublicDataset.name: PublicDatasetType` (e.g. `"sharegpt"`) | `PublicDataset.dataset: PublicDatasetType` |
| (no top-level name) | All dataset variants gain `name: str` (user-facing identifier, used in `phase.dataset = "main"` references) |

Example migration:

```yaml
# OLD
datasets:
  main:
    type: public
    name: sharegpt
    entries: 1000

# NEW
datasets:
  - name: main         # user-facing identifier (was the dict key)
    type: public
    dataset: sharegpt  # the HF dataset enum (was: name)
    entries: 1000
```

This is the only existing field that collides; `SyntheticDataset`, `FileDataset`, and `ComposedDataset` had no top-level `name` field.

---

## File Structure

**Core schema:**
- `src/aiperf/config/dataset.py` — Add `name: Annotated[str, Field(min_length=1, ...)]` to each of `SyntheticDataset`, `FileDataset`, `PublicDataset`, `ComposedDataset`. Rename `PublicDataset.name` → `PublicDataset.dataset`.
- `src/aiperf/config/config.py` — `BenchmarkConfig.datasets: list[DatasetConfig]`; replace `parse_datasets` (mode=before) with one that accepts list shape; add `validate_datasets_unique_names` (mode=after); update `validate_dataset_references` and `validate_phase_dataset_compatibility` to lookup by `.name`.
- `src/aiperf/config/_benchmark_normalizers.py` — `_normalize_dataset_and_phases` builds list when `dataset:` shorthand present; `parse_datasets_input` builds list-of-validated-models.
- `src/aiperf/config/_benchmark_helpers.py` — `get_default_dataset_name()` returns `self.datasets[0].name`; `_dataset_by_name(name)` helper for lookups; error messages list `[d.name for d in self.datasets]`.
- `src/aiperf/config/cli_converter.py:88` — `"datasets": [{"name": "main", **ds}]`.

**Consumers:**
- `src/aiperf/config/_dataset_resolver.py:44` — `for ds in run.cfg.datasets:` (use `ds.name` inside).
- `src/aiperf/cli_commands/config_cli.py:61` — `[d.name for d in config.datasets]`.
- Other small consumers per grep.

**Templates (9 files with `datasets:` blocks):**
- `embeddings.yaml`, `long_context.yaml`, `audio_multimodal.yaml`, `multimodal_vision.yaml`, `kv_cache_test.yaml`, `jinja2_variables.yaml`, `sweep_distributions.yaml`, `scenario_workload_profiles.yaml`, `warmup_profiling.yaml` (and any others that grep finds).
- Plus the singular `dataset:` shorthand templates (no change to YAML — normalizer wraps).

**Schema:**
- `src/aiperf/config/schema/aiperf-config.schema.json` — auto-regen via `make generate-config-schema`.
- CRD: `spec.benchmark` is opaque; no CRD-level change.

**Tests:**
- `tests/unit/config/test_datasets_list_shape.py` (NEW) — contract tests.
- All `tests/unit/`, `tests/component_integration/`, `tests/kubernetes/`, `tests/harness/` files that build dict-shape `datasets={...}` literals — full grep needed.

**Docs:**
- `docs/tutorials/yaml-config.md`, `docs/tutorials/template-endpoint.md`, `docs/tutorials/distributions.md` (any `datasets:` example).
- `src/aiperf/config/schema/README.md`, `src/aiperf/config/templates/README.md`.
- `src/aiperf/config/__init__.py:25-31` (the docstring example).
- `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` — only if they mention `datasets` shape (likely no).

---

## Per-task discipline

- **One** `uv run pytest -n auto tests/unit/ -x` invocation per task max (per memory `feedback_plan_ceremony_minimalism.md`). No subfolder splits.
- **`git commit -s --no-verify`** if dispatched as parallel agents (per memory `feedback_precommit_auto_stash_destroys_parallel_agents.md`). For sequential single-agent execution, regular commits are fine but pre-commit may auto-regen the schema mid-task — use `--no-verify` if templates are intentionally stale within a task.
- **No `git stash`, no `git restore`** (shell-blocked, per user's #1 rule).
- **Sign-off** required on every commit (`-s`).

---

## Task Decomposition

### Task 1: Lock the new shape with a contract test

**Files:**
- Create: `tests/unit/config/test_datasets_list_shape.py`

- [ ] **Step 1: Write the failing contract tests**

```python
# tests/unit/config/test_datasets_list_shape.py
"""Contract tests for the list-of-named-datasets schema (post-refactor)."""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig


_BASE: dict = {
    "models": "mock",
    "endpoint": {"urls": ["http://x:8000/v1/chat/completions"], "streaming": True},
    "phases": [
        {"name": "profiling", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def _cfg(datasets):
    return BenchmarkConfig.model_validate({**_BASE, "datasets": datasets})


def test_datasets_accepts_list_with_name_field():
    cfg = _cfg([
        {"name": "main", "type": "synthetic", "prompts": {"isl": {"mean": 128}}},
        {"name": "eval", "type": "synthetic", "prompts": {"isl": {"mean": 64}}},
    ])
    assert isinstance(cfg.datasets, list)
    assert [d.name for d in cfg.datasets] == ["main", "eval"]


def test_datasets_preserves_input_order():
    cfg = _cfg([
        {"name": "zebra", "type": "synthetic"},
        {"name": "alpha", "type": "synthetic"},
    ])
    # Insertion order — alphabetization would invert these.
    assert cfg.datasets[0].name == "zebra"
    assert cfg.datasets[1].name == "alpha"


def test_datasets_default_dataset_is_first_in_list():
    cfg = _cfg([
        {"name": "primary", "type": "synthetic"},
        {"name": "fallback", "type": "synthetic"},
    ])
    assert cfg.get_default_dataset_name() == "primary"


def test_datasets_rejects_dict_shape():
    with pytest.raises(ValueError, match="datasets must be a list"):
        _cfg({"main": {"type": "synthetic"}})


def test_datasets_rejects_missing_name():
    with pytest.raises(ValueError, match="name"):
        _cfg([{"type": "synthetic"}])


def test_datasets_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate dataset name"):
        _cfg([
            {"name": "d", "type": "synthetic"},
            {"name": "d", "type": "synthetic"},
        ])


def test_datasets_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 1 item"):
        _cfg([])


def test_phase_dataset_reference_resolves_by_name():
    """A phase referencing dataset='eval' must resolve to the eval entry by name lookup."""
    cfg = BenchmarkConfig.model_validate({
        **{k: v for k, v in _BASE.items() if k != "phases"},
        "datasets": [
            {"name": "main", "type": "synthetic"},
            {"name": "eval", "type": "synthetic"},
        ],
        "phases": [{
            "name": "p", "type": "concurrency", "requests": 1, "concurrency": 1,
            "dataset": "eval",
        }],
    })
    assert cfg.phases[0].dataset == "eval"


def test_public_dataset_uses_dataset_field_not_name():
    """PublicDataset.name was renamed to .dataset to free up `name` for the outer identifier."""
    cfg = _cfg([
        {"name": "my_public", "type": "public", "dataset": "sharegpt"},
    ])
    assert cfg.datasets[0].name == "my_public"
    assert cfg.datasets[0].dataset == "sharegpt"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest -n auto tests/unit/config/test_datasets_list_shape.py -x
```

Expected: all 9 tests FAIL — current schema is `dict[str, DatasetConfig]` and `PublicDataset.name` still exists.

- [ ] **Step 3: Commit the contract**

```bash
git add tests/unit/config/test_datasets_list_shape.py
git commit -s -m "test(config): contract for datasets as list with name field"
```

---

### Task 2: Add `name` to all `DatasetConfig` subclasses, rename `PublicDataset.name` → `dataset`

**Files:**
- Modify: `src/aiperf/config/dataset.py` — add `name` field to `SyntheticDataset`, `FileDataset`, `PublicDataset`, `ComposedDataset`. Rename `PublicDataset.name` → `PublicDataset.dataset`.

- [ ] **Step 1: Add the `name` field to each subclass**

In `src/aiperf/config/dataset.py`, locate each of the four classes (`SyntheticDataset`, `FileDataset`, `PublicDataset`, `ComposedDataset`). Right after each class's `model_config = ConfigDict(extra="forbid")`, add:

```python
    name: Annotated[
        str,
        Field(
            min_length=1,
            description="Dataset identifier — unique within the benchmark's datasets list. "
            "Used in `phase.dataset = '<name>'` references and in result file paths. "
            "Common names: 'main', 'eval', 'warmup_data'.",
        ),
    ]
```

- [ ] **Step 2: Rename `PublicDataset.name` → `PublicDataset.dataset`**

In the `PublicDataset` class definition, change the existing field (currently around line 303):

```python
# OLD
    name: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "AIPerf automatically downloads and parses these datasets. "
        ),
    ]

# NEW
    dataset: Annotated[
        PublicDatasetType,
        Field(
            description="Pre-configured public dataset to download and use for benchmarking. "
            "Name of the HuggingFace public dataset enum (e.g. 'sharegpt', 'alpaca'). "
            "AIPerf automatically downloads and parses these datasets.",
        ),
    ]
```

- [ ] **Step 3: Find and update every consumer of `PublicDataset.name`**

```bash
grep -rnE "PublicDataset.*\.name\b|public_dataset\.name\b|\.name.*PublicDatasetType" src/aiperf/ tests/ 2>/dev/null
```

For each match, change `.name` → `.dataset`. Watch out for false positives (the new outer `name` field on the same class is also called `name` — disambiguate by reading context).

Quick paths likely affected (audit each):
- `src/aiperf/dataset/loader/` — anywhere a `PublicDataset` config is consumed to fetch the HF dataset.
- `src/aiperf/config/resolvers.py` — if it pattern-matches on dataset type.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "refactor(config): add 'name' to all DatasetConfig subclasses; rename PublicDataset.name→dataset"
```

(Tests still red — that's fine; we'll go green after Task 3.)

---

### Task 3: Switch `BenchmarkConfig.datasets` to `list[DatasetConfig]`

**Files:**
- Modify: `src/aiperf/config/config.py:147-155` (field type), `~320-327` (parse_datasets validator), `~336-348` (validate_dataset_references), `~373-389` (validate_phase_dataset_compatibility), and add `validate_datasets_unique_names`.
- Modify: `src/aiperf/config/_benchmark_normalizers.py` — `_normalize_dataset_and_phases` and `parse_datasets_input` build list shape.
- Modify: `src/aiperf/config/_benchmark_helpers.py` — `get_default_dataset_name`, `_dataset_by_name`, and any error messages that previously did `sorted(self.datasets.keys())`.

- [ ] **Step 1: Change the field type and add uniqueness validator**

In `src/aiperf/config/config.py`, replace the `datasets:` field definition (currently at lines 147-155):

```python
    datasets: Annotated[
        list[DatasetConfig],
        Field(
            min_length=1,
            description="Named dataset configurations. Each entry must have a unique 'name' "
            "(e.g. 'main', 'eval'). Phases reference datasets by name "
            "(`phase.dataset = '<name>'`); when omitted, the FIRST dataset in the list is used. "
            "Singular `dataset:` shorthand at the BenchmarkConfig top level is normalized to "
            "a one-entry list with name='default'.",
        ),
    ]
```

Replace the `parse_datasets` validator (currently a `@field_validator("datasets", mode="before")` calling `parse_datasets_input`):

```python
    @field_validator("datasets", mode="before")
    @classmethod
    def parse_datasets(cls, v: Any) -> list[Any]:
        """Parse dataset configurations into a list shape, validating each item has a name."""
        return parse_datasets_input(v)
```

Add a uniqueness check `model_validator(mode="after")` (place near `validate_phase_names_unique`):

```python
    @model_validator(mode="after")
    def validate_datasets_unique_names(self) -> Self:
        """Reject duplicate dataset names — they must be unique within the list."""
        seen: set[str] = set()
        for ds in self.datasets:
            if ds.name in seen:
                raise ValueError(
                    f"duplicate dataset name '{ds.name}' — names must be unique. "
                    f"Found names: {[d.name for d in self.datasets]}"
                )
            seen.add(ds.name)
        return self
```

Update `validate_dataset_references` to compute the name set from the list:

```python
    @model_validator(mode="after")
    def validate_dataset_references(self) -> Self:
        """Validate that all dataset references in phase configs exist."""
        dataset_names = {d.name for d in self.datasets}
        for phase in self.phases:
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Phase config '{phase.name}' references undefined dataset "
                    f"'{phase.dataset}'. Available datasets: {sorted(dataset_names)}"
                )
        return self
```

Update `validate_phase_dataset_compatibility` to look up by name (it currently does `self.datasets.get(dataset_name)`):

```python
    @model_validator(mode="after")
    def validate_phase_dataset_compatibility(self) -> Self:
        from aiperf.config.resolved import check_phase_dataset_compatibility
        by_name = {d.name: d for d in self.datasets}
        for phase in self.phases:
            dataset_name = phase.dataset or self.get_default_dataset_name()
            ds = by_name.get(dataset_name)
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

Locate `_normalize_dataset_and_phases`. Modify the dataset shorthand branch (currently `data["datasets"] = {"default": data.pop("dataset")}`):

```python
def _normalize_dataset_and_phases(data: dict[str, Any]) -> None:
    if "dataset" in data and "datasets" not in data:
        ds = data.pop("dataset")
        if isinstance(ds, dict):
            ds = {"name": "default", **ds}
        data["datasets"] = [ds]

    if "phases" in data:
        phases = data["phases"]
        if isinstance(phases, dict) and "type" in phases:
            data["phases"] = [{"name": "default", **phases}]
```

Modify `parse_datasets_input` to build a list (currently returns `dict[str, Any]`):

```python
def parse_datasets_input(v: Any) -> list[Any]:
    """Parse dataset configurations into a list, handling composed datasets.

    Composed datasets don't have a 'type' field but have 'source' and 'augment'.
    Accepts already-constructed Pydantic models for programmatic use.
    """
    from aiperf.config.dataset import (
        ComposedDataset,
        FileDataset,
        PublicDataset,
        SyntheticDataset,
    )

    dataset_types = (SyntheticDataset, FileDataset, PublicDataset, ComposedDataset)

    if not isinstance(v, list):
        raise ValueError(
            f"datasets must be a list of named entries (was a dict in earlier versions); "
            f"got {type(v).__name__}. Use [{{'name': 'main', 'type': 'synthetic', ...}}, ...]. "
            f"See docs/tutorials/yaml-config.md#datasets."
        )

    return [
        _normalize_single_dataset_listed(idx, item, dataset_types)
        for idx, item in enumerate(v)
    ]


def _normalize_single_dataset_listed(
    idx: int, config: Any, dataset_types: tuple
) -> Any:
    if isinstance(config, dataset_types):
        return config
    if not isinstance(config, dict):
        raise ValueError(
            f"datasets[{idx}] must be a dictionary or Pydantic model"
        )
    if "name" not in config:
        raise ValueError(
            f"datasets[{idx}] is missing required 'name' field. "
            f"Each dataset entry needs a name (e.g. 'main', 'eval')."
        )
    name = config["name"]
    _hoist_synthetic_prompt_fields(config)

    is_composed = "source" in config and "augment" in config and "type" not in config
    if is_composed:
        return config
    if "type" not in config:
        config["type"] = "synthetic"
    return config
```

(Keep `_hoist_synthetic_prompt_fields` untouched — it operates on a single dataset's body, name-agnostic.)

- [ ] **Step 3: Update `_benchmark_helpers.py`**

Find the helper methods (currently around lines 47-57). Replace:

```python
# OLD
def _validate_dataset_name(self, name: str) -> None:
    if name not in self.datasets:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {sorted(self.datasets.keys())}"
        )

def get_default_dataset_name(self) -> str:
    return next(iter(self.datasets.keys()))

def _get_default_dataset(self) -> DatasetConfig:
    return next(iter(self.datasets.values()))

# NEW
def _validate_dataset_name(self, name: str) -> None:
    available = [d.name for d in self.datasets]  # type: ignore[attr-defined]
    if name not in available:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {sorted(available)}"
        )

def get_default_dataset_name(self) -> str:
    """Returns the name of the first dataset in the list (the default)."""
    return self.datasets[0].name  # type: ignore[attr-defined]

def _get_default_dataset(self):  # type: ignore[no-untyped-def]
    return self.datasets[0]  # type: ignore[attr-defined]

def _dataset_by_name(self, name: str):  # type: ignore[no-untyped-def]
    """Look up a dataset by name; raises if not found."""
    for d in self.datasets:  # type: ignore[attr-defined]
        if d.name == name:
            return d
    available = [d.name for d in self.datasets]  # type: ignore[attr-defined]
    raise KeyError(
        f"Dataset '{name}' not found. Available: {sorted(available)}"
    )
```

- [ ] **Step 4: Update `cli_converter.py`**

In `src/aiperf/config/cli_converter.py` (line ~88), change:

```python
# OLD
"datasets": {"main": ds},

# NEW
"datasets": [{"name": "main", **ds}],
```

(Read the surrounding code first — `ds` is built earlier in the function. Confirm it's a dict so the `**ds` spread works; if it's already a Pydantic model, use `{"name": "main", **ds.model_dump()}` instead.)

- [ ] **Step 5: Run the contract tests + the full unit suite**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -50
```

Expected: contract tests from Task 1 PASS. Many existing unit tests with dict-shape `datasets={...}` literals will FAIL — that's the breaking change surface; Task 4 fixes those.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -s -m "refactor(config): datasets is now list[DatasetConfig] with required 'name' field"
```

---

### Task 4: Fix consumers of `config.datasets`

**Files:**
- Modify: `src/aiperf/config/_dataset_resolver.py:44` — `for name, ds in run.cfg.datasets.items():` → list iteration with `ds.name`.
- Modify: `src/aiperf/cli_commands/config_cli.py:61` — `[d.name for d in config.datasets]`.
- Modify: any other consumer found via grep.

- [ ] **Step 1: Sweep `.datasets.items()/.values()/.keys()` and `cfg.datasets[<key>]`**

```bash
grep -rnE "\.datasets\.(items|values|keys)\(\)|cfg\.datasets\[\"|config\.datasets\[\"|run\.cfg\.datasets\[" src/aiperf/ 2>/dev/null
```

For each match, apply the mechanical translation:

| Old | New |
|---|---|
| `for name, ds in cfg.datasets.items():` | `for ds in cfg.datasets:` (use `ds.name`) |
| `for ds in cfg.datasets.values():` | `for ds in cfg.datasets:` |
| `for name in cfg.datasets.keys():` | `for ds in cfg.datasets:` (use `ds.name`) |
| `cfg.datasets[name]` | `cfg._dataset_by_name(name)` (or inline lookup) |
| `name in cfg.datasets` | `any(d.name == name for d in cfg.datasets)` |
| `sorted(cfg.datasets.keys())` | `sorted(d.name for d in cfg.datasets)` |
| `cfg.datasets[name] = ...` | rebuild list (rare) |

Specific known callsites:
- `src/aiperf/config/_dataset_resolver.py:44`: `for name, ds in run.cfg.datasets.items():` → `for ds in run.cfg.datasets:` (then use `ds.name` where `name` was used).
- `src/aiperf/cli_commands/config_cli.py:61`: `print(f"  Datasets: {list(config.datasets.keys())}")` → `print(f"  Datasets: {[d.name for d in config.datasets]}")`.

- [ ] **Step 2: Run the full unit suite**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -30
```

Expected: most green. Test-fixture failures (dict-shape literals built in tests/unit/) are fixed in Task 5.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -s -m "refactor: update all config.datasets consumers to iterate list"
```

---

### Task 5: Migrate unit-test fixtures and harness builders

**Files:**
- Modify: every `tests/unit/` and `tests/harness/` file that builds dict-shape `datasets={...}` or `"datasets": {...}`.

- [ ] **Step 1: Find every dict-shape datasets literal**

```bash
grep -rnE "datasets\s*=\s*\{|\"datasets\":\s*\{" tests/unit/ tests/harness/ tests/component_integration/ 2>/dev/null
```

Known callsites (audit each — there may be more):
- `tests/unit/test_cli_runner.py:24`
- `tests/unit/test_cli_runner_macos.py:19`
- `tests/unit/conftest.py:508`
- `tests/unit/api/conftest.py:197`
- `tests/unit/api/routers/conftest.py:44`
- `tests/unit/api/test_metrics_utils.py:28`
- `tests/unit/api/test_dashboard_js.py:218,246,632`
- `tests/harness/operator.py:37,62,84,140`
- `tests/harness/k8s.py:219`
- `tests/component_integration/` — anything that imports configs.

- [ ] **Step 2: Translate each**

```python
# OLD
datasets={"main": {"type": "synthetic", "prompts": {"isl": {"mean": 128}}}}

# NEW
datasets=[{"name": "main", "type": "synthetic", "prompts": {"isl": {"mean": 128}}}]
```

For Public dataset usage, also rename `name` → `dataset`:

```python
# OLD (if any)
datasets={"my_pub": {"type": "public", "name": "sharegpt"}}

# NEW
datasets=[{"name": "my_pub", "type": "public", "dataset": "sharegpt"}]
```

For programmatic Pydantic model construction:

```python
# OLD
SyntheticDataset(type="synthetic", entries=100)
# (often inserted into a dict at the call site)

# NEW
SyntheticDataset(name="main", type="synthetic", entries=100)
```

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -30
```

Expected: **fully green**.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "test(config): migrate datasets dict literals to list-with-name shape"
```

---

### Task 6: Migrate YAML templates

**Files:** `src/aiperf/config/templates/*.yaml` — every template with a `datasets:` block.

- [ ] **Step 1: List templates needing edits**

```bash
grep -lE "^datasets:" src/aiperf/config/templates/*.yaml
```

Known: `embeddings.yaml`, `long_context.yaml`, `audio_multimodal.yaml`, `multimodal_vision.yaml`, `kv_cache_test.yaml`, `jinja2_variables.yaml`, `sweep_distributions.yaml`, `scenario_workload_profiles.yaml`, `warmup_profiling.yaml`.

- [ ] **Step 2: For each, transform the block**

Example — `embeddings.yaml`:

```yaml
# OLD
datasets:
  single:
    type: synthetic
    entries: 1000
    prompts:
      isl: {mean: 256, stddev: 50}
      batchSize: 1
  batch_10:
    type: synthetic
    entries: 500
    prompts:
      isl: {mean: 128, stddev: 30}
      batchSize: 10

# NEW
datasets:
  - name: single
    type: synthetic
    entries: 1000
    prompts:
      isl: {mean: 256, stddev: 50}
      batchSize: 1
  - name: batch_10
    type: synthetic
    entries: 500
    prompts:
      isl: {mean: 128, stddev: 30}
      batchSize: 10
```

For PublicDataset usage in any template, also rename the field:

```yaml
# OLD
datasets:
  open:
    type: public
    name: sharegpt

# NEW
datasets:
  - name: open
    type: public
    dataset: sharegpt
```

**Special case — `scenario_workload_profiles.yaml`:** has nested `sweep.runs[].datasets.workload` overrides. After this task, the sweep override merger from the phases refactor (Task 14 of that plan, now landed) handles list-of-named-dicts via name match — convert nested overrides to list shape too:

```yaml
# OLD
runs:
  - name: chatbot
    datasets:
      workload:
        prompts: {isl: {mean: 128}}

# NEW
runs:
  - name: chatbot
    datasets:
      - name: workload
        prompts: {isl: {mean: 128}}
```

**Templates using singular `dataset:` shorthand** stay unchanged — the normalizer wraps them.

- [ ] **Step 3: Validate templates parse**

```bash
uv run aiperf plugins --validate
make validate-plugin-schemas
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/config/templates/
git commit -s -m "templates: migrate datasets dict to list-with-name across all examples"
```

---

### Task 7: Migrate non-unit tests + k8s fixtures

**Files:**
- `tests/component_integration/`, `tests/integration/`, `tests/kubernetes/`, `tests/operator/` (if exists), `helm/` / `deploy/`, `recipes/*/perf.yaml`.

- [ ] **Step 1: Find dict-shape datasets in non-unit tests + k8s fixtures**

```bash
grep -rnE "datasets\s*=\s*\{|\"datasets\":\s*\{|^\s*datasets:\s*$" \
  tests/component_integration/ tests/integration/ tests/kubernetes/ \
  helm/ deploy/ recipes/ 2>/dev/null
```

For YAML files, use the line-after check to confirm dict-of-named (not the singular shorthand).

- [ ] **Step 2: Translate each match**

Same mechanical translation as Tasks 5 + 6.

Known callsites:
- `tests/kubernetes/test_edge_cases.py:112,148`
- `tests/kubernetes/test_kueue_integration.py:57,122,660`
- `tests/kubernetes/helpers/operator.py:95`
- `recipes/*/perf.yaml` — 27 files; many likely use the singular `dataset:` shorthand and need no change. Audit each.

- [ ] **Step 3: Run each suite once**

ONE invocation per suite (per memory rule). Skip `tests/kubernetes/` if Kind cluster isn't available — collect-only is sufficient evidence:

```bash
uv run pytest -n auto tests/component_integration/ -x 2>&1 | tail -20
uv run pytest -n auto -m integration -x 2>&1 | tail -20
uv run pytest -n auto tests/kubernetes/ --collect-only 2>&1 | tail -10
```

Pre-existing failures (e.g. `test_adaptive_convergence`, `test_multi_run_confidence`) from the orchestrator async work are unrelated — document and skip.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -s -m "test: migrate non-unit datasets fixtures + k8s manifests to list shape"
```

---

### Task 8: Regenerate JSON schema; verify CRD opacity

**Files:**
- Regenerate: `src/aiperf/config/schema/aiperf-config.schema.json`.
- Confirm CRD: `deploy/helm/aiperf-operator/templates/crd.yaml` `spec.benchmark` retains `x-kubernetes-preserve-unknown-fields: true` (no schema-level dataset change expected).

- [ ] **Step 1: Regenerate**

```bash
make generate-config-schema
make generate-crd
make generate-all-docs
make generate-all-plugin-files
```

- [ ] **Step 2: Diff and sanity-check**

```bash
git diff --stat
git diff src/aiperf/config/schema/aiperf-config.schema.json | head -100
```

Expected:
- `"datasets"` definition: `"type": "object"` + `additionalProperties` → `"type": "array"` + `items`.
- Each dataset variant (`SyntheticDataset`, `FileDataset`, `PublicDataset`, `ComposedDataset`) gains a required `name` property.
- `PublicDataset` schema: `"name"` (the old PublicDatasetType field) renamed to `"dataset"`.
- `aiperfjob` CRD: no diff (opacity preserved).

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/config/schema/aiperf-config.schema.json
git commit -s -m "chore: regenerate config schema after datasets list migration"
```

---

### Task 9: Migrate user-facing docs

**Files:**
- `docs/tutorials/yaml-config.md` — datasets section + any example.
- `docs/tutorials/template-endpoint.md`, `docs/tutorials/distributions.md` — examples.
- `src/aiperf/config/schema/README.md`, `src/aiperf/config/templates/README.md`.
- `src/aiperf/config/__init__.py:25-31` — module docstring example.
- `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` (only if they mention datasets shape — likely no, but grep to confirm).

- [ ] **Step 1: Find every `datasets:` example in docs**

```bash
grep -rnB1 -A8 "^datasets:" docs/tutorials/ src/aiperf/config/schema/README.md src/aiperf/config/templates/README.md
grep -nE "datasets\s*=\s*\{|\"datasets\":\s*\{" src/aiperf/config/__init__.py
grep -nE "datasets" CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc 2>/dev/null
```

- [ ] **Step 2: Migrate each example**

YAML examples: dict → list-with-name (same as Task 6). Update any prose that says "dict of named datasets" or similar to "list of named datasets" / "list ordered, first is default".

`src/aiperf/config/__init__.py:25-31` docstring:

```python
# OLD
...     datasets={"main": {"type": "synthetic", "count": 1000, "prompts": {"isl": 512}}},

# NEW
...     datasets=[{"name": "main", "type": "synthetic", "count": 1000, "prompts": {"isl": 512}}],
```

If `CLAUDE.md` etc. have nothing about datasets shape, skip the three-file sync (it'll be a no-op).

- [ ] **Step 3: Commit**

```bash
git add docs/ src/aiperf/config/__init__.py src/aiperf/config/schema/README.md src/aiperf/config/templates/README.md
# add CLAUDE.md/.github/copilot-instructions.md/.cursor/rules/python.mdc if changed
git commit -s -m "docs: migrate datasets dict examples to list-with-name shape"
```

---

### Task 10: Live DGX validation

**Files:** none — live verification.

This validates the WHOLE refactor on the actual cluster — the bug being fixed manifests at the CRD-storage boundary.

- [ ] **Step 1: Build and push fresh ARM64 image**

Per `~/.claude/workflows/aiperf-dgx/build-and-push-arm64.md`:

```bash
artifacts/publish_aiperf_arm64.py
```

Capture the new image tag.

- [ ] **Step 2: Submit smoke job with multi-dataset list-shape config**

`/tmp/smoke-datasets.yaml` (substitute `<TAG>`):

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: smoke-datasets
  namespace: acasagrande-aiperf-bench
spec:
  image: nvcr.io/nvidian/dynamo-dev/aiperf:<TAG>
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
    - name: zebra
      type: synthetic
      prompts: { isl: { mean: 128 } }
    - name: alpha
      type: synthetic
      prompts: { isl: { mean: 64 } }
    phases:
    - name: warmup
      type: concurrency
      excludeFromResults: true
      concurrency: 8
      requests: 500
      dataset: alpha
    - name: profiling
      type: concurrency
      concurrency: 50
      requests: 20000
      dataset: zebra
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

`zebra` (alphabetically last) is intentionally before `alpha` — under the old dict shape, K8s would alphabetize this to `[alpha, zebra]` and the implicit-default would be `alpha`. With list shape, `zebra` MUST remain first.

```bash
KUBE_CONTEXT=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-01
kubectl --context $KUBE_CONTEXT apply -f /tmp/smoke-datasets.yaml
```

- [ ] **Step 3: Verify CR storage retains list order**

```bash
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  get aiperfjob smoke-datasets \
  -o jsonpath='{.spec.benchmark.datasets}' | python -m json.tool
```

**Required:** JSON array, `[0].name == "zebra"`, `[1].name == "alpha"`. If you see them alphabetized OR an object/dict, REPORT BLOCKED.

- [ ] **Step 4: Verify both phases ran with their explicit dataset references**

```bash
CTRL=$(kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  get pods -o name | grep smoke-datasets-controller | head -1)
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  logs $CTRL -c dataset-manager --tail=200 2>&1 | grep -iE "dataset|loaded"
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench \
  get aiperfjob smoke-datasets -o jsonpath='{.status.phase}{"\n"}{.status.summary}{"\n"}'
```

Expected: dataset-manager logs both `zebra` and `alpha` loaded; final phase is `Completed` with non-empty `summary`.

- [ ] **Step 5: Cleanup**

```bash
kubectl --context $KUBE_CONTEXT -n acasagrande-aiperf-bench delete aiperfjob smoke-datasets
```

- [ ] **Step 6: Empty verification commit**

```bash
git commit --allow-empty -s -m "$(cat <<'EOF'
verify(k8s): list-shape datasets preserve order at CRD boundary

Submitted AIPerfJob smoke-datasets with datasets=[zebra, alpha]
(alphabetical inverse of CRD insertion). CR storage retained list order
(vs old dict shape which would have alphabetized to [alpha, zebra] and
silently changed the implicit-default dataset). Phase 'warmup' loaded
'alpha' explicitly; phase 'profiling' loaded 'zebra' explicitly. Job
completed with both datasets correctly resolved by name.

Image tag: <TAG>
EOF
)"
```

---

### Task 11: Final shipping gate

**Files:** none — verification gate only.

- [ ] **Step 1: Run full pre-commit**

```bash
pre-commit run --all-files 2>&1 | tail -30
```

Auto-format / regenerated artifacts: commit them as a chore commit if any landed.

- [ ] **Step 2: Ergonomics floor**

```bash
make check-ergonomics
make check-ruff-baselined
```

Expected: zero NEW violations attributable to this refactor. Pre-existing violations in unrelated files: skip and document.

- [ ] **Step 3: Full unit suite + targeted sweep tests**

```bash
uv run pytest -n auto tests/unit/ -x 2>&1 | tail -10
```

Expected: green except possibly the known `test_error_queue` xdist flake (per memory).

- [ ] **Step 4: LLM-ergonomics spot-check**

This refactor changes a public API (`BenchmarkConfig.datasets` shape; `PublicDataset.name` → `.dataset`). Per CLAUDE.md, the `/aiperf-llm-ergonomics-review` review is required.

Spot-check the new error messages in `parse_datasets_input`, `validate_datasets_unique_names`, and `_dataset_by_name` — they should name the operation, the input, and a likely cause/next-step (e.g. point to the docs section). Tighten any vague messages.

```bash
grep -nE "raise (ValueError|KeyError|TypeError)" \
  src/aiperf/config/_benchmark_normalizers.py \
  src/aiperf/config/_benchmark_helpers.py \
  src/aiperf/config/config.py | head
```

- [ ] **Step 5: Summary commit**

```bash
git commit --allow-empty -s -m "$(cat <<'EOF'
verify: final shipping gate green for datasets dict→list migration

- pre-commit run --all-files: clean
- make check-ergonomics: no new violations
- make check-ruff-baselined: no new violations
- pytest -n auto tests/unit/: all green (1 known xdist flake at most)
- LLM-ergonomics spot-check: error messages reviewed

Bug fix verified live on DGX (see Task 10 verification commit).
EOF
)"
```

---

## Open Questions / Risks

1. **PublicDataset.name rename collisions:** the rename to `.dataset` could conflict with how loaders consume the public dataset spec. Audit everything under `src/aiperf/dataset/loader/` before Task 2 commit. If loaders reference `.name` to fetch HF data, those callsites need both name (the new outer identifier) AND dataset (the new inner field) renames.

2. **Recipes (`recipes/*/perf.yaml`):** 27 files were already migrated by the phases-refactor agent for test_recipes.py. Many use singular `dataset:` shorthand (no change needed); some may use plural `datasets:` (need migration). Audit during Task 7.

3. **Sweep override semantics for datasets:** the phases refactor introduced name-targeted merge for `phases:` overrides. The same merger should auto-handle `datasets:` overrides since `_is_named_dict_list` gates by structure (any list of dicts each with a `name`). Verify by adding a sweep-with-dataset-override test. If broken, add a Task 7.5 for the sweep adjustment.

4. **`run.cfg.datasets[name]` callsites in places I didn't grep deeply** (e.g. `src/aiperf/operator/`, `src/aiperf/api/`): the Task 4 grep should catch them, but a final `pre-commit run --all-files` check after Task 4 is the safety net.

---

## Self-Review

**Spec coverage:** All audit findings from the conversation are covered: schema flip (Task 3), `PublicDataset.name` rename (Task 2), normalizers (Task 3), every consumer (Task 4), templates (Task 6), tests (Tasks 5, 7), schema regen (Task 8), docs (Task 9), live DGX validation (Task 10), final gate (Task 11). The "datasets" CRD bug (alphabetical reordering of map keys → wrong implicit default) is directly tested in Task 10 with `zebra` deliberately placed before `alpha`.

**Placeholder scan:** Each task step has concrete code or exact commands. No "TBD" / "implement later" / "fill in details". The grep audits in Tasks 4-7-9 are concrete commands, not vague directives.

**Type consistency:** `BasePhaseConfig.name`-style pattern reused — `name: Annotated[str, Field(min_length=1, ...)]` on each subclass. `BenchmarkConfig.datasets: list[DatasetConfig]` matches `phases: list[PhaseConfig]`. `get_default_dataset_name()` returns `self.datasets[0].name` consistently. `_dataset_by_name(name)` is the new lookup helper used in Task 4 translations.

---
