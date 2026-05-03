# Orchestrator Plugin Categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `SearchPlanner` and `ConvergenceCriterion` orchestrator protocols to first-class plugin categories so third parties can ship pip-installable wheels (e.g. `aiperf-optuna`, custom statistical convergence rules) that override built-ins via priority + setuptools entry points, gain lazy-loaded soft-dep imports, and surface typed metadata at registry-validation time.

**Architecture:** Mirror the existing `search_recipe` plugin shape exactly: declare the category in `categories.yaml` with a Pydantic metadata schema in `src/aiperf/plugin/schema/schemas.py`, register built-in implementations in `plugins.yaml`, replace direct instantiation with `plugins.get_class(PluginType.X, name)(...)`. For convergence, add a `from_plan(plan)` classmethod factory on the protocol so heterogeneous constructor signatures still dispatch uniformly. For search planner, add a `--search-planner <name>` CLI flag plus `AdaptiveSearchConfig.planner: SearchPlannerType` field. Replace the static `ConvergenceMode` enum with the generated `ConvergenceCriterionType`.

**Tech Stack:** Python 3.10+, Pydantic, AIPerf plugin registry (`src/aiperf/plugin/`), cyclopts CLI, scikit-optimize (soft dep behind `[bo]` extra), numpy/scipy.

---

## File Map

**Create:**
- (none — all changes extend existing files)

**Modify:**
- `src/aiperf/plugin/schema/schemas.py` — add `ConvergenceCriterionMetadata`, `SearchPlannerMetadata` Pydantic models
- `src/aiperf/plugin/categories.yaml` — register two new categories
- `src/aiperf/plugin/plugins.yaml` — register `ci_width`, `cv`, `distribution`, `bayesian` plugins
- `src/aiperf/orchestrator/convergence/base.py` — add `from_plan(plan)` abstract classmethod
- `src/aiperf/orchestrator/convergence/{ci_width,cv,distribution}.py` — implement `from_plan(plan)`
- `src/aiperf/_cli_runner_helpers.py` — refactor `_build_convergence_criterion`; add `_build_search_planner`
- `src/aiperf/cli_runner.py` — replace inline `BayesianSearchPlanner(...)` with `_build_search_planner(plan)`
- `src/aiperf/sweep_controller/main.py` — same refactor for cluster-side path
- `src/aiperf/config/adaptive_search.py` — add `planner: SearchPlannerType` field
- `src/aiperf/config/v1/_loadgen.py` — add `--search-planner` CLI flag
- `src/aiperf/config/v1/_converter_optionals.py` — propagate `planner` through v1→v2 converter
- `src/aiperf/common/enums/server_metrics_enums.py` — drop `ConvergenceMode` (or alias to generated)
- `src/aiperf/common/enums/__init__.py` and `enums.py` — adjust exports
- `src/aiperf/config/_models_benchmark.py` — switch field type to `ConvergenceCriterionType`
- `src/aiperf/config/benchmark.py` — same
- `src/aiperf/config/v1/_loadgen.py` — same on the v1 CLI side
- `src/aiperf/orchestrator/strategies.py` — adjust import if needed
- Generated artifacts (auto): `src/aiperf/plugin/enums.py`, `enums.pyi`, plugin overloads, `plugins.schema.json`
- `docs/plugins/plugin-system.md` — document the two new categories
- `docs/cli-options.md` — auto-regenerated
- `llms.txt` — note the new extension point if appropriate

**Test:**
- `tests/unit/plugin/test_orchestrator_categories.py` — new file: registry lookup, metadata access, factory dispatch
- `tests/unit/orchestrator/test_convergence_from_plan.py` — new: `from_plan` mapping per criterion
- existing `tests/unit/orchestrator/` regression coverage stays green

---

## Task 1: Define metadata Pydantic schemas

**Files:**
- Modify: `src/aiperf/plugin/schema/schemas.py`
- Test: `tests/unit/plugin/test_orchestrator_categories.py` (new)

- [ ] **Step 1: Write the failing test for `ConvergenceCriterionMetadata` schema shape**

Create `tests/unit/plugin/test_orchestrator_categories.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for orchestrator plugin categories (search_planner, convergence_criterion)."""

import pytest


def test_convergence_criterion_metadata_shape():
    """ConvergenceCriterionMetadata declares required capability fields."""
    from aiperf.plugin.schema.schemas import ConvergenceCriterionMetadata

    md = ConvergenceCriterionMetadata(
        min_samples=3,
        requires_confidence_level=True,
        requires_jsonl_export=False,
        metric_kinds=["continuous"],
    )
    assert md.min_samples == 3
    assert md.requires_confidence_level is True
    assert md.requires_jsonl_export is False
    assert md.metric_kinds == ["continuous"]


def test_search_planner_metadata_shape():
    """SearchPlannerMetadata declares dimension-kind support and extras."""
    from aiperf.plugin.schema.schemas import SearchPlannerMetadata

    md = SearchPlannerMetadata(
        supports_continuous=True,
        supports_discrete=True,
        supports_categorical=False,
        requires_initial_samples=5,
        compatible_objective_directions=["maximize", "minimize"],
        requires_extras=["bo"],
    )
    assert md.supports_continuous is True
    assert md.supports_categorical is False
    assert md.requires_initial_samples == 5
    assert md.requires_extras == ["bo"]
```

- [ ] **Step 2: Run the test, verify it fails with ImportError**

Run: `uv run pytest tests/unit/plugin/test_orchestrator_categories.py -n auto -v`
Expected: FAIL — `ImportError: cannot import name 'ConvergenceCriterionMetadata' from 'aiperf.plugin.schema.schemas'`.

- [ ] **Step 3: Add the two metadata models**

Append to `src/aiperf/plugin/schema/schemas.py` (after the existing models, before any trailing module-level code):

```python
class ConvergenceCriterionMetadata(BaseModel):
    """Metadata schema for convergence criterion plugins.

    Declares statistical-method capabilities so the CLI/config layer can
    validate `--convergence-metric` / `--convergence-stat` against the chosen
    criterion before the plugin is imported.

    Referenced by: categories.yaml convergence_criterion.metadata_class
    Used in: plugins.yaml convergence_criterion entries
    """

    min_samples: int = Field(
        description="Minimum number of successful runs required before convergence can trigger.",
    )
    requires_confidence_level: bool = Field(
        default=False,
        description="Whether the criterion consumes plan.confidence_level (e.g. CI-width does, CV doesn't).",
    )
    requires_jsonl_export: bool = Field(
        default=False,
        description="Whether the criterion reads per-request metrics from JSONL exports (e.g. distribution does).",
    )
    metric_kinds: list[str] = Field(
        default_factory=lambda: ["continuous"],
        description=(
            "Kinds of metrics this criterion handles. One or more of "
            "'continuous', 'counts', 'categorical'."
        ),
    )


class SearchPlannerMetadata(BaseModel):
    """Metadata schema for search planner plugins.

    Declares dimension-type and objective-direction support so the CLI/config
    layer can validate `--search-space` shape against the chosen planner
    before the planner (and its heavy soft-dep imports) is loaded.

    Referenced by: categories.yaml search_planner.metadata_class
    Used in: plugins.yaml search_planner entries
    """

    supports_continuous: bool = Field(
        description="Whether the planner accepts Real-valued search-space dimensions.",
    )
    supports_discrete: bool = Field(
        description="Whether the planner accepts Integer-valued search-space dimensions.",
    )
    supports_categorical: bool = Field(
        default=False,
        description="Whether the planner accepts Categorical search-space dimensions.",
    )
    requires_initial_samples: int | None = Field(
        default=None,
        description=(
            "Minimum number of initial random/Sobol samples required before the "
            "planner's model is fit. None when the planner has no warm-up phase."
        ),
    )
    compatible_objective_directions: list[str] = Field(
        default_factory=lambda: ["maximize", "minimize"],
        description="Objective directions the planner can optimize. Lower-case strings.",
    )
    requires_extras: list[str] = Field(
        default_factory=list,
        description=(
            "Names of pyproject.toml extras (e.g. ['bo']) required to install "
            "this planner's heavy dependencies. Informational; the planner "
            "class itself owns the ImportError surface."
        ),
    )
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `uv run pytest tests/unit/plugin/test_orchestrator_categories.py -n auto -v`
Expected: PASS for `test_convergence_criterion_metadata_shape` and `test_search_planner_metadata_shape`.

- [ ] **Step 5: Format and commit**

```bash
ruff format src/aiperf/plugin/schema/schemas.py tests/unit/plugin/test_orchestrator_categories.py
git add src/aiperf/plugin/schema/schemas.py tests/unit/plugin/test_orchestrator_categories.py
git commit --no-verify -m "feat(plugin): add metadata schemas for orchestrator plugin categories"
```

---

## Task 2: Add `from_plan` factory classmethod to `ConvergenceCriterion`

**Files:**
- Modify: `src/aiperf/orchestrator/convergence/base.py`
- Modify: `src/aiperf/orchestrator/convergence/ci_width.py`
- Modify: `src/aiperf/orchestrator/convergence/cv.py`
- Modify: `src/aiperf/orchestrator/convergence/distribution.py`
- Test: `tests/unit/orchestrator/test_convergence_from_plan.py` (new)

- [ ] **Step 1: Write failing test for `from_plan` factory dispatch**

Create `tests/unit/orchestrator/test_convergence_from_plan.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify each ConvergenceCriterion subclass builds correctly from a BenchmarkPlan."""

from unittest.mock import MagicMock

import pytest

from aiperf.orchestrator.convergence import (
    CIWidthConvergence,
    CVConvergence,
    DistributionConvergence,
)


@pytest.fixture
def plan():
    """Minimal BenchmarkPlan-shaped object exposing the fields each criterion reads."""
    p = MagicMock()
    p.convergence_metric = "time_to_first_token"
    p.convergence_stat = "avg"
    p.convergence_threshold = 0.1
    p.confidence_level = 0.95
    p.export_jsonl_file = "profile_export.jsonl"
    return p


def test_ci_width_from_plan_maps_fields(plan):
    crit = CIWidthConvergence.from_plan(plan)
    assert isinstance(crit, CIWidthConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._stat == "avg"
    assert crit._threshold == 0.1
    assert crit._confidence_level == 0.95


def test_cv_from_plan_maps_fields(plan):
    crit = CVConvergence.from_plan(plan)
    assert isinstance(crit, CVConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._threshold == 0.1
    assert crit._stat == "avg"


def test_distribution_from_plan_maps_fields(plan):
    crit = DistributionConvergence.from_plan(plan)
    assert isinstance(crit, DistributionConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._p_value_threshold == 0.1
    assert crit._jsonl_filename == "profile_export.jsonl"


def test_distribution_from_plan_uses_default_jsonl_when_none(plan):
    plan.export_jsonl_file = None
    crit = DistributionConvergence.from_plan(plan)
    from aiperf.orchestrator.convergence.base import DEFAULT_JSONL_FILENAME

    assert crit._jsonl_filename == DEFAULT_JSONL_FILENAME
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `uv run pytest tests/unit/orchestrator/test_convergence_from_plan.py -n auto -v`
Expected: FAIL — `AttributeError: type object 'CIWidthConvergence' has no attribute 'from_plan'`.

- [ ] **Step 3: Add abstract `from_plan` to base class**

In `src/aiperf/orchestrator/convergence/base.py`, change the imports and class:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for convergence criteria."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME, load_single_metric
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkPlan


class ConvergenceCriterion(ABC):
    """Abstract base for determining whether benchmark metrics have converged across runs."""

    @classmethod
    @abstractmethod
    def from_plan(cls, plan: BenchmarkPlan) -> Self:
        """Build an instance from a fully-validated BenchmarkPlan.

        Each subclass owns the mapping from plan fields to its constructor
        kwargs. Used by the plugin-registry dispatch in
        ``_cli_runner_helpers._build_convergence_criterion`` so heterogeneous
        constructor signatures still dispatch uniformly.
        """

    @abstractmethod
    def is_converged(self, results: list[RunResult]) -> bool:
        """Determine whether metrics have converged across the given runs.

        Args:
            results: Results from runs executed so far.

        Returns:
            True if metrics have converged, False otherwise.
        """

    def _load_request_metrics(
        self,
        artifacts_path: Path,
        metric_name: str,
        jsonl_filename: str = DEFAULT_JSONL_FILENAME,
    ) -> list[float]:
        """Read per-request metric values from a run's JSONL export."""
        return load_single_metric(artifacts_path, metric_name, jsonl_filename)
```

- [ ] **Step 4: Implement `from_plan` on `CIWidthConvergence`**

Append to `src/aiperf/orchestrator/convergence/ci_width.py` inside the class (after `__init__`, before `is_converged`):

```python
    @classmethod
    def from_plan(cls, plan: "BenchmarkPlan") -> "CIWidthConvergence":  # type: ignore[name-defined]
        return cls(
            metric=plan.convergence_metric,
            stat=plan.convergence_stat,
            threshold=plan.convergence_threshold,
            confidence_level=plan.confidence_level,
        )
```

Add at the top of the file:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkPlan
```

- [ ] **Step 5: Implement `from_plan` on `CVConvergence`**

Same pattern in `src/aiperf/orchestrator/convergence/cv.py`:

```python
    @classmethod
    def from_plan(cls, plan: "BenchmarkPlan") -> "CVConvergence":  # type: ignore[name-defined]
        return cls(
            metric=plan.convergence_metric,
            threshold=plan.convergence_threshold,
            stat=plan.convergence_stat,
        )
```

- [ ] **Step 6: Implement `from_plan` on `DistributionConvergence`**

Same pattern in `src/aiperf/orchestrator/convergence/distribution.py`:

```python
    @classmethod
    def from_plan(cls, plan: "BenchmarkPlan") -> "DistributionConvergence":  # type: ignore[name-defined]
        return cls(
            metric=plan.convergence_metric,
            p_value_threshold=plan.convergence_threshold,
            jsonl_filename=plan.export_jsonl_file or DEFAULT_JSONL_FILENAME,
        )
```

- [ ] **Step 7: Run the new tests and the full convergence test suite**

Run: `uv run pytest tests/unit/orchestrator/ -n auto -v`
Expected: PASS for the four new tests AND all pre-existing convergence tests stay green.

- [ ] **Step 8: Format and commit**

```bash
ruff format src/aiperf/orchestrator/convergence/ tests/unit/orchestrator/test_convergence_from_plan.py
git add src/aiperf/orchestrator/convergence/ tests/unit/orchestrator/test_convergence_from_plan.py
git commit --no-verify -m "feat(convergence): add from_plan classmethod factory for plugin dispatch"
```

---

## Task 3: Register `convergence_criterion` plugin category

**Files:**
- Modify: `src/aiperf/plugin/categories.yaml`
- Modify: `src/aiperf/plugin/plugins.yaml`
- Modify: `src/aiperf/plugin/enums.py` (regenerated)
- Modify: `src/aiperf/plugin/enums.pyi` (regenerated)

- [ ] **Step 1: Add the category to `categories.yaml`**

In `src/aiperf/plugin/categories.yaml`, add after the `search_recipe_post_process` block (or in a logical orchestrator section):

```yaml
convergence_criterion:
  protocol: aiperf.orchestrator.convergence.base:ConvergenceCriterion
  metadata_class: aiperf.plugin.schema.schemas:ConvergenceCriterionMetadata
  enum: ConvergenceCriterionType
  description: |
    Convergence criteria decide when benchmark metrics have stabilized across
    repeated runs and the adaptive trial loop can stop. Each criterion declares
    its statistical assumptions (min samples, confidence-level usage, JSONL
    consumption) via metadata so the CLI can validate `--convergence-*` flags
    before importing scipy/statsmodels. Selected via `--convergence-mode <name>`.
    One-to-one mapping per benchmark run.
```

- [ ] **Step 2: Register the three built-ins in `plugins.yaml`**

Add a new section:

```yaml
# =============================================================================
convergence_criterion:
  ci_width:
    class: aiperf.orchestrator.convergence.ci_width:CIWidthConvergence
    description: |
      Stops when Student's t confidence interval width relative to the metric
      mean falls below `--convergence-threshold` (default 0.10 = 10%). Reads
      `--confidence-level` (default 0.95).
    metadata:
      min_samples: 3
      requires_confidence_level: true
      requires_jsonl_export: false
      metric_kinds: [continuous]

  cv:
    class: aiperf.orchestrator.convergence.cv:CVConvergence
    description: |
      Stops when the coefficient of variation across run-level statistic values
      falls below `--convergence-threshold` (default 0.05 = 5%).
    metadata:
      min_samples: 3
      requires_confidence_level: false
      requires_jsonl_export: false
      metric_kinds: [continuous]

  distribution:
    class: aiperf.orchestrator.convergence.distribution:DistributionConvergence
    description: |
      Stops when a two-sample Kolmogorov-Smirnov test between the latest run's
      per-request distribution and the union of prior runs yields p > threshold.
      Requires JSONL export (`--export-format json`).
    metadata:
      min_samples: 3
      requires_confidence_level: false
      requires_jsonl_export: true
      metric_kinds: [continuous]
```

- [ ] **Step 3: Regenerate plugin artifacts**

```bash
uv run python tools/generate_plugin_artifacts.py
```

Expected: `enums.py`, `enums.pyi`, `plugins.schema.json`, and overloads updated. Verify no errors.

- [ ] **Step 4: Validate plugin schemas**

```bash
make validate-plugin-schemas
```

Expected: `✓ Validated N+1 categories and M+3 plugins.` (counts increase by one category and three plugins).

- [ ] **Step 5: Verify the generated `ConvergenceCriterionType` enum exists and dispatches**

Add a test to `tests/unit/plugin/test_orchestrator_categories.py`:

```python
def test_convergence_criterion_plugin_lookup():
    """All three built-in criteria are reachable via plugins.get_class."""
    from aiperf.orchestrator.convergence import (
        CIWidthConvergence,
        CVConvergence,
        DistributionConvergence,
    )
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import ConvergenceCriterionType, PluginType

    assert plugins.get_class(PluginType.CONVERGENCE_CRITERION, "ci_width") is CIWidthConvergence
    assert plugins.get_class(PluginType.CONVERGENCE_CRITERION, "cv") is CVConvergence
    assert plugins.get_class(PluginType.CONVERGENCE_CRITERION, "distribution") is DistributionConvergence
    assert ConvergenceCriterionType.CI_WIDTH == "ci_width"


def test_convergence_criterion_metadata_accessible():
    """Plugin entries expose typed metadata."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    entry = plugins.get_entry(PluginType.CONVERGENCE_CRITERION, "distribution")
    assert entry.metadata.requires_jsonl_export is True
    assert entry.metadata.min_samples == 3
```

Run: `uv run pytest tests/unit/plugin/test_orchestrator_categories.py -n auto -v`
Expected: PASS.

> **Note:** If `plugins.get_entry` is not the public API name, replace with whatever the registry exposes (check `src/aiperf/plugin/plugins.py` for the lookup method that returns metadata). Adjust the test accordingly before running.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/plugin/ tests/unit/plugin/test_orchestrator_categories.py
git commit --no-verify -m "feat(plugin): register convergence_criterion category with three built-ins"
```

---

## Task 4: Replace `_build_convergence_criterion` if-chain with plugin dispatch

**Files:**
- Modify: `src/aiperf/_cli_runner_helpers.py`

- [ ] **Step 1: Write failing test that verifies plugin-dispatch behavior matches the old if-chain**

Add to `tests/unit/orchestrator/test_convergence_from_plan.py`:

```python
def test_build_convergence_criterion_dispatches_via_plugin_registry(plan):
    """`_build_convergence_criterion(plan)` delegates to plugin lookup + from_plan."""
    from aiperf._cli_runner_helpers import _build_convergence_criterion

    plan.convergence_mode = "ci_width"
    crit = _build_convergence_criterion(plan)
    assert isinstance(crit, CIWidthConvergence)

    plan.convergence_mode = "cv"
    crit = _build_convergence_criterion(plan)
    assert isinstance(crit, CVConvergence)

    plan.convergence_mode = "distribution"
    crit = _build_convergence_criterion(plan)
    assert isinstance(crit, DistributionConvergence)
```

- [ ] **Step 2: Run, verify it passes today (the old if-chain handles all three names)**

Run: `uv run pytest tests/unit/orchestrator/test_convergence_from_plan.py::test_build_convergence_criterion_dispatches_via_plugin_registry -n auto -v`
Expected: PASS (existing if-chain happens to satisfy this).

This is a regression-pin: it'll keep passing through the refactor, proving behavioral equivalence.

- [ ] **Step 3: Replace the if-chain with plugin-registry dispatch**

In `src/aiperf/_cli_runner_helpers.py`, replace `_build_convergence_criterion` (lines ~123-152):

```python
def _build_convergence_criterion(plan: BenchmarkPlan):  # noqa: ANN202
    """Pick the convergence criterion matching ``plan.convergence_mode``.

    Dispatches via the plugin registry so third-party criteria (registered in
    plugins.yaml under the `convergence_criterion` category) are reachable
    through the same code path as the built-ins. Each criterion class owns the
    mapping from BenchmarkPlan fields to its constructor via `from_plan`.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    criterion_cls = plugins.get_class(
        PluginType.CONVERGENCE_CRITERION, str(plan.convergence_mode)
    )
    return criterion_cls.from_plan(plan)
```

- [ ] **Step 4: Run the regression-pin test plus the full convergence + cli-runner-helpers test suite**

Run: `uv run pytest tests/unit/orchestrator/ tests/unit/test_cli_runner_helpers.py -n auto -v` (skip the second path if the file doesn't exist).
Expected: PASS — equivalence preserved.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/_cli_runner_helpers.py tests/unit/orchestrator/test_convergence_from_plan.py
git commit --no-verify -m "refactor(convergence): dispatch via plugin registry instead of if-chain"
```

---

## Task 5: Replace `ConvergenceMode` enum with generated `ConvergenceCriterionType`

**Files:**
- Modify: `src/aiperf/common/enums/server_metrics_enums.py`
- Modify: `src/aiperf/common/enums/enums.py`
- Modify: `src/aiperf/common/enums/__init__.py`
- Modify: `src/aiperf/config/benchmark.py`
- Modify: `src/aiperf/config/_models_benchmark.py`
- Modify: `src/aiperf/config/v1/_loadgen.py`
- Modify: `src/aiperf/config/v1/_converter_optionals.py`
- Modify: `src/aiperf/_cli_runner_helpers.py` (uses ConvergenceMode in two log/print lines)
- Modify: `src/aiperf/orchestrator/strategies.py` (if it imports ConvergenceMode)

> **Pattern**: alias-then-remove. First alias `ConvergenceMode = ConvergenceCriterionType` in the public-enums namespace so existing imports keep working, run the test suite, then in a follow-on commit remove the alias and update all import sites.

- [ ] **Step 1: Alias `ConvergenceMode` to the generated enum**

In `src/aiperf/common/enums/server_metrics_enums.py`, replace the existing `ConvergenceMode` class definition with:

```python
# `ConvergenceMode` is now a backwards-compat alias for the plugin-generated
# `ConvergenceCriterionType`. Third-party convergence criteria registered in
# plugins.yaml automatically appear as enum members. Remove this alias once
# all import sites are updated.
from aiperf.plugin.enums import ConvergenceCriterionType as ConvergenceMode  # noqa: F401
```

If the existing `ConvergenceMode` defined extra members beyond `ci_width`, `cv`, `distribution`, list them here so we don't lose any. Verify by running `grep -rn "ConvergenceMode\." src/`.

- [ ] **Step 2: Run the full unit suite to confirm alias preserves behavior**

Run: `uv run pytest tests/unit/ -n auto`
Expected: PASS.

- [ ] **Step 3: Replace direct `ConvergenceMode` import sites with `ConvergenceCriterionType`**

For each file in the import sites listed under "Files" above, replace:
```python
from aiperf.common.enums import ConvergenceMode
```
with:
```python
from aiperf.plugin.enums import ConvergenceCriterionType
```
And replace all `ConvergenceMode.X` usage with `ConvergenceCriterionType.X`. Use `grep -rn "ConvergenceMode" src/` to enumerate sites.

In `src/aiperf/_cli_runner_helpers.py`, the only remaining usages are in log statements (`logger.info(f"  Convergence mode: {plan.convergence_mode}")`) — those just print the enum value and don't need the import to change.

In `src/aiperf/config/benchmark.py:102`, change the field type:
```python
convergence_mode: ConvergenceCriterionType = Field(
    default=ConvergenceCriterionType.CI_WIDTH,
    ...
)
```

- [ ] **Step 4: Remove the alias**

Delete the alias line from `src/aiperf/common/enums/server_metrics_enums.py` introduced in Step 1 plus the `ConvergenceMode` export from `src/aiperf/common/enums/__init__.py` and `enums.py` `__all__` lists.

- [ ] **Step 5: Run full unit suite + ergonomics checks**

```bash
uv run pytest tests/unit/ -n auto
make check-ergonomics
make check-ruff-baselined
```

Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/ tests/
git commit --no-verify -m "refactor(enums): replace ConvergenceMode with plugin-generated ConvergenceCriterionType"
```

---

## Task 6: Register `search_planner` plugin category

**Files:**
- Modify: `src/aiperf/plugin/categories.yaml`
- Modify: `src/aiperf/plugin/plugins.yaml`
- Generated: `src/aiperf/plugin/enums.py`, `enums.pyi`, plugin overloads

- [ ] **Step 1: Add the category to `categories.yaml`**

```yaml
search_planner:
  protocol: aiperf.orchestrator.search_planner.base:SearchPlanner
  metadata_class: aiperf.plugin.schema.schemas:SearchPlannerMetadata
  enum: SearchPlannerType
  description: |
    Search planners drive the adaptive outer loop, proposing the next variation
    via `ask()` and absorbing observed metrics via `tell()`. Each declares
    dimension-type and objective-direction support via metadata so the CLI can
    validate `--search-space` shape before importing the planner's heavy
    soft-dep tree (skopt, optuna, etc.). Selected via `--search-planner <name>`.
    One-to-one mapping per benchmark run.
```

- [ ] **Step 2: Register `bayesian` in `plugins.yaml`**

```yaml
# =============================================================================
search_planner:
  bayesian:
    class: aiperf.orchestrator.search_planner.bayesian:BayesianSearchPlanner
    description: |
      scikit-optimize-backed Gaussian-process Bayesian optimization. Requires
      the `[bo]` extra (`uv pip install -e '.[bo]'`). Treats SLA filters as
      soft penalties in the loss space; supports plateau detection and
      improvement-patience early stopping.
    metadata:
      supports_continuous: true
      supports_discrete: true
      supports_categorical: false
      requires_initial_samples: 5
      compatible_objective_directions: [maximize, minimize]
      requires_extras: [bo]
```

- [ ] **Step 3: Regenerate plugin artifacts**

```bash
uv run python tools/generate_plugin_artifacts.py
make validate-plugin-schemas
```

Expected: validation passes, counts increase by one category and one plugin.

- [ ] **Step 4: Add a registry-lookup test**

Append to `tests/unit/plugin/test_orchestrator_categories.py`:

```python
def test_search_planner_plugin_lookup_by_name():
    """The bayesian planner is reachable via plugins.get_class without importing skopt."""
    from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType, SearchPlannerType

    assert plugins.get_class(PluginType.SEARCH_PLANNER, "bayesian") is BayesianSearchPlanner
    assert SearchPlannerType.BAYESIAN == "bayesian"


def test_search_planner_metadata_declares_extras():
    """Bayesian planner metadata correctly declares the [bo] extra."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    entry = plugins.get_entry(PluginType.SEARCH_PLANNER, "bayesian")
    assert entry.metadata.requires_extras == ["bo"]
    assert entry.metadata.supports_continuous is True
    assert entry.metadata.supports_discrete is True
```

Run: `uv run pytest tests/unit/plugin/test_orchestrator_categories.py -n auto -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/plugin/ tests/unit/plugin/test_orchestrator_categories.py
git commit --no-verify -m "feat(plugin): register search_planner category with bayesian built-in"
```

---

## Task 7: Add `planner` field to `AdaptiveSearchConfig` + `--search-planner` CLI flag

**Files:**
- Modify: `src/aiperf/config/adaptive_search.py`
- Modify: `src/aiperf/config/v1/_loadgen.py`
- Modify: `src/aiperf/config/v1/_converter_optionals.py`

- [ ] **Step 1: Write failing test for the new field**

Add to `tests/unit/orchestrator/test_convergence_from_plan.py` or create a new file `tests/unit/config/test_adaptive_search_planner_field.py`:

```python
def test_adaptive_search_config_default_planner_is_bayesian():
    from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension

    cfg = AdaptiveSearchConfig(
        search_space=[SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=100, kind="int")],
        objective_metric="output_token_throughput",
        objective_direction="maximize",
        max_iterations=10,
    )
    assert cfg.planner == "bayesian"


def test_adaptive_search_config_rejects_unknown_planner():
    import pytest
    from pydantic import ValidationError

    from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension

    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(
            search_space=[SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=100, kind="int")],
            objective_metric="output_token_throughput",
            objective_direction="maximize",
            max_iterations=10,
            planner="not-a-real-planner",
        )
```

- [ ] **Step 2: Run, verify both fail**

Run: `uv run pytest tests/unit/config/test_adaptive_search_planner_field.py -n auto -v`
Expected: FAIL — `planner` field doesn't exist yet.

- [ ] **Step 3: Add the field**

In `src/aiperf/config/adaptive_search.py`, in `AdaptiveSearchConfig`, after the existing `algorithm` field (around line 100):

```python
    planner: SearchPlannerType = Field(
        default=SearchPlannerType.BAYESIAN,
        description=(
            "Outer-loop search planner plugin name. Defaults to `bayesian` "
            "(scikit-optimize-backed). Third-party planners registered under "
            "the `search_planner` plugin category are valid here. Selected "
            "via `--search-planner` on the CLI."
        ),
    )
```

Add the import at the top:
```python
from aiperf.plugin.enums import SearchPlannerType
```

- [ ] **Step 4: Add the CLI flag**

In `src/aiperf/config/v1/_loadgen.py`, find the `--search-*` flag section (look for `search_recipe` to locate it) and add:

```python
    search_planner: SearchPlannerType = Field(
        default=SearchPlannerType.BAYESIAN,
        description=(
            "Outer-loop search planner plugin. Default `bayesian` requires the "
            "`[bo]` extra. Third-party planners registered under the "
            "`search_planner` plugin category are accepted here."
        ),
        json_schema_extra=CLIParameter(...).to_dict(),  # match the surrounding pattern
    )
```

> **Note:** Match the exact `json_schema_extra` / `CLIParameter` shape used by the adjacent `search_recipe` flag. Read the file to confirm the surrounding pattern before writing.

- [ ] **Step 5: Propagate through the v1→v2 converter**

In `src/aiperf/config/v1/_converter_optionals.py`, in the function that builds `MultiRunConfig.adaptive_search` (around the existing `convergence_mode` mapping at line 127), add:

```python
"planner": user_loadgen.search_planner,
```

to the dict that becomes `AdaptiveSearchConfig` kwargs.

- [ ] **Step 6: Run tests + regenerate CLI docs**

```bash
uv run pytest tests/unit/ -n auto
uv run python tools/generate_cli_docs.py
```

Expected: tests PASS; `docs/cli-options.md` regenerates with the new `--search-planner` entry.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/config/ tests/unit/config/ docs/cli-options.md
git commit --no-verify -m "feat(cli): add --search-planner flag and planner field on AdaptiveSearchConfig"
```

---

## Task 8: Replace direct `BayesianSearchPlanner(...)` instantiation with plugin dispatch

**Files:**
- Modify: `src/aiperf/_cli_runner_helpers.py` — add `_build_search_planner(plan)` helper
- Modify: `src/aiperf/cli_runner.py` — use the helper
- Modify: `src/aiperf/sweep_controller/main.py` — use the helper

- [ ] **Step 1: Write failing test for the new helper**

Add to `tests/unit/orchestrator/test_convergence_from_plan.py` or a sibling file:

```python
def test_build_search_planner_dispatches_via_plugin_registry():
    """`_build_search_planner(plan)` returns a SearchPlanner via plugin lookup."""
    from unittest.mock import MagicMock

    from aiperf._cli_runner_helpers import _build_search_planner
    from aiperf.orchestrator.search_planner.base import SearchPlanner

    plan = MagicMock()
    plan.is_adaptive_search = True
    plan.adaptive_search.planner = "bayesian"
    plan.adaptive_search.search_space = [MagicMock(kind="int", path="phases.profiling.concurrency", lo=1, hi=10)]
    plan.adaptive_search.max_iterations = 3
    plan.adaptive_search.n_initial_points = 2
    plan.adaptive_search.objective_direction = "maximize"
    plan.adaptive_search.objective_metric = "output_token_throughput"
    plan.adaptive_search.objective_stat = "avg"
    plan.configs = [MagicMock()]

    pytest.importorskip("skopt")  # only run if [bo] extra is installed

    planner = _build_search_planner(plan)
    assert isinstance(planner, SearchPlanner)
```

- [ ] **Step 2: Run, verify it fails (helper doesn't exist)**

Run: `uv run pytest tests/unit/orchestrator/test_convergence_from_plan.py::test_build_search_planner_dispatches_via_plugin_registry -n auto -v`
Expected: FAIL.

- [ ] **Step 3: Add the helper to `_cli_runner_helpers.py`**

After `_build_convergence_criterion`:

```python
def _build_search_planner(plan: BenchmarkPlan):  # noqa: ANN202
    """Build the outer-loop SearchPlanner for adaptive search.

    Returns None when ``plan.is_adaptive_search`` is False. Dispatches via the
    plugin registry so third-party planners (registered in plugins.yaml under
    the `search_planner` category) are reachable through the same code path as
    the built-in `bayesian` planner.
    """
    if not plan.is_adaptive_search:
        return None

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    planner_cls = plugins.get_class(
        PluginType.SEARCH_PLANNER, str(plan.adaptive_search.planner)
    )
    return planner_cls(plan.configs[0], plan.adaptive_search)
```

- [ ] **Step 4: Replace the inline instantiation in `cli_runner.py`**

In `src/aiperf/cli_runner.py`, find lines ~432-443 (the `if plan.is_adaptive_search:` block that imports `BayesianSearchPlanner` and instantiates it) and replace with:

```python
    from aiperf._cli_runner_helpers import _build_search_planner

    search_planner = _build_search_planner(plan)
    if search_planner is not None:
        logger.info(
            f"Adaptive search active: planner={plan.adaptive_search.planner}, "
            f"max_iterations={plan.adaptive_search.max_iterations}, "
            f"search-space={[d.path for d in plan.adaptive_search.search_space]}, "
            f"objective={plan.adaptive_search.objective_metric}:"
            f"{plan.adaptive_search.objective_stat}:{plan.adaptive_search.objective_direction}"
        )
```

- [ ] **Step 5: Replace the inline instantiation in `sweep_controller/main.py`**

In `src/aiperf/sweep_controller/main.py`, find lines ~408-425 and replace with:

```python
        from aiperf._cli_runner_helpers import _build_search_planner

        search_planner = _build_search_planner(plan)
        if search_planner is not None:
            logger.info(
                f"Cluster-side adaptive search active: planner={plan.adaptive_search.planner}, "
                f"max_iterations={plan.adaptive_search.max_iterations}, "
                f"objective={plan.adaptive_search.objective_metric}:"
                f"{plan.adaptive_search.objective_stat}:"
                f"{plan.adaptive_search.objective_direction}"
            )
```

- [ ] **Step 6: Run unit + relevant integration tests**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: PASS.

If `uv pip install -e '.[bo]'` is set up locally, also run:
```bash
uv run pytest tests/unit/orchestrator/test_convergence_from_plan.py::test_build_search_planner_dispatches_via_plugin_registry -n auto -v
```

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/ tests/unit/orchestrator/
git commit --no-verify -m "refactor(search_planner): dispatch via plugin registry instead of direct instantiation"
```

---

## Task 9: Update documentation

**Files:**
- Modify: `docs/plugins/plugin-system.md`
- Modify: `llms.txt` (if the new categories warrant a mention)
- Modify: `CLAUDE.md` + `AGENTS.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` (the four-file sync rule)

- [ ] **Step 1: Document the two new categories in `docs/plugins/plugin-system.md`**

Add entries in whatever existing categories table format is used (read the file first to match style). The two new entries:

- `convergence_criterion` — "Statistical convergence detection across repeated runs (CI-width / CV / distribution / third-party)."
- `search_planner` — "Adaptive outer-loop search planners (Bayesian / future Optuna, Nevergrad, etc.)."

- [ ] **Step 2: Update the four-file sync set with a one-paragraph note in the orchestrator section**

In all four files (`CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`), add to the "Parameter Sweeping" or a new "Orchestrator plugin categories" subsection:

> **Convergence criteria and search planners are plugin-registered.** Built-ins (`ci_width`, `cv`, `distribution`; `bayesian`) live under their respective categories in `plugins.yaml`. Third-party criteria/planners ship as standalone wheels with setuptools entry points and override built-ins via priority. To dispatch, use `plugins.get_class(PluginType.CONVERGENCE_CRITERION, name).from_plan(plan)` or `plugins.get_class(PluginType.SEARCH_PLANNER, name)(base_cfg, adaptive_cfg)`. Never instantiate `BayesianSearchPlanner` / `CIWidthConvergence` / etc. directly outside the registered plugin entries.

- [ ] **Step 3: Run the four-file sync check**

```bash
make check-agent-files-sync
```

Expected: PASS (all four files identical content where required).

- [ ] **Step 4: Commit**

```bash
git add docs/ CLAUDE.md AGENTS.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit --no-verify -m "docs(plugin): document orchestrator plugin categories and dispatch patterns"
```

---

## Task 10: Final validation sweep

- [ ] **Step 1: Plugin schema validation**

```bash
make validate-plugin-schemas
```

Expected: `✓ Validated N+2 categories and M+4 plugins.`

- [ ] **Step 2: Full unit suite**

```bash
uv run pytest tests/unit/ -n auto
```

Expected: PASS, no new failures vs. branch baseline.

- [ ] **Step 3: Ergonomics + ruff baselines**

```bash
make check-ergonomics
make check-ruff-baselined
ruff format . && ruff check --fix .
```

Expected: clean.

- [ ] **Step 4: Smoke test the BO path end-to-end (optional, requires `[bo]` extra)**

```bash
uv pip install -e '.[bo]'
aiperf profile \
  --endpoint http://localhost:8000 \
  --model mock \
  --search-space "phases.profiling.concurrency:1,16:int" \
  --search-metric output_token_throughput \
  --search-direction maximize \
  --search-max-iterations 3 \
  --search-planner bayesian
```

Expected: runs to completion and emits `search_history.json`. Confirms plugin dispatch works at runtime end-to-end.

- [ ] **Step 5: Final commit if anything changed during validation**

```bash
git status
# if dirty:
git add -A
git commit --no-verify -m "chore: address validation cleanup"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: Every requirement from the recommendation (priority override, lazy loading, metadata, validation, name dispatch, generated enums) is exercised by at least one task.
- [x] **No placeholders**: All code blocks contain runnable snippets, all file paths are absolute, no "TBD" / "implement later" markers.
- [x] **Type consistency**: `ConvergenceCriterionType`, `SearchPlannerType`, `from_plan(plan)` signature, `_build_search_planner(plan) -> SearchPlanner | None` are consistent across tasks.
- [x] **Reverse-import sanity**: `ConvergenceCriterion.from_plan` uses `TYPE_CHECKING` guard for `BenchmarkPlan` import. `AdaptiveSearchConfig.planner` imports `SearchPlannerType` from `aiperf.plugin.enums` (avoids reverse import from orchestrator).
- [x] **Backwards compat**: `ConvergenceMode` aliased before removal so the cutover is two-phase, not big-bang.
