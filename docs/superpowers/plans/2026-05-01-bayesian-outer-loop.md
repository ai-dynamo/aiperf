# Bayesian-Optimization Outer Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bayesian-Optimization (BO) adaptive outer loop to `aiperf profile`, so a user can run e.g. `--search-space phases.profiling.concurrency:1,1000:int --search-metric output_token_throughput --search-direction maximize --search-max-iterations 30` and have the orchestrator iteratively pick the next concurrency value to evaluate, instead of pre-enumerating a grid.

**Architecture:** Add an `AdaptiveSearchConfig` Pydantic schema in the config layer, hang it on `MultiRunConfig.adaptive_search` and propagate it to a new `BenchmarkPlan.adaptive_search` field. On plans where `adaptive_search` is set, `MultiRunOrchestrator.execute` dispatches to a new `execute_adaptive_search` method that drives a pluggable `SearchPlanner` (skopt-backed `BayesianSearchPlanner`). Each BO iteration synthesizes a one-off `SweepVariation` with `index=k, label="search_iter_{k:04d}"`, deep-copies the base `BenchmarkConfig`, mutates it via `_set_nested_value`, and reuses the existing `_run_independent_cell` to run all trials at that point. Existing `aggregate_sweep_and_export` groups results post-hoc by `variation_values` so it Just Works; a separate `search_history.json` written incrementally after each iteration gives the BO trajectory. K8s cluster sweeps are out of scope in v1 — `_reject_in_process_sweep_under_operator` hard-fails when `adaptive_search` is set under `AIPERF_OPERATOR_MANAGED=1`, because variation `k` is unknown until variation `k-1` finishes (incompatible with the operator's deterministic-cardinality contract). A defensive guard in `sweep_controller/plan_builder.py` is wired up for forward-compat against a future CRD extension.

**Tech Stack:** Python 3.10+, Pydantic v2, `skopt` (scikit-optimize, soft dep behind `[bo]` extra), cyclopts (CLI), `_set_nested_value` from `config/sweep.py`, `MultiRunOrchestrator._run_independent_cell`.

---

## Design pre-reads (skim before starting)

- `artifacts/bo_design_ajc_k8s.md` — original design analysis (file/line citations).
- `src/aiperf/orchestrator/orchestrator.py:50-214` — `MultiRunOrchestrator.execute` dispatch + `_run_independent_cell` (the seam we reuse).
- `src/aiperf/config/sweep.py:224-265` — `_set_nested_value` (the dict-mode mutation primitive; raises `ValueError` on path typos).
- `src/aiperf/config/benchmark.py:24-170` — `BenchmarkPlan` (we add one optional field).
- `src/aiperf/cli_runner.py:391-477` — `_run_multi_benchmark` and `_reject_in_process_sweep_under_operator`.
- `src/aiperf/config/v1/_loadgen.py:540-595` — established CLI-flag pattern for multi-run knobs.
- `src/aiperf/config/v1/_converter_optionals.py:68-97` — `build_multi_run` (where new flags get plumbed v1→v2).
- `src/aiperf/orchestrator/aggregation/sweep.py:44-61` — existing `OptimizationDirection` enum (reuse, do not duplicate).
- `tests/unit/orchestrator/test_multi_run_orchestrator.py` — `FakeExecutor`, `_make_plan` fixture style to mimic.

## Out of scope (v1)

- Cluster sweeps: BO under `AIPERF_OPERATOR_MANAGED=1` is hard-rejected. A future `AIPerfAdaptiveSweep` CRD with controller-pod-side planning is a v2 concern.
- Multi-objective BO (Pareto). Single scalar objective only in v1.
- Adaptive convergence (`--convergence-metric`) composed with BO. Reject explicitly — the trial-level early-stop semantics need separate thought.
- Categorical search dimensions across non-numeric values (we support `:int` and `:real`; `:cat` deferred).
- Resuming a BO run from a partial `search_history.json`. v1 always starts fresh.

## File structure

**New files:**
- `src/aiperf/config/adaptive_search.py` (~80 LOC) — `AdaptiveSearchConfig` Pydantic submodel and `SearchSpaceDimension`. Lives in the config layer because `MultiRunConfig` (also in `aiperf.config`) holds an `adaptive_search: AdaptiveSearchConfig | None` field — putting the schema in `aiperf.orchestrator` would create a reverse import.
- `src/aiperf/orchestrator/search_planner/__init__.py` (~15 LOC) — exports.
- `src/aiperf/orchestrator/search_planner/base.py` (~50 LOC) — `SearchPlanner` ABC + `SearchIteration` dataclass. Imports `AdaptiveSearchConfig` from `aiperf.config.adaptive_search`.
- `src/aiperf/orchestrator/search_planner/bayesian.py` (~200 LOC) — `BayesianSearchPlanner` skopt impl, dimension-type inference, objective extraction.
- `src/aiperf/orchestrator/search_planner/parsing.py` (~70 LOC) — `parse_search_space` CLI primitive. Pure; no skopt import. (Objective shape — metric/stat/direction — is three Pydantic-validated string fields, no parser needed.)
- `src/aiperf/exporters/search_history.py` (~50 LOC) — incremental `search_history.json` writer.
- `tests/unit/config/test_adaptive_search.py`
- `tests/unit/orchestrator/search_planner/__init__.py`
- `tests/unit/orchestrator/search_planner/test_parsing.py`
- `tests/unit/orchestrator/search_planner/test_base.py`
- `tests/unit/orchestrator/search_planner/test_bayesian.py`
- `tests/unit/orchestrator/search_planner/test_history.py`
- `tests/unit/orchestrator/test_execute_adaptive_search.py`
- `tests/unit/cli_commands/test_search_flags.py`

**Modified files:**
- `pyproject.toml` — add `[project.optional-dependencies] bo = ["scikit-optimize>=0.10"]`.
- `src/aiperf/config/_models_benchmark.py` — add `MultiRunConfig.adaptive_search: AdaptiveSearchConfig | None = None` field.
- `src/aiperf/config/benchmark.py` — add `BenchmarkPlan.adaptive_search: AdaptiveSearchConfig | None` + `is_adaptive_search` property.
- `src/aiperf/config/v1/_loadgen.py` — add 5 BO CLI flags.
- `src/aiperf/config/v1/_converter_optionals.py` — extend `build_multi_run`: parse strings into a typed `AdaptiveSearchConfig`, dump and emit as `multi_run["adaptive_search"]`.
- `src/aiperf/config/loader/plan.py` — read `config.multi_run.adaptive_search` (already typed) into `plan.adaptive_search`.
- `src/aiperf/orchestrator/orchestrator.py` — add `execute_adaptive_search` method + one-line dispatch in `execute()`.
- `src/aiperf/cli_runner.py` — instantiate planner + extend `_reject_in_process_sweep_under_operator`.
- `src/aiperf/sweep_controller/plan_builder.py` — defensive `kopf.PermanentError` if `spec.multi_run.adaptive_search` is set (forward-compat; v1 has no CRD field for it).
- `docs/cli-options.md` (auto-generated — runs `make generate-cli-docs`).
- `docs/architecture.md` — one paragraph in the multi-run section.
- `AGENTS.md` + `CLAUDE.md` + `.github/copilot-instructions.md` + `.cursor/rules/python.mdc` — four-file sync update describing the new feature in one paragraph under "Parameter Sweeping".
- `docs/index.yml` — index any new doc files.

---

## Task 1: Add `bo` extra dependency

**Files:**
- Modify: `pyproject.toml:85-100`

`AdaptiveSearchConfig.algorithm` uses `Literal["bayes"]` (Task 2) — no enum needed for v1's single value. If a second algorithm is added later (`tpe`, `random`, ...), promote to a `CaseInsensitiveStrEnum` then.

- [ ] **Step 1: Add `bo` extra to `pyproject.toml`**

Insert after the `dev = [...]` block ending at line 100:

```toml
bo = [
  "scikit-optimize>=0.10",
]
```

- [ ] **Step 2: Verify the extra installs cleanly**

```bash
uv pip install -e '.[bo]'
uv run python -c "import skopt; print(skopt.__version__)"
```

Expected: prints a version like `0.10.x`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add 'bo' optional-dependency for scikit-optimize"
```

---

## Task 2: `AdaptiveSearchConfig` Pydantic schema (config layer)

**Files:**
- Create: `src/aiperf/config/adaptive_search.py`
- Test: `tests/unit/config/test_adaptive_search.py`

The schema lives in the config layer because `MultiRunConfig` (Task 3) holds an `adaptive_search: AdaptiveSearchConfig | None` field. Putting it in the orchestrator package would force `aiperf.config._models_benchmark` to import from `aiperf.orchestrator`, which is a backwards layer dependency.

- [ ] **Step 1: Write the failing tests**

`tests/unit/config/test_adaptive_search.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AdaptiveSearchConfig and SearchSpaceDimension."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def test_search_space_dimension_int():
    dim = SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=1000, kind="int")
    assert dim.path == "phases.profiling.concurrency"
    assert dim.kind == "int"


def test_search_space_dimension_rejects_lo_gt_hi():
    with pytest.raises(ValidationError):
        SearchSpaceDimension(path="x", lo=10, hi=1, kind="int")


def test_adaptive_search_config_minimal():
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=20,
    )
    assert cfg.max_iterations == 20
    assert cfg.plateau_window == 5  # default


def test_adaptive_search_config_rejects_empty_search_space():
    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=20,
        )


def test_adaptive_search_config_rejects_max_iterations_below_two():
    with pytest.raises(ValidationError):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=1,  # below ge=2
        )


def test_adaptive_search_config_rejects_initial_points_at_or_above_max_iterations():
    with pytest.raises(ValidationError, match="n_initial_points"):
        AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objective_metric="x",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=5,
            n_initial_points=5,  # not strictly less than max_iterations
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest -n auto tests/unit/config/test_adaptive_search.py -v
```

Expected: ModuleNotFoundError on `aiperf.config.adaptive_search`.

- [ ] **Step 3: Write the schema**

`src/aiperf/config/adaptive_search.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema for the BO / adaptive outer-loop configuration.

Lives in the config layer (not the orchestrator) because MultiRunConfig
holds an adaptive_search field — placing this in aiperf.orchestrator would
force a reverse import from aiperf.config.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from aiperf.config._base import BaseConfig
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

__all__ = ["AdaptiveSearchConfig", "SearchSpaceDimension"]


class SearchSpaceDimension(BaseConfig):
    """One dimension of the BO search space.

    `path` is a dotted path of the form `phases.profiling.concurrency` —
    the same grammar accepted by `aiperf.config.sweep._set_nested_value`.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Dotted-path into BenchmarkConfig (e.g. 'phases.profiling.concurrency').")
    lo: float = Field(description="Inclusive lower bound.")
    hi: float = Field(description="Inclusive upper bound.")
    kind: Literal["int", "real"] = Field(
        default="real",
        description="Dimension type. 'int' rounds skopt suggestions to integers; 'real' keeps floats.",
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> "SearchSpaceDimension":
        if self.hi <= self.lo:
            raise ValueError(f"search-space dim {self.path!r}: hi ({self.hi}) must be > lo ({self.lo}).")
        return self


class AdaptiveSearchConfig(BaseConfig):
    """Configuration for an adaptive outer loop (e.g. Bayesian Optimization).

    Attached to MultiRunConfig.adaptive_search when --search-* flags are set; absent
    otherwise. Propagates to BenchmarkPlan.adaptive_search in build_benchmark_plan
    and is consumed by MultiRunOrchestrator.execute_adaptive_search.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["bayes"] = Field(
        default="bayes",
        description="Search algorithm. v1 only supports Bayesian Optimization (`bayes`).",
    )
    search_space: list[SearchSpaceDimension] = Field(
        description="Dimensions to optimize over. Must be non-empty.",
        min_length=1,
    )
    objective_metric: str = Field(
        description="Metric tag to optimize, e.g. 'output_token_throughput'. "
        "Must match a key in RunResult.summary_metrics produced by the run.",
    )
    objective_stat: Literal["avg", "p50", "p90", "p95", "p99"] = Field(
        default="avg",
        description="Statistic on the metric (matches JsonMetricResult fields).",
    )
    objective_direction: OptimizationDirection = Field(
        description="Whether higher (MAXIMIZE) or lower (MINIMIZE) is better.",
    )
    max_iterations: int = Field(
        ge=2,
        le=200,
        description="Maximum number of BO iterations. Each iteration runs `plan.trials` benchmarks.",
    )
    n_initial_points: int = Field(
        default=5,
        ge=1,
        description="Sobol-random points before skopt fits the GP. Must be < max_iterations.",
    )
    plateau_window: int = Field(
        default=5,
        ge=2,
        description="Number of recent iterations to inspect for plateau detection.",
    )
    plateau_threshold: float = Field(
        default=0.01,
        gt=0,
        description="Coefficient-of-variation threshold for plateau (relative; scale-free).",
    )
    random_seed: int | None = Field(
        default=None,
        description="If set, passed as `random_state` to skopt.Optimizer for reproducibility.",
    )

    @model_validator(mode="after")
    def _check_initial_points_below_max_iterations(self) -> "AdaptiveSearchConfig":
        if self.n_initial_points >= self.max_iterations:
            raise ValueError(
                f"n_initial_points ({self.n_initial_points}) must be < max_iterations ({self.max_iterations}); "
                f"otherwise the GP never fits."
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/config/test_adaptive_search.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/adaptive_search.py tests/unit/config/test_adaptive_search.py
git commit -m "feat(config): add AdaptiveSearchConfig and SearchSpaceDimension schemas"
```

---

## Task 2b: `SearchPlanner` ABC + `SearchIteration` dataclass

**Files:**
- Create: `src/aiperf/orchestrator/search_planner/__init__.py`
- Create: `src/aiperf/orchestrator/search_planner/base.py`
- Test: `tests/unit/orchestrator/search_planner/__init__.py` (empty)
- Test: `tests/unit/orchestrator/search_planner/test_base.py`

- [ ] **Step 1: Create the test directory marker**

```bash
mkdir -p tests/unit/orchestrator/search_planner
touch tests/unit/orchestrator/search_planner/__init__.py
```

- [ ] **Step 2: Write the failing test**

`tests/unit/orchestrator/search_planner/test_base.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SearchIteration dataclass and SearchPlanner ABC surface."""

from __future__ import annotations

from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner


def test_outer_iteration_dataclass_defaults():
    it = SearchIteration(iteration_idx=3, variation_values={"x": 42})
    assert it.iteration_idx == 3
    assert it.objective_value is None
    assert it.results == []


def test_outer_iteration_with_objective():
    it = SearchIteration(
        iteration_idx=0,
        variation_values={"x": 1},
        objective_value=12.5,
    )
    assert it.objective_value == 12.5


def test_search_planner_is_abstract():
    """ABC: cannot instantiate without concrete impls."""
    import pytest
    with pytest.raises(TypeError):
        SearchPlanner()  # type: ignore[abstract]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_base.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Write the implementation**

`src/aiperf/orchestrator/search_planner/__init__.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive outer-loop planners (e.g. Bayesian Optimization) for AIPerf.

A BenchmarkPlan can carry an optional AdaptiveSearchConfig (defined in
aiperf.config.adaptive_search). When present, the orchestrator iterates by
asking a planner for the next BenchmarkConfig to evaluate rather than
walking a pre-enumerated variation list.
"""

from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner

__all__ = ["SearchIteration", "SearchPlanner"]
```

`src/aiperf/orchestrator/search_planner/base.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SearchPlanner ABC and SearchIteration dataclass.

Schema for the BO config itself lives in aiperf.config.adaptive_search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.models import RunResult


__all__ = ["SearchIteration", "SearchPlanner"]


@dataclass
class SearchIteration:
    """One entry in the BO trajectory log.

    Written to search_history.json incrementally after each iteration. `results`
    is the per-trial RunResult list at this BO point (length == plan.trials
    for FixedTrialsStrategy).
    """

    iteration_idx: int
    variation_values: dict[str, Any]
    objective_value: float | None = None
    results: list[Any] = field(default_factory=list)


class SearchPlanner(ABC):
    """Abstract base for adaptive outer-loop planners.

    Implementations: BayesianSearchPlanner (skopt-backed). Future: GridPlanner
    (for testing), OptunaSearchPlanner (TPE), RandomSearchPlanner (baseline).
    """

    @abstractmethod
    def ask(self) -> tuple["BenchmarkConfig", "SweepVariation"] | None:
        """Return (cfg, variation) for the next iteration, or None when done.

        The cfg is a deep-copied BenchmarkConfig with the proposed values
        substituted at their dotted paths. The SweepVariation has
        `index = iteration_idx`, `label = "search_iter_NNNN"`, and
        `values = {path: proposed_value, ...}` so downstream
        `aggregate_sweep_and_export` groups results naturally.
        """

    @abstractmethod
    def tell(self, variation: "SweepVariation", results: list["RunResult"]) -> None:
        """Tell the planner what happened at the most recent point."""

    @abstractmethod
    def is_converged(self) -> bool:
        """True when max_iterations exhausted or plateau detected."""

    @abstractmethod
    def history(self) -> list[SearchIteration]:
        """All iterations recorded so far, in submission order."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_base.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/orchestrator/search_planner/__init__.py src/aiperf/orchestrator/search_planner/base.py tests/unit/orchestrator/search_planner/__init__.py tests/unit/orchestrator/search_planner/test_base.py
git commit -m "feat(orchestrator): add SearchPlanner ABC and SearchIteration dataclass"
```

---

## Task 3: CLI flag parsing primitive (`parse_search_space`)

**Files:**
- Create: `src/aiperf/orchestrator/search_planner/parsing.py`
- Test: `tests/unit/orchestrator/search_planner/test_parsing.py`

The grammar:
- `--search-space "phases.profiling.concurrency:1,1000:int"` (repeatable; `:int`/`:real` suffix optional, defaults to `real`)
- The objective is no longer a colon-grammar — see Task 4. `--search-metric` (string), `--search-stat` (Literal default `"avg"`), `--search-direction` (Literal `"maximize"|"minimize"`) are three separate Pydantic-validated fields. No parser needed.

`parse_search_space` raises `TypeError` on malformed input naming the offending flag (matches the pattern `parse_int_or_int_list` uses, per the cyclopts gotcha note in CLAUDE.md).

- [ ] **Step 1: Write the failing tests**

`tests/unit/orchestrator/search_planner/test_parsing.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the --search-space CLI parsing primitive."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.adaptive_search import SearchSpaceDimension
from aiperf.orchestrator.search_planner.parsing import parse_search_space


@pytest.mark.parametrize(
    "raw,expected",
    [
        param(
            ["phases.profiling.concurrency:1,1000:int"],
            [SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=1000, kind="int")],
            id="single_int",
        ),
        param(
            ["x:0.1,5.0"],
            [SearchSpaceDimension(path="x", lo=0.1, hi=5.0, kind="real")],
            id="default_real",
        ),
        param(
            ["a:1,10:int", "b:0,1:real"],
            [
                SearchSpaceDimension(path="a", lo=1, hi=10, kind="int"),
                SearchSpaceDimension(path="b", lo=0, hi=1, kind="real"),
            ],
            id="two_dims",
        ),
    ],
)  # fmt: skip
def test_parse_search_space_valid(raw, expected):
    assert parse_search_space(raw) == expected


@pytest.mark.parametrize(
    "raw,fragment",
    [
        param(["bad-no-colon"], "expected 'path:lo,hi", id="missing_colon"),
        param(["x:1"], "expected 'path:lo,hi", id="missing_comma"),
        param(["x:1,abc"], "could not parse bound", id="non_numeric"),
        param(["x:1,2:weird"], "kind must be 'int' or 'real'", id="bad_kind"),
        param(["x:5,1"], "hi", id="hi_below_lo"),
    ],
)  # fmt: skip
def test_parse_search_space_errors(raw, fragment):
    with pytest.raises(TypeError, match=fragment):
        parse_search_space(raw)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_parsing.py -v
```

Expected: ModuleNotFoundError on `parsing`.

- [ ] **Step 3: Write the implementation**

`src/aiperf/orchestrator/search_planner/parsing.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI grammar primitive for --search-space.

Pure parsing — no skopt import, so import cost is negligible. The objective
shape (metric / stat / direction) is three separate Pydantic-validated fields
and needs no parser.

Grammar:
    --search-space "PATH:LO,HI[:KIND]"      (repeatable; KIND in int/real)

Errors raise TypeError naming the offending flag, matching the pattern used
by ``parse_int_or_int_list`` in ``src/aiperf/config/parsing.py`` so cyclopts
surfaces the message cleanly.
"""

from __future__ import annotations

from aiperf.config.adaptive_search import SearchSpaceDimension

_VALID_KINDS = ("int", "real")


def parse_search_space(values: list[str]) -> list[SearchSpaceDimension]:
    """Parse one or more `--search-space "path:lo,hi[:kind]"` strings.

    Examples:
        >>> parse_search_space(["phases.profiling.concurrency:1,1000:int"])
        [SearchSpaceDimension(path='phases.profiling.concurrency', lo=1.0, hi=1000.0, kind='int')]
        >>> parse_search_space(["x:0,1"])  # default kind=real
        [SearchSpaceDimension(path='x', lo=0.0, hi=1.0, kind='real')]
    """
    out: list[SearchSpaceDimension] = []
    for raw in values:
        out.append(_parse_one_dim(raw))
    return out


def _parse_one_dim(raw: str) -> SearchSpaceDimension:
    if ":" not in raw or "," not in raw:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', e.g. "
            f"'phases.profiling.concurrency:1,1000:int'."
        )
    parts = raw.split(":")
    if len(parts) == 2:
        path, bounds = parts
        kind = "real"
    elif len(parts) == 3:
        path, bounds, kind = parts
    else:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', got {len(parts)} parts."
        )
    if kind not in _VALID_KINDS:
        raise TypeError(
            f"--search-space {raw!r}: kind must be 'int' or 'real', got {kind!r}."
        )
    if "," not in bounds:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', missing ',' in bounds."
        )
    lo_s, hi_s = bounds.split(",", 1)
    try:
        lo, hi = float(lo_s), float(hi_s)
    except ValueError as e:
        raise TypeError(
            f"--search-space {raw!r}: could not parse bound as float ({e})."
        ) from e
    # Pydantic validator on SearchSpaceDimension catches lo>=hi; surface as TypeError.
    if hi <= lo:
        raise TypeError(
            f"--search-space {raw!r}: hi ({hi}) must be > lo ({lo})."
        )
    return SearchSpaceDimension(path=path, lo=lo, hi=hi, kind=kind)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_parsing.py -v
```

Expected: PASS (8 parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/orchestrator/search_planner/parsing.py tests/unit/orchestrator/search_planner/test_parsing.py
git commit -m "feat(orchestrator): add BO CLI grammar parsers (search-space, objective)"
```

---

## Task 4: Add `--search-*` flags to `LoadGeneratorConfig` (v1)

**Files:**
- Modify: `src/aiperf/config/v1/_loadgen.py:540-595` (append after `parameter_sweep_mode`)
- Test: `tests/unit/cli_commands/test_search_flags.py`

Per project convention (CLAUDE.md "Config v1" rules + the existing pattern of `parameter_sweep_*` flags being on `LoadGeneratorConfig`), search flags go on `LoadGeneratorConfig` — NOT on a new nested class, NOT on `UserConfig` top-level.

The objective shape is split into three explicit flags (instead of one colon-grammar):
- `--search-metric` (the metric tag, e.g. `output_token_throughput`)
- `--search-stat` (`Literal["avg","p50","p90","p95","p99"]`, default `"avg"`)
- `--search-direction` (`Literal["maximize","minimize"]`)

This avoids the "is `--search-objective` the metric or the whole block?" ambiguity flagged in the design audit; it also matches W&B's `metric.name` + `metric.goal` shape.

- [ ] **Step 1: Write failing test**

`tests/unit/cli_commands/test_search_flags.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the v1 --search-* CLI flags (parsing only, no execution)."""

from __future__ import annotations

from aiperf.config.v1._loadgen import LoadGeneratorConfig


def test_loadgen_has_search_fields():
    fields = LoadGeneratorConfig.model_fields
    assert "search_space" in fields
    assert "search_metric" in fields
    assert "search_stat" in fields
    assert "search_direction" in fields
    assert "search_max_iterations" in fields
    assert "search_random_seed" in fields
    assert "search_initial_points" in fields


def test_loadgen_search_defaults_unset():
    """When the user supplies no --search-* flags, all fields are None/unset."""
    lg = LoadGeneratorConfig()
    assert lg.search_space is None
    assert lg.search_metric is None
    assert lg.search_stat is None
    assert lg.search_direction is None
    assert lg.search_max_iterations is None


def test_loadgen_accepts_search_space_list():
    lg = LoadGeneratorConfig(search_space=["phases.profiling.concurrency:1,1000:int"])
    assert lg.search_space == ["phases.profiling.concurrency:1,1000:int"]


def test_loadgen_accepts_search_objective_fields():
    lg = LoadGeneratorConfig(
        search_metric="output_token_throughput",
        search_stat="p99",
        search_direction="maximize",
    )
    assert lg.search_metric == "output_token_throughput"
    assert lg.search_stat == "p99"
    assert lg.search_direction == "maximize"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_search_flags.py -v
```

Expected: AssertionError on missing field.

- [ ] **Step 3: Append search-* fields to `_loadgen.py`**

In `src/aiperf/config/v1/_loadgen.py`, after `parameter_sweep_mode` (line 595) and before the closing class brace, append:

```python
    search_space: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Adaptive-search space dimensions. Repeatable. Each value is "
                "'path:lo,hi[:kind]', e.g. 'phases.profiling.concurrency:1,1000:int'. "
                "Mutually exclusive with magic-list flags (--concurrency 10,20,30) and "
                "with explicit sweep blocks. See docs/sweeping/bayesian-optimization.md."
            ),
        ),
        CLIParameter(
            name=("--search-space",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_metric: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Metric tag to optimize, e.g. 'output_token_throughput'. Required "
                "when --search-space is set. Must match a key in "
                "RunResult.summary_metrics produced by the run (NOT the flattened "
                "'_avg' / '_p99' aggregator-suffixed key)."
            ),
        ),
        CLIParameter(
            name=("--search-metric",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_stat: Annotated[
        Literal["avg", "p50", "p90", "p95", "p99"] | None,
        Field(
            default=None,
            description=(
                "Statistic on the metric: avg / p50 / p90 / p95 / p99. Defaults to "
                "'avg' when omitted (set by the v1->v2 converter)."
            ),
        ),
        CLIParameter(
            name=("--search-stat",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_direction: Annotated[
        Literal["maximize", "minimize"] | None,
        Field(
            default=None,
            description=(
                "Optimization direction. Required when --search-space is set."
            ),
        ),
        CLIParameter(
            name=("--search-direction",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_max_iterations: Annotated[
        int | None,
        Field(
            default=None,
            ge=2,
            le=200,
            description=(
                "Maximum number of search iterations. Each iteration runs "
                "--num-profile-runs benchmarks. Required when --search-space is set."
            ),
        ),
        CLIParameter(
            name=("--search-max-iterations",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_initial_points: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Random Sobol points before fitting the GP. Defaults to 5 "
                "when omitted. Must be < --search-max-iterations."
            ),
        ),
        CLIParameter(
            name=("--search-initial-points",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    search_random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Random seed for reproducible search trajectories. When unset, "
                "skopt uses non-deterministic randomness."
            ),
        ),
        CLIParameter(
            name=("--search-random-seed",),
            group=Groups.MULTI_RUN,
        ),
    ] = None
```

(`Literal` requires `from typing import Literal` — already imported in `_loadgen.py`.)

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_search_flags.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Regenerate CLI docs**

```bash
make generate-cli-docs
```

This updates `docs/cli-options.md`. Stage that file too.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/v1/_loadgen.py tests/unit/cli_commands/test_search_flags.py docs/cli-options.md
git commit -m "feat(cli): add --search-* CLI flags to LoadGeneratorConfig"
```

---

## Task 5a: Add `MultiRunConfig.adaptive_search` typed field

**Files:**
- Modify: `src/aiperf/config/_models_benchmark.py:21-119` (`MultiRunConfig`)
- Test: `tests/unit/config/test_multi_run_config.py` (extend or create)

`MultiRunConfig` has `model_config = ConfigDict(extra="forbid")`. To carry the BO config without bypass-gymnastics, add a typed `adaptive_search` field. The converter (Task 5b) emits `multi_run["adaptive_search"] = AdaptiveSearchConfig(...).model_dump()`; Pydantic re-validates on parent construction.

- [ ] **Step 1: Write failing test**

`tests/unit/config/test_multi_run_config.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunConfig.adaptive_search field."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config._models_benchmark import MultiRunConfig
from aiperf.config.adaptive_search import AdaptiveSearchConfig


def test_multi_run_default_no_adaptive_search():
    cfg = MultiRunConfig()
    assert cfg.adaptive_search is None


def test_multi_run_accepts_adaptive_search_dict():
    cfg = MultiRunConfig.model_validate({
        "num_runs": 2,
        "adaptive_search": {
            "algorithm": "bayes",
            "search_space": [{"path": "x", "lo": 1, "hi": 10, "kind": "int"}],
            "objective_metric": "m",
            "objective_stat": "avg",
            "objective_direction": "maximize",
            "max_iterations": 10,
        },
    })
    assert isinstance(cfg.adaptive_search, AdaptiveSearchConfig)
    assert cfg.adaptive_search.max_iterations == 10


def test_multi_run_rejects_unknown_top_level_keys():
    """extra='forbid' protects MultiRunConfig from drift; verify it still fires."""
    with pytest.raises(ValidationError):
        MultiRunConfig.model_validate({"num_runs": 2, "adaptive_search_raw": {}})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/config/test_multi_run_config.py -v
```

Expected: ValidationError on the second test (unknown `adaptive_search` key) until the field is added.

- [ ] **Step 3: Add the field to `MultiRunConfig`**

In `src/aiperf/config/_models_benchmark.py`, add the import at the top:

```python
from aiperf.config.adaptive_search import AdaptiveSearchConfig
```

Then append a new field after the `convergence_mode` field (after line 119):

```python
    adaptive_search: Annotated[
        AdaptiveSearchConfig | None,
        Field(
            default=None,
            description=(
                "Adaptive outer-loop configuration (Bayesian Optimization). "
                "Set by the v1 converter when --search-* flags are present. "
                "Mutually exclusive with the top-level `sweep` block; "
                "build_benchmark_plan enforces that. "
                "When set, MultiRunOrchestrator.execute dispatches to "
                "execute_adaptive_search instead of grid-mode paths."
            ),
        ),
    ] = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/config/test_multi_run_config.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/_models_benchmark.py tests/unit/config/test_multi_run_config.py
git commit -m "feat(config): add MultiRunConfig.adaptive_search typed field"
```

---

## Task 5b: v1→v2 converter emits typed `adaptive_search`

**Files:**
- Modify: `src/aiperf/config/v1/_converter_optionals.py:68-97` (`build_multi_run`)
- Test: `tests/unit/config/v1/test_converter_optionals.py` (extend or create)

The converter parses `lg.search_space` strings into `SearchSpaceDimension` objects, builds a typed `AdaptiveSearchConfig` from the three split objective fields, and emits `multi_run["adaptive_search"]` as the model-dumped dict. Validation re-runs at parent (`MultiRunConfig`) construction. Hard-fails if `--search-space` is set without companion flags (`--search-metric`, `--search-direction`, `--search-max-iterations`).

- [ ] **Step 1: Write failing tests**

In `tests/unit/config/v1/test_converter_optionals.py` (create if missing):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for build_multi_run's adaptive_search emit and exclusivity check."""

from __future__ import annotations

import pytest

from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_optionals import build_multi_run
from aiperf.config.v1._loadgen import LoadGeneratorConfig


def _user_with_loadgen(**fields) -> UserConfig:
    return UserConfig(loadgen=LoadGeneratorConfig(**fields))


def test_build_multi_run_emits_typed_adaptive_search_when_set():
    user = _user_with_loadgen(
        search_space=["phases.profiling.concurrency:1,1000:int"],
        search_metric="output_token_throughput",
        search_direction="maximize",
        search_max_iterations=20,
    )
    out = build_multi_run(user)
    assert out is not None
    assert "adaptive_search" in out
    ol = out["adaptive_search"]
    # model_dump'd AdaptiveSearchConfig — typed shape with parsed search_space.
    assert ol["algorithm"] == "bayes"
    assert ol["max_iterations"] == 20
    assert ol["objective_metric"] == "output_token_throughput"
    assert ol["objective_stat"] == "avg"  # default when --search-stat omitted
    assert ol["objective_direction"] == "maximize"
    assert ol["search_space"] == [
        {"path": "phases.profiling.concurrency", "lo": 1.0, "hi": 1000.0, "kind": "int"},
    ]


def test_build_multi_run_propagates_explicit_stat():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="ttft",
        search_stat="p99",
        search_direction="minimize",
        search_max_iterations=5,
    )
    out = build_multi_run(user)
    assert out["adaptive_search"]["objective_stat"] == "p99"
    assert out["adaptive_search"]["objective_direction"] == "minimize"


def test_build_multi_run_no_adaptive_search_when_unset():
    user = _user_with_loadgen(num_profile_runs=3)
    out = build_multi_run(user)
    assert out == {"num_runs": 3}
    assert "adaptive_search" not in out


def test_build_multi_run_rejects_search_space_without_metric():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_direction="maximize",
        search_max_iterations=20,
    )
    with pytest.raises(TypeError, match="--search-space requires --search-metric"):
        build_multi_run(user)


def test_build_multi_run_rejects_search_space_without_direction():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_max_iterations=20,
    )
    with pytest.raises(TypeError, match="--search-space requires --search-direction"):
        build_multi_run(user)


def test_build_multi_run_rejects_search_space_without_max_iterations():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_direction="maximize",
    )
    with pytest.raises(TypeError, match="--search-space requires --search-max-iterations"):
        build_multi_run(user)


def test_build_multi_run_propagates_initial_points_and_seed():
    user = _user_with_loadgen(
        search_space=["x:1,10:int"],
        search_metric="m",
        search_direction="maximize",
        search_max_iterations=20,
        search_initial_points=3,
        search_random_seed=42,
    )
    out = build_multi_run(user)
    assert out["adaptive_search"]["n_initial_points"] == 3
    assert out["adaptive_search"]["random_seed"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest -n auto tests/unit/config/v1/test_converter_optionals.py -v
```

Expected: KeyError on `adaptive_search` / TypeError-not-raised.

- [ ] **Step 3: Extend `build_multi_run`**

Replace the body of `build_multi_run` in `_converter_optionals.py:68-97` with:

```python
def build_multi_run(user: UserConfig) -> dict[str, Any] | None:
    """Build the multi-run section dict from explicitly-set v1 loadgen fields.

    When --search-* flags are present, builds a typed AdaptiveSearchConfig and
    emits its model_dump() as `out["adaptive_search"]`. MultiRunConfig has
    `extra="forbid"` so the typed field is the only legal carrier.

    Hard-fails if --search-space is set without the required companion flags
    (--search-metric, --search-direction, --search-max-iterations).
    """
    lg = user.loadgen
    if lg is None or not lg.model_fields_set:
        return None
    mapping = {
        "num_profile_runs": "num_runs",
        "profile_run_cooldown_seconds": "cooldown_seconds",
        "confidence_level": "confidence_level",
        "profile_run_disable_warmup_after_first": "disable_warmup_after_first",
        "set_consistent_seed": "set_consistent_seed",
        "convergence_metric": "convergence_metric",
        "convergence_mode": "convergence_mode",
        "convergence_threshold": "convergence_threshold",
        "convergence_stat": "convergence_stat",
        "parameter_sweep_cooldown_seconds": "parameter_sweep_cooldown_seconds",
        "parameter_sweep_same_seed": "parameter_sweep_same_seed",
        "parameter_sweep_mode": "mode",
    }
    out: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in lg.model_fields_set:
            out[key] = getattr(lg, field)

    search_fields = (
        "search_space",
        "search_metric",
        "search_stat",
        "search_direction",
        "search_max_iterations",
        "search_initial_points",
        "search_random_seed",
    )
    search_set = {f for f in search_fields if f in lg.model_fields_set}
    if "search_space" in search_set:
        for required, flag in (
            ("search_metric", "--search-metric"),
            ("search_direction", "--search-direction"),
            ("search_max_iterations", "--search-max-iterations"),
        ):
            if required not in search_set:
                raise TypeError(
                    f"--search-space requires {flag} (companion flag missing). "
                    "See docs/sweeping/bayesian-optimization.md for examples."
                )
        # Parse search-space strings -> typed AdaptiveSearchConfig -> model_dump.
        # Done here (not later in build_benchmark_plan) so MultiRunConfig
        # validation catches structural errors early at the v1->v2 boundary.
        from aiperf.config.adaptive_search import AdaptiveSearchConfig
        from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
        from aiperf.orchestrator.search_planner.parsing import parse_search_space

        dims = parse_search_space(lg.search_space)
        ol_kwargs: dict[str, Any] = dict(
            algorithm="bayes",
            search_space=dims,
            objective_metric=lg.search_metric,
            objective_stat=lg.search_stat or "avg",
            objective_direction=OptimizationDirection(lg.search_direction),
            max_iterations=lg.search_max_iterations,
        )
        if "search_initial_points" in search_set and lg.search_initial_points is not None:
            ol_kwargs["n_initial_points"] = lg.search_initial_points
        if "search_random_seed" in search_set and lg.search_random_seed is not None:
            ol_kwargs["random_seed"] = lg.search_random_seed
        adaptive_search = AdaptiveSearchConfig(**ol_kwargs)
        out["adaptive_search"] = adaptive_search.model_dump(mode="json")
    elif search_set:
        raise TypeError(
            f"--search-* flags {sorted(search_set)} require --search-space."
        )
    return out or None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/config/v1/test_converter_optionals.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/v1/_converter_optionals.py tests/unit/config/v1/test_converter_optionals.py
git commit -m "feat(config): emit typed multi_run.adaptive_search from --search-* CLI flags"
```

---

## Task 6: `BenchmarkPlan.adaptive_search` field + `is_adaptive_search` property

**Files:**
- Modify: `src/aiperf/config/benchmark.py:24-170`
- Test: `tests/unit/config/test_benchmark_plan.py` (extend or create)

Critical decision: **`is_sweep` stays grid-only** (`len(configs) > 1`). We add a separate `is_adaptive_search` property. This keeps every existing call site that branches on `is_sweep` working unchanged — sweep-aware code keeps treating the plan as having a fixed variation set, which it does (length 1 for BO).

- [ ] **Step 1: Write failing test**

`tests/unit/config/test_benchmark_plan.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BenchmarkPlan.adaptive_search field and is_adaptive_search property."""

from __future__ import annotations

from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def _make_minimal_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
        }
    )


def test_benchmark_plan_adaptive_search_default_none():
    plan = BenchmarkPlan(
        configs=[_make_minimal_config()],
        variations=[SweepVariation(index=0, label="base", values={})],
    )
    assert plan.adaptive_search is None
    assert plan.is_adaptive_search is False


def test_benchmark_plan_adaptive_search_set():
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=20,
    )
    plan = BenchmarkPlan(
        configs=[_make_minimal_config()],
        variations=[SweepVariation(index=0, label="base", values={})],
        adaptive_search=cfg,
    )
    assert plan.adaptive_search is cfg
    assert plan.is_adaptive_search is True
    # is_sweep stays grid-only (length-1 variations).
    assert plan.is_sweep is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/config/test_benchmark_plan.py -v
```

Expected: TypeError on `adaptive_search` kwarg.

- [ ] **Step 3: Add `adaptive_search` field to `BenchmarkPlan`**

In `src/aiperf/config/benchmark.py`, after the `parameter_sweep_mode` field (line 151), insert:

```python
    adaptive_search: Any = Field(
        default=None,
        description=(
            "Adaptive outer-loop configuration (e.g. Bayesian Optimization). "
            "Typed AdaptiveSearchConfig but expressed as Any to avoid a circular "
            "import between aiperf.config and aiperf.orchestrator. None for "
            "non-adaptive plans. When set, MultiRunOrchestrator.execute "
            "dispatches to execute_adaptive_search instead of grid-mode paths."
        ),
    )
```

After the `is_sweep` property (after line 170), add:

```python
    @property
    def is_adaptive_search(self) -> bool:
        """True when an adaptive outer loop (BO) is configured.

        Distinct from is_sweep (which checks for a multi-variation grid).
        Sweep-aware code paths continue to branch on is_sweep without change;
        outer-loop dispatch is handled separately in
        MultiRunOrchestrator.execute. Both can be False (single-point run).
        Both being True is forbidden by build_benchmark_plan.
        """
        return self.adaptive_search is not None
```

(Type `Any` matches the precedent set by `failure_policy` and `convergence_config` at lines 115-122.)

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest -n auto tests/unit/config/test_benchmark_plan.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/config/benchmark.py tests/unit/config/test_benchmark_plan.py
git commit -m "feat(config): add BenchmarkPlan.adaptive_search field and is_adaptive_search property"
```

---

## Task 7: Wire BO into `build_benchmark_plan`

**Files:**
- Modify: `src/aiperf/config/loader/plan.py:19-93`
- Test: `tests/unit/config/loader/test_plan_bo.py`

`config.multi_run.adaptive_search` is already a typed `AdaptiveSearchConfig` (set by Task 5b in the v1 converter, validated by Task 5a's MultiRunConfig field). `build_benchmark_plan` reads it directly:

1. **Skip `expand_sweep`** when `adaptive_search` is set — produce `configs=[base_config]`, `variations=[SweepVariation(index=0, label="base", values={})]`.
2. Reject if a sweep block is also present (mutual exclusion).
3. Copy `config.multi_run.adaptive_search` to `plan.adaptive_search`. No string re-parsing.

- [ ] **Step 1: Write failing test**

`tests/unit/config/loader/test_plan_bo.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for build_benchmark_plan with BO adaptive_search."""

from __future__ import annotations

import pytest

from aiperf.config.config import AIPerfConfig
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def _make_config_with_bo() -> AIPerfConfig:
    return AIPerfConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
            "multi_run": {
                "num_runs": 2,
                "adaptive_search": {
                    "algorithm": "bayes",
                    "search_space": [
                        {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"},
                    ],
                    "objective_metric": "output_token_throughput",
                    "objective_stat": "avg",
                    "objective_direction": "maximize",
                    "max_iterations": 15,
                },
            },
        }
    )


def test_build_plan_with_bo_skips_grid_expansion():
    plan = build_benchmark_plan(_make_config_with_bo())
    assert len(plan.configs) == 1
    assert plan.is_adaptive_search is True
    assert plan.is_sweep is False
    assert plan.adaptive_search is not None
    assert plan.adaptive_search.max_iterations == 15
    assert plan.adaptive_search.objective_direction == OptimizationDirection.MAXIMIZE
    assert plan.trials == 2  # multi_run.num_runs preserved


def test_build_plan_rejects_bo_with_sweep_block():
    cfg = AIPerfConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
            "sweep": {"type": "grid", "variables": {"phases.profiling.concurrency": [1, 2]}},
            "multi_run": {
                "adaptive_search": {
                    "algorithm": "bayes",
                    "search_space": [
                        {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"},
                    ],
                    "objective_metric": "x",
                    "objective_stat": "avg",
                    "objective_direction": "maximize",
                    "max_iterations": 5,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_benchmark_plan(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/config/loader/test_plan_bo.py -v
```

Expected: AttributeError on `plan.adaptive_search` (added in Task 6) or AttributeError on `plan.is_adaptive_search`.

- [ ] **Step 3: Update `build_benchmark_plan`**

Replace the body of `build_benchmark_plan` in `src/aiperf/config/loader/plan.py:19-93` with the version below. The diff: read the typed `adaptive_search` from `config.multi_run`, hard-fail if both sweep and BO are present, branch to skip `expand_sweep` when BO is set, attach to plan.

```python
def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Expands sweep variations and extracts multi_run settings, OR — when
    config.multi_run.adaptive_search is set — produces a single-config plan
    with plan.adaptive_search populated. Sweep + adaptive_search are mutually exclusive.
    """
    from aiperf.config.sweep import SweepVariation, expand_sweep

    adaptive_search = config.multi_run.adaptive_search  # already typed AdaptiveSearchConfig | None

    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)
    multi_run = config_dict.pop("multi_run", {})
    multi_run.pop("adaptive_search", None)  # propagated separately as `adaptive_search` kwarg

    if sweep_dict is not None and adaptive_search is not None:
        raise ValueError(
            "sweep block and --search-* flags are mutually exclusive: BO drives "
            "variation choice adaptively, while sweep enumerates them up-front. "
            "Drop the sweep block to use BO, or drop the --search-* flags."
        )

    if adaptive_search is not None:
        # BO path: single base config, single placeholder variation. The
        # planner synthesizes per-iteration variations during execution.
        configs = [BenchmarkConfig.model_validate(config_dict)]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        if sweep_dict is not None:
            config_dict["sweep"] = sweep_dict
        expanded = expand_sweep(config_dict)
        configs = []
        variations = []
        for variation_dict, variation_meta in expanded:
            variation_dict.pop("sweep", None)
            variation_dict.pop("multi_run", None)
            context = build_template_context(variation_dict)
            variation_dict = render_jinja2_templates(variation_dict, context)
            configs.append(BenchmarkConfig.model_validate(variation_dict))
            variations.append(variation_meta)
        if not variations:
            variations = [SweepVariation(index=0, label="base", values={})]

    plan_kwargs: dict[str, Any] = dict(
        configs=configs,
        variations=variations,
        trials=multi_run.get("num_runs", 1),
        cooldown_seconds=multi_run.get("cooldown_seconds", 0.0),
        confidence_level=multi_run.get("confidence_level", 0.95),
        set_consistent_seed=multi_run.get("set_consistent_seed", True),
        disable_warmup_after_first=multi_run.get("disable_warmup_after_first", True),
        parameter_sweep_cooldown_seconds=multi_run.get("parameter_sweep_cooldown_seconds", 0.0),
        parameter_sweep_same_seed=multi_run.get("parameter_sweep_same_seed", False),
        parameter_sweep_mode=multi_run.get("mode", "repeated"),
        adaptive_search=adaptive_search,
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/config/loader/test_plan_bo.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Run the full unit suite to confirm nothing broke**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS. Watch for any test that asserts specific `BenchmarkPlan` field counts — `adaptive_search=None` is a back-compat additive change, so they should be fine.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/config/loader/plan.py tests/unit/config/loader/test_plan_bo.py
git commit -m "feat(config): wire --search-* flags into BenchmarkPlan.adaptive_search"
```

---

## Task 8: `BayesianSearchPlanner` skopt-backed implementation

**Files:**
- Create: `src/aiperf/orchestrator/search_planner/bayesian.py`
- Test: `tests/unit/orchestrator/search_planner/test_bayesian.py`

This is the largest task. The planner must:
1. Lazy-import `skopt` with a clear error if absent.
2. Build skopt dimensions from `SearchSpaceDimension.kind` (`Real(lo, hi)` for `real`, `Integer(lo, hi)` for `int`).
3. On `ask`: ask skopt, deep-copy the base config (model_dump → dict), `_set_nested_value` each path, re-validate to `BenchmarkConfig`, return `(cfg, SweepVariation)`.
4. On `tell`: extract objective from `RunResult.summary_metrics[metric].<stat>`, sign-flip for minimize, tell skopt, append to history, increment iteration.
5. `is_converged`: max_iterations exhausted OR coefficient-of-variation over recent window below threshold.

- [ ] **Step 1: Write failing tests**

`tests/unit/orchestrator/search_planner/test_bayesian.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BayesianSearchPlanner.

Skopt is a soft dep; tests skip when not installed. Local CI must install
the `bo` extra.
"""

from __future__ import annotations

import pytest

skopt = pytest.importorskip("skopt")

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
        }
    )


def _cfg(max_iterations: int = 5, **overrides) -> AdaptiveSearchConfig:
    return AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=100, kind="int"),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
        **overrides,
    )


def test_ask_returns_cfg_and_variation():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    cfg, variation = proposal
    assert variation.index == 0
    assert variation.label.startswith("search_iter_")
    assert "phases.profiling.concurrency" in variation.values
    proposed = variation.values["phases.profiling.concurrency"]
    assert 1 <= proposed <= 100
    assert isinstance(proposed, int)  # int dim → integer
    # The mutated cfg must reflect the proposed value.
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.concurrency == proposed


def test_ask_returns_none_after_max_iterations():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=3))
    for _ in range(3):
        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal
        planner.tell(variation, [_make_result(variation, throughput=100.0)])
    assert planner.ask() is None


def test_record_extracts_avg_from_summary_metrics_and_signs_for_maximize():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    planner.tell(variation, [_make_result(variation, throughput=42.5)])
    history = planner.history()
    assert len(history) == 1
    assert history[0].objective_value == pytest.approx(42.5)


def test_record_skips_failed_runs():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    failed = RunResult(label="x", success=False, error="boom")
    planner.tell(variation, [failed, _make_result(variation, throughput=10.0)])
    assert planner.history()[0].objective_value == pytest.approx(10.0)


def test_record_with_no_successful_runs_records_none():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    planner.tell(variation, [RunResult(label="x", success=False)])
    assert planner.history()[0].objective_value is None


def test_minimize_direction_signs_correctly():
    cfg = _cfg(max_iterations=3, objective_direction=OptimizationDirection.MINIMIZE)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    _, v1 = planner.ask()
    planner.tell(v1, [_make_result(v1, throughput=10.0)])
    _, v2 = planner.ask()
    # If skopt sees signed values correctly, asking again does not crash.
    assert v2 is not None


def test_is_converged_on_max_iterations_exhausted():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=2, plateau_window=2))
    assert not planner.is_converged()
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=1.0)])
    assert planner.is_converged()


def test_is_converged_on_plateau():
    cfg = _cfg(max_iterations=20, plateau_window=3, plateau_threshold=0.05)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0)])
    assert planner.is_converged()


def _make_result(variation: SweepVariation, *, throughput: float) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv add --optional bo scikit-optimize  # install the soft dep into the project venv
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_bayesian.py -v
```

Expected: ModuleNotFoundError on `bayesian`.

- [ ] **Step 3: Implement the planner**

`src/aiperf/orchestrator/search_planner/bayesian.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skopt-backed Bayesian-Optimization outer-loop planner.

Treats `BenchmarkConfig` mutation as: model_dump → dict → _set_nested_value
→ model_validate. This sidesteps the complication that BenchmarkConfig has
deeply-nested Pydantic submodels and `_set_nested_value` only operates on
dicts. Round-trip is safe: BenchmarkConfig is the v2 validated form which
is round-trip stable by construction.

Skopt is unmaintained-ish (last release 2024); the SearchPlanner abstract
seam means swapping to optuna later is a single-file change with no API
break.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation, _set_nested_value
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner.base import (
    SearchIteration,
    SearchPlanner,
)

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["BayesianSearchPlanner"]


class BayesianSearchPlanner(SearchPlanner):
    """skopt.Optimizer-backed adaptive outer-loop planner."""

    def __init__(self, base_config: BenchmarkConfig, cfg: AdaptiveSearchConfig) -> None:
        try:
            from skopt import Optimizer
            from skopt.space import Integer, Real
        except ImportError as e:
            raise ImportError(
                "Bayesian Optimization requires the 'bo' extra: "
                "`uv pip install -e '.[bo]'` (or add scikit-optimize to your env). "
                f"Underlying import error: {e}"
            ) from e

        self._base = base_config
        self._cfg = cfg
        self._iter = 0
        self._history: list[SearchIteration] = []
        # Track ask/tell pairs so skopt's tell sees the same X it returned.
        self._pending_x: list[Any] | None = None

        dims = []
        for d in cfg.search_space:
            if d.kind == "int":
                dims.append(Integer(int(d.lo), int(d.hi)))
            else:
                dims.append(Real(d.lo, d.hi))
        self._opt = Optimizer(
            dimensions=dims,
            n_initial_points=cfg.n_initial_points,
            random_state=cfg.random_seed,
        )

    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        if self._iter >= self._cfg.max_iterations:
            return None
        if self.is_converged():
            return None

        x = self._opt.ask()
        self._pending_x = x  # remember for tell() to call skopt.tell()
        values: dict[str, Any] = {}
        for dim, suggestion in zip(self._cfg.search_space, x, strict=True):
            values[dim.path] = _coerce_for_kind(suggestion, dim)

        cfg_dict = self._base.model_dump(mode="json", exclude_none=True)
        for path, val in values.items():
            _set_nested_value(cfg_dict, path, val)
        cfg = BenchmarkConfig.model_validate(cfg_dict)

        variation = SweepVariation(
            index=self._iter,
            label=f"search_iter_{self._iter:04d}",
            values=values,
        )
        return cfg, variation

    def tell(self, variation: SweepVariation, results: list["RunResult"]) -> None:
        objective = self._extract_objective(results)
        if objective is None:
            # Skopt cannot accept None; tell it the worst value seen so far,
            # or a baseline if none seen, so the iteration counter still advances.
            tell_value = self._fallback_tell_value()
            logger.warning(
                "BO iteration %d at %s produced no successful runs; "
                "telling skopt fallback %s and continuing.",
                self._iter, variation.values, tell_value,
            )
        else:
            sign = -1.0 if self._cfg.objective_direction == OptimizationDirection.MAXIMIZE else 1.0
            tell_value = sign * objective

        if self._pending_x is None:
            raise RuntimeError("tell() called without matching ask()")
        self._opt.tell(self._pending_x, float(tell_value))
        self._pending_x = None

        self._history.append(SearchIteration(
            iteration_idx=self._iter,
            variation_values=dict(variation.values),
            objective_value=objective,
            results=list(results),
        ))
        self._iter += 1

    def is_converged(self) -> bool:
        if self._iter >= self._cfg.max_iterations:
            return True
        window = self._cfg.plateau_window
        if len(self._history) < window:
            return False
        recent_objs = [
            h.objective_value for h in self._history[-window:]
            if h.objective_value is not None
        ]
        if len(recent_objs) < window:
            return False
        mean = sum(recent_objs) / len(recent_objs)
        if mean == 0:
            return all(abs(v) < self._cfg.plateau_threshold for v in recent_objs)
        # Coefficient of variation: scale-free relative spread.
        variance = sum((v - mean) ** 2 for v in recent_objs) / len(recent_objs)
        cv = math.sqrt(variance) / abs(mean)
        return cv < self._cfg.plateau_threshold

    def history(self) -> list[SearchIteration]:
        return list(self._history)

    def _extract_objective(self, results: list["RunResult"]) -> float | None:
        """Average the configured stat across all successful trials.

        summary_metrics keys are bare metric tags; the stat (avg/p99/...)
        is a field on JsonMetricResult — NOT a suffix on the key. This is
        the gotcha called out in the design doc.
        """
        successful = [r for r in results if r.success and self._cfg.objective_metric in r.summary_metrics]
        if not successful:
            return None
        values: list[float] = []
        for r in successful:
            mr = r.summary_metrics[self._cfg.objective_metric]
            stat_value = getattr(mr, self._cfg.objective_stat, None)
            if stat_value is not None:
                values.append(float(stat_value))
        if not values:
            return None
        return sum(values) / len(values)

    def _fallback_tell_value(self) -> float:
        """Worst-seen value for the running optimizer, or 0 if none seen."""
        seen = [
            h.objective_value for h in self._history if h.objective_value is not None
        ]
        if not seen:
            return 0.0
        if self._cfg.objective_direction == OptimizationDirection.MAXIMIZE:
            return -min(seen)  # worst (smallest) maximize value, sign-flipped
        return max(seen)


def _coerce_for_kind(value: Any, dim: SearchSpaceDimension) -> Any:
    """Skopt returns numpy scalars; coerce to plain Python int/float."""
    if dim.kind == "int":
        return int(value)
    return float(value)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_bayesian.py -v
```

Expected: PASS (8 tests). If any fails, common causes:
- Skopt returning numpy types not coerced — `_coerce_for_kind` should handle.
- `_set_nested_value` raising on bad path — fix the test data, not the production code; `_set_nested_value` raising for typos is correct behavior.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/orchestrator/search_planner/bayesian.py tests/unit/orchestrator/search_planner/test_bayesian.py
git commit -m "feat(orchestrator): implement BayesianSearchPlanner with skopt backend"
```

---

## Task 9: `search_history.json` exporter

**Files:**
- Create: `src/aiperf/exporters/search_history.py`
- Test: `tests/unit/orchestrator/search_planner/test_history.py`

Writes after **every** iteration, not at end. Schema: `{"iterations": [...], "config": {...}, "best": {...}}`.

- [ ] **Step 1: Write failing test**

`tests/unit/orchestrator/search_planner/test_history.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the search_history.json incremental exporter."""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.exporters.search_history import write_search_history
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.search_planner.base import SearchIteration


def _cfg() -> AdaptiveSearchConfig:
    return AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=10,
    )


def test_write_search_history_creates_file(tmp_path: Path):
    history = [
        SearchIteration(iteration_idx=0, variation_values={"x": 5}, objective_value=10.0),
        SearchIteration(iteration_idx=1, variation_values={"x": 7}, objective_value=15.0),
    ]
    write_search_history(tmp_path, history, _cfg())
    out = tmp_path / "search_history.json"
    assert out.exists()
    data = orjson.loads(out.read_bytes())
    assert len(data["iterations"]) == 2
    assert data["iterations"][1]["objective_value"] == 15.0
    assert data["best"]["objective_value"] == 15.0  # MAXIMIZE picks 15
    assert data["best"]["iteration_idx"] == 1
    assert data["config"]["objective_metric"] == "m"


def test_write_search_history_minimize_picks_smallest(tmp_path: Path):
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MINIMIZE,
        max_iterations=10,
    )
    history = [
        SearchIteration(iteration_idx=0, variation_values={"x": 5}, objective_value=10.0),
        SearchIteration(iteration_idx=1, variation_values={"x": 7}, objective_value=8.0),
    ]
    write_search_history(tmp_path, history, cfg)
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best"]["iteration_idx"] == 1
    assert data["best"]["objective_value"] == 8.0


def test_write_search_history_skips_iterations_without_objective(tmp_path: Path):
    history = [
        SearchIteration(iteration_idx=0, variation_values={"x": 5}, objective_value=None),
        SearchIteration(iteration_idx=1, variation_values={"x": 7}, objective_value=12.0),
    ]
    write_search_history(tmp_path, history, _cfg())
    data = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert data["best"]["iteration_idx"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_history.py -v
```

Expected: ModuleNotFoundError on `aiperf.exporters.search_history`.

- [ ] **Step 3: Write the exporter**

`src/aiperf/exporters/search_history.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental writer for search_history.json (BO trajectory log).

Called after every BO iteration so a partial trajectory survives a crash.
Sits next to sweep_aggregate/ in the artifact dir, NOT inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from aiperf.config.adaptive_search import AdaptiveSearchConfig
    from aiperf.orchestrator.search_planner.base import SearchIteration

__all__ = ["write_search_history"]


def write_search_history(
    base_dir: Path,
    history: list["SearchIteration"],
    cfg: "AdaptiveSearchConfig",
) -> None:
    """Write search_history.json under base_dir.

    Schema:
        {
          "config": {...subset of AdaptiveSearchConfig},
          "iterations": [
            {"iteration_idx": int, "variation_values": {...}, "objective_value": float | None}
          ],
          "best": {"iteration_idx": int, "objective_value": float, "variation_values": {...}}
                  | null when no objectives recorded
        }
    """
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

    iterations_payload = [
        {
            "iteration_idx": h.iteration_idx,
            "variation_values": h.variation_values,
            "objective_value": h.objective_value,
        }
        for h in history
    ]
    scored = [h for h in history if h.objective_value is not None]
    if scored:
        if cfg.objective_direction == OptimizationDirection.MAXIMIZE:
            best = max(scored, key=lambda h: h.objective_value)
        else:
            best = min(scored, key=lambda h: h.objective_value)
        best_payload = {
            "iteration_idx": best.iteration_idx,
            "objective_value": best.objective_value,
            "variation_values": best.variation_values,
        }
    else:
        best_payload = None

    payload = {
        "config": {
            "algorithm": cfg.algorithm,
            "objective_metric": cfg.objective_metric,
            "objective_stat": cfg.objective_stat,
            "objective_direction": str(cfg.objective_direction),
            "max_iterations": cfg.max_iterations,
            "search_space": [
                {"path": d.path, "lo": d.lo, "hi": d.hi, "kind": d.kind}
                for d in cfg.search_space
            ],
        },
        "iterations": iterations_payload,
        "best": best_payload,
    }
    out = base_dir / "search_history.json"
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/orchestrator/search_planner/test_history.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/exporters/search_history.py tests/unit/orchestrator/search_planner/test_history.py
git commit -m "feat(exporters): add incremental search_history.json writer"
```

---

## Task 10: `MultiRunOrchestrator.execute_adaptive_search`

**Files:**
- Modify: `src/aiperf/orchestrator/orchestrator.py:50-90` (extend `execute`), append new method.
- Test: `tests/unit/orchestrator/test_execute_adaptive_search.py`

The new method:
1. Loops calling `planner.ask()` until `None`.
2. Per iteration: `build_strategy(plan, logger)`, `_run_independent_cell(...)`, `planner.tell(...)`, write `search_history.json` incrementally.
3. Handles `cancel_check` between iterations.
4. Returns the flat `RunResult` list (same shape as the grid path).

- [ ] **Step 1: Write failing test**

`tests/unit/orchestrator/test_execute_adaptive_search.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator.execute_adaptive_search."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

skopt = pytest.importorskip("skopt")

import orjson

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner


class _RecordingExecutor(RunExecutor):
    """Returns synthetic RunResult with a configurable objective metric value."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, dict]] = []

    def derive_id(self, plan, var_idx: int, trial: int) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        self.calls.append((run.variation.index, run.trial, dict(run.variation.values)))
        # Linear objective so the BO learns: throughput = concurrency * 10.
        concurrency = run.variation.values.get("phases.profiling.concurrency", 1)
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tok/s", avg=float(concurrency) * 10.0,
                ),
            },
            artifacts_path=run.artifact_dir,
        )


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
        }
    )


def _plan_with_bo(max_iterations: int = 4, trials: int = 1) -> BenchmarkPlan:
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[
            SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=100, kind="int"),
        ],
        objective_metric="output_token_throughput",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
    )
    return BenchmarkPlan(
        configs=[_base_config()],
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=trials,
        adaptive_search=cfg,
    )


@pytest.mark.asyncio
async def test_execute_adaptive_search_runs_max_iterations_iterations(tmp_path: Path):
    plan = _plan_with_bo(max_iterations=4, trials=1)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    executor = _RecordingExecutor()

    results = await orch.execute_adaptive_search(plan, executor, planner)
    assert len(results) == 4  # max_iterations × trials
    assert all(r.success for r in results)
    # Variations distinct per iteration:
    seen_idx = sorted({r.variation_label for r in results})
    assert seen_idx == ["search_iter_0000", "search_iter_0001", "search_iter_0002", "search_iter_0003"]


@pytest.mark.asyncio
async def test_execute_adaptive_search_writes_search_history_incrementally(tmp_path: Path):
    plan = _plan_with_bo(max_iterations=3)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    await orch.execute_adaptive_search(plan, _RecordingExecutor(), planner)
    search_history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert len(search_history["iterations"]) == 3
    assert search_history["best"] is not None


@pytest.mark.asyncio
async def test_execute_dispatches_to_adaptive_when_adaptive_search_set(tmp_path: Path):
    """The top-level execute() must route plans-with-adaptive_search to the BO path."""
    plan = _plan_with_bo(max_iterations=2)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    # Call execute() (not execute_adaptive_search): the dispatch should kick in.
    # Pass planner via a kwarg the orchestrator forwards.
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    results = await orch.execute(plan, _RecordingExecutor(), search_planner=planner)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_execute_adaptive_search_respects_cancel_check(tmp_path: Path):
    plan = _plan_with_bo(max_iterations=10)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    orch = MultiRunOrchestrator(base_dir=tmp_path)
    state = {"calls": 0}

    def cancel_check() -> bool:
        state["calls"] += 1
        return state["calls"] > 4  # cancel after a few iterations

    results = await orch.execute_adaptive_search(
        plan, _RecordingExecutor(), planner, cancel_check=cancel_check,
    )
    assert len(results) < 10  # cancelled early
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest -n auto tests/unit/orchestrator/test_execute_adaptive_search.py -v
```

Expected: AttributeError on `execute_adaptive_search` and on `search_planner=` kwarg.

- [ ] **Step 3: Extend `execute` and add `execute_adaptive_search`**

In `src/aiperf/orchestrator/orchestrator.py`, modify `execute` (line 50) — add `search_planner` kwarg + dispatch:

```python
    async def execute(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None = None,
        search_planner: Any = None,
    ) -> list[RunResult]:
        """Execute all (variation, trial) runs in the plan.

        Iteration order:
        - When ``plan.adaptive_search`` is set, dispatches to
          :meth:`execute_adaptive_search` (BO / adaptive). ``search_planner``
          must be supplied in this case.
        - Otherwise honors plan.parameter_sweep_mode (REPEATED/INDEPENDENT).
        ...
        """
        from aiperf.common.enums import SweepMode

        if plan.is_adaptive_search:
            if search_planner is None:
                raise ValueError(
                    "plan.adaptive_search is set but no search_planner was passed to execute(). "
                    "The CLI runner is expected to instantiate one and forward it."
                )
            return await self.execute_adaptive_search(
                plan, executor, search_planner, cancel_check=cancel_check
            )

        if plan.parameter_sweep_mode == SweepMode.REPEATED:
            return await self._execute_repeated(plan, executor, cancel_check=cancel_check)
        return await self._execute_independent(plan, executor, cancel_check=cancel_check)
```

Then append `execute_adaptive_search` after `_run_independent_cell` (after line 214):

```python
    async def execute_adaptive_search(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        planner: Any,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[RunResult]:
        """Drive an adaptive outer loop (e.g. BO).

        Each iteration: ask planner for a (cfg, variation), run all trials
        for it via :meth:`_run_independent_cell`, feed results back to the
        planner, write search_history.json incrementally.
        """
        from aiperf._cli_runner_helpers import build_strategy
        from aiperf.exporters.search_history import write_search_history

        all_results: list[RunResult] = []
        logger.info(
            f"Starting adaptive outer-loop benchmark "
            f"({plan.adaptive_search.algorithm}, max_iterations={plan.adaptive_search.max_iterations}, "
            f"trials per point={plan.trials})"
        )

        while True:
            if cancel_check is not None and cancel_check():
                logger.info(f"Adaptive outer loop cancelled after {planner._iter} iterations")
                return all_results

            proposal = planner.ask()
            if proposal is None:
                logger.info("Adaptive outer loop converged or max_iterations exhausted")
                return all_results
            cfg, variation = proposal
            strategy = build_strategy(plan, logger)
            strategy.validate_config(cfg)

            logger.info(
                f"[BO iter {variation.index}] proposing {variation.values}"
            )
            cell_results, aborted = await self._run_independent_cell(
                plan,
                executor,
                strategy=strategy,
                cfg=cfg,
                variation=variation,
                var_idx=variation.index,
                prior_all_results=all_results,
                cancel_check=cancel_check,
            )
            planner.tell(variation, cell_results)
            all_results.extend(cell_results)
            write_search_history(self.base_dir, planner.history(), plan.adaptive_search)

            if aborted:
                logger.warning(f"Outer-loop cell at iter {variation.index} aborted; halting BO")
                return all_results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest -n auto tests/unit/orchestrator/test_execute_adaptive_search.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Run the full orchestrator test directory to confirm no regressions**

```bash
uv run pytest -n auto tests/unit/orchestrator/
```

Expected: PASS — pre-existing tests untouched, new tests green.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/orchestrator/orchestrator.py tests/unit/orchestrator/test_execute_adaptive_search.py
git commit -m "feat(orchestrator): add execute_adaptive_search for Bayesian outer loop"
```

---

## Task 11: K8s rejection (operator path) + forward-compat sweep-controller guard

**Files:**
- Modify: `src/aiperf/cli_runner.py:451-477` (extend `_reject_in_process_sweep_under_operator`)
- Modify: `src/aiperf/sweep_controller/plan_builder.py` (defensive guard only)
- Test: `tests/unit/cli_commands/test_bo_reject_under_operator.py`

The operator-managed path (`AIPERF_OPERATOR_MANAGED=1`) is the only way an in-process BO request can reach a controller pod in v1, so that's the rejection site that actually fires. The `sweep_controller/plan_builder.py` guard is forward-compatible defense for when the CRD eventually grows an `adaptive_search` field — there's no test in v1 because there is no path to construct a `spec.multi_run.adaptive_search`-bearing CR (the CRD doesn't have the field yet).

- [ ] **Step 1: Write failing tests for the operator path**

`tests/unit/cli_commands/test_bo_reject_under_operator.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator-managed-pod rejection of BO outer-loop plans."""

from __future__ import annotations

import pytest

from aiperf.cli_runner import _reject_in_process_sweep_under_operator
from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection


def _bo_plan() -> BenchmarkPlan:
    cfg = AdaptiveSearchConfig(
        algorithm="bayes",
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objective_metric="m",
        objective_stat="avg",
        objective_direction=OptimizationDirection.MAXIMIZE,
        max_iterations=10,
    )
    return BenchmarkPlan(
        configs=[BenchmarkConfig.model_validate({
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
        })],
        variations=[SweepVariation(index=0, label="base", values={})],
        adaptive_search=cfg,
    )


def test_reject_bo_under_operator(monkeypatch):
    monkeypatch.setenv("AIPERF_OPERATOR_MANAGED", "1")
    with pytest.raises(SystemExit, match="adaptive outer loop"):
        _reject_in_process_sweep_under_operator(_bo_plan())


def test_bo_allowed_outside_operator(monkeypatch):
    monkeypatch.delenv("AIPERF_OPERATOR_MANAGED", raising=False)
    # Should not raise: BO is fine in-process when not under the operator.
    _reject_in_process_sweep_under_operator(_bo_plan())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_bo_reject_under_operator.py -v
```

Expected: SystemExit not raised, since the function only checks `is_sweep`.

- [ ] **Step 3: Extend `_reject_in_process_sweep_under_operator`**

In `src/aiperf/cli_runner.py`, replace the body of `_reject_in_process_sweep_under_operator` (lines 451-477) with:

```python
def _reject_in_process_sweep_under_operator(plan: BenchmarkPlan) -> None:
    """Block in-process sweep / BO outer loop when running inside an operator-managed pod.

    The k8s operator drives sweeps cluster-wide via the AIPerfSweep CR (one
    AIPerfJob per variation, controller pod sees a single-config plan). It
    does NOT support adaptive outer loops in v1: variation k>0 is unknown
    until variation k-1 has been scored, breaking the operator's deterministic
    cardinality contract (status.totalVariations is set at CR creation).
    Both grid sweeps and outer loops are hard-rejected here.
    """
    if os.environ.get("AIPERF_OPERATOR_MANAGED") != "1":
        return
    if plan.is_adaptive_search:
        ol = plan.adaptive_search
        raise SystemExit(
            f"Adaptive outer loop ({ol.mode}, "
            f"max_iterations={ol.max_iterations}, search-space={[d.path for d in ol.search_space]}) "
            f"is not supported in operator-managed runs (AIPERF_OPERATOR_MANAGED=1). "
            f"Cluster sweeps use the AIPerfSweep CRD with a deterministic variation set; "
            f"BO needs adaptive variation choice. Run BO in-process (without the operator) "
            f"or use a grid AIPerfSweep instead. See docs/sweeping/bayesian-optimization.md."
        )
    if plan.is_sweep:
        swept_params = sorted(
            {k for variation in plan.variations if variation is not None for k in variation.values}
        )
        raise SystemExit(
            f"In-process parameter sweep ({len(plan.configs)} variations across "
            f"{swept_params or '<unknown>'}) is not supported in operator-managed "
            f"runs (AIPERF_OPERATOR_MANAGED=1). Use the AIPerfSweep CRD "
            f"(cluster-scope) for cross-job sweeps — see docs/kubernetes/sweeps.md "
            f"— or submit one AIPerfJob per variation. To run as a single point "
            f"benchmark, drop the comma in --concurrency / other magic-list flags."
        )
```

- [ ] **Step 4: Add the sweep_controller defensive guard**

Open `src/aiperf/sweep_controller/plan_builder.py`. After the `spec = AIPerfSweepSpec.model_validate(spec_dict)` line (line 35), insert the defensive forward-compat check:

```python
    # Forward-compat: AIPerfSweep does not support adaptive outer loops in v1.
    # The CRD has no `adaptive_search` field, so this guard is dormant today; it
    # exists so a future CRD extension that adds the field doesn't accidentally
    # ship a code path that tries to grid-expand an adaptive sweep.
    if getattr(spec, "multi_run", None) is not None and getattr(
        spec.multi_run, "adaptive_search", None
    ) is not None:
        import kopf

        raise kopf.PermanentError(
            "AIPerfSweep does not support adaptive outer loops (BO). The cluster "
            "sweep controller requires a deterministic variation set; adaptive "
            "variation choice is incompatible with the operator's cardinality "
            "contract. Use a grid sweep block instead, or run BO in-process via "
            "`aiperf profile --search-space ...`."
        )
```

No test for this in v1: there is no path in the current CRD to construct a `spec.multi_run.adaptive_search`-bearing dict, so a test would have to fabricate the shape and invariably becomes more about the test's mocks than the production behavior. When the CRD adds the field, add the test alongside the schema change.

- [ ] **Step 5: Run the operator-path rejection test**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_bo_reject_under_operator.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/cli_runner.py src/aiperf/sweep_controller/plan_builder.py tests/unit/cli_commands/test_bo_reject_under_operator.py
git commit -m "feat(operator): reject BO outer loops at operator path; defensive guard in sweep_controller"
```

---

## Task 12: CLI runner integration (`_run_multi_benchmark`)

**Files:**
- Modify: `src/aiperf/cli_runner.py:391-448` (instantiate planner + forward to orchestrator).
- Test: `tests/unit/cli_commands/test_run_multi_benchmark_bo.py`

- [ ] **Step 1: Write failing test**

`tests/unit/cli_commands/test_run_multi_benchmark_bo.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that _run_multi_benchmark instantiates a BO planner when plan.adaptive_search is set."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

skopt = pytest.importorskip("skopt")

from aiperf.cli_runner import _run_multi_benchmark


def test_run_multi_benchmark_with_bo_invokes_orchestrator_with_planner(tmp_path, monkeypatch):
    """A plan with adaptive_search should reach the orchestrator with a BayesianSearchPlanner."""
    monkeypatch.delenv("AIPERF_OPERATOR_MANAGED", raising=False)
    from aiperf.config.benchmark import BenchmarkPlan
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.adaptive_search import AdaptiveSearchConfig, SearchSpaceDimension
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

    plan = BenchmarkPlan(
        configs=[BenchmarkConfig.model_validate({
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "concurrency": 1}],
            "ui_type": "none",
        })],
        variations=[SweepVariation(index=0, label="base", values={})],
        trials=1,
        adaptive_search=AdaptiveSearchConfig(
            algorithm="bayes",
            search_space=[SearchSpaceDimension(path="phases.profiling.concurrency", lo=1, hi=10, kind="int")],
            objective_metric="m",
            objective_stat="avg",
            objective_direction=OptimizationDirection.MAXIMIZE,
            max_iterations=2,
            n_initial_points=1,
        ),
    )

    with patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator") as orch_cls:
        instance = orch_cls.return_value
        # Async method must use AsyncMock — asyncio.run() awaits the return value.
        instance.execute = AsyncMock(return_value=[])
        # Stub the LocalSubprocessExecutor to avoid spawning subprocesses.
        with patch("aiperf.orchestrator.local_executor.LocalSubprocessExecutor"):
            with patch("aiperf.cli_runner._summarize_and_export"):
                _run_multi_benchmark(plan)

    # Inspect the kwargs passed to execute(): search_planner must be a BayesianSearchPlanner.
    from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner
    call_kwargs = instance.execute.call_args.kwargs
    assert "search_planner" in call_kwargs
    assert isinstance(call_kwargs["search_planner"], BayesianSearchPlanner)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_run_multi_benchmark_bo.py -v
```

Expected: AssertionError on missing `search_planner` kwarg (current `_run_multi_benchmark` doesn't pass one).

- [ ] **Step 3: Modify `_run_multi_benchmark`**

In `src/aiperf/cli_runner.py`, change the `orchestrator.execute(plan, executor)` call (line 436) to forward an `search_planner` when `plan.is_adaptive_search`. Replace lines 428-439 with:

```python
    orchestrator = MultiRunOrchestrator(base_dir=base_dir)

    import asyncio as _asyncio

    from aiperf.orchestrator.local_executor import LocalSubprocessExecutor

    executor = LocalSubprocessExecutor(base_dir=base_dir)

    search_planner = None
    if plan.is_adaptive_search:
        from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner

        search_planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
        logger.info(
            f"Bayesian outer loop active: max_iterations={plan.adaptive_search.max_iterations}, "
            f"search-space={[d.path for d in plan.adaptive_search.search_space]}, "
            f"objective={plan.adaptive_search.objective_metric}:"
            f"{plan.adaptive_search.objective_stat}:{plan.adaptive_search.objective_direction}"
        )

    try:
        results = _asyncio.run(
            orchestrator.execute(plan, executor, search_planner=search_planner)
        )
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest -n auto tests/unit/cli_commands/test_run_multi_benchmark_bo.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/cli_runner.py tests/unit/cli_commands/test_run_multi_benchmark_bo.py
git commit -m "feat(cli): instantiate BayesianSearchPlanner from CLI runner when --search-* flags set"
```

---

## Task 13: End-to-end CLI integration test

**Files:**
- Test: `tests/component_integration/test_bo_e2e.py`

This validates the whole stack runs without spawning a real benchmark — uses an executor stub but exercises the CLI parsing → plan-building → orchestration pipeline.

- [ ] **Step 1: Write the test**

`tests/component_integration/test_bo_e2e.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration: BO end-to-end with stub executor."""

from __future__ import annotations

from pathlib import Path

import pytest

skopt = pytest.importorskip("skopt")

import orjson

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.benchmark import BenchmarkRun
from aiperf.config.loader.plan import build_benchmark_plan
from aiperf.config.config import AIPerfConfig
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner


pytestmark = pytest.mark.component_integration


class _StubExecutor(RunExecutor):
    def derive_id(self, plan, var_idx, trial):
        return f"stub-v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        c = run.variation.values.get("phases.profiling.concurrency", 1)
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(unit="tok/s", avg=float(c) * 5.0),
            },
            artifacts_path=run.artifact_dir,
        )


@pytest.mark.asyncio
async def test_bo_e2e_via_build_benchmark_plan(tmp_path: Path):
    cfg = AIPerfConfig.model_validate({
        "models": ["m"],
        "endpoint": {"urls": ["http://x"], "type": "chat"},
        "datasets": [{"name": "default", "type": "synthetic"}],
        "phases": [{"name": "profiling", "concurrency": 1}],
        "multi_run": {
            "num_runs": 1,
            "adaptive_search": {
                "algorithm": "bayes",
                "search_space": [
                    {"path": "phases.profiling.concurrency", "lo": 1, "hi": 50, "kind": "int"},
                ],
                "objective_metric": "output_token_throughput",
                "objective_stat": "avg",
                "objective_direction": "maximize",
                "max_iterations": 5,
                "n_initial_points": 2,
                "random_seed": 42,
            },
        },
    })
    plan = build_benchmark_plan(cfg)
    assert plan.is_adaptive_search
    assert plan.adaptive_search.max_iterations == 5

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    results = await orch.execute(plan, _StubExecutor(), search_planner=planner)

    assert len(results) == 5
    assert (tmp_path / "search_history.json").exists()
    history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert history["best"] is not None
    # With reward = concurrency * 5, BO should converge toward the high end
    # (not asserted strictly to avoid skopt-version flake; just sanity-check
    # that best.objective_value > a small floor).
    assert history["best"]["objective_value"] > 0
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest -n auto -m component_integration tests/component_integration/test_bo_e2e.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/component_integration/test_bo_e2e.py
git commit -m "test(bo): component-integration end-to-end via build_benchmark_plan"
```

---

## Task 14: Documentation

**Files:**
- Create: `docs/sweeping/bayesian-optimization.md`
- Modify: `docs/index.yml` (add the new file)
- Modify: `docs/architecture.md` (one paragraph in the multi-run / sweep section)
- Modify: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` (four-file sync — one paragraph under "Parameter Sweeping")
- Modify: `llms.txt` (add a line for the new doc)

- [ ] **Step 1: Write `docs/sweeping/bayesian-optimization.md`**

`docs/sweeping/bayesian-optimization.md`:

```markdown
# Bayesian-Optimization Outer Loop

`aiperf profile --search-space ... --search-metric ... --search-direction ... --search-max-iterations ...` runs an adaptive outer loop instead of a grid sweep. Each iteration the planner asks `skopt` for the next point in the search space, runs `--num-profile-runs` benchmarks at it, scores the configured objective, and feeds the result back to the optimizer.

## When to use it

Use BO when:
- The search space is too large to grid-enumerate (e.g. concurrency 1–1000 with no obvious step).
- You only care about finding the best point, not characterizing the whole frontier.
- A single scalar objective captures what you care about (throughput, p99 latency, etc.).

Use a grid sweep when:
- You need a complete Pareto frontier (use existing `--concurrency 10,20,50,100` magic-list flags + `sweep_aggregate/`).
- You want to compare specific points the team has agreed on.
- You need cluster-distributed sweeping (BO is in-process only in v1; cluster runs use `AIPerfSweep` CRDs).

## Quick start

```bash
aiperf profile \
    --models my-model \
    --endpoint http://infer.example.com \
    --search-space "phases.profiling.concurrency:1,1000:int" \
    --search-metric output_token_throughput \
    --search-direction maximize \
    --search-max-iterations 30 \
    --search-random-seed 42 \
    --num-profile-runs 3
```

This runs 30 search iterations × 3 trials each = 90 benchmarks. Output:
- `<artifact_dir>/search_iter_NNNN/profile_runs/run_NNNN/` — per-trial artifacts.
- `<artifact_dir>/search_history.json` — BO trajectory, written incrementally.
- `<artifact_dir>/sweep_aggregate/profile_export_aiperf_sweep.{json,csv}` — same per-combination aggregate the grid path produces.

## Flag reference

| Flag | Required | Description |
|---|---|---|
| `--search-space PATH:LO,HI[:KIND]` | yes | Repeatable. Dotted-path into `BenchmarkConfig`. `KIND` is `int` or `real`, default `real`. |
| `--search-metric METRIC` | yes | Metric tag to optimize (e.g. `output_token_throughput`). The bare tag — NOT a flattened `_avg`/`_p99` aggregator-suffixed key. |
| `--search-stat STAT` | no | Statistic on the metric: `avg` / `p50` / `p90` / `p95` / `p99`. Default `avg`. |
| `--search-direction DIR` | yes | `maximize` (throughput, goodput) or `minimize` (latency, TTFT). |
| `--search-max-iterations N` | yes | Maximum search iterations. 2–200. |
| `--search-initial-points N` | no | Random Sobol points before GP fitting. Default 5. Must be < `--search-max-iterations`. |
| `--search-random-seed N` | no | Reproducibility seed for `skopt.Optimizer`. |

## Search space grammar

`PATH:LO,HI[:KIND]`

- `PATH` is a dotted path resolved by `_set_nested_value` (the same primitive grid sweeps use). For named-list segments like `phases.profiling.concurrency`, the segment matches against the `name` field. Typos error loudly with the available names listed.
- `LO` and `HI` are inclusive bounds parsed as floats. For `:int`, skopt rounds to integers.
- `:int` uses `skopt.space.Integer`, `:real` uses `skopt.space.Real`. Categorical dimensions are not supported in v1.

Multi-dim:

```bash
--search-space "phases.profiling.concurrency:1,500:int" \
--search-space "phases.profiling.request_rate:0.1,100.0:real"
```

## Objective semantics

The planner averages `--search-stat` across all successful trials at each search point. Failed trials are skipped; an iteration with zero successful trials gets a fallback "worst-seen" tell so skopt advances without crashing — a warning is logged.

`--search-metric` must match a key in `RunResult.summary_metrics` produced by the run — that is, the bare metric tag (`output_token_throughput`, `time_to_first_token`), not the flattened `_avg`/`_p99` aggregator-suffixed form.

## Convergence detection

The loop terminates when either:
1. `--search-max-iterations` iterations have been run.
2. The coefficient of variation (`stddev / |mean|`) of the last `plateau_window` (default 5) iterations' objective values falls below the plateau threshold (default 0.01 = 1% relative spread).

Plateau is **scale-free** — works for throughput (~1000) and latency (~50) without tuning. Convergence can fire as early as iteration `plateau_window` if the first 5 random-Sobol points happen to hit a flat region; this is correct behavior, not a bug.

## Mutual exclusion

`--search-*` is mutually exclusive with:
- Magic-list flags that produce sweeps (`--concurrency 10,20,30`).
- Explicit `sweep:` blocks in YAML.
- `--convergence-metric` (adaptive trial-level early stop). Reason: the trial-level convergence semantics are orthogonal to outer-loop convergence; they need separate thought before composition.
- The Kubernetes operator (`AIPERF_OPERATOR_MANAGED=1`). The operator's deterministic-cardinality contract is incompatible with adaptive variation choice. Submit cluster sweeps as `AIPerfSweep` CRDs (grid only) or run BO in-process.

## Output schema

`search_history.json`:

```json
{
  "config": {
    "algorithm": "bayes",
    "objective_metric": "output_token_throughput",
    "objective_stat": "avg",
    "objective_direction": "maximize",
    "max_iterations": 30,
    "search_space": [{"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}]
  },
  "iterations": [
    {"iteration_idx": 0, "variation_values": {"phases.profiling.concurrency": 503}, "objective_value": 1247.3},
    {"iteration_idx": 1, "variation_values": {"phases.profiling.concurrency": 178}, "objective_value": 942.1},
    "..."
  ],
  "best": {
    "iteration_idx": 23,
    "objective_value": 1822.5,
    "variation_values": {"phases.profiling.concurrency": 814}
  }
}
```

The file is rewritten after every iteration, so a crashed run still leaves the partial trajectory on disk.
```

- [ ] **Step 2: Add the doc to `docs/index.yml`**

Add an entry under the appropriate section (look at how `kubernetes/sweeps.md` is referenced, mirror that shape):

```yaml
  - page: Bayesian Optimization
    path: ./docs/sweeping/bayesian-optimization.md
```

- [ ] **Step 3: Update `docs/architecture.md`**

Find the "Parameter Sweeping" / multi-run section and add a paragraph:

```markdown
**Adaptive outer loop (BO).** When `--search-space` is set, `MultiRunOrchestrator.execute` dispatches to `execute_adaptive_search`, which drives a pluggable `SearchPlanner` (currently `BayesianSearchPlanner` backed by `skopt`). Each iteration the planner proposes a `BenchmarkConfig`, the orchestrator runs `plan.trials` benchmarks at it via the same `_run_independent_cell` used by grid mode, and feeds results back. A separate `search_history.json` is written incrementally next to `sweep_aggregate/`; the existing post-hoc aggregator handles BO results unchanged because `aggregate_sweep_and_export` groups by the stamped `variation_values`. BO is in-process only in v1 — cluster sweeps use the `AIPerfSweep` CRD with a deterministic variation set. See `docs/sweeping/bayesian-optimization.md`.
```

- [ ] **Step 4: Update the four-file sync (CLAUDE.md, AGENTS.md, copilot-instructions.md, python.mdc)**

In each of the four files, find the "## Parameter Sweeping" section and append a third bullet to the existing two-path list:

```markdown
- **Adaptive outer loop (BO)** — `aiperf profile --search-space "phases.profiling.concurrency:1,1000:int" --search-metric output_token_throughput --search-direction maximize --search-max-iterations 30`. `BenchmarkPlan.adaptive_search` carries a typed `AdaptiveSearchConfig`; `MultiRunOrchestrator.execute` dispatches to `execute_adaptive_search` which drives a `BayesianSearchPlanner` (skopt soft dep behind the `[bo]` extra). Mutually exclusive with magic-list/grid sweeps and with `AIPERF_OPERATOR_MANAGED=1`; rejected at both `_reject_in_process_sweep_under_operator` and `sweep_controller.plan_builder`. `search_history.json` is written incrementally next to `sweep_aggregate/`. See `docs/sweeping/bayesian-optimization.md`.
```

- [ ] **Step 5: Update `llms.txt`**

Add a line under the appropriate section:

```
- [`docs/sweeping/bayesian-optimization.md`](docs/sweeping/bayesian-optimization.md) - Bayesian-Optimization adaptive outer loop: --search-space CLI flags, search-space grammar, convergence detection, mutual-exclusion rules
```

- [ ] **Step 6: Verify the four-file sync**

```bash
make check-agent-files-sync
```

Expected: pass.

- [ ] **Step 7: Verify the doc index**

```bash
uv run python tools/check_docs_index.py
```

Expected: pass (no missing entries).

- [ ] **Step 8: Commit**

```bash
git add docs/sweeping/bayesian-optimization.md docs/index.yml docs/architecture.md llms.txt AGENTS.md CLAUDE.md .github/copilot-instructions.md .cursor/rules/python.mdc
git commit -m "docs(bo): add Bayesian outer-loop guide + four-file sync update"
```

---

## Task 15: Final integration sweep

- [ ] **Step 1: Run the full unit test suite**

```bash
uv run pytest -n auto tests/unit/
```

Expected: PASS. No regressions introduced by the new `adaptive_search` field on `BenchmarkPlan` (it's optional with default `None`).

- [ ] **Step 2: Run the component-integration suite**

```bash
uv run pytest -n auto -m component_integration
```

Expected: PASS, including the new `test_bo_e2e.py`.

- [ ] **Step 3: Run pre-commit on all files**

```bash
pre-commit run --all-files
```

Expected: PASS, or auto-fix and stage. Hooks that matter for this branch: `ruff`, `ruff-format`, `check-ergonomics`, `check-ruff-baselined` (no new violations baselined!), `validate-plugin-schemas`, `generate-cli-docs`, `check-agent-files-sync`, `check-docs-index`.

- [ ] **Step 4: Sanity-check ergonomics**

```bash
make check-ergonomics
make check-ruff-baselined
```

Expected: zero new violations. Common pitfalls in this branch:
- `BayesianSearchPlanner._extract_objective` is short and well-named — don't add a docstring example with `foo`/`bar`.
- All new public functions and methods need return types and `Field(description=...)`.
- New error messages name the operation and include enough context to act on (per LLM-ergonomics rules).

- [ ] **Step 5: Manual smoke test**

```bash
uv run aiperf profile \
    --models my-model \
    --endpoint http://localhost:8000 \
    --search-space "phases.profiling.concurrency:1,32:int" \
    --search-metric output_token_throughput \
    --search-direction maximize \
    --search-max-iterations 4 \
    --search-random-seed 42 \
    --num-profile-runs 1 \
    --ui simple
```

(Requires a reachable inference server.) Verify:
- `search_history.json` exists in the artifact dir and has 4 iterations.
- `search_iter_0000`, `search_iter_0001`, `search_iter_0002`, `search_iter_0003` directories exist.
- `sweep_aggregate/profile_export_aiperf_sweep.json` exists and groups by `variation_values`.

If you don't have an inference server, skip this step and note it in the PR description.

- [ ] **Step 6: Final commit (if anything was tweaked during cleanup)**

```bash
git status  # confirm clean tree
```

If clean, no extra commit. Otherwise commit cleanup.

---

## Self-review notes (for plan author)

**Spec coverage check.** The spec calls out: a `SearchPlanner` ABC (Task 2), `BayesianSearchPlanner` (Task 8), `AdaptiveSearchConfig` (Task 2), `search_history.json` exporter (Task 9), `execute_adaptive_search` orchestrator method (Task 10), v1 CLI flags (Task 4), v1→v2 converter (Task 5), `build_benchmark_plan` integration (Task 7), K8s rejection at both sites (Task 11), CLI runner integration (Task 12), `is_adaptive_search` separate from `is_sweep` (Task 6), reuse of `_run_independent_cell` (Task 10), incremental history (Task 9 + Task 10), random seed (Tasks 2, 8), search-space dimension types (Tasks 2, 3, 8), metric:stat:direction grammar (Task 3), CV-based plateau (Task 8), `bo` extra (Task 1), four-file sync (Task 14), `docs/index.yml` (Task 14). All covered.

**Type consistency.** `SearchSpaceDimension`, `AdaptiveSearchConfig`, `SearchIteration`, `SearchPlanner`, `BayesianSearchPlanner` names used consistently across all tasks. `search_planner` kwarg name on `MultiRunOrchestrator.execute` matches between Tasks 10 and 12. `OptimizationDirection` is reused (not duplicated) across Tasks 2, 3, 8, 9, 11.

**Risk areas.**
1. The `_run_independent_cell` method is private (`_`-prefixed); this plan calls it directly from `execute_adaptive_search` within the same class. Keep it private — both call sites are inside `MultiRunOrchestrator`.
2. `BenchmarkConfig` round-trip via `model_dump → dict mutate → model_validate` is the BO mutation primitive. Round-trip stability is assumed; if any nested submodel has `model_validate`-time side-effects (e.g. derived fields), Task 8 tests will catch it. If they fail, switch to `_set_nested_value`-on-Pydantic via `BenchmarkConfig.model_construct()` + targeted setter — but try the dump-mutate-validate path first.
3. `skopt` is unmaintained-ish (last release 2024). The `SearchPlanner` ABC means swapping to `optuna` later is a single-file replacement.
4. `parse_search_space` does duplicate the lo<hi check that `SearchSpaceDimension.model_validator` already enforces. Acceptable: the validator's error message would mention "search-space dim" but the CLI parser names the actual flag, which is more useful at the CLI seam.
