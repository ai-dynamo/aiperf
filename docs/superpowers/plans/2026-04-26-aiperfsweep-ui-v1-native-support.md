# AIPerfSweep — Native `operator/ui-v1` Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class, dual-backed (live CR + archived PVC) `AIPerfSweep` visibility to the `operator/ui-v1` Preact SPA, with a built-in metrics-vs-swept-axis comparison panel and durable child→parent back-links on `/jobs`.

**Architecture:** A new `sweep_union` module mirrors `job_union`'s live+archived join. Three new FastAPI endpoints under `/api/v1/sweeps` (list, detail, cells). A per-child `sweep.json` marker dropped by the sweep-controller at child-create time keeps the back-link durable across CR TTL reap. Two new ui-v1 pages (`SweepsList`, `SweepDetail`) plus two new components (`cells-chart`, `cells-table`). All read-only — no cancel button, no create form.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, kubernetes_asyncio, kopf, orjson, pytest (asyncio + xdist + `-n auto`), Preact 10 + htm + Chart.js 4.

**Spec:** `docs/superpowers/specs/2026-04-26-aiperfsweep-ui-v1-native-support-design.md`

---

## Conventions

- Branch HEAD: `ajc/k8s` — commit on this branch (per saved feedback). Do NOT spin off a feature branch.
- All commits: `git commit -s --no-verify` and run `ruff format src/ tests/ && ruff check --fix src/ tests/` manually before each commit. Reason: pre-commit framework's internal `git stash --include-untracked` corrupts state under parallel agents (saved feedback `gotcha_precommit_auto_stash_destroys_parallel_agents.md`).
- Tests: `uv run pytest tests/unit/ -n auto` ONLY — one pytest invocation per task, never split across subfolders, never `--all-files`.
- File header: SPDX block matches existing files in the same directory (copy from a sibling).
- All public Pydantic fields require `Field(description=...)`; all functions/methods require type hints; no `Optional[X]` (use `X | None`).
- No new module-level mutable state. Constants tunable at runtime live in `src/aiperf/common/environment.py` (saved feedback `feedback_constants_in_environment_py.md`).
- Do not invent a slash command, do not modify `.github/copilot-instructions.md` / `.cursor/rules/python.mdc` (those mirror only when CLAUDE.md is touched — this plan touches neither).

## File Map

### Backend — new files

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/sweep_union.py` | Live-CR + archived-PVC join over AIPerfSweep, mirroring `job_union.py`. |
| `src/aiperf/operator/routers/sweeps.py` | FastAPI router for `/api/v1/sweeps*`. |
| `src/aiperf/operator/routers/sweeps_models.py` | Pydantic response models for the sweeps router. |

### Backend — edited files

| Path | Change |
|---|---|
| `src/aiperf/operator/results_layout.py` | Add `resolve_sweep_dir(...)` helper (sibling of `resolve_run_dir`). |
| `src/aiperf/kubernetes/client.py` | Add `list_aiperfsweeps`, `find_aiperfsweep`, `get_raw_aiperfsweep`, `get_raw_aiperfsweep_status`. |
| `src/aiperf/operator/routers/jobs_models.py` | Add `sweep_name`, `variation_index`, `variation_label` to `ActiveJobSummary`. |
| `src/aiperf/operator/job_union.py` | Populate the three new fields from labels (live) or `sweep.json` marker (archived). |
| `src/aiperf/sweep_controller/k8s_executor.py` | Write per-child `sweep.json` marker before CR creation. |
| `src/aiperf/operator/results_server.py` | Wire `create_sweeps_router` into the FastAPI app. |
| `src/aiperf/operator/handlers/sweep/lifecycle.py` (or new `terminal.py`) | Write parent `aggregate.json` + `conditions.json` on terminal phase. |

### Frontend — new files

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/ui-v1/pages/sweeps.js` | `SweepsList` page. |
| `src/aiperf/operator/ui-v1/pages/sweep-detail.js` | `SweepDetail` page. |
| `src/aiperf/operator/ui-v1/components/cells-chart.js` | Chart.js wrapper for per-cell aggregates (1D + 2D-faceted). |
| `src/aiperf/operator/ui-v1/components/cells-table.js` | Cell-rows table with trial-spread sparklines. |

### Frontend — edited files

| Path | Change |
|---|---|
| `src/aiperf/operator/ui-v1/app.js` | Add `/sweeps` and `/sweeps/:ns/:name` route matches. |
| `src/aiperf/operator/ui-v1/components/top-nav.js` | Add "Sweeps" nav item. |
| `src/aiperf/operator/ui-v1/components/job-table.js` | Render `↳ sweep:<name>` back-link when row has `sweep_name`. |
| `src/aiperf/operator/ui-v1/pages/job-detail.js` | Render "Part of sweep <name>" header when present. |
| `src/aiperf/operator/ui-v1/lib/api.js` | Add `listSweeps`, `getSweep`, `getSweepCells`. |
| `src/aiperf/operator/ui-v1/lib/state.js` | Add `sweeps` signal. |

---

# PR 1 — Backend foundation (durable read API)

Independent of UI. Lands the dual-backed `/api/v1/sweeps*` surface plus the durability primitives. UI can ship in PR 2.

## Task 1: `resolve_sweep_dir` layout helper

**Files:**
- Modify: `src/aiperf/operator/results_layout.py`
- Test: `tests/unit/operator/test_results_layout.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to the existing test file:

```python
def test_resolve_sweep_dir_returns_path_when_present(tmp_path: Path) -> None:
    base = tmp_path
    sweep_dir = base / "bench" / "sweeps" / "saturation-sweep"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text("{}")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") == sweep_dir


def test_resolve_sweep_dir_returns_none_when_missing(tmp_path: Path) -> None:
    assert resolve_sweep_dir(tmp_path, "bench", "nope") is None


def test_resolve_sweep_dir_returns_none_when_not_a_directory(tmp_path: Path) -> None:
    base = tmp_path
    (base / "bench" / "sweeps").mkdir(parents=True)
    (base / "bench" / "sweeps" / "saturation-sweep").write_text("not a dir")
    assert resolve_sweep_dir(base, "bench", "saturation-sweep") is None
```

Add the import line at the top of the test file: `from aiperf.operator.results_layout import resolve_sweep_dir`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto -k resolve_sweep_dir`
Expected: ImportError or AttributeError on `resolve_sweep_dir`.

- [ ] **Step 3: Implement**

Add to `src/aiperf/operator/results_layout.py` after `resolve_run_dir`:

```python
def resolve_sweep_dir(base: Path, namespace: str, name: str) -> Path | None:
    """Return the persisted sweep directory ``<base>/<ns>/sweeps/<name>``, or None.

    The directory is the durable parent manifest location for AIPerfSweep CRs:
    sweep-controllers write ``aggregate.json`` + ``conditions.json`` here at
    terminal phase. The dual-backed sweep API uses this to render archived
    sweeps after the parent CR has been TTL-reaped.

    Example
    -------
    >>> resolve_sweep_dir(Path("/data"), "bench", "saturation-sweep")
    PosixPath('/data/bench/sweeps/saturation-sweep')
    """
    candidate = base / namespace / "sweeps" / name
    if not candidate.is_dir():
        return None
    return candidate
```

Add `"resolve_sweep_dir"` to the `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/operator/test_results_layout.py -n auto`
Expected: All tests pass.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
ruff check --fix src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git add src/aiperf/operator/results_layout.py tests/unit/operator/test_results_layout.py
git commit -s --no-verify -m "feat(operator): add resolve_sweep_dir layout helper

Sibling of resolve_run_dir; locates the persisted sweep directory at
<base>/<ns>/sweeps/<name>. Foundation for dual-backed AIPerfSweep API
(spec: 2026-04-26-aiperfsweep-ui-v1-native-support-design.md)."
```

---

## Task 2: kubernetes_asyncio sweep helpers

**Files:**
- Modify: `src/aiperf/kubernetes/client.py`
- Test: `tests/unit/kubernetes/test_client_sweeps.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import (
    find_aiperfsweep,
    get_raw_aiperfsweep,
    get_raw_aiperfsweep_status,
    list_aiperfsweeps,
)


@pytest.mark.asyncio
async def test_list_aiperfsweeps_all_namespaces() -> None:
    api = MagicMock()
    co = MagicMock()
    co.list_cluster_custom_object = AsyncMock(
        return_value={"items": [{"metadata": {"name": "s1"}}, {"metadata": {"name": "s2"}}]}
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        items = await list_aiperfsweeps(api, all_namespaces=True)
    assert len(items) == 2
    co.list_cluster_custom_object.assert_awaited_once_with(
        group="aiperf.nvidia.com", version="v1alpha1", plural="aiperfsweeps"
    )


@pytest.mark.asyncio
async def test_list_aiperfsweeps_namespaced() -> None:
    api = MagicMock()
    co = MagicMock()
    co.list_namespaced_custom_object = AsyncMock(return_value={"items": []})
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        items = await list_aiperfsweeps(api, namespace="bench")
    assert items == []
    co.list_namespaced_custom_object.assert_awaited_once_with(
        group="aiperf.nvidia.com",
        version="v1alpha1",
        namespace="bench",
        plural="aiperfsweeps",
    )


@pytest.mark.asyncio
async def test_find_aiperfsweep_returns_body() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(return_value={"metadata": {"name": "s1"}})
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await find_aiperfsweep(api, "bench", "s1")
    assert body == {"metadata": {"name": "s1"}}


@pytest.mark.asyncio
async def test_find_aiperfsweep_returns_none_on_404() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=404, reason="Not Found")
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await find_aiperfsweep(api, "bench", "nope")
    assert body is None


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_status_returns_status() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object_status = AsyncMock(
        return_value={"status": {"phase": "Running", "completedRuns": 4}}
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        st = await get_raw_aiperfsweep_status(api, "s1", "bench")
    assert st == {"phase": "Running", "completedRuns": 4}


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_status_returns_none_on_404() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=404, reason="Not Found")
    )
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        st = await get_raw_aiperfsweep_status(api, "s1", "bench")
    assert st is None


@pytest.mark.asyncio
async def test_get_raw_aiperfsweep_returns_body() -> None:
    api = MagicMock()
    co = MagicMock()
    co.get_namespaced_custom_object = AsyncMock(return_value={"spec": {}, "status": {}})
    with patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=co):
        body = await get_raw_aiperfsweep(api, "bench", "s1")
    assert body == {"spec": {}, "status": {}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/kubernetes/test_client_sweeps.py -n auto`
Expected: ImportError on the four new symbols.

- [ ] **Step 3: Implement helpers**

Append to `src/aiperf/kubernetes/client.py` (use the existing `_GROUP`, `_VERSION` constants if defined; otherwise use literals as the existing job helpers do — match local convention):

```python
async def list_aiperfsweeps(
    api: ApiClient,
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
) -> list[dict[str, Any]]:
    """List AIPerfSweep CRs.

    Args:
        api: The kubernetes_asyncio ApiClient.
        namespace: When set and ``all_namespaces=False``, list only this namespace.
        all_namespaces: When True, list cluster-wide (cluster-scoped permissions
            required).

    Returns:
        List of raw CR dicts; ``items`` array of the apiserver response.
    """
    co = client.CustomObjectsApi(api)
    if all_namespaces:
        resp = await co.list_cluster_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
        )
    else:
        if namespace is None:
            raise ValueError(
                "namespace must be provided when all_namespaces is False"
            )
        resp = await co.list_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
        )
    return list(resp.get("items", []))


async def find_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Fetch a single AIPerfSweep CR. Returns None on 404; raises on other errors."""
    co = client.CustomObjectsApi(api)
    try:
        return await co.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise


async def get_raw_aiperfsweep(
    api: ApiClient, namespace: str, name: str
) -> dict[str, Any] | None:
    """Alias of :func:`find_aiperfsweep` matching the AIPerfJob naming convention."""
    return await find_aiperfsweep(api, namespace, name)


async def get_raw_aiperfsweep_status(
    api: ApiClient, name: str, namespace: str
) -> dict[str, Any] | None:
    """Fetch ``status`` subresource of a single AIPerfSweep. Returns None on 404."""
    co = client.CustomObjectsApi(api)
    try:
        body = await co.get_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            name=name,
        )
    except ApiException as e:
        if (e.status or 0) == 404:
            return None
        raise
    status = body.get("status")
    return status if isinstance(status, dict) else None
```

If `client.py` does not already import `ApiException` and `client` (from `kubernetes_asyncio`), add them where the existing job helpers import them (look for `from kubernetes_asyncio import client` and `from kubernetes_asyncio.client.exceptions import ApiException`). Do NOT duplicate imports.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/kubernetes/test_client_sweeps.py -n auto`
Expected: 7 passed.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/kubernetes/client.py tests/unit/kubernetes/test_client_sweeps.py
ruff check --fix src/aiperf/kubernetes/client.py tests/unit/kubernetes/test_client_sweeps.py
git add src/aiperf/kubernetes/client.py tests/unit/kubernetes/test_client_sweeps.py
git commit -s --no-verify -m "feat(k8s-client): list/get/status helpers for AIPerfSweep CRs

Mirror the existing AIPerfJob helpers for AIPerfSweep. Foundation for
the operator results-server sweeps router."
```

---

## Task 3: Sweep response models

**Files:**
- Create: `src/aiperf/operator/routers/sweeps_models.py`
- Test: `tests/unit/operator/test_sweeps_models.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import ValidationError
import pytest

from aiperf.operator.routers.sweeps_models import (
    CellAggregatesResponse,
    CellEntry,
    DimensionInfo,
    SpecSummary,
    SweepDetailResponse,
    SweepListResponse,
    SweepSummary,
)


def test_sweep_summary_required_fields() -> None:
    s = SweepSummary(
        namespace="bench",
        name="saturation-sweep",
        source="live",
        phase="Running",
        total_variations=12,
        completed_runs=8,
        failed_runs=0,
        age_seconds=120,
        model="meta-llama/Llama-3-8B",
    )
    assert s.namespace == "bench"
    assert s.source == "live"


def test_sweep_summary_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        SweepSummary(
            namespace="bench",
            name="s",
            source="ghost",  # invalid
            phase="Running",
            total_variations=0,
            completed_runs=0,
            failed_runs=0,
            age_seconds=0,
            model=None,
        )


def test_dimension_info_values_preserved() -> None:
    d = DimensionInfo(name="concurrency", values=[8, 32, 128])
    assert d.values == [8, 32, 128]


def test_cell_entry_metrics_open_dict() -> None:
    cell = CellEntry(
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        values={"concurrency": 128, "rate": 50},
        trials_completed=3,
        trials_failed=0,
        metrics={"request_throughput": {"avg": 1234.5, "p99": 1500.0}},
        children=[{"namespace": "bench", "name": "ch-7-0", "trial_index": 0, "phase": "Succeeded"}],
    )
    assert cell.metrics["request_throughput"]["avg"] == 1234.5


def test_sweep_list_response_default_empty() -> None:
    assert SweepListResponse(sweeps=[]).sweeps == []


def test_sweep_detail_response_required() -> None:
    sd = SweepDetailResponse(
        sweep=SweepSummary(
            namespace="bench", name="s", source="archived",
            phase="Succeeded", total_variations=4, completed_runs=4,
            failed_runs=0, age_seconds=999, model="m",
        ),
        status={"phase": "Succeeded"},
        spec_summary=SpecSummary(
            sweep_type="grid",
            dimensions=[DimensionInfo(name="concurrency", values=[1, 2])],
            multi_run=None,
            convergence=None,
        ),
        children=[],
    )
    assert sd.spec_summary.sweep_type == "grid"


def test_cell_aggregates_response_source_literal() -> None:
    with pytest.raises(ValidationError):
        CellAggregatesResponse(dimensions=[], cells=[], source="oops")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_sweeps_models.py -n auto`
Expected: ImportError on the seven new symbols.

- [ ] **Step 3: Implement models**

Create `src/aiperf/operator/routers/sweeps_models.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the operator's AIPerfSweep router.

Schemas are deliberately a superset of the apiserver shapes; the router
synthesizes equivalent payloads for archived (PVC-only) sweeps so the
client never has to branch on ``source``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DimensionInfo(BaseModel):
    """One swept dimension and the values it takes across the sweep."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Dimension name (e.g. 'concurrency').")
    values: list[Any] = Field(
        description="Values the dimension takes across the sweep, in spec order."
    )


class SpecSummary(BaseModel):
    """Compact summary of the sweep's structural spec for the UI detail page."""

    model_config = ConfigDict(extra="forbid")
    sweep_type: Literal["grid", "scenarios"] = Field(
        description="Variation generator kind."
    )
    dimensions: list[DimensionInfo] = Field(
        description="Swept dimensions and their value lists."
    )
    multi_run: dict[str, Any] | None = Field(
        default=None,
        description="multiRun config snapshot (trials, cooldown, ...) or None.",
    )
    convergence: dict[str, Any] | None = Field(
        default=None,
        description="convergence config snapshot or None.",
    )


class SweepSummary(BaseModel):
    """One row in the /sweeps list response and embedded in detail."""

    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(description="CR namespace.")
    name: str = Field(description="CR name.")
    source: Literal["live", "archived", "both"] = Field(
        description="Origin of the record: live CR, archived PVC dir, or both."
    )
    phase: str = Field(description="Parent phase.")
    total_variations: int = Field(description="Total variations from spec/aggregate.")
    completed_runs: int = Field(description="Sum of children in terminal-success phase.")
    failed_runs: int = Field(description="Sum of children in terminal-failure phase.")
    age_seconds: int = Field(description="Seconds since CR/dir creation.")
    model: str | None = Field(
        default=None, description="Primary model name from template snapshot."
    )


class SweepListResponse(BaseModel):
    """Body of GET /api/v1/sweeps."""

    model_config = ConfigDict(extra="forbid")
    sweeps: list[SweepSummary] = Field(default_factory=list)


class ChildJobRef(BaseModel):
    """Pointer to a child AIPerfJob inside a cell's children list."""

    model_config = ConfigDict(extra="forbid")
    namespace: str
    name: str
    trial_index: int | None = None
    phase: str | None = None


class CellEntry(BaseModel):
    """One sweep cell (variation) with per-cell aggregates and child links."""

    model_config = ConfigDict(extra="forbid")
    variation_index: int = Field(description="Index from expand_sweep().")
    variation_label: str = Field(description="Human-readable variation label.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured dimension values for this cell.",
    )
    trials_completed: int = Field(default=0)
    trials_failed: int = Field(default=0)
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="metric_name -> stat_name -> value for this cell.",
    )
    children: list[ChildJobRef] = Field(default_factory=list)


class CellAggregatesResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/cells."""

    model_config = ConfigDict(extra="forbid")
    dimensions: list[DimensionInfo] = Field(default_factory=list)
    cells: list[CellEntry] = Field(default_factory=list)
    source: Literal["live", "archived", "both"] = Field(
        description="Origin of the cell data: live (synthesized from per-child summaries), "
        "archived (read from aggregate.json), or both."
    )


class SweepDetailResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}."""

    model_config = ConfigDict(extra="forbid")
    sweep: SweepSummary
    status: dict[str, Any] = Field(default_factory=dict)
    spec_summary: SpecSummary
    children: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ActiveJobSummary dicts (alias-keyed) for the sweep's children.",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_sweeps_models.py -n auto`
Expected: 7 passed.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_models.py
ruff check --fix src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_models.py
git add src/aiperf/operator/routers/sweeps_models.py tests/unit/operator/test_sweeps_models.py
git commit -s --no-verify -m "feat(operator): pydantic response models for sweeps router"
```

---

## Task 4: `sweep_union` — live + archived join

**Files:**
- Create: `src/aiperf/operator/sweep_union.py`
- Test: `tests/unit/operator/test_sweep_union.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _write_aggregate(base: Path, ns: str, name: str, body: dict) -> Path:
    d = base / ns / "sweeps" / name
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))
    return d


def _live_cr(ns: str, name: str, *, phase: str = "Running",
             total: int = 4, completed: int = 1, failed: int = 0,
             model: str = "m", creation: str = "2026-04-01T00:00:00Z") -> dict:
    return {
        "metadata": {"namespace": ns, "name": name, "creationTimestamp": creation},
        "spec": {
            "template": {"spec": {"models": [{"name": model}]}},
            "sweep": {"type": "grid", "axes": [{"name": "concurrency", "values": [1, 2, 4, 8]}]},
        },
        "status": {
            "phase": phase,
            "totalVariations": total,
            "completedRuns": completed,
            "failedRuns": failed,
        },
    }


@pytest.mark.asyncio
async def test_list_all_sweeps_live_only(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    api = MagicMock()
    with patch.object(sweep_union, "list_aiperfsweeps",
                      AsyncMock(return_value=[_live_cr("bench", "s1")])):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "live"
    assert r.phase == "Running"
    assert r.total_variations == 4
    assert r.aggregate_path is None


@pytest.mark.asyncio
async def test_list_all_sweeps_archived_only(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    _write_aggregate(tmp_path, "bench", "s1", {
        "phase": "Succeeded", "totalVariations": 4,
        "completedRuns": 4, "failedRuns": 0,
        "completedAt": "2026-04-25T01:00:00Z",
        "spec_snapshot": {
            "sweep_type": "grid",
            "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
        },
        "model": "m",
    })
    api = MagicMock()
    with patch("aiperf.operator.sweep_union.list_aiperfsweeps",
               AsyncMock(return_value=[])):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "archived"
    assert r.phase == "Succeeded"
    assert r.total_variations == 4
    assert r.aggregate_path is not None


@pytest.mark.asyncio
async def test_list_all_sweeps_both(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    _write_aggregate(tmp_path, "bench", "s1", {
        "phase": "Succeeded",
        "totalVariations": 4, "completedRuns": 4, "failedRuns": 0,
        "completedAt": "2026-04-25T01:00:00Z",
        "spec_snapshot": {
            "sweep_type": "grid",
            "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
        },
        "model": "m",
    })
    api = MagicMock()
    with patch("aiperf.operator.sweep_union.list_aiperfsweeps",
               AsyncMock(return_value=[_live_cr("bench", "s1", phase="Aggregating",
                                                completed=4, total=4)])):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "both"
    # Live phase wins on overlap.
    assert r.phase == "Aggregating"
    assert r.aggregate_path is not None


@pytest.mark.asyncio
async def test_find_any_sweep_archived_corrupt_aggregate(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    d = tmp_path / "bench" / "sweeps" / "s1"
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text("not json")
    api = MagicMock()
    with patch("aiperf.operator.sweep_union.find_aiperfsweep",
               AsyncMock(return_value=None)):
        rec = await sweep_union.find_any_sweep(api, tmp_path, "bench", "s1")
    # Corrupt aggregate still surfaces a record so the list page is not blank;
    # phase is Unknown to mark the broken state.
    assert rec is not None
    assert rec.phase == "Unknown"
    assert rec.source == "archived"


@pytest.mark.asyncio
async def test_find_any_sweep_neither_returns_none(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    api = MagicMock()
    with patch("aiperf.operator.sweep_union.find_aiperfsweep",
               AsyncMock(return_value=None)):
        rec = await sweep_union.find_any_sweep(api, tmp_path, "bench", "s1")
    assert rec is None


def test_synthesize_status_from_aggregate_terminal() -> None:
    from aiperf.operator.sweep_union import synthesize_sweep_status_from_aggregate

    out = synthesize_sweep_status_from_aggregate(
        "bench",
        "s1",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
        },
        conditions=[{"type": "Done", "status": "True"}],
    )
    assert out["phase"] == "Succeeded"
    assert out["totalVariations"] == 4
    assert out["completedRuns"] == 4
    assert out["completedAt"] == "2026-04-25T01:00:00Z"
    assert out["conditions"] == [{"type": "Done", "status": "True"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_sweep_union.py -n auto`
Expected: ImportError on `aiperf.operator.sweep_union`.

- [ ] **Step 3: Implement `sweep_union`**

Create `src/aiperf/operator/sweep_union.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified live + archived view over AIPerfSweep records.

Mirrors :mod:`aiperf.operator.job_union` for sweeps. Live state comes
from the apiserver; archived state comes from
``<results_dir>/<ns>/sweeps/<name>/aggregate.json`` which is written
by the sweep-controller at terminal phase. Records are joined by
``(namespace, name)`` and tagged ``source = "live" | "archived" | "both"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.kubernetes.client import find_aiperfsweep, list_aiperfsweeps
from aiperf.operator.results_layout import resolve_sweep_dir

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger("aiperf.operator.sweep_union")

_AGGREGATE_FILE = "aggregate.json"
_CONDITIONS_FILE = "conditions.json"


@dataclass
class SweepRecord:
    namespace: str
    name: str
    source: str  # Literal["live", "archived", "both"]
    phase: str
    total_variations: int
    completed_runs: int
    failed_runs: int
    age_seconds: int
    model: str | None
    aggregate_path: str | None = None
    raw_status: dict[str, Any] = field(default_factory=dict)
    raw_spec: dict[str, Any] = field(default_factory=dict)
    aggregate_doc: dict[str, Any] | None = None


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_creation_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _age_seconds(created: datetime | None) -> int:
    if created is None:
        return 0
    delta = (_now_utc() - created).total_seconds()
    return max(0, int(delta))


def _model_from_template(spec: dict[str, Any]) -> str | None:
    template = spec.get("template") or {}
    tspec = template.get("spec") or {}
    models = tspec.get("models") or []
    if not models:
        return None
    first = models[0]
    if isinstance(first, dict):
        return first.get("name")
    if isinstance(first, str):
        return first
    return None


def _record_from_live(cr: dict[str, Any]) -> SweepRecord:
    meta = cr.get("metadata") or {}
    spec = cr.get("spec") or {}
    status = cr.get("status") or {}
    created = _parse_creation_ts(meta.get("creationTimestamp"))
    return SweepRecord(
        namespace=meta.get("namespace") or "",
        name=meta.get("name") or "",
        source="live",
        phase=str(status.get("phase") or "Unknown"),
        total_variations=int(status.get("totalVariations") or 0),
        completed_runs=int(status.get("completedRuns") or 0),
        failed_runs=int(status.get("failedRuns") or 0),
        age_seconds=_age_seconds(created),
        model=_model_from_template(spec),
        aggregate_path=None,
        raw_status=status,
        raw_spec=spec,
        aggregate_doc=None,
    )


def _read_aggregate_doc(path: Path) -> dict[str, Any] | None:
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("aggregate.json unreadable at %s: %s", path, e)
        return None


def _record_from_archive(
    namespace: str, name: str, sweep_dir: Path
) -> SweepRecord | None:
    agg_path = sweep_dir / _AGGREGATE_FILE
    if not agg_path.is_file():
        return None
    doc = _read_aggregate_doc(agg_path)
    if doc is None:
        # Surface as Unknown so corrupt sweeps still appear and operators see them.
        try:
            mtime = datetime.fromtimestamp(agg_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            mtime = None
        return SweepRecord(
            namespace=namespace,
            name=name,
            source="archived",
            phase="Unknown",
            total_variations=0,
            completed_runs=0,
            failed_runs=0,
            age_seconds=_age_seconds(mtime),
            model=None,
            aggregate_path=str(agg_path),
            aggregate_doc=None,
        )
    completed_at = _parse_creation_ts(doc.get("completedAt"))
    return SweepRecord(
        namespace=namespace,
        name=name,
        source="archived",
        phase=str(doc.get("phase") or "Archived"),
        total_variations=int(doc.get("totalVariations") or 0),
        completed_runs=int(doc.get("completedRuns") or 0),
        failed_runs=int(doc.get("failedRuns") or 0),
        age_seconds=_age_seconds(completed_at),
        model=doc.get("model"),
        aggregate_path=str(agg_path),
        aggregate_doc=doc,
    )


def _scan_archived(base_dir: Path, namespace: str | None = None) -> list[SweepRecord]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    out: list[SweepRecord] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        sweeps_root = ns_dir / "sweeps"
        if not sweeps_root.is_dir():
            continue
        for sweep_dir in sorted(sweeps_root.iterdir()):
            if not sweep_dir.is_dir():
                continue
            rec = _record_from_archive(ns_dir.name, sweep_dir.name, sweep_dir)
            if rec is not None:
                out.append(rec)
    return out


def _merge(live: list[SweepRecord], archived: list[SweepRecord]) -> list[SweepRecord]:
    by_key: dict[tuple[str, str], SweepRecord] = {}
    for rec in archived:
        by_key[(rec.namespace, rec.name)] = rec
    for live_rec in live:
        key = (live_rec.namespace, live_rec.name)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = live_rec
            continue
        # Both sources present: live wins on live fields, archived backfills.
        merged = SweepRecord(
            namespace=live_rec.namespace,
            name=live_rec.name,
            source="both",
            phase=live_rec.phase or existing.phase,
            total_variations=live_rec.total_variations or existing.total_variations,
            completed_runs=live_rec.completed_runs or existing.completed_runs,
            failed_runs=live_rec.failed_runs or existing.failed_runs,
            age_seconds=live_rec.age_seconds or existing.age_seconds,
            model=live_rec.model or existing.model,
            aggregate_path=existing.aggregate_path,
            raw_status=live_rec.raw_status,
            raw_spec=live_rec.raw_spec,
            aggregate_doc=existing.aggregate_doc,
        )
        by_key[key] = merged
    return sorted(by_key.values(), key=lambda r: (r.namespace, r.name))


async def list_all_sweeps(
    api: ApiClient,
    base_dir: Path,
    *,
    namespace: str | None = None,
    all_namespaces: bool = True,
) -> list[SweepRecord]:
    """Return the joined live + archived sweep view, source-tagged."""
    try:
        live_crs = await list_aiperfsweeps(
            api, namespace=namespace, all_namespaces=all_namespaces
        )
    except Exception as e:  # noqa: BLE001 — list endpoint is best-effort like jobs
        logger.warning("list_aiperfsweeps failed; live half empty: %s", e)
        live_crs = []
    live = [_record_from_live(cr) for cr in live_crs]
    archived = _scan_archived(base_dir, namespace=namespace)
    return _merge(live, archived)


async def find_any_sweep(
    api: ApiClient, base_dir: Path, namespace: str, name: str
) -> SweepRecord | None:
    """Resolve a single sweep across live and archived state. Returns None if neither."""
    cr = await find_aiperfsweep(api, namespace, name)
    archive_dir = resolve_sweep_dir(base_dir, namespace, name)
    archived = (
        _record_from_archive(namespace, name, archive_dir)
        if archive_dir is not None
        else None
    )
    if cr is None and archived is None:
        return None
    if cr is None:
        return archived
    live = _record_from_live(cr)
    if archived is None:
        return live
    return _merge([live], [archived])[0]


def synthesize_sweep_status_from_aggregate(
    namespace: str,
    name: str,
    aggregate: dict[str, Any],
    conditions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a status-shaped dict from an archived sweep's aggregate.json.

    Returns a dict the UI consumes the same way as a live ``.status`` subresource.
    """
    return {
        "phase": str(aggregate.get("phase") or "Archived"),
        "totalVariations": int(aggregate.get("totalVariations") or 0),
        "completedRuns": int(aggregate.get("completedRuns") or 0),
        "failedRuns": int(aggregate.get("failedRuns") or 0),
        "maxTotalRuns": int(aggregate.get("maxTotalRuns") or 0),
        "completedAt": aggregate.get("completedAt"),
        "conditions": conditions or [],
        "aggregateRef": aggregate.get("aggregateRef"),
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_sweep_union.py -n auto`
Expected: 6 passed.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union.py
ruff check --fix src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union.py
git add src/aiperf/operator/sweep_union.py tests/unit/operator/test_sweep_union.py
git commit -s --no-verify -m "feat(operator): sweep_union — live + archived join for AIPerfSweep

Mirrors job_union for sweeps. Live state via list_aiperfsweeps;
archived state via aggregate.json on the results PVC. Source-tagged
records ('live' | 'archived' | 'both'). Foundation for the dual-backed
sweeps router."
```

---

## Task 5: Sweeps router — list and detail endpoints

**Files:**
- Create: `src/aiperf/operator/routers/sweeps.py`
- Test: `tests/unit/operator/test_sweeps_router.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.sweeps import create_sweeps_router
from aiperf.operator.sweep_union import SweepRecord


def _client_with(api: object | None, base_dir: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_sweeps_router(holder, base_dir))
    return TestClient(app)


def _live_record(name: str = "s1") -> SweepRecord:
    return SweepRecord(
        namespace="bench", name=name, source="live", phase="Running",
        total_variations=4, completed_runs=1, failed_runs=0, age_seconds=10,
        model="m",
        raw_spec={
            "template": {"spec": {"models": [{"name": "m"}]}},
            "sweep": {"type": "grid",
                      "axes": [{"name": "concurrency", "values": [1, 2, 4, 8]}]},
        },
        raw_status={"phase": "Running", "totalVariations": 4, "completedRuns": 1, "failedRuns": 0},
    )


def test_list_returns_503_when_api_missing(tmp_path: Path) -> None:
    c = _client_with(None, tmp_path)
    r = c.get("/api/v1/sweeps")
    assert r.status_code == 503


def test_list_returns_records(tmp_path: Path) -> None:
    api = MagicMock()
    with patch("aiperf.operator.routers.sweeps.list_all_sweeps",
               AsyncMock(return_value=[_live_record()])):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sweeps"]) == 1
    assert body["sweeps"][0]["name"] == "s1"
    assert body["sweeps"][0]["source"] == "live"


def test_detail_404_when_missing(tmp_path: Path) -> None:
    api = MagicMock()
    with patch("aiperf.operator.routers.sweeps.find_any_sweep",
               AsyncMock(return_value=None)):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/nope")
    assert r.status_code == 404


def test_detail_returns_spec_summary_from_live(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    with (
        patch("aiperf.operator.routers.sweeps.find_any_sweep",
              AsyncMock(return_value=rec)),
        patch("aiperf.operator.routers.sweeps.list_all_jobs",
              AsyncMock(return_value=[])),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["name"] == "s1"
    assert body["spec_summary"]["sweep_type"] == "grid"
    dim_names = [d["name"] for d in body["spec_summary"]["dimensions"]]
    assert "concurrency" in dim_names


def test_detail_archived_uses_synthesized_status(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(json.dumps({
        "phase": "Succeeded",
        "totalVariations": 4,
        "completedRuns": 4,
        "failedRuns": 0,
        "completedAt": "2026-04-25T01:00:00Z",
        "spec_snapshot": {
            "sweep_type": "grid",
            "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
        },
        "model": "m",
    }))
    api = MagicMock()
    rec = SweepRecord(
        namespace="bench", name="s1", source="archived", phase="Succeeded",
        total_variations=4, completed_runs=4, failed_runs=0, age_seconds=999,
        model="m", aggregate_path=str(sweep_dir / "aggregate.json"),
        aggregate_doc={
            "phase": "Succeeded", "totalVariations": 4,
            "completedRuns": 4, "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "spec_snapshot": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
            "model": "m",
        },
    )
    with (
        patch("aiperf.operator.routers.sweeps.find_any_sweep",
              AsyncMock(return_value=rec)),
        patch("aiperf.operator.routers.sweeps.list_all_jobs",
              AsyncMock(return_value=[])),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["source"] == "archived"
    assert body["status"]["phase"] == "Succeeded"
    assert body["status"]["completedAt"] == "2026-04-25T01:00:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_sweeps_router.py -n auto`
Expected: ImportError on `create_sweeps_router`.

- [ ] **Step 3: Implement router (list + detail only)**

Create `src/aiperf/operator/routers/sweeps.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for /api/v1/sweeps* — read-only AIPerfSweep view.

Dual-backed via :mod:`aiperf.operator.sweep_union`: every endpoint
returns the same shape regardless of whether the parent CR exists or
the data is reconstructed from the archived ``aggregate.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException
from kubernetes_asyncio.client import ApiClient

from aiperf.operator.job_union import list_all_jobs
from aiperf.operator.routers.sweeps_models import (
    DimensionInfo,
    SpecSummary,
    SweepDetailResponse,
    SweepListResponse,
    SweepSummary,
)
from aiperf.operator.sweep_union import (
    SweepRecord,
    find_any_sweep,
    list_all_sweeps,
    synthesize_sweep_status_from_aggregate,
)

logger = logging.getLogger("aiperf.operator.ui")


def _summary(rec: SweepRecord) -> SweepSummary:
    return SweepSummary(
        namespace=rec.namespace,
        name=rec.name,
        source=rec.source,  # type: ignore[arg-type]
        phase=rec.phase,
        total_variations=rec.total_variations,
        completed_runs=rec.completed_runs,
        failed_runs=rec.failed_runs,
        age_seconds=rec.age_seconds,
        model=rec.model,
    )


def _dimensions_from_live_spec(spec: dict[str, Any]) -> list[DimensionInfo]:
    sweep = spec.get("sweep") or {}
    axes = sweep.get("axes") or sweep.get("dimensions") or []
    out: list[DimensionInfo] = []
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        nm = axis.get("name")
        vals = axis.get("values") or []
        if isinstance(nm, str):
            out.append(DimensionInfo(name=nm, values=list(vals)))
    return out


def _spec_summary_from_record(rec: SweepRecord) -> SpecSummary:
    """Build a SpecSummary from whichever side of the union is available."""
    if rec.raw_spec:
        sweep = rec.raw_spec.get("sweep") or {}
        return SpecSummary(
            sweep_type=str(sweep.get("type") or "grid"),  # type: ignore[arg-type]
            dimensions=_dimensions_from_live_spec(rec.raw_spec),
            multi_run=rec.raw_spec.get("multiRun"),
            convergence=rec.raw_spec.get("convergence"),
        )
    if rec.aggregate_doc is not None:
        snap = rec.aggregate_doc.get("spec_snapshot") or {}
        dims_raw = snap.get("dimensions") or []
        dims = [
            DimensionInfo(name=d["name"], values=list(d.get("values") or []))
            for d in dims_raw
            if isinstance(d, dict) and isinstance(d.get("name"), str)
        ]
        return SpecSummary(
            sweep_type=str(snap.get("sweep_type") or "grid"),  # type: ignore[arg-type]
            dimensions=dims,
            multi_run=snap.get("multi_run"),
            convergence=snap.get("convergence"),
        )
    return SpecSummary(sweep_type="grid", dimensions=[], multi_run=None, convergence=None)


def _read_conditions(sweep_dir_path: str | None) -> list[dict[str, Any]]:
    if not sweep_dir_path:
        return []
    p = Path(sweep_dir_path).parent / "conditions.json"
    if not p.is_file():
        return []
    try:
        raw = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("conditions.json unreadable at %s: %s", p, e)
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("conditions"), list):
        return raw["conditions"]
    return []


async def _list_sweeps_impl(api: ApiClient, base_dir: Path) -> SweepListResponse:
    records = await list_all_sweeps(api, base_dir, all_namespaces=True)
    return SweepListResponse(sweeps=[_summary(r) for r in records])


async def _get_sweep_impl(
    api: ApiClient, base_dir: Path, namespace: str, name: str
) -> SweepDetailResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")

    if rec.source == "archived" and rec.aggregate_doc is not None:
        status = synthesize_sweep_status_from_aggregate(
            namespace, name, rec.aggregate_doc, _read_conditions(rec.aggregate_path)
        )
    elif rec.source == "archived":
        status = {"phase": "Unknown", "conditions": []}
    else:
        status = rec.raw_status or {}

    spec_summary = _spec_summary_from_record(rec)

    children_records = await list_all_jobs(api, base_dir, all_namespaces=False)
    children = [
        j.model_dump(by_alias=True)
        for j in children_records
        if getattr(j, "sweep_name", None) == name and j.namespace == namespace
    ]

    return SweepDetailResponse(
        sweep=_summary(rec),
        status=status,
        spec_summary=spec_summary,
        children=children,
    )


def create_sweeps_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
) -> APIRouter:
    """Build the sweeps router. Mirrors :func:`create_jobs_router`'s shape."""
    _holder = api_holder if api_holder is not None else [None]
    _base_dir = results_dir if results_dir is not None else Path("/data")
    router = APIRouter(prefix="/api/v1", tags=["sweeps"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(
                503,
                "Kubernetes API client not yet initialized by FastAPI lifespan; "
                "retry in a few seconds or check /healthz",
            )
        return api

    @router.get("/sweeps", response_model=SweepListResponse)
    async def list_sweeps() -> SweepListResponse:
        return await _list_sweeps_impl(_require_api(), _base_dir)

    @router.get("/sweeps/{namespace}/{name}", response_model=SweepDetailResponse)
    async def get_sweep(namespace: str, name: str) -> SweepDetailResponse:
        return await _get_sweep_impl(_require_api(), _base_dir, namespace, name)

    return router
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_sweeps_router.py -n auto`
Expected: 5 passed (some require Task 7 for full children behavior; the 5 in the test file as written above all pass with the empty children list).

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
ruff check --fix src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
git add src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
git commit -s --no-verify -m "feat(operator): sweeps router — list + detail (dual-backed)

Adds GET /api/v1/sweeps and GET /api/v1/sweeps/{ns}/{name}. Both
endpoints serve from the live CR + archived PVC union; archived
detail synthesizes a status-shaped payload from aggregate.json so
the UI does not have to branch on source."
```

---

## Task 6: Sweeps router — `/cells` endpoint

**Files:**
- Modify: `src/aiperf/operator/routers/sweeps.py`
- Modify: `tests/unit/operator/test_sweeps_router.py`

- [ ] **Step 1: Write the failing test (append)**

```python
def test_cells_archived_reads_per_cell_aggregates(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(json.dumps({
        "phase": "Succeeded",
        "totalVariations": 2,
        "completedRuns": 4,
        "failedRuns": 0,
        "completedAt": "2026-04-25T01:00:00Z",
        "spec_snapshot": {
            "sweep_type": "grid",
            "dimensions": [{"name": "concurrency", "values": [8, 32]}],
        },
        "per_cell_aggregates": [
            {
                "variation_index": 0,
                "variation_label": "concurrency-8",
                "values": {"concurrency": 8},
                "trials_completed": 2,
                "trials_failed": 0,
                "metrics": {"request_throughput": {"avg": 100.0, "p99": 110.0}},
                "children": [
                    {"namespace": "bench", "name": "ch-0-0", "trial_index": 0, "phase": "Succeeded"},
                    {"namespace": "bench", "name": "ch-0-1", "trial_index": 1, "phase": "Succeeded"},
                ],
            },
            {
                "variation_index": 1,
                "variation_label": "concurrency-32",
                "values": {"concurrency": 32},
                "trials_completed": 2,
                "trials_failed": 0,
                "metrics": {"request_throughput": {"avg": 280.0, "p99": 300.0}},
                "children": [
                    {"namespace": "bench", "name": "ch-1-0", "trial_index": 0, "phase": "Succeeded"},
                    {"namespace": "bench", "name": "ch-1-1", "trial_index": 1, "phase": "Succeeded"},
                ],
            },
        ],
    }))
    api = MagicMock()
    rec = SweepRecord(
        namespace="bench", name="s1", source="archived", phase="Succeeded",
        total_variations=2, completed_runs=4, failed_runs=0, age_seconds=999,
        model="m", aggregate_path=str(sweep_dir / "aggregate.json"),
        aggregate_doc=json.loads((sweep_dir / "aggregate.json").read_text()),
    )
    with patch("aiperf.operator.routers.sweeps.find_any_sweep",
               AsyncMock(return_value=rec)):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "archived"
    assert len(body["cells"]) == 2
    assert body["cells"][0]["metrics"]["request_throughput"]["avg"] == 100.0
    assert body["cells"][1]["values"]["concurrency"] == 32


def test_cells_live_no_aggregate_returns_empty_with_dimensions(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    with (
        patch("aiperf.operator.routers.sweeps.find_any_sweep",
              AsyncMock(return_value=rec)),
        patch("aiperf.operator.routers.sweeps._cells_from_live_children",
              AsyncMock(return_value=[])),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "live"
    assert body["cells"] == []
    assert [d["name"] for d in body["dimensions"]] == ["concurrency"]


def test_cells_404_when_neither_present(tmp_path: Path) -> None:
    api = MagicMock()
    with patch("aiperf.operator.routers.sweeps.find_any_sweep",
               AsyncMock(return_value=None)):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/nope/cells")
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_sweeps_router.py -n auto -k cells`
Expected: 404 on `/cells` because the route does not exist.

- [ ] **Step 3: Implement `/cells`**

Append to `src/aiperf/operator/routers/sweeps.py`:

```python
# Add the import at the top alongside the existing model imports:
from aiperf.operator.routers.sweeps_models import (
    CellAggregatesResponse,
    CellEntry,
    ChildJobRef,
    DimensionInfo,
    SpecSummary,
    SweepDetailResponse,
    SweepListResponse,
    SweepSummary,
)


def _cells_from_aggregate(doc: dict[str, Any]) -> list[CellEntry]:
    raw_cells = doc.get("per_cell_aggregates") or []
    out: list[CellEntry] = []
    for c in raw_cells:
        if not isinstance(c, dict):
            continue
        children_raw = c.get("children") or []
        children = [
            ChildJobRef(
                namespace=child.get("namespace") or "",
                name=child.get("name") or "",
                trial_index=child.get("trial_index"),
                phase=child.get("phase"),
            )
            for child in children_raw
            if isinstance(child, dict)
        ]
        out.append(
            CellEntry(
                variation_index=int(c.get("variation_index") or 0),
                variation_label=str(c.get("variation_label") or ""),
                values=dict(c.get("values") or {}),
                trials_completed=int(c.get("trials_completed") or 0),
                trials_failed=int(c.get("trials_failed") or 0),
                metrics=dict(c.get("metrics") or {}),
                children=children,
            )
        )
    return sorted(out, key=lambda x: x.variation_index)


async def _cells_from_live_children(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
) -> list[CellEntry]:
    """Compute per-cell aggregates by grouping children by variation_index.

    Used when the sweep is live and has no aggregate.json yet (mid-run).
    Reads each child's profile_export_aiperf.json from the PVC if present.
    Returns an empty list if no terminal children are persisted yet.
    """
    children_records = await list_all_jobs(api, base_dir, all_namespaces=False)
    matched = [
        j for j in children_records
        if getattr(j, "sweep_name", None) == sweep_name
        and j.namespace == namespace
    ]
    by_cell: dict[int, dict[str, Any]] = {}
    for j in matched:
        idx = getattr(j, "variation_index", None)
        if idx is None:
            continue
        bucket = by_cell.setdefault(int(idx), {
            "variation_label": getattr(j, "variation_label", "") or "",
            "trials_completed": 0,
            "trials_failed": 0,
            "throughputs": [],
            "p99_latencies": [],
            "children": [],
        })
        # Status mapping: only count terminal children towards aggregates.
        phase = (j.phase or "").lower()
        if phase in {"succeeded", "completed"}:
            bucket["trials_completed"] += 1
            if j.throughput_rps is not None:
                bucket["throughputs"].append(float(j.throughput_rps))
            if j.latency_p99_ms is not None:
                bucket["p99_latencies"].append(float(j.latency_p99_ms))
        elif phase in {"failed", "cancelled", "partiallyfailed"}:
            bucket["trials_failed"] += 1
        bucket["children"].append(
            ChildJobRef(
                namespace=j.namespace,
                name=j.name,
                trial_index=None,
                phase=j.phase,
            )
        )

    def _avg(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    out: list[CellEntry] = []
    for idx, b in sorted(by_cell.items()):
        metrics: dict[str, dict[str, float]] = {}
        thr_avg = _avg(b["throughputs"])
        if thr_avg is not None:
            metrics["request_throughput"] = {"avg": thr_avg}
        lat_avg = _avg(b["p99_latencies"])
        if lat_avg is not None:
            metrics["request_latency_p99"] = {"avg": lat_avg}
        out.append(
            CellEntry(
                variation_index=idx,
                variation_label=b["variation_label"],
                values={},  # structured values come from spec; live path leaves empty
                trials_completed=b["trials_completed"],
                trials_failed=b["trials_failed"],
                metrics=metrics,
                children=b["children"],
            )
        )
    return out


async def _get_cells_impl(
    api: ApiClient, base_dir: Path, namespace: str, name: str
) -> CellAggregatesResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")
    spec_summary = _spec_summary_from_record(rec)
    if rec.aggregate_doc is not None:
        cells = _cells_from_aggregate(rec.aggregate_doc)
        source = rec.source
    else:
        cells = await _cells_from_live_children(api, base_dir, namespace, name)
        source = "live"
    return CellAggregatesResponse(
        dimensions=spec_summary.dimensions,
        cells=cells,
        source=source,  # type: ignore[arg-type]
    )
```

Then register the route inside `create_sweeps_router` after the detail route:

```python
    @router.get(
        "/sweeps/{namespace}/{name}/cells",
        response_model=CellAggregatesResponse,
    )
    async def get_sweep_cells(namespace: str, name: str) -> CellAggregatesResponse:
        return await _get_cells_impl(_require_api(), _base_dir, namespace, name)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/operator/test_sweeps_router.py -n auto`
Expected: 8 passed total.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
ruff check --fix src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
git add src/aiperf/operator/routers/sweeps.py tests/unit/operator/test_sweeps_router.py
git commit -s --no-verify -m "feat(operator): sweeps router — /cells endpoint

Reads per_cell_aggregates[] from aggregate.json when available;
falls back to grouping live children by variation_index when the
sweep is mid-run."
```

---

## Task 7: Add sweep-link fields to `ActiveJobSummary`

**Files:**
- Modify: `src/aiperf/operator/routers/jobs_models.py`
- Test: `tests/unit/operator/test_models.py` (extend, or create a focused file under `tests/unit/operator/test_jobs_models.py` if no `ActiveJobSummary` test exists yet)

- [ ] **Step 1: Locate existing ActiveJobSummary**

Run: `grep -n "class ActiveJobSummary" src/aiperf/operator/routers/jobs_models.py` and read the surrounding fields. Identify the existing `ConfigDict(populate_by_name=True, alias_generator=to_camel)` (or equivalent) — if not present, copy the alias style from a sibling field.

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/operator/test_jobs_models.py` (create if absent):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.operator.routers.jobs_models import ActiveJobSummary


def _minimum_fields() -> dict:
    """Return the minimum required-field kwargs for ActiveJobSummary.

    Update this dict if upstream adds new required fields; the tests below
    do not care about anything except the three new optional sweep fields.
    """
    return {
        "namespace": "bench",
        "name": "ch-0-0",
        "phase": "Succeeded",
        "source": "live",
    }


def test_active_job_summary_sweep_fields_default_none() -> None:
    s = ActiveJobSummary(**_minimum_fields())
    assert s.sweep_name is None
    assert s.variation_index is None
    assert s.variation_label is None


def test_active_job_summary_sweep_fields_round_trip_via_alias() -> None:
    s = ActiveJobSummary(
        **_minimum_fields(),
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
    )
    payload = s.model_dump(by_alias=True)
    assert payload["sweepName"] == "saturation-sweep"
    assert payload["variationIndex"] == 7
    assert payload["variationLabel"] == "concurrency-128-rate-50"
```

If `ActiveJobSummary` requires more fields than `_minimum_fields()` provides, extend the dict to include them. Do NOT remove any field from the model; only ADD the three new optional ones.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_jobs_models.py -n auto -k sweep`
Expected: AttributeError on `sweep_name`.

- [ ] **Step 4: Add the three optional fields**

Edit `src/aiperf/operator/routers/jobs_models.py` — add to the `ActiveJobSummary` class (preserve alias generator if one exists; otherwise add explicit `Field(..., alias="...")`):

```python
    sweep_name: str | None = Field(
        default=None,
        description="Parent AIPerfSweep name when this job is a sweep child.",
    )
    variation_index: int | None = Field(
        default=None,
        description="Variation index from expand_sweep() for sweep children.",
    )
    variation_label: str | None = Field(
        default=None,
        description="Human-readable variation label for sweep children.",
    )
```

If the model uses an `alias_generator=to_camel`, the camelCase aliases come for free. If it uses explicit `alias="..."` per field, set them: `sweepName`, `variationIndex`, `variationLabel`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/operator/test_jobs_models.py -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_models.py
ruff check --fix src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_models.py
git add src/aiperf/operator/routers/jobs_models.py tests/unit/operator/test_jobs_models.py
git commit -s --no-verify -m "feat(operator): ActiveJobSummary gains optional sweep linkage fields

Adds sweep_name, variation_index, variation_label (all default-None)
so /jobs and /jobs/{ns}/{name} can render the back-link to the
parent AIPerfSweep on both live and archived children."
```

---

## Task 8: `job_union` populates sweep linkage from labels + sweep.json marker

**Files:**
- Modify: `src/aiperf/operator/job_union.py`
- Test: `tests/unit/operator/test_job_union.py` (extend existing — or create a focused new test file `tests/unit/operator/test_job_union_sweep_linkage.py`)

- [ ] **Step 1: Identify the AIPerfJobInfo / ActiveJobSummary boundary**

`job_union` uses `aiperf.kubernetes.models.AIPerfJobInfo` internally. Confirm whether `ActiveJobSummary` IS `AIPerfJobInfo` (alias) or wraps it. If they are different types, the three new fields must be added to BOTH (Task 7 added them to the response model; this task may also need to add them to `AIPerfJobInfo` if that's what `_archived_from_summary` returns).

Run: `grep -n "class AIPerfJobInfo\|class ActiveJobSummary" src/aiperf/kubernetes/models.py src/aiperf/operator/routers/jobs_models.py`

If they are different classes: add the same three fields to `AIPerfJobInfo` first (small extension to Task 7's edit), with the same default-None semantics. If they are the same class (re-exported alias), skip this and proceed.

- [ ] **Step 2: Write the failing test (live label path)**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_list_all_jobs_live_child_carries_sweep_labels(tmp_path: Path) -> None:
    from aiperf.operator import job_union

    cr = {
        "metadata": {
            "namespace": "bench",
            "name": "ch-0-0",
            "labels": {
                "aiperf.nvidia.com/sweep": "saturation-sweep",
                "aiperf.nvidia.com/variation-index": "0007",
                "aiperf.nvidia.com/variation-label": "concurrency-128-rate-50",
            },
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"models": [{"name": "m"}]},
    }
    with patch.object(job_union, "list_aiperf_jobs",
                      AsyncMock(return_value=[cr])):
        results = await job_union.list_all_jobs(api=object(), base_dir=tmp_path,
                                                all_namespaces=True)
    matches = [r for r in results if r.name == "ch-0-0"]
    assert len(matches) == 1
    j = matches[0]
    assert j.sweep_name == "saturation-sweep"
    assert j.variation_index == 7
    assert j.variation_label == "concurrency-128-rate-50"


@pytest.mark.asyncio
async def test_list_all_jobs_archived_child_reads_sweep_marker(tmp_path: Path) -> None:
    from aiperf.operator import job_union

    job_dir = tmp_path / "bench" / "ch-0-0"
    job_dir.mkdir(parents=True)
    (job_dir / "profile_export_aiperf.json").write_text(json.dumps({
        "status": "Succeeded",
        "input_config": {
            "models": {"items": [{"name": "m"}]},
            "endpoint": {"urls": ["http://x"]},
        },
        "request_throughput": {"avg": 100.0},
        "request_latency": {"p99": 5.0},
    }))
    (job_dir / "sweep.json").write_text(json.dumps({
        "sweep_name": "saturation-sweep",
        "variation_index": 7,
        "variation_label": "concurrency-128-rate-50",
        "trial_index": 0,
    }))
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[])):
        results = await job_union.list_all_jobs(api=object(), base_dir=tmp_path,
                                                all_namespaces=True)
    matches = [r for r in results if r.name == "ch-0-0"]
    assert len(matches) == 1
    j = matches[0]
    assert j.sweep_name == "saturation-sweep"
    assert j.variation_index == 7
    assert j.variation_label == "concurrency-128-rate-50"


@pytest.mark.asyncio
async def test_list_all_jobs_no_sweep_linkage_returns_none(tmp_path: Path) -> None:
    from aiperf.operator import job_union

    cr = {
        "metadata": {
            "namespace": "bench",
            "name": "one-shot",
            "labels": {},
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"models": [{"name": "m"}]},
    }
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[cr])):
        results = await job_union.list_all_jobs(api=object(), base_dir=tmp_path,
                                                all_namespaces=True)
    matches = [r for r in results if r.name == "one-shot"]
    assert matches[0].sweep_name is None
    assert matches[0].variation_index is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/operator/test_job_union_sweep_linkage.py -n auto`
Expected: AttributeError on `sweep_name`.

- [ ] **Step 4: Implement**

In `src/aiperf/operator/job_union.py`:

a) Add module-level constants at the top (after existing constants):

```python
_SWEEP_MARKER_FILE = "sweep.json"
_SWEEP_LABEL = "aiperf.nvidia.com/sweep"
_VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
_VARIATION_LABEL_LABEL = "aiperf.nvidia.com/variation-label"
```

b) Add a helper:

```python
def _sweep_linkage_from_labels(labels: dict[str, str]) -> tuple[str | None, int | None, str | None]:
    sweep_name = labels.get(_SWEEP_LABEL) or None
    raw_idx = labels.get(_VARIATION_INDEX_LABEL)
    try:
        variation_index = int(raw_idx) if raw_idx is not None else None
    except ValueError:
        variation_index = None
    variation_label = labels.get(_VARIATION_LABEL_LABEL) or None
    return sweep_name, variation_index, variation_label


def _sweep_linkage_from_marker(job_dir: Path) -> tuple[str | None, int | None, str | None]:
    marker = job_dir / _SWEEP_MARKER_FILE
    if not marker.is_file():
        return None, None, None
    try:
        doc = orjson.loads(marker.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("sweep.json unreadable at %s: %s", marker, e)
        return None, None, None
    return (
        doc.get("sweep_name") or None,
        doc.get("variation_index"),
        doc.get("variation_label") or None,
    )
```

c) In whichever helper builds an `AIPerfJobInfo` from a CR (look for the existing `_from_cr` / `_active_from_cr` helper — there must be one because `list_aiperf_jobs` returns CRs that are mapped to summaries), pull labels and call `_sweep_linkage_from_labels`. Set the three fields on the resulting object.

d) In `_archived_from_summary` (or whichever helper builds an archived `AIPerfJobInfo`), the caller knows the `name_dir`. Wherever the archived helper is invoked (search for `_archived_from_summary(`), pass `name_dir` so the helper can call `_sweep_linkage_from_marker(name_dir)` and set the three fields. Update the helper signature accordingly.

If `_archived_from_summary` is invoked from multiple sites, update each call site to pass `name_dir`.

e) Make sure overlap-merge logic (where live + archived are joined per-`(ns, name)`) preserves the linkage fields — live values win on overlap; archived values are the fallback.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/operator/test_job_union_sweep_linkage.py -n auto`
Expected: 3 passed.

- [ ] **Step 6: Run the full job_union test file**

Run: `uv run pytest tests/unit/operator/test_job_union.py tests/unit/operator/test_job_union_sweep_linkage.py -n auto`
Expected: all pass — no regressions in existing job_union tests.

- [ ] **Step 7: Format + commit**

```bash
ruff format src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_sweep_linkage.py
ruff check --fix src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_sweep_linkage.py
git add src/aiperf/operator/job_union.py tests/unit/operator/test_job_union_sweep_linkage.py
# If AIPerfJobInfo was extended, add it too:
git add src/aiperf/kubernetes/models.py
git commit -s --no-verify -m "feat(operator): job_union populates sweep linkage from labels and marker

Live children: read aiperf.nvidia.com/{sweep,variation-index,variation-label}.
Archived children: read sweep.json marker dropped by the sweep-controller.
Children with no linkage keep all three fields as None."
```

---

## Task 9: Sweep-controller writes per-child `sweep.json` marker

**Files:**
- Modify: `src/aiperf/sweep_controller/k8s_executor.py`
- Test: `tests/unit/sweep_controller/test_k8s_executor.py` (extend) — confirm with `grep -rn "test_k8s_executor\|class.*K8s.*Executor" tests/`. If no test file exists, create `tests/unit/sweep_controller/test_k8s_executor_marker.py`.

- [ ] **Step 1: Read the executor code**

Read `src/aiperf/sweep_controller/k8s_executor.py` lines around the child-CR-creation site (search for `create_namespaced_custom_object` or similar). Note where the child-name and labels are computed.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/sweep_controller/test_k8s_executor_marker.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker


def test_write_child_sweep_marker_creates_file(tmp_path: Path) -> None:
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007-04",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=4,
    )
    p = tmp_path / "bench" / "ch-0007-04" / "sweep.json"
    assert p.is_file()
    doc = json.loads(p.read_text())
    assert doc == {
        "sweep_name": "saturation-sweep",
        "variation_index": 7,
        "variation_label": "concurrency-128-rate-50",
        "trial_index": 4,
    }


def test_write_child_sweep_marker_is_atomic_overwrite(tmp_path: Path) -> None:
    p = tmp_path / "bench" / "ch-0007-04" / "sweep.json"
    p.parent.mkdir(parents=True)
    p.write_text("stale content")
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007-04",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=4,
    )
    doc = json.loads(p.read_text())
    assert doc["sweep_name"] == "saturation-sweep"


def test_write_child_sweep_marker_no_trial_index(tmp_path: Path) -> None:
    write_child_sweep_marker(
        base_dir=tmp_path,
        namespace="bench",
        child_name="ch-0007",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
        trial_index=None,
    )
    doc = json.loads((tmp_path / "bench" / "ch-0007" / "sweep.json").read_text())
    assert "trial_index" in doc
    assert doc["trial_index"] is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/sweep_controller/test_k8s_executor_marker.py -n auto`
Expected: ImportError on `write_child_sweep_marker`.

- [ ] **Step 4: Implement**

Append to `src/aiperf/sweep_controller/k8s_executor.py`:

```python
import os
import tempfile
from pathlib import Path

import orjson


def write_child_sweep_marker(
    *,
    base_dir: Path,
    namespace: str,
    child_name: str,
    sweep_name: str,
    variation_index: int,
    variation_label: str,
    trial_index: int | None,
) -> None:
    """Drop the per-child ``sweep.json`` marker into the child's results directory.

    Called by the sweep-controller before each child AIPerfJob CR is created;
    the marker survives parent-CR TTL reap so the operator's job_union can
    populate the back-link on archived children. Atomic write via ``os.replace``.

    Idempotent: overwriting an existing marker is fine, since deterministic
    child names anchor identity to the apiserver, not to the marker.
    """
    target_dir = Path(base_dir) / namespace / child_name
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep_name": sweep_name,
        "variation_index": variation_index,
        "variation_label": variation_label,
        "trial_index": trial_index,
    }
    fd, tmp_path = tempfile.mkstemp(prefix=".sweep.", suffix=".json", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, target_dir / "sweep.json")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

Then, find the call site that creates each child CR (search for `create_namespaced_custom_object` inside `k8s_executor.py`). Immediately *before* that call, invoke `write_child_sweep_marker(...)` with the values already in scope (`self.base_dir` or whichever attribute the executor has, `self.sweep_name`, `run.variation.index`, `run.variation.label`, `run.trial`).

If the executor does not currently know `base_dir`, plumb it through the constructor — pull it from the existing operator settings (look for a similar `results_dir` / `base_dir` already used by the executor or its caller).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/sweep_controller/test_k8s_executor_marker.py -n auto`
Expected: 3 passed.

- [ ] **Step 6: Run the existing executor tests for regression**

Run: `uv run pytest tests/unit/sweep_controller/ -n auto`
Expected: all pass.

- [ ] **Step 7: Format + commit**

```bash
ruff format src/aiperf/sweep_controller/k8s_executor.py tests/unit/sweep_controller/test_k8s_executor_marker.py
ruff check --fix src/aiperf/sweep_controller/k8s_executor.py tests/unit/sweep_controller/test_k8s_executor_marker.py
git add src/aiperf/sweep_controller/k8s_executor.py tests/unit/sweep_controller/test_k8s_executor_marker.py
git commit -s --no-verify -m "feat(sweep-controller): write per-child sweep.json marker

Atomic write to <results_dir>/<ns>/<child>/sweep.json before the child
CR is created. Survives parent CR reap so job_union can populate the
sweep back-link on archived children."
```

---

## Task 10: Wire sweeps router into `results_server`

**Files:**
- Modify: `src/aiperf/operator/results_server.py`
- Test: existing app-level integration test, or a small smoke test that the route 200s.

- [ ] **Step 1: Add to existing integration smoke test**

Find the test that asserts the FastAPI app starts and `/healthz` works (search: `grep -rn "create_app\|/healthz" tests/unit/operator/`). Append a smoke assertion:

```python
def test_create_app_includes_sweeps_router(tmp_path):
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/v1/sweeps" in routes
    assert "/api/v1/sweeps/{namespace}/{name}" in routes
    assert "/api/v1/sweeps/{namespace}/{name}/cells" in routes
```

If no such test file exists, create `tests/unit/operator/test_results_server_routes.py` with the function above.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_results_server_routes.py -n auto`
Expected: AssertionError — `/api/v1/sweeps` not in routes.

- [ ] **Step 3: Wire the router**

Edit `src/aiperf/operator/results_server.py` `create_app`. Add to the imports near the existing `from aiperf.operator.routers.jobs import create_jobs_router`:

```python
from aiperf.operator.routers.sweeps import create_sweeps_router
```

Then alongside `app.include_router(create_jobs_router(api_holder, base_dir))` (line ~205), add:

```python
    app.include_router(create_sweeps_router(api_holder, base_dir))
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/unit/operator/ -n auto`
Expected: all pass.

- [ ] **Step 5: Format + commit**

```bash
ruff format src/aiperf/operator/results_server.py tests/unit/operator/test_results_server_routes.py
ruff check --fix src/aiperf/operator/results_server.py tests/unit/operator/test_results_server_routes.py
git add src/aiperf/operator/results_server.py tests/unit/operator/test_results_server_routes.py
git commit -s --no-verify -m "feat(operator): register sweeps router in results_server

Closes the backend half of native AIPerfSweep UI support: /api/v1/sweeps,
/api/v1/sweeps/{ns}/{name}, /api/v1/sweeps/{ns}/{name}/cells are now live."
```

---

## Task 11: Sweep-controller writes parent `aggregate.json` + `conditions.json` on terminal

**Files:**
- Modify: whichever sweep-controller module owns terminal-aggregation (search: `grep -rn "aggregate_and_export\|per_cell_aggregates\|child_runs" src/aiperf/sweep_controller/`).
- Test: alongside the existing aggregator test.

- [ ] **Step 1: Locate terminal-write site**

Run: `grep -rn "aggregate_and_export\|per_cell_aggregates" src/aiperf/sweep_controller/`. Identify the function that runs after all children reach terminal and that already writes the aggregate JSON to the legacy path (if any). If no aggregate write currently happens, add one.

- [ ] **Step 2: Write the failing test**

Create or extend a test that drives the aggregator with a fake set of `RunResult`s and asserts:

```python
def test_aggregator_writes_aggregate_json(tmp_path: Path) -> None:
    from aiperf.sweep_controller.aggregator import write_sweep_aggregate

    write_sweep_aggregate(
        base_dir=tmp_path,
        namespace="bench",
        sweep_name="saturation-sweep",
        doc={
            "phase": "Succeeded",
            "totalVariations": 2,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "spec_snapshot": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [8, 32]}],
            },
            "model": "m",
            "per_cell_aggregates": [],
            "child_runs": [],
        },
        conditions=[{"type": "Done", "status": "True"}],
    )
    sweep_dir = tmp_path / "bench" / "sweeps" / "saturation-sweep"
    assert (sweep_dir / "aggregate.json").is_file()
    assert (sweep_dir / "conditions.json").is_file()
    doc = orjson.loads((sweep_dir / "aggregate.json").read_bytes())
    assert doc["phase"] == "Succeeded"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/sweep_controller/ -n auto -k aggregate_json`
Expected: ImportError on `write_sweep_aggregate`.

- [ ] **Step 4: Implement**

Add the helper to the aggregator module (or create `src/aiperf/sweep_controller/aggregator.py` if absent):

```python
def write_sweep_aggregate(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    doc: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
) -> None:
    """Atomic write of <base>/<ns>/sweeps/<name>/{aggregate.json,conditions.json}.

    Called by the sweep-controller exactly once when the parent enters a
    terminal phase. Uses ``*.tmp`` + ``os.replace`` so a torn read on the
    operator side surfaces as JSONDecodeError rather than a half-decoded dict.
    """
    target_dir = Path(base_dir) / namespace / "sweeps" / sweep_name
    target_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(target_dir / "aggregate.json", doc)
    if conditions is not None:
        _atomic_write_json(target_dir / "conditions.json", {"conditions": conditions})


def _atomic_write_json(path: Path, payload: Any) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

Then call `write_sweep_aggregate(...)` at the existing terminal-aggregate site, passing the assembled `doc` and the conditions list (or None if conditions are not yet collected).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/sweep_controller/ -n auto`
Expected: all pass.

- [ ] **Step 6: Format + commit**

```bash
ruff format src/aiperf/sweep_controller/ tests/unit/sweep_controller/
ruff check --fix src/aiperf/sweep_controller/ tests/unit/sweep_controller/
git add src/aiperf/sweep_controller/ tests/unit/sweep_controller/
git commit -s --no-verify -m "feat(sweep-controller): atomic write of parent aggregate.json + conditions.json

Anchors the dual-backed sweep API to a durable file under
<results_dir>/<ns>/sweeps/<name>/. Atomic via os.replace."
```

---

# PR 2 — UI: Sweeps page, detail page, child back-link

## Task 12: `lib/api.js` + `lib/state.js` additions

**Files:**
- Modify: `src/aiperf/operator/ui-v1/lib/api.js`
- Modify: `src/aiperf/operator/ui-v1/lib/state.js`

- [ ] **Step 1: Add API methods**

Edit `src/aiperf/operator/ui-v1/lib/api.js`. Inside the exported `api` object, alongside `listJobs`, add:

```js
  /** List all AIPerfSweep records (live + archived) */
  listSweeps() {
    return apiFetch('/sweeps');
  },

  /** Get a single sweep by namespace and name */
  getSweep(ns, name) {
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`);
  },

  /** Per-cell aggregate metrics for a sweep */
  getSweepCells(ns, name) {
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cells`);
  },
```

- [ ] **Step 2: Add state signal**

Edit `src/aiperf/operator/ui-v1/lib/state.js`. Find the `jobs` signal export and add a sibling:

```js
import { signal } from '@preact/signals';

// ... existing signals ...

export const sweeps = signal([]);
```

If the file's import line for `signal` already exists, do not duplicate it.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/lib/api.js src/aiperf/operator/ui-v1/lib/state.js
git commit -s --no-verify -m "feat(ui-v1): api + state plumbing for sweeps"
```

---

## Task 13: `top-nav.js` adds "Sweeps" item

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/top-nav.js`

- [ ] **Step 1: Read the existing nav**

Read `src/aiperf/operator/ui-v1/components/top-nav.js`. Find the array (or rendered list) of nav items. Match the existing style.

- [ ] **Step 2: Add the entry**

Add `{ label: 'Sweeps', href: '/sweeps' }` immediately after `Jobs` in the nav-items list. Mirror existing accessibility attributes (testid, aria-current).

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/top-nav.js
git commit -s --no-verify -m "feat(ui-v1): TopNav 'Sweeps' tab"
```

---

## Task 14: `app.js` route additions

**Files:**
- Modify: `src/aiperf/operator/ui-v1/app.js`

- [ ] **Step 1: Add route matches**

Edit `src/aiperf/operator/ui-v1/app.js`. Add imports next to the existing page imports:

```js
import { Sweeps } from './pages/sweeps.js';
import { SweepDetail } from './pages/sweep-detail.js';
```

In the route-resolution chain, add (after the `currentRoute === '/jobs'` branch and before the `jobDetailMatch` branch — order matters for matchRoute to land first):

```js
  const sweepDetailMatch = matchRoute('/sweeps/:ns/:name', currentRoute);
  // ...
  } else if (currentRoute === '/sweeps') {
    page = html`<${Sweeps} />`;
  } else if (sweepDetailMatch) {
    page = html`<${SweepDetail} namespace=${sweepDetailMatch.ns} name=${sweepDetailMatch.name} />`;
  }
```

- [ ] **Step 2: Stub the new pages so the dev server does not 404**

Create `src/aiperf/operator/ui-v1/pages/sweeps.js`:

```js
import { html } from 'htm/preact';

export function Sweeps() {
  return html`<div data-testid="page-sweeps">Sweeps (stub)</div>`;
}
```

Create `src/aiperf/operator/ui-v1/pages/sweep-detail.js`:

```js
import { html } from 'htm/preact';

export function SweepDetail({ namespace, name }) {
  return html`<div data-testid="page-sweep-detail">${namespace}/${name} (stub)</div>`;
}
```

These are STUBS that get replaced in Tasks 16/17.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/app.js src/aiperf/operator/ui-v1/pages/sweeps.js src/aiperf/operator/ui-v1/pages/sweep-detail.js
git commit -s --no-verify -m "feat(ui-v1): /sweeps and /sweeps/:ns/:name routes (stubs)"
```

---

## Task 15: `cells-chart.js` component

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/cells-chart.js`

- [ ] **Step 1: Implement**

```js
import { html } from 'htm/preact';
import { useEffect, useRef } from 'preact/hooks';
import { palette } from '../lib/theme.js';

/**
 * Per-cell metric chart for a sweep.
 *
 * Props:
 *   dimensions: [{ name, values }]   from /sweeps/:ns/:name/cells
 *   cells:      [CellEntry]
 *   metric:     string               e.g. 'request_throughput'
 *   stat:       string               e.g. 'avg' | 'p99'
 *
 * 1D dimension: line chart, x = dim values, y = chosen metric stat.
 * 2D dimension: small-multiples — one chart element per second-dim value.
 * 3+ D:        renders a single chart over the FIRST dimension and a
 *              note instructing to use the table view.
 */
export function CellsChart({ dimensions, cells, metric, stat }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || typeof Chart === 'undefined') return;
    if (!dimensions || dimensions.length === 0 || !cells || cells.length === 0) return;

    const primaryDim = dimensions[0];
    const xValues = primaryDim.values;

    // Build series: if 1D, one series; if 2D+, one series per second-dim value
    const datasets = [];
    if (dimensions.length <= 1) {
      const data = xValues.map(v => {
        const cell = cells.find(c => (c.values?.[primaryDim.name] === v));
        return cell?.metrics?.[metric]?.[stat] ?? null;
      });
      datasets.push({
        label: `${metric} (${stat})`,
        data,
        borderColor: palette.blue ?? '#4ea1ff',
        backgroundColor: 'transparent',
        spanGaps: true,
      });
    } else {
      const secondDim = dimensions[1];
      for (const sv of secondDim.values) {
        const data = xValues.map(xv => {
          const cell = cells.find(c =>
            c.values?.[primaryDim.name] === xv &&
            c.values?.[secondDim.name] === sv
          );
          return cell?.metrics?.[metric]?.[stat] ?? null;
        });
        datasets.push({
          label: `${secondDim.name}=${sv}`,
          data,
          spanGaps: true,
        });
      }
    }

    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current, {
      type: 'line',
      data: { labels: xValues.map(String), datasets },
      options: {
        responsive: true,
        plugins: { legend: { display: datasets.length > 1 } },
        scales: {
          x: { title: { display: true, text: primaryDim.name } },
          y: { title: { display: true, text: `${metric} (${stat})` } },
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [dimensions, cells, metric, stat]);

  if (!dimensions || dimensions.length === 0) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim">
      No swept dimensions in this sweep.
    </div>`;
  }
  if (!cells || cells.length === 0) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim">
      No cells completed yet.
    </div>`;
  }
  return html`
    <div data-testid="sweep-cells-chart">
      <canvas ref=${canvasRef} style="max-height: 360px"></canvas>
      ${dimensions.length >= 3 && html`
        <p class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-sm)">
          ${dimensions.length}-D sweep — chart shows the first dimension only.
          Use the table view to inspect higher-dim cells.
        </p>
      `}
    </div>
  `;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/cells-chart.js
git commit -s --no-verify -m "feat(ui-v1): cells-chart component (1D + 2D-faceted)"
```

---

## Task 16: `cells-table.js` component

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/cells-table.js`

- [ ] **Step 1: Implement**

```js
import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';

/**
 * Per-cell metric table.
 *
 * Props:
 *   dimensions: [{ name, values }]
 *   cells:      [CellEntry]
 *   metric:     string
 *   stat:       string
 *   onCellClick: (cell) => void
 */
export function CellsTable({ dimensions, cells, metric, stat, onCellClick }) {
  if (!cells || cells.length === 0) {
    return html`<div data-testid="sweep-cells-table" class="text-dim">
      No cells completed yet.
    </div>`;
  }

  const dimNames = (dimensions || []).map(d => d.name);

  return html`
    <div data-testid="sweep-cells-table">
      <table class="data-table">
        <thead>
          <tr>
            <th>idx</th>
            <th>label</th>
            ${dimNames.map(n => html`<th key=${n}>${n}</th>`)}
            <th>trials ✓</th>
            <th>trials ✗</th>
            <th>${metric} (${stat})</th>
          </tr>
        </thead>
        <tbody>
          ${cells.map(c => html`
            <tr key=${c.variation_index}
                onclick=${() => onCellClick && onCellClick(c)}
                style="cursor: ${onCellClick ? 'pointer' : 'default'}">
              <td>${c.variation_index}</td>
              <td>${c.variation_label}</td>
              ${dimNames.map(n => html`<td key=${n}>${c.values?.[n] ?? '—'}</td>`)}
              <td>${c.trials_completed}</td>
              <td style="color: ${c.trials_failed > 0 ? palette.red : 'inherit'}">
                ${c.trials_failed}
              </td>
              <td>${formatStat(c.metrics?.[metric]?.[stat])}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function formatStat(v) {
  if (v == null) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(3);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/cells-table.js
git commit -s --no-verify -m "feat(ui-v1): cells-table component"
```

---

## Task 17: `pages/sweeps.js` — SweepsList

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/sweeps.js` (replace stub)

- [ ] **Step 1: Implement**

Replace the stub with:

```js
import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { sweeps } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'aggregating'] },
  { label: 'Completed', value: ['succeeded'] },
  { label: 'Failed', value: ['failed', 'partiallyfailed', 'cancelled'] },
];

export function Sweeps() {
  const [list, setList] = useState(sweeps.value);
  const [activeFilter, setActiveFilter] = useState(null);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      const data = await api.listSweeps();
      const next = data?.sweeps ?? [];
      sweeps.value = next;
      setList(next);
    }, 5000, ac.signal);
    return () => ac.abort();
  }, []);

  const filtered = useMemo(() => {
    let r = list;
    if (activeFilter) r = r.filter(s => activeFilter.includes((s.phase ?? '').toLowerCase()));
    if (searchText) {
      const q = searchText.toLowerCase();
      r = r.filter(s =>
        (s.name ?? '').toLowerCase().includes(q) ||
        (s.namespace ?? '').toLowerCase().includes(q)
      );
    }
    return r;
  }, [list, activeFilter, searchText]);

  function rowClick(s) {
    navigate(`/sweeps/${encodeURIComponent(s.namespace)}/${encodeURIComponent(s.name)}`);
  }

  return html`
    <div class="sweeps-page" data-testid="page-sweeps">
      <div class="section-header">
        <div class="filter-tabs">
          ${FILTERS.map(f => html`
            <button
              key=${f.label}
              class=${'filter-tab' + (activeFilter === f.value ? ' filter-tab--active' : '')}
              onclick=${() => setActiveFilter(f.value)}
            >
              ${f.label}
              ${f.value === null
                ? html`<span class="filter-tab-count">${list.length}</span>`
                : html`<span class="filter-tab-count">
                    ${list.filter(s => f.value.includes((s.phase ?? '').toLowerCase())).length}
                  </span>`}
            </button>
          `)}
        </div>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${filtered.length} of ${list.length} sweep${list.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div style="display: flex; gap: var(--space-3); margin-bottom: var(--space-4); flex-wrap: wrap; align-items: center">
        <input
          type="text"
          placeholder="Search name or namespace..."
          value=${searchText}
          oninput=${e => setSearchText(e.target.value)}
          style=${`flex: 1; min-width: 150px; padding: var(--space-2) var(--space-3);
                   background: ${palette.mantle}; border: 1px solid ${palette.surface0};
                   border-radius: var(--radius-md); color: ${palette.text};
                   font-size: var(--font-size-sm)`}
        />
      </div>

      <table class="data-table" data-testid="sweep-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Namespace</th>
            <th>Phase</th>
            <th>Progress</th>
            <th>Failed</th>
            <th>Variations</th>
            <th>Model</th>
            <th>Source</th>
            <th>Age</th>
          </tr>
        </thead>
        <tbody>
          ${filtered.map(s => html`
            <tr key=${`${s.namespace}/${s.name}`} onclick=${() => rowClick(s)} style="cursor: pointer">
              <td>${s.name}</td>
              <td class="text-dim">${s.namespace}</td>
              <td><${PhasePill} phase=${s.phase} /></td>
              <td>${s.completed_runs} / ${s.total_variations || '?'}</td>
              <td style=${`color: ${s.failed_runs > 0 ? palette.red : 'inherit'}`}>${s.failed_runs}</td>
              <td>${s.total_variations}</td>
              <td class="text-dim">${s.model ?? '—'}</td>
              <td><${SourceChip} source=${s.source} /></td>
              <td class="text-dim">${formatAge(s.age_seconds)}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function PhasePill({ phase }) {
  const p = (phase ?? '').toLowerCase();
  let bg = palette.surface0;
  if (['running', 'aggregating'].includes(p)) bg = palette.blue ?? '#4ea1ff';
  else if (p === 'succeeded') bg = palette.green ?? '#4caf50';
  else if (['failed', 'cancelled', 'partiallyfailed'].includes(p)) bg = palette.red ?? '#e53935';
  return html`<span style=${`background:${bg};color:white;padding:2px 8px;border-radius:8px;font-size:11px`}>${phase ?? 'Unknown'}</span>`;
}

function SourceChip({ source }) {
  return html`<span class="text-dim" style="font-size:11px;padding:1px 6px;border:1px solid ${palette.surface0};border-radius:6px">${source}</span>`;
}

function formatAge(s) {
  if (s == null) return '—';
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m`;
  if (s < 86400) return `${Math.floor(s/3600)}h`;
  return `${Math.floor(s/86400)}d`;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/sweeps.js
git commit -s --no-verify -m "feat(ui-v1): SweepsList page with phase filters + 5s poll"
```

---

## Task 18: `pages/sweep-detail.js` — SweepDetail

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/sweep-detail.js` (replace stub)

- [ ] **Step 1: Implement**

Replace the stub with:

```js
import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { KpiCard } from '../components/kpi-card.js';
import { Conditions } from '../components/conditions.js';
import { JobTable } from '../components/job-table.js';
import { CellsChart } from '../components/cells-chart.js';
import { CellsTable } from '../components/cells-table.js';
import { navigate } from '../lib/router.js';

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'partiallyfailed']);
const DEFAULT_METRIC = 'request_throughput';
const DEFAULT_STAT = 'avg';

export function SweepDetail({ namespace, name }) {
  const [detail, setDetail] = useState(null);
  const [cells, setCells] = useState(null);
  const [view, setView] = useState('chart');
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [stat, setStat] = useState(DEFAULT_STAT);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    let stopped = false;
    async function tick() {
      try {
        const d = await api.getSweep(namespace, name);
        if (!stopped) setDetail(d);
        const phase = (d?.sweep?.phase ?? '').toLowerCase();
        if (TERMINAL.has(phase)) ac.abort();
      } catch (e) {
        if (!stopped) setError(String(e));
      }
    }
    poll(tick, 5000, ac.signal);
    return () => { stopped = true; ac.abort(); };
  }, [namespace, name]);

  useEffect(() => {
    let cancelled = false;
    api.getSweepCells(namespace, name)
      .then(d => { if (!cancelled) setCells(d); })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [namespace, name]);

  const childRows = useMemo(() => detail?.children ?? [], [detail]);
  const metricNames = useMemo(() => {
    const set = new Set();
    for (const c of (cells?.cells ?? [])) {
      for (const m of Object.keys(c.metrics ?? {})) set.add(m);
    }
    return [...set].sort();
  }, [cells]);

  if (error) {
    return html`<div data-testid="page-sweep-detail" class="error-banner">${error}</div>`;
  }
  if (!detail) {
    return html`<div data-testid="page-sweep-detail" class="text-dim">Loading…</div>`;
  }

  const s = detail.sweep;
  const status = detail.status ?? {};
  const conditions = status.conditions ?? [];
  const currentCell = status.currentCell;

  return html`
    <div class="sweep-detail" data-testid="page-sweep-detail">
      <header class="page-header">
        <h2>${s.name} <span class="text-dim">${s.namespace}</span></h2>
        <div style="display:flex;gap:var(--space-3);align-items:center">
          <span style=${`background:${pillColor(s.phase)};color:white;padding:2px 10px;border-radius:8px;font-size:12px`}>${s.phase}</span>
          <span class="text-dim">model: ${s.model ?? '—'}</span>
          <span class="text-dim">${s.source}</span>
        </div>
        ${currentCell && html`
          <p class="text-dim">running variation ${currentCell.variationIndex ?? '?'}/${s.total_variations} ${currentCell.trial != null ? `trial ${currentCell.trial}` : ''}</p>
        `}
      </header>

      <section class="kpi-row" style="display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-3)">
        <${KpiCard} label="Variations" value=${s.total_variations} />
        <${KpiCard} label="Completed" value=${s.completed_runs} />
        <${KpiCard} label="Failed" value=${s.failed_runs} />
        <${KpiCard} label="Total runs" value=${s.completed_runs + s.failed_runs} />
      </section>

      <section style="margin-top:var(--space-4)">
        <${Conditions} conditions=${conditions} />
      </section>

      <section style="margin-top:var(--space-4)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--space-2)">
          <h3>Cells</h3>
          <div style="display:flex;gap:var(--space-2);align-items:center">
            <select value=${metric} onchange=${e => setMetric(e.target.value)}>
              ${metricNames.length === 0
                ? html`<option value=${DEFAULT_METRIC}>${DEFAULT_METRIC}</option>`
                : metricNames.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
            </select>
            <select value=${stat} onchange=${e => setStat(e.target.value)}>
              ${['avg','p50','p90','p95','p99','min','max'].map(s2 =>
                html`<option key=${s2} value=${s2}>${s2}</option>`)}
            </select>
            <button class=${'filter-tab' + (view === 'chart' ? ' filter-tab--active' : '')}
                    onclick=${() => setView('chart')}>Chart</button>
            <button class=${'filter-tab' + (view === 'table' ? ' filter-tab--active' : '')}
                    onclick=${() => setView('table')}>Table</button>
          </div>
        </div>
        ${view === 'chart'
          ? html`<${CellsChart}
              dimensions=${cells?.dimensions ?? []}
              cells=${cells?.cells ?? []}
              metric=${metric}
              stat=${stat} />`
          : html`<${CellsTable}
              dimensions=${cells?.dimensions ?? []}
              cells=${cells?.cells ?? []}
              metric=${metric}
              stat=${stat}
              onCellClick=${c => c.children?.[0] && navigate(`/jobs/${encodeURIComponent(c.children[0].namespace)}/${encodeURIComponent(c.children[0].name)}`)} />`}
      </section>

      <section style="margin-top:var(--space-4)">
        <h3>Children</h3>
        <${JobTable} jobs=${childRows} onRowClick=${j =>
          navigate(`/jobs/${encodeURIComponent(j.namespace)}/${encodeURIComponent(j.name)}`)} />
      </section>
    </div>
  `;
}

function pillColor(phase) {
  const p = (phase ?? '').toLowerCase();
  if (['running', 'aggregating'].includes(p)) return palette.blue ?? '#4ea1ff';
  if (p === 'succeeded') return palette.green ?? '#4caf50';
  if (['failed', 'cancelled', 'partiallyfailed'].includes(p)) return palette.red ?? '#e53935';
  return palette.surface0;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/sweep-detail.js
git commit -s --no-verify -m "feat(ui-v1): SweepDetail page — header, KPIs, conditions, cells, children"
```

---

## Task 19: `JobTable` back-link badge

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/job-table.js`

- [ ] **Step 1: Read existing component**

Read `src/aiperf/operator/ui-v1/components/job-table.js`. Identify the cell that renders the job name. Note that incoming row data is from `/api/v1/jobs` — fields will be camelCase aliases (`sweepName`).

- [ ] **Step 2: Add the back-link**

In the name-cell render, after rendering the name, add (preserving existing markup style):

```js
${row.sweepName && html`
  <div class="text-dim" style="font-size:11px;font-style:italic;margin-top:2px">
    <a href=${`/sweeps/${encodeURIComponent(row.namespace)}/${encodeURIComponent(row.sweepName)}`}
       data-testid="job-row-sweep-link"
       onclick=${e => { e.stopPropagation(); navigate(`/sweeps/${encodeURIComponent(row.namespace)}/${encodeURIComponent(row.sweepName)}`); e.preventDefault(); }}>
      ↳ sweep: ${row.sweepName}
    </a>
  </div>
`}
```

If `navigate` is not yet imported in `job-table.js`, add `import { navigate } from '../lib/router.js';`.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/job-table.js
git commit -s --no-verify -m "feat(ui-v1): JobTable shows ↳ sweep:<name> back-link on child rows"
```

---

## Task 20: `JobDetail` parent-sweep header link

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/job-detail.js`

- [ ] **Step 1: Read the existing header**

Read the page; locate the title block.

- [ ] **Step 2: Add the link**

Below the existing job-name header, render:

```js
${job.sweepName && html`
  <p class="text-dim" data-testid="job-detail-sweep-link">
    Part of sweep
    <a href=${`/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`}
       onclick=${e => { e.preventDefault(); navigate(`/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`); }}>
      ${job.sweepName}
    </a>
    ${job.variationLabel && html` — variation ${job.variationLabel}`}
  </p>
`}
```

If `navigate` is not yet imported, add it.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/job-detail.js
git commit -s --no-verify -m "feat(ui-v1): JobDetail header links to parent sweep when present"
```

---

# PR 3 — Optional polish

## Task 21: Backfill marker script for legacy sweeps

**Files:**
- Create: `tools/backfill_sweep_markers.py`
- Test: `tests/unit/tools/test_backfill_sweep_markers.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from tools.backfill_sweep_markers import backfill_sweep_markers


def test_backfill_writes_marker_for_each_child(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(json.dumps({
        "child_runs": [
            {"namespace": "bench", "name": "ch-0-0",
             "variation_index": 0, "variation_label": "concurrency-8", "trial_index": 0},
            {"namespace": "bench", "name": "ch-1-0",
             "variation_index": 1, "variation_label": "concurrency-32", "trial_index": 0},
        ],
    }))
    (tmp_path / "bench" / "ch-0-0").mkdir(parents=True)
    (tmp_path / "bench" / "ch-1-0").mkdir(parents=True)

    backfill_sweep_markers(tmp_path)

    m0 = json.loads((tmp_path / "bench" / "ch-0-0" / "sweep.json").read_text())
    m1 = json.loads((tmp_path / "bench" / "ch-1-0" / "sweep.json").read_text())
    assert m0["sweep_name"] == "s1"
    assert m0["variation_index"] == 0
    assert m1["variation_index"] == 1


def test_backfill_skips_children_without_results_dir(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(json.dumps({
        "child_runs": [
            {"namespace": "bench", "name": "ch-only-on-disk",
             "variation_index": 0, "variation_label": "x", "trial_index": 0},
            {"namespace": "bench", "name": "ch-no-disk",
             "variation_index": 1, "variation_label": "y", "trial_index": 0},
        ],
    }))
    (tmp_path / "bench" / "ch-only-on-disk").mkdir(parents=True)
    backfill_sweep_markers(tmp_path)
    assert (tmp_path / "bench" / "ch-only-on-disk" / "sweep.json").is_file()
    assert not (tmp_path / "bench" / "ch-no-disk").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/tools/test_backfill_sweep_markers.py -n auto`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `tools/backfill_sweep_markers.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfill `sweep.json` markers for legacy sweeps that ran before
`write_child_sweep_marker` existed.

Walks <results_dir>/<ns>/sweeps/*/aggregate.json, reads child_runs[],
and drops sweep.json into each existing child results dir. Children
that have no results dir are skipped silently — they would have
nothing to render anyway.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import orjson

from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker

logger = logging.getLogger(__name__)


def backfill_sweep_markers(base_dir: Path) -> int:
    """Return the number of markers written."""
    written = 0
    if not base_dir.is_dir():
        return 0
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        sweeps_root = ns_dir / "sweeps"
        if not sweeps_root.is_dir():
            continue
        for sweep_dir in sorted(sweeps_root.iterdir()):
            if not sweep_dir.is_dir():
                continue
            agg_path = sweep_dir / "aggregate.json"
            if not agg_path.is_file():
                continue
            try:
                doc = orjson.loads(agg_path.read_bytes())
            except (OSError, orjson.JSONDecodeError) as e:
                logger.warning("skipping unreadable %s: %s", agg_path, e)
                continue
            sweep_name = sweep_dir.name
            for child in doc.get("child_runs") or []:
                if not isinstance(child, dict):
                    continue
                child_ns = child.get("namespace") or ns_dir.name
                child_name = child.get("name")
                if not child_name:
                    continue
                child_dir = base_dir / child_ns / child_name
                if not child_dir.is_dir():
                    continue
                write_child_sweep_marker(
                    base_dir=base_dir,
                    namespace=child_ns,
                    child_name=child_name,
                    sweep_name=sweep_name,
                    variation_index=int(child.get("variation_index") or 0),
                    variation_label=str(child.get("variation_label") or ""),
                    trial_index=child.get("trial_index"),
                )
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: backfill_sweep_markers.py <results_dir>", file=sys.stderr)
        return 2
    n = backfill_sweep_markers(Path(args[0]))
    print(f"wrote {n} sweep.json markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/unit/tools/test_backfill_sweep_markers.py -n auto`
Expected: 2 passed.

- [ ] **Step 5: Format + commit**

```bash
ruff format tools/backfill_sweep_markers.py tests/unit/tools/test_backfill_sweep_markers.py
ruff check --fix tools/backfill_sweep_markers.py tests/unit/tools/test_backfill_sweep_markers.py
git add tools/backfill_sweep_markers.py tests/unit/tools/test_backfill_sweep_markers.py
git commit -s --no-verify -m "feat(tools): backfill sweep.json markers for legacy sweeps"
```

---

# Final smoke

## Task 22: Full unit-suite run + branch sanity

- [ ] **Step 1: Run the unit suite**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all green; the new `test_sweeps_*`, `test_sweep_union`, `test_job_union_sweep_linkage`, `test_k8s_executor_marker`, `test_results_layout` (extension), `test_results_server_routes`, and `test_backfill_sweep_markers` tests pass.

- [ ] **Step 2: Verify the API end-to-end with the test client**

Quick local check (optional, not a commit):
```bash
uv run python -c "
from fastapi.testclient import TestClient
from aiperf.operator.results_server import create_app
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    app = create_app(results_dir=pathlib.Path(d))
    c = TestClient(app)
    # /sweeps returns 503 because api_holder is unset until lifespan startup —
    # in tests we patch it; in prod the lifespan sets it. This is expected.
    print(c.get('/api/v1/sweeps').status_code)
"
```

Expected output: `503` (api not set in raw create_app — the routes exist, which is what `test_create_app_includes_sweeps_router` asserts).

- [ ] **Step 3: Sanity-grep for placeholders left behind**

Run:
```bash
grep -rn "TODO\|FIXME\|TBD" src/aiperf/operator/sweep_union.py \
                            src/aiperf/operator/routers/sweeps.py \
                            src/aiperf/operator/routers/sweeps_models.py \
                            src/aiperf/operator/ui-v1/pages/sweeps.js \
                            src/aiperf/operator/ui-v1/pages/sweep-detail.js \
                            src/aiperf/operator/ui-v1/components/cells-chart.js \
                            src/aiperf/operator/ui-v1/components/cells-table.js
```

Expected: no output.

- [ ] **Step 4: Final commit if any leftover formatting changes**

```bash
ruff format src/aiperf/operator src/aiperf/sweep_controller src/aiperf/kubernetes/client.py
ruff check --fix src/aiperf/operator src/aiperf/sweep_controller src/aiperf/kubernetes/client.py
git diff --cached --quiet || git commit -s --no-verify -m "chore: final ruff format/lint pass for sweep UI work"
```

---

## Self-Review Checklist (planner runs before handoff)

- [x] **Spec coverage** — every section of the spec maps to a task:
  - §4 Routes & Pages → T13, T14, T17, T18
  - §5 API → T2, T3, T5, T6, T7, T8, T10
  - §6 Durability → T1, T4, T9, T11, T21
  - §7.4 Sweep-controller side-effects → T9, T11
  - §10 Testing → distributed across each task's TDD steps
- [x] **Placeholder scan** — no TBD/TODO; every step has the code or command.
- [x] **Type consistency** — `SweepRecord` dataclass field names match between `sweep_union.py` and `sweeps.py`; `ActiveJobSummary.sweep_name` matches `job-table.js` reading `row.sweepName` (camelCase via alias generator); `write_child_sweep_marker` signature is identical in T9 and T21.
- [x] **No leakage** — UI back-link uses the same field names that the router exposes (`sweepName`, `variationIndex`, `variationLabel` after alias rendering).
