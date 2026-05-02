# Run Picker Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual `EpochSelector` + `RunSelectorCard` controls on the job-detail page (ui-v1) with a single `RunPicker` dropdown, fed by a normalized server-side `status` enum per epoch.

**Architecture:** Backend extends `GET /api/v1/jobs/{ns}/{name}/epochs` to return a precomputed `status` enum (`running`/`succeeded`/`failed`/`cancelled`/`unknown`) plus `startedAt`/`endedAt`, derived once on the server from the SQLite runs index reconciled against the live CR. Frontend deletes the two existing pickers and renders a single new `RunPicker` component (button + custom popover, keyboard-accessible) in the title row. Same-config epochs only.

**Tech Stack:** Python 3.10+ FastAPI/Pydantic on the backend; vanilla JS with `htm/preact` (zero build step) on the frontend; pytest with `node --input-type=module` subprocess for JS unit tests.

**Spec:** `docs/superpowers/specs/2026-05-01-run-picker-redesign.md`

**Parallelism note:** Tasks 1–2 (backend model + helper) and Task 4 (new frontend component) are independent — they can be dispatched concurrently. Task 3 depends on 1 and 2. Tasks 5–6 depend on 4 and on the shape from Task 1. The recommended dispatch order is `{1, 2, 4} → 3 → 5 → 6`.

---

## File Structure

**Created:**
- `src/aiperf/operator/ui-v1/components/run-picker.js` — the new component (button + popover, ~250 lines).
- `tests/unit/operator/test_derive_run_status.py` — exhaustive unit tests for the status derivation helper.
- `tests/unit/ui/test_operator_run_picker.py` — JS-via-pytest tests for `RunPicker` and supporting helpers.

**Modified:**
- `src/aiperf/operator/routers/jobs_models.py` — extend `JobEpochSummary` with `status`/`started_at`/`ended_at`.
- `src/aiperf/operator/routers/jobs.py` — add `derive_run_status` helper; rewrite `_list_job_epochs_impl` to read rich rows from `runs_index` and reconcile with the CR; thread `api` parameter through and update the route handler.
- `src/aiperf/operator/ui-v1/lib/api.js` — update the docstring on `getJobEpochs` to mention the new fields.
- `src/aiperf/operator/ui-v1/lib/run-selector.js` — delete `buildRunSelectorRows`; keep `runHref`.
- `src/aiperf/operator/ui-v1/pages/job-detail.js` — delete `RunSelectorCard` definition (lines 1521–1572), delete its render site (~lines 2193–2200), delete the `EpochSelector` import and callsite (line 18 import; line 2116 callsite), insert `RunPicker` import + callsite at the same spot.
- `tests/unit/operator/test_jobs_router_epochs.py` — extend with cases that assert the new response fields and the running/index-miss/failed-row paths.

**Deleted:**
- `src/aiperf/operator/ui-v1/components/epoch-selector.js` — entirely.
- `tests/unit/ui/test_operator_run_selector.py` — entirely (the function it tests is removed).

---

## Task 1: Backend — extend `JobEpochSummary` response model

**Files:**
- Modify: `src/aiperf/operator/routers/jobs_models.py:192-209`

- [ ] **Step 1: Add the new fields to the Pydantic model**

In `src/aiperf/operator/routers/jobs_models.py`, replace the `JobEpochSummary` class with:

```python
class JobEpochSummary(AIPerfBaseModel):
    """One epoch entry in the job-history listing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
        alias_generator=to_camel,
    )

    epoch: str = Field(description="Decimal-seconds epoch identifier.")
    is_latest: bool = Field(
        description="Whether this is the current latest epoch (per latest.txt)."
    )
    mtime_epoch: int = Field(description="UNIX seconds of the run dir's mtime.")
    file_count: int = Field(
        description="Number of files persisted under this epoch dir."
    )
    status: Literal["running", "succeeded", "failed", "cancelled", "unknown"] = Field(
        default="unknown",
        description=(
            "Normalized run status. 'running' for the live in-flight epoch; "
            "'succeeded'/'failed'/'cancelled' for terminal phases; "
            "'unknown' when the runs index hasn't ingested this epoch yet."
        ),
    )
    started_at: int | None = Field(
        default=None,
        description="UNIX seconds when this run started, or None if unknown.",
    )
    ended_at: int | None = Field(
        default=None,
        description="UNIX seconds when this run ended, or None if still running / unknown.",
    )
```

Add `from typing import Literal` to the imports at the top of the file if not already present.

- [ ] **Step 2: Verify the response model still serializes today's data**

Run: `uv run pytest tests/unit/operator/test_jobs_router_epochs.py::test_list_job_epochs -n auto -v`
Expected: PASS — the existing test asserts the legacy fields and shouldn't be sensitive to new optional fields with defaults.

- [ ] **Step 3: Commit**

```bash
git add src/aiperf/operator/routers/jobs_models.py
git commit -s -m "feat(operator-api): widen JobEpochSummary with status/startedAt/endedAt"
```

---

## Task 2: Backend — `derive_run_status` helper + exhaustive unit tests

**Files:**
- Modify: `src/aiperf/operator/routers/jobs.py` (add helper near `_list_job_epochs_impl` at ~line 386)
- Test: `tests/unit/operator/test_derive_run_status.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/operator/test_derive_run_status.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for derive_run_status — the per-epoch status normalizer."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.operator.routers.jobs import derive_run_status
from aiperf.operator.runs_index_models import RunIndexRow


def _row(*, epoch: str = "1714069400", phase: str = "Succeeded",
         error: str | None = None, is_latest: bool = True) -> RunIndexRow:
    return RunIndexRow(
        namespace="bench",
        job_id="j1",
        epoch=epoch,
        phase=phase,
        is_latest=is_latest,
        start_time=None,
        end_time=None,
        created_unix=0,
        mtime_epoch=0,
        error=error,
        model=None,
        endpoint=None,
        gpu_count=0,
        gpu_name=None,
        file_count=0,
        total_size_bytes=0,
        sweep_namespace=None,
        sweep_name=None,
        sweep_epoch=None,
        sweep_variation_idx=None,
    )


@pytest.mark.parametrize(
    "row, live_running_epoch, expected",
    [
        param(_row(epoch="100", phase="Running"), "100", "running",
              id="live-running-wins-over-phase"),
        param(_row(epoch="100", phase="Succeeded"), "100", "running",
              id="live-running-wins-even-when-index-stale"),
        param(_row(epoch="100", phase="Succeeded"), None, "succeeded",
              id="phase-succeeded-no-live"),
        param(_row(epoch="100", phase="Succeeded"), "999", "succeeded",
              id="phase-succeeded-different-live-epoch"),
        param(_row(epoch="100", phase="Failed"), None, "failed",
              id="phase-failed"),
        param(_row(epoch="100", phase="Cancelled"), None, "cancelled",
              id="phase-cancelled"),
        param(_row(epoch="100", phase="Succeeded", error="boom"), None, "failed",
              id="error-overrides-succeeded"),
        param(_row(epoch="100", phase="Pending"), None, "unknown",
              id="phase-pending-falls-to-unknown"),
        param(_row(epoch="100", phase=""), None, "unknown",
              id="empty-phase-falls-to-unknown"),
        param(_row(epoch="100", phase="SUCCEEDED"), None, "succeeded",
              id="phase-case-insensitive-uppercase"),
        param(_row(epoch="100", phase="failed"), None, "failed",
              id="phase-case-insensitive-lowercase"),
    ],
)  # fmt: skip
def test_derive_run_status(row: RunIndexRow, live_running_epoch: str | None,
                           expected: str) -> None:
    assert derive_run_status(row, live_running_epoch=live_running_epoch) == expected
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `uv run pytest tests/unit/operator/test_derive_run_status.py -n auto -v`
Expected: FAIL with `ImportError: cannot import name 'derive_run_status'`.

- [ ] **Step 3: Add the helper**

In `src/aiperf/operator/routers/jobs.py`, add this helper near `_list_job_epochs_impl` (immediately above it, around line 386):

```python
def derive_run_status(
    row: RunIndexRow,
    *,
    live_running_epoch: str | None,
) -> Literal["running", "succeeded", "failed", "cancelled", "unknown"]:
    """Reconcile a runs-index row with the live CR into a single status enum.

    The live in-flight epoch always reports ``"running"`` even if the index
    row's ``phase`` lags behind (the index is updated on completion; the CR
    is the truth-of-the-moment for "is this epoch alive right now?"). For
    every other row, ``error`` overrides phase (a row that finished with an
    error is failed, regardless of the phase column), and unknown phases
    fall through to ``"unknown"`` rather than guessing.

    Example:
        >>> row = _some_row(phase="Succeeded")
        >>> derive_run_status(row, live_running_epoch=None)
        'succeeded'
    """
    if live_running_epoch is not None and row.epoch == live_running_epoch:
        return "running"
    if row.error:
        return "failed"
    phase = (row.phase or "").lower()
    if phase == "succeeded":
        return "succeeded"
    if phase == "failed":
        return "failed"
    if phase == "cancelled":
        return "cancelled"
    return "unknown"
```

Add the necessary imports at the top of `jobs.py`:
- `from typing import Literal` (if not already imported)
- `from aiperf.operator.runs_index_models import RunIndexRow`

- [ ] **Step 4: Run tests — confirm they pass**

Run: `uv run pytest tests/unit/operator/test_derive_run_status.py -n auto -v`
Expected: PASS, all 11 cases.

- [ ] **Step 5: Lint**

Run: `ruff format src/aiperf/operator/routers/jobs.py tests/unit/operator/test_derive_run_status.py && ruff check --fix src/aiperf/operator/routers/jobs.py tests/unit/operator/test_derive_run_status.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/routers/jobs.py tests/unit/operator/test_derive_run_status.py
git commit -s -m "feat(operator-api): add derive_run_status helper for epoch listing"
```

---

## Task 3: Backend — switch `_list_job_epochs_impl` to runs index, thread CR access

**Files:**
- Modify: `src/aiperf/operator/routers/jobs.py:386-413` (impl) and `:707-712` (route handler)
- Test: `tests/unit/operator/test_jobs_router_epochs.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/unit/operator/test_jobs_router_epochs.py` (after the existing `test_list_job_epochs`):

```python
def test_list_job_epochs_returns_status_unknown_when_index_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """Index-miss path: every row returns status=unknown."""
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")
    _patch_no_live_cr(monkeypatch)
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["epochs"]) == 1
    e = body["epochs"][0]
    assert e["status"] == "unknown"
    assert e["startedAt"] is None
    assert e["endedAt"] is None


def test_list_job_epochs_running_overrides_index_phase(
    tmp_path: Path, monkeypatch
) -> None:
    """Live in-flight epoch reports status=running even if index phase is stale."""
    import asyncio

    from aiperf.operator import runs_index
    from aiperf.operator import job_union as ju

    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")

    db = tmp_path / "_index.sqlite"
    asyncio.run(runs_index.open(db))
    try:
        asyncio.run(
            runs_index.upsert_run_created(
                "bench", "j1", "1714069400", spec={"models": {"items": [{"name": "m"}]}}
            )
        )
        # Stale phase: index says Succeeded; CR will say Running below.
        asyncio.run(
            runs_index.upsert_run_phase(
                "bench", "j1", "1714069400", phase="Succeeded"
            )
        )
        asyncio.run(runs_index.set_latest("bench", "j1", "1714069400"))

        async def _running_cr(*_args, **_kwargs):
            return {
                "metadata": {"name": "j1", "namespace": "bench"},
                "status": {"phase": "Running", "runEpoch": 1714069400},
            }

        monkeypatch.setattr(ju, "find_aiperf_job", _running_cr)

        api = MagicMock()
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1/epochs")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["epochs"][0]["status"] == "running"
    finally:
        asyncio.run(runs_index.close())


def test_list_job_epochs_phase_failed(tmp_path: Path, monkeypatch) -> None:
    """Phase=Failed in the index produces status=failed in the response."""
    import asyncio

    from aiperf.operator import runs_index

    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")
    _patch_no_live_cr(monkeypatch)

    db = tmp_path / "_index.sqlite"
    asyncio.run(runs_index.open(db))
    try:
        asyncio.run(
            runs_index.upsert_run_created(
                "bench", "j1", "1714069400", spec={"models": {"items": [{"name": "m"}]}}
            )
        )
        asyncio.run(
            runs_index.upsert_run_completed(
                "bench", "j1", "1714069400",
                summary_blob=b"",
                metrics={"metrics": {}},
                files=[],
                mtime_epoch=2,
                start_time="2026-05-01T00:00:00+00:00",
                end_time="2026-05-01T00:05:00+00:00",
                total_size_bytes=0,
                phase="Failed",
            )
        )
        asyncio.run(runs_index.set_latest("bench", "j1", "1714069400"))
        api = MagicMock()
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1/epochs")
        body = r.json()
        assert body["epochs"][0]["status"] == "failed"
        # ISO timestamps from the index are surfaced as unix-seconds ints.
        assert body["epochs"][0]["startedAt"] == 1746057600
        assert body["epochs"][0]["endedAt"] == 1746057900
    finally:
        asyncio.run(runs_index.close())
```

Note: the actual public-write API surface in `runs_index.py` is `upsert_run_created` / `upsert_run_phase` / `upsert_run_completed` / `set_latest`. The writer should re-read those signatures before adapting and adjust kwarg names if the file has drifted.

- [ ] **Step 2: Run tests — confirm they fail**

Run: `uv run pytest tests/unit/operator/test_jobs_router_epochs.py -n auto -v -k "status_unknown or running_overrides or phase_failed"`
Expected: FAIL — current impl returns no `status` field; assertions on `status` and timestamp fields raise `KeyError` or `assert e["status"] == "unknown"` against a default that hasn't yet been wired.

- [ ] **Step 3: Rewrite the impl**

In `src/aiperf/operator/routers/jobs.py`, replace `_list_job_epochs_impl` (currently around line 386) with:

```python
async def _list_job_epochs_impl(
    api: ApiClient | None,
    base_dir: Path,
    namespace: str,
    name: str,
) -> JobEpochsResponse:
    """Body of GET /api/v1/jobs/{namespace}/{name}/epochs.

    Reads rich rows from the runs SQLite index and reconciles each row's
    ``phase`` / ``error`` against the live CR's ``status.runEpoch`` to
    produce a single normalized ``status`` enum per epoch. Falls back to a
    disk walk (``list_runs_async``) when the index has no rows for this
    job — those rows report ``status='unknown'`` and ``started_at``/
    ``ended_at`` of ``None``.

    Order is ascending by ``mtime_epoch`` so the latest entry sits at the
    tail; this matches the prior contract.

    Returns an empty list when neither the index nor the disk has rows
    (job has never been persisted, or PVC directory was reaped).
    """
    from aiperf.operator import job_union as ju
    from aiperf.operator import runs_index

    # Resolve the live in-flight epoch from the CR (None if not running).
    live_running_epoch: str | None = None
    if api is not None:
        try:
            cr = await ju.find_aiperf_job(api, namespace, name)
        except Exception:  # noqa: BLE001 — UI surface, never block on CR errors
            cr = None
        if cr is not None:
            cr_status = (cr.get("status") or {}) if isinstance(cr, dict) else {}
            if cr_status.get("phase") == "Running":
                run_epoch = cr_status.get("runEpoch")
                if run_epoch is not None:
                    live_running_epoch = str(run_epoch)

    # Index-first read.
    rich_rows: list[RunIndexRow] = []
    try:
        rich_rows = await runs_index.list_runs_for_job(namespace, name)
    except Exception:  # noqa: BLE001 — index unavailable degrades to disk
        rich_rows = []

    if rich_rows:
        rich_rows.sort(key=lambda r: r.mtime_epoch or 0)
        return JobEpochsResponse(
            epochs=[
                JobEpochSummary(
                    epoch=r.epoch,
                    is_latest=bool(r.is_latest),
                    mtime_epoch=int(r.mtime_epoch or 0),
                    file_count=r.file_count,
                    status=derive_run_status(r, live_running_epoch=live_running_epoch),
                    started_at=_iso_to_unix(r.start_time),
                    ended_at=_iso_to_unix(r.end_time),
                )
                for r in rich_rows
            ]
        )

    # Disk fallback — index has nothing for this job.
    runs = await list_runs_async(base_dir, namespace, name)
    return JobEpochsResponse(
        epochs=[
            JobEpochSummary(
                epoch=r.epoch,
                is_latest=r.is_latest,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
                status=("running"
                        if live_running_epoch is not None and r.epoch == live_running_epoch
                        else "unknown"),
                started_at=None,
                ended_at=None,
            )
            for r in reversed(runs)
        ]
    )


def _iso_to_unix(ts: str | None) -> int | None:
    """Parse a ``2026-05-01T00:00:00Z`` style timestamp to unix seconds; None on miss."""
    if not ts:
        return None
    try:
        from datetime import datetime
        # Accept both 'Z' suffix and explicit offsets.
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Update the route handler to thread the API client**

In the same file, around line 711, change the route handler:

```python
@router.get(
    "/jobs/{namespace}/{name}/epochs",
    response_model=JobEpochsResponse,
    response_model_by_alias=True,
)
async def list_job_epochs(namespace: str, name: str) -> JobEpochsResponse:
    return await _list_job_epochs_impl(_optional_api(), _results_dir, namespace, name)
```

Where `_optional_api()` returns the cached API client if available or `None` (this is the existing pattern — search for `_require_api` callers to see whether an `_optional_api`-style helper already exists; if not, the simplest approach is to call `_require_api()` inside a `try` and let the route handler fall back to `None` on `RuntimeError`):

```python
def _optional_api() -> ApiClient | None:
    try:
        return _require_api()
    except RuntimeError:
        return None
```

Place `_optional_api` near `_require_api` in the file.

- [ ] **Step 5: Run tests — confirm new tests pass**

Run: `uv run pytest tests/unit/operator/test_jobs_router_epochs.py -n auto -v`
Expected: PASS — all original tests plus the three new ones.

- [ ] **Step 6: Lint**

Run: `ruff format src/aiperf/operator/routers/jobs.py tests/unit/operator/test_jobs_router_epochs.py && ruff check --fix src/aiperf/operator/routers/jobs.py tests/unit/operator/test_jobs_router_epochs.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/aiperf/operator/routers/jobs.py tests/unit/operator/test_jobs_router_epochs.py
git commit -s -m "feat(operator-api): epoch listing reconciles index phase with live CR"
```

---

## Task 4: Frontend — `RunPicker` component (new file) + tests

**Files:**
- Create: `src/aiperf/operator/ui-v1/components/run-picker.js`
- Test: `tests/unit/ui/test_operator_run_picker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/ui/test_operator_run_picker.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ui-v1 RunPicker (component + helpers).

The component is exercised by importing pure helpers via ``node`` and
asserting the JSON-serialized output. Render assertions live in the
integration smoke tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

RUN_PICKER_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui-v1"
    / "components"
    / "run-picker.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _epochs_fixture() -> list[dict]:
    return [
        {"epoch": "1000", "isLatest": False, "mtimeEpoch": 1000,
         "fileCount": 3, "status": "succeeded",
         "startedAt": 999, "endedAt": 1001},
        {"epoch": "2000", "isLatest": False, "mtimeEpoch": 2000,
         "fileCount": 5, "status": "failed",
         "startedAt": 1999, "endedAt": 2002},
        {"epoch": "3000", "isLatest": True, "mtimeEpoch": 3000,
         "fileCount": 7, "status": "running",
         "startedAt": 2999, "endedAt": None},
    ]


def test_build_picker_rows_orders_newest_first_with_ordinal_labels() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench',
          name: 'j1',
          epochs: {fixture},
          current: undefined,
        }});
        console.log(JSON.stringify(rows));
    """
    rows = json.loads(_run_node(script))
    assert [r["label"] for r in rows] == ["Run 3", "Run 2", "Run 1"]
    assert [r["status"] for r in rows] == ["running", "failed", "succeeded"]
    assert [r["isLatest"] for r in rows] == [True, False, False]
    assert [r["selected"] for r in rows] == [True, False, False]
    assert rows[0]["href"] == "#/jobs/bench/j1"
    assert rows[1]["href"] == "#/jobs/bench/j1/runs/2000"


def test_build_picker_rows_marks_pinned_older_run_selected() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench',
          name: 'j1',
          epochs: {fixture},
          current: '2000',
        }});
        console.log(JSON.stringify(rows.map(r => ({{
          label: r.label, selected: r.selected, isLatest: r.isLatest,
        }}))));
    """
    rows = json.loads(_run_node(script))
    assert rows == [
        {"label": "Run 3", "selected": False, "isLatest": True},
        {"label": "Run 2", "selected": True, "isLatest": False},
        {"label": "Run 1", "selected": False, "isLatest": False},
    ]


def test_build_button_label_for_each_state() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const epochs = {fixture};
        const cases = [
          // viewing latest, running
          {{ current: undefined, now: 3060 }},
          // viewing older
          {{ current: '2000', now: 5602 }},
          // viewing latest after completion (mock different epochs)
          {{
            current: undefined, now: 3700,
            epochs: [{{ ...epochs[2], status: 'succeeded', endedAt: 3000 }}, ...epochs.slice(0, 2)],
          }},
        ];
        const out = cases.map(c => buildButtonLabel({{
          epochs: c.epochs ?? epochs,
          current: c.current, now: c.now,
        }}));
        console.log(JSON.stringify(out));
    """
    out = json.loads(_run_node(script))
    # Running latest: numeric "Run 3 · running"
    assert out[0]["text"].startswith("Run 3")
    assert "running" in out[0]["text"]
    assert out[0]["status"] == "running"
    assert out[0]["isLatest"] is True
    # Viewing older: includes "not latest"
    assert out[1]["isLatest"] is False
    assert out[1]["notLatest"] is True
    assert out[1]["text"].startswith("Run 2")
    # Latest completed: relative-time format
    assert out[2]["status"] == "succeeded"
    assert out[2]["isLatest"] is True


def test_build_button_label_single_epoch_renders_inert() -> None:
    fixture = json.dumps([_epochs_fixture()[2]])
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const out = buildButtonLabel({{
          epochs: {fixture}, current: undefined, now: 3060,
        }});
        console.log(JSON.stringify(out));
    """
    out = json.loads(_run_node(script))
    assert out["inert"] is True


def test_build_button_label_zero_epochs_returns_null() -> None:
    script = f"""
        import {{ buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const out = buildButtonLabel({{ epochs: [], current: undefined, now: 0 }});
        console.log(JSON.stringify(out));
    """
    assert _run_node(script) == "null"


def test_build_picker_rows_handles_stale_pinned_epoch() -> None:
    fixture = json.dumps(_epochs_fixture())
    script = f"""
        import {{ buildPickerRows, buildButtonLabel }} from {RUN_PICKER_PATH.as_uri()!r};
        const rows = buildPickerRows({{
          namespace: 'bench', name: 'j1',
          epochs: {fixture}, current: '9999',
        }});
        const label = buildButtonLabel({{
          epochs: {fixture}, current: '9999', now: 5000,
        }});
        console.log(JSON.stringify({{
          rowCount: rows.length, anySelected: rows.some(r => r.selected),
          label: label,
        }}));
    """
    out = json.loads(_run_node(script))
    # Orphan epochs are not added as menu rows.
    assert out["rowCount"] == 3
    assert out["anySelected"] is False
    assert out["label"]["text"].startswith("Run ?")
    assert out["label"]["status"] == "unknown"
    assert out["label"]["notLatest"] is True
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `uv run pytest tests/unit/ui/test_operator_run_picker.py -n auto -v`
Expected: FAIL — `run-picker.js` doesn't exist yet, node import errors.

- [ ] **Step 3: Implement `run-picker.js`**

Create `src/aiperf/operator/ui-v1/components/run-picker.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { html } from 'htm/preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import { runHref } from '../lib/run-selector.js';
import { palette } from '../lib/theme.js';

const STATUS_COLORS = {
  running:   { dot: '#38bdf8', glow: 'rgba(56,189,248,0.25)', pulse: true },
  succeeded: { dot: '#22c55e' },
  failed:    { dot: '#ef4444' },
  cancelled: { dot: '#f59e0b' },
  unknown:   { dot: '#6b7280' },
};

function formatRelativeTime(unixSeconds, now) {
  if (unixSeconds == null) return '';
  const delta = Math.max(0, now - unixSeconds);
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString([], {
    month: 'short', day: 'numeric',
  });
}

/**
 * Pure helper — returns the menu rows in newest-first order with ordinal
 * "Run N" labels (oldest = Run 1). Exported so unit tests can assert the
 * shape without needing a DOM.
 */
export function buildPickerRows({ namespace, name, epochs, current }) {
  const ascending = [...(epochs || [])].sort(
    (a, b) => (a?.mtimeEpoch ?? 0) - (b?.mtimeEpoch ?? 0)
  );
  // Ordinal: oldest = Run 1, newest = Run M.
  const ordinalByEpoch = new Map();
  ascending.forEach((e, i) => ordinalByEpoch.set(String(e.epoch), i + 1));

  const desc = [...ascending].reverse();
  return desc.map(e => {
    const epochStr = String(e.epoch);
    return {
      epoch: epochStr,
      label: `Run ${ordinalByEpoch.get(epochStr)}`,
      status: e.status || 'unknown',
      isLatest: Boolean(e.isLatest),
      selected: current != null && current === epochStr,
      href: e.isLatest ? runHref(namespace, name) : runHref(namespace, name, epochStr),
      startedAt: e.startedAt ?? null,
      mtimeEpoch: e.mtimeEpoch ?? null,
    };
  });
}

/**
 * Pure helper — returns ``{text, status, isLatest, notLatest, inert}`` describing
 * the collapsed button content, or ``null`` when the picker should not render.
 */
export function buildButtonLabel({ epochs, current, now }) {
  if (!epochs || epochs.length === 0) return null;

  const ascending = [...epochs].sort(
    (a, b) => (a?.mtimeEpoch ?? 0) - (b?.mtimeEpoch ?? 0)
  );
  const ordinalByEpoch = new Map();
  ascending.forEach((e, i) => ordinalByEpoch.set(String(e.epoch), i + 1));

  const latest = ascending[ascending.length - 1];
  const latestEpoch = latest ? String(latest.epoch) : null;
  const viewingLatest = current == null || current === latestEpoch;
  const inert = epochs.length === 1;

  if (viewingLatest && latest) {
    const ord = ordinalByEpoch.get(latestEpoch);
    const status = latest.status || 'unknown';
    const text = status === 'running'
      ? `Run ${ord} · running`
      : `Run ${ord} · ${formatRelativeTime(latest.endedAt ?? latest.startedAt ?? latest.mtimeEpoch, now)}`;
    return { text, status, isLatest: true, notLatest: false, inert };
  }

  // Viewing pinned older epoch (or stale/orphan).
  const found = ascending.find(e => String(e.epoch) === String(current));
  if (!found) {
    return {
      text: `Run ?(${current}) · unknown`,
      status: 'unknown',
      isLatest: false,
      notLatest: true,
      inert: false,
    };
  }
  const ord = ordinalByEpoch.get(String(found.epoch));
  const rel = formatRelativeTime(found.endedAt ?? found.startedAt ?? found.mtimeEpoch, now);
  return {
    text: `Run ${ord} · ${rel} · not latest`,
    status: found.status || 'unknown',
    isLatest: false,
    notLatest: true,
    inert,
  };
}

export function RunPicker({ namespace, name, epochs, current, onPick }) {
  const [open, setOpen] = useState(false);
  const [focusIdx, setFocusIdx] = useState(0);
  const wrapRef = useRef(null);
  const now = Math.floor(Date.now() / 1000);

  const label = buildButtonLabel({ epochs, current, now });
  const rows = buildPickerRows({ namespace, name, epochs, current });

  useEffect(() => {
    if (!open) return undefined;
    function onDocClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    function onKey(e) {
      if (e.key === 'Escape') { setOpen(false); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); setFocusIdx(i => Math.min(rows.length - 1, i + 1)); }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setFocusIdx(i => Math.max(0, i - 1)); }
      if (e.key === 'Enter') {
        e.preventDefault();
        const r = rows[focusIdx];
        if (r) { onPick(r.isLatest ? undefined : r.epoch); setOpen(false); }
      }
      if (e.key === 'Tab') setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, rows, focusIdx, onPick]);

  const closeAndPick = useCallback((epoch) => {
    onPick(epoch);
    setOpen(false);
  }, [onPick]);

  if (label == null) return null;

  const dotStyle = (status) => {
    const c = STATUS_COLORS[status] || STATUS_COLORS.unknown;
    const base = `display:inline-block;width:8px;height:8px;border-radius:50%;background:${c.dot};vertical-align:middle;`;
    if (c.pulse) {
      return base + `box-shadow:0 0 0 2px ${c.glow};animation:pulse 1.4s ease-in-out infinite;`;
    }
    return base;
  };

  const showJumpToLatest = !label.isLatest && rows.some(r => r.isLatest);

  return html`
    <div data-testid="job-detail-run-picker" ref=${wrapRef}
         style="position:relative;display:inline-flex;align-items:center;gap:var(--space-2)">
      <button
        type="button"
        aria-haspopup=${label.inert ? 'false' : 'listbox'}
        aria-expanded=${open ? 'true' : 'false'}
        aria-disabled=${label.inert ? 'true' : 'false'}
        onclick=${() => { if (!label.inert) setOpen(o => !o); }}
        title="Pick which run to view"
        style=${'display:inline-flex;align-items:center;gap:6px;padding:4px 10px;'
          + 'background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.12);'
          + 'border-radius:999px;color:var(--text);font-size:11px;'
          + (label.inert ? 'cursor:default;opacity:0.85;' : 'cursor:pointer;')}
      >
        <span style=${dotStyle(label.status)}></span>
        <span>${label.text}</span>
        ${!label.inert && html`<span style="opacity:0.6">▾</span>`}
      </button>
      ${open && html`
        <div role="listbox"
             style=${'position:absolute;top:100%;left:0;margin-top:4px;'
               + 'background:#1a1d24;border:1px solid rgba(255,255,255,0.12);'
               + 'border-radius:6px;padding:4px;min-width:280px;max-height:60vh;'
               + 'overflow-y:auto;z-index:50'}>
          ${showJumpToLatest && html`
            <button
              type="button"
              data-testid="job-detail-run-picker-jump-latest"
              onclick=${() => closeAndPick(undefined)}
              style=${'display:flex;width:100%;align-items:center;gap:8px;padding:8px;'
                + 'background:none;border:none;color:' + palette.blue + ';'
                + 'font-size:11px;cursor:pointer;text-align:left'}
            >↩ Jump to latest</button>
          `}
          ${rows.map((r, i) => html`
            <button
              key=${r.epoch}
              type="button"
              role="option"
              data-testid="job-detail-run-picker-row"
              aria-selected=${r.selected ? 'true' : 'false'}
              onclick=${() => closeAndPick(r.isLatest ? undefined : r.epoch)}
              onfocus=${() => setFocusIdx(i)}
              tabindex=${i === focusIdx ? 0 : -1}
              title=${`Epoch ${r.epoch}`}
              style=${'display:flex;width:100%;align-items:center;gap:10px;padding:8px;'
                + 'background:' + (r.selected ? 'rgba(56,189,248,0.10)' : 'transparent') + ';'
                + 'border:none;border-radius:4px;color:var(--text);font-size:11px;'
                + 'cursor:pointer;text-align:left'}
            >
              <span style=${dotStyle(r.status)}></span>
              <span style="font-weight:600">${r.label}</span>
              ${r.isLatest && html`<span style=${'font-size:10px;padding:1px 6px;border-radius:999px;'
                + 'background:rgba(56,189,248,0.18);color:#7dd3fc'}>latest</span>`}
              <span style="margin-left:auto;opacity:0.7">
                ${formatRelativeTime(r.startedAt ?? r.mtimeEpoch, now)}
              </span>
            </button>
          `)}
        </div>
      `}
    </div>
  `;
}
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `uv run pytest tests/unit/ui/test_operator_run_picker.py -n auto -v`
Expected: PASS, all 6 cases.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/run-picker.js tests/unit/ui/test_operator_run_picker.py
git commit -s -m "feat(ui-v1): add RunPicker dropdown component"
```

---

## Task 5: Frontend — wire `RunPicker` into `job-detail.js`, remove `RunSelectorCard`

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/job-detail.js` (delete `RunSelectorCard` definition, replace `EpochSelector` callsite, delete `RunSelectorCard` callsite)
- Modify: `src/aiperf/operator/ui-v1/lib/api.js` (update docstring on `getJobEpochs`)

- [ ] **Step 1: Replace the `EpochSelector` import and the inline callsite**

In `src/aiperf/operator/ui-v1/pages/job-detail.js`:

Change the import at line 18:
```javascript
// Old:
import { EpochSelector } from '../components/epoch-selector.js';
// New:
import { RunPicker } from '../components/run-picker.js';
```

Change the callsite at line 2116:
```javascript
// Old:
<${EpochSelector} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
// New:
<${RunPicker} namespace=${namespace} name=${name} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
```

- [ ] **Step 2: Delete the `RunSelectorCard` definition and its callsite**

In `src/aiperf/operator/ui-v1/pages/job-detail.js`:

- Delete the `RunSelectorCard` function definition (currently around lines 1521–1572).
- Delete the comment block above it (lines 1502–1519) describing "Similar runs" — wait, re-read that comment first; if it documents `SimilarRunsLink` (which stays), keep it. The block from `function formatRunSelectorTime` through the end of `RunSelectorCard` is what's removed.
- Delete the `RunSelectorCard` render site (currently around lines 2193–2200).
- Delete the import `import { buildRunSelectorRows } from '../lib/run-selector.js';` at line 5 (unused after this change).

- [ ] **Step 3: Update `lib/api.js` docstring**

In `src/aiperf/operator/ui-v1/lib/api.js`, lines 40–43:
```javascript
// Old:
  /** List the persisted run epochs for a job */
  getJobEpochs(ns, name) {
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },
// New:
  /**
   * List the persisted run epochs for a job. Each entry carries:
   * { epoch, isLatest, mtimeEpoch, fileCount, status, startedAt, endedAt }
   * where status is one of running/succeeded/failed/cancelled/unknown.
   */
  getJobEpochs(ns, name) {
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },
```

- [ ] **Step 4: Verify the page still loads (syntax)**

Run: `node --check src/aiperf/operator/ui-v1/pages/job-detail.js`
Expected: no output (file parses).

- [ ] **Step 5: Run any integration smoke that exercises job-detail**

Run: `uv run pytest tests/unit/ui/ -n auto -v -k "job_detail or job-detail"`
Expected: PASS. If a test asserts on the deleted `data-testid="job-detail-run-selector"` or `data-testid="run-selector-live"`, update it to use `data-testid="job-detail-run-picker"` and `data-testid="job-detail-run-picker-row"` instead.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/job-detail.js src/aiperf/operator/ui-v1/lib/api.js
git commit -s -m "feat(ui-v1): replace EpochSelector + RunSelectorCard with RunPicker on job-detail"
```

---

## Task 6: Frontend — delete `EpochSelector`, prune `buildRunSelectorRows`, drop the old test

**Files:**
- Delete: `src/aiperf/operator/ui-v1/components/epoch-selector.js`
- Modify: `src/aiperf/operator/ui-v1/lib/run-selector.js` (remove `buildRunSelectorRows`)
- Delete: `tests/unit/ui/test_operator_run_selector.py`

- [ ] **Step 1: Confirm there are no remaining callers**

Run: `grep -rn "EpochSelector\|buildRunSelectorRows\|epoch-selector" src/aiperf/operator/ui-v1/ tests/ 2>/dev/null`
Expected: empty (or only a remaining `runHref` import, which is fine).

- [ ] **Step 2: Delete `epoch-selector.js`**

Run: `rm src/aiperf/operator/ui-v1/components/epoch-selector.js`

- [ ] **Step 3: Trim `lib/run-selector.js` to keep only `runHref`**

Replace `src/aiperf/operator/ui-v1/lib/run-selector.js` with:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export function runHref(namespace, name, epoch = null) {
  const base = `#/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
  return epoch == null ? base : `${base}/runs/${encodeURIComponent(epoch)}`;
}
```

- [ ] **Step 4: Delete the obsolete test file**

Run: `rm tests/unit/ui/test_operator_run_selector.py`

- [ ] **Step 5: Run the full ui-v1 test slice**

Run: `uv run pytest tests/unit/ui/ -n auto -v`
Expected: PASS.

- [ ] **Step 6: Run pre-commit on touched files**

Run: `pre-commit run --files src/aiperf/operator/routers/jobs.py src/aiperf/operator/routers/jobs_models.py src/aiperf/operator/ui-v1/components/run-picker.js src/aiperf/operator/ui-v1/lib/run-selector.js src/aiperf/operator/ui-v1/lib/api.js src/aiperf/operator/ui-v1/pages/job-detail.js tests/unit/operator/test_jobs_router_epochs.py tests/unit/operator/test_derive_run_status.py tests/unit/ui/test_operator_run_picker.py`
Expected: all hooks pass.

- [ ] **Step 7: Commit**

```bash
git add -A src/aiperf/operator/ui-v1/components/epoch-selector.js src/aiperf/operator/ui-v1/lib/run-selector.js tests/unit/ui/test_operator_run_selector.py
git commit -s -m "chore(ui-v1): drop EpochSelector and buildRunSelectorRows, both unused"
```

---

## Verification (end-of-plan)

- [ ] **Run the full unit suite:** `uv run pytest tests/unit/ -n auto`
  Expected: PASS, no regressions outside the touched modules.

- [ ] **Manual smoke (if a dev cluster is reachable):**
  - Open the operator UI, navigate to a job-detail page with multiple runs.
  - Confirm exactly one Run picker is visible in the title row, no full-width pills bar below.
  - Confirm the dot color matches each run's status; latest run carries a `latest` badge in the menu.
  - Click an older run; confirm the URL switches to `/runs/<epoch>` and the button reads "Run K · ... · not latest".
  - Click "↩ Jump to latest"; confirm URL drops back to `/jobs/<ns>/<name>` and the button no longer shows "not latest".
  - With keyboard: `Tab` to the button, `Enter` to open, `↑`/`↓` to navigate, `Enter` to select, `Esc` to dismiss.

- [ ] **Final commit if any drift was caught during verification.**
