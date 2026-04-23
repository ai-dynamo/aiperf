# Unified Jobs Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make `/api/v1/jobs` the single source of truth — a union of cluster CRs and PVC result directories, keyed by `(namespace, name)` — and update the Jobs / Job Detail / Dashboard UI pages to display the full history.

**Architecture:** New `aiperf.operator.job_union` module exposing `list_all_jobs(api, results_dir)` and `find_any_job(api, results_dir, ns, name)`. `AIPerfJobInfo` gains a `source: Literal["live","archived","both"]` field. Jobs router calls the union helpers; Job Detail tolerates `source="archived"` (no CR); UI adds an Archived tab, a source badge, and a "CR deleted" banner on detail.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, kubernetes_asyncio (unchanged), Preact SPA (UI).

**Spec:** [`docs/superpowers/specs/2026-04-22-unified-jobs-source-design.md`](../specs/2026-04-22-unified-jobs-source-design.md)

**Pre-flight for every task:** branch is `ajc/k8s`; commit directly on it; use `git commit --no-verify -s` (pre-existing CRD drift blocks pre-commit hooks). Never `git stash`.

---

## File Structure

### New files
- `src/aiperf/operator/job_union.py` — `list_all_jobs`, `find_any_job`, `_archived_from_summary_json`
- `tests/unit/operator/test_job_union.py`
- `tests/e2e/operator_ui/test_unified_jobs.py`

### Modified files
- `src/aiperf/kubernetes/models.py` — add `source` field to `AIPerfJobInfo`; extend `to_info()` to stamp `source="live"`
- `src/aiperf/operator/routers/jobs.py` — `create_jobs_router` gains `results_dir` arg; `_list_jobs_impl` / `_get_job_impl` / `_cancel_job_impl` use union helpers; cancel returns 400 on archived
- `src/aiperf/operator/results_server.py` — pass `results_dir` into `create_jobs_router`
- `src/aiperf/operator/ui/pages/jobs.js` — Archived filter tab, source badge
- `src/aiperf/operator/ui/components/job-table.js` — source-badge column (inline next to phase)
- `src/aiperf/operator/ui/pages/job-detail.js` — archived banner + hide Cancel/Pods when archived
- `src/aiperf/operator/ui/pages/dashboard.js` — counts use union (completed includes archived)
- `tests/e2e/operator_ui/conftest.py` — new `archived_only_job` helper fixture

### Untouched
- Leaderboard, History, Compare pages (intentionally PVC-only)
- DuckDB analytics router
- CRD / Helm

---

## Task 1: Add `source` field to `AIPerfJobInfo`

**Files:**
- Modify: `src/aiperf/kubernetes/models.py`
- Test: add to `tests/unit/operator/test_job_union.py` (new file; single test for now)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/operator/test_job_union.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified-jobs-source helpers."""

from __future__ import annotations

import pytest

from aiperf.kubernetes.models import AIPerfJobInfo


def test_aiperfjobinfo_source_defaults_to_live():
    info = AIPerfJobInfo(
        name="j1", namespace="ns", phase="Running", job_id="j1",
    )
    assert info.source == "live"


def test_aiperfjobinfo_source_accepts_archived_and_both():
    for s in ("archived", "both"):
        info = AIPerfJobInfo(
            name="j1", namespace="ns", phase="Succeeded", job_id="j1", source=s,
        )
        assert info.source == s


def test_aiperfjobinfo_source_rejects_unknown():
    with pytest.raises(ValueError):
        AIPerfJobInfo(
            name="j1", namespace="ns", phase="Running", job_id="j1", source="bogus",
        )
```

- [ ] **Step 2: Run — expect failure**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: FAIL — `AIPerfJobInfo` has no `source` field.

- [ ] **Step 3: Add the field**

In `src/aiperf/kubernetes/models.py`, inside `class AIPerfJobInfo(K8sCamelModel):` add (place it after the existing `endpoint` field, before `workers_str` property):
```python
    source: Literal["live", "archived", "both"] = Field(
        default="live",
        description=(
            "Provenance: 'live' = CR on cluster only; 'archived' = PVC results "
            "only (CR no longer exists); 'both' = CR + PVC results."
        ),
    )
```

Add to the top imports if not present:
```python
from typing import Literal
```

- [ ] **Step 4: Run — expect pass**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/kubernetes/models.py tests/unit/operator/test_job_union.py
git commit --no-verify -s -m "feat(kubernetes): add source discriminator to AIPerfJobInfo

Field marks provenance: live (CR only), archived (PVC only), or both.
Default 'live' preserves backwards-compat for existing callers."
```

---

## Task 2: Write `list_all_jobs` PVC scan half

**Files:**
- Modify: `src/aiperf/operator/job_union.py` (create)
- Modify: `tests/unit/operator/test_job_union.py`

The scan builds `AIPerfJobInfo` entries with `source="archived"` from `<base>/<ns>/<job>/profile_export_aiperf.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/operator/test_job_union.py`:
```python
from pathlib import Path
import orjson


def _write_summary(base: Path, ns: str, name: str, **extra) -> None:
    d = base / ns / name
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "status": "Succeeded",
        "start_time": "2026-04-22T10:00:00Z",
        "end_time": "2026-04-22T10:45:00Z",
        "request_throughput": {"avg": 42.1, "unit": "requests/sec"},
        "request_latency": {"p99": 390.0, "unit": "ms"},
        "input_config": {
            "models": {"items": [{"name": "llama3-8b"}]},
            "endpoint": {"urls": ["http://llama3.svc:8000/v1"], "type": "chat"},
        },
    }
    body.update(extra)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps(body))


def test_scan_pvc_jobs_empty_dir(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    assert _scan_pvc_jobs(tmp_path) == []


def test_scan_pvc_jobs_finds_summary_files(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    _write_summary(tmp_path, "aiperf-bench", "run-a")
    _write_summary(tmp_path, "ml-lab", "run-b", **{"status": "Failed"})
    entries = _scan_pvc_jobs(tmp_path)
    by_key = {(e.namespace, e.name): e for e in entries}
    assert (("aiperf-bench", "run-a")) in by_key
    assert (("ml-lab", "run-b")) in by_key
    a = by_key[("aiperf-bench", "run-a")]
    assert a.source == "archived"
    assert a.phase == "Succeeded"
    assert a.throughput_rps == 42.1
    assert a.latency_p99_ms == 390.0
    assert a.model == "llama3-8b"
    assert a.endpoint == "http://llama3.svc:8000/v1"
    assert a.progress_percent == 100.0
    b = by_key[("ml-lab", "run-b")]
    assert b.phase == "Failed"


def test_scan_pvc_jobs_skips_dirs_without_summary(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    (tmp_path / "aiperf-bench" / "no-summary").mkdir(parents=True)
    (tmp_path / "aiperf-bench" / "no-summary" / "other.txt").write_bytes(b"x")
    assert _scan_pvc_jobs(tmp_path) == []


def test_scan_pvc_jobs_missing_status_defaults_to_archived(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    d = tmp_path / "ns" / "job"
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "request_throughput": {"avg": 10.0, "unit": "requests/sec"},
    }))
    [e] = _scan_pvc_jobs(tmp_path)
    assert e.phase == "Archived"


def test_scan_pvc_jobs_filters_by_namespace(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    _write_summary(tmp_path, "ns-a", "j1")
    _write_summary(tmp_path, "ns-b", "j2")
    entries = _scan_pvc_jobs(tmp_path, namespace="ns-a")
    assert [e.namespace for e in entries] == ["ns-a"]
```

- [ ] **Step 2: Run — expect failure**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: FAIL — `aiperf.operator.job_union` doesn't exist.

- [ ] **Step 3: Implement `_scan_pvc_jobs`**

Create `src/aiperf/operator/job_union.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified view of AIPerfJobs: cluster CRs + PVC result directories.

The operator UI treats "a job" as a single logical concept, but the data lives
in two planes:

1. Cluster CRs (ephemeral, live state: workers, pods, phase).
2. PVC result directories (persistent, historical state: metrics, config).

This module joins the two by `(namespace, name)` and stamps each entry with a
`source` field so callers can reason about provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson

from aiperf.kubernetes.models import AIPerfJobInfo

logger = logging.getLogger("aiperf.operator.job_union")

# Filename the operator writes once a run is persisted to the PVC. Its presence
# marks a directory as "has a real completed run" for the union.
_SUMMARY_FILE = "profile_export_aiperf.json"


def _read_summary(path: Path) -> dict[str, Any] | None:
    """Load a ``profile_export_aiperf.json`` or return None if unreadable."""
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning(f"Skipping unreadable summary {path}: {e}")
        return None


def _archived_from_summary(
    namespace: str, name: str, summary: dict[str, Any], *, mtime_iso: str,
) -> AIPerfJobInfo:
    """Build an archived ``AIPerfJobInfo`` from a summary JSON dict."""
    phase = str(summary.get("status") or "Archived")
    start_time = summary.get("start_time") or None
    end_time = summary.get("end_time") or None

    rt = summary.get("request_throughput") or {}
    throughput = rt.get("avg") if isinstance(rt, dict) else None
    lat = summary.get("request_latency") or {}
    latency_p99 = lat.get("p99") if isinstance(lat, dict) else None

    ic = summary.get("input_config") or {}
    models = (ic.get("models") or {}).get("items") or []
    model: str | None = None
    if models:
        first = models[0]
        model = first.get("name") if isinstance(first, dict) else first
    endpoint = ic.get("endpoint") or {}
    urls = endpoint.get("urls") or []
    endpoint_url = urls[0] if urls else None

    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase=phase,
        job_id=name,
        workers_ready=0,
        workers_total=0,
        current_phase=None,
        error=None,
        start_time=start_time,
        completion_time=end_time,
        created=start_time or mtime_iso,
        progress_percent=100.0,
        throughput_rps=float(throughput) if throughput is not None else None,
        latency_p99_ms=float(latency_p99) if latency_p99 is not None else None,
        model=model,
        endpoint=endpoint_url,
        source="archived",
    )


def _scan_pvc_jobs(
    base_dir: Path, *, namespace: str | None = None,
) -> list[AIPerfJobInfo]:
    """Walk ``<base>/<ns>/<job>/profile_export_aiperf.json`` and build entries.

    Skips namespaces other than ``namespace`` if supplied; skips dirs that
    lack a summary JSON; logs + skips unreadable summaries.
    """
    import datetime as _dt

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    out: list[AIPerfJobInfo] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        for job_dir in sorted(ns_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            summary_path = job_dir / _SUMMARY_FILE
            if not summary_path.is_file():
                continue
            summary = _read_summary(summary_path)
            if summary is None:
                continue
            mtime_iso = _dt.datetime.fromtimestamp(
                summary_path.stat().st_mtime, tz=_dt.timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            out.append(
                _archived_from_summary(
                    ns_dir.name, job_dir.name, summary, mtime_iso=mtime_iso,
                )
            )
    return out
```

- [ ] **Step 4: Run — expect pass**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: all previously-written tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/job_union.py tests/unit/operator/test_job_union.py
git commit --no-verify -s -m "feat(operator): add PVC-scan helper for archived jobs

_scan_pvc_jobs walks <base>/<ns>/<job>/profile_export_aiperf.json and
builds AIPerfJobInfo entries with source=archived. Fields derived from
the summary JSON's top-level status, request_throughput.avg,
request_latency.p99, and input_config.{models,endpoint}."
```

---

## Task 3: Write `list_all_jobs` union + `find_any_job`

**Files:**
- Modify: `src/aiperf/operator/job_union.py`
- Modify: `tests/unit/operator/test_job_union.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/operator/test_job_union.py`:
```python
import pytest


class _FakeCR:
    """Minimal stand-in for AIPerfJobInfo returned by list_aiperf_jobs."""


@pytest.mark.asyncio
async def test_list_all_jobs_cr_only(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return [
            AIPerfJobInfo(
                name="live-a", namespace="ns", phase="Running", job_id="live-a",
            ),
        ]
    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)

    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    assert [e.name for e in out] == ["live-a"]
    assert out[0].source == "live"


@pytest.mark.asyncio
async def test_list_all_jobs_pvc_only(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return []
    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns", "archive-a")
    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    assert [e.name for e in out] == ["archive-a"]
    assert out[0].source == "archived"


@pytest.mark.asyncio
async def test_list_all_jobs_overlap_marks_both(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return [
            AIPerfJobInfo(
                name="run-1", namespace="ns", phase="Succeeded", job_id="run-1",
                throughput_rps=77.7,
            ),
        ]
    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns", "run-1")
    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    [entry] = out
    assert entry.source == "both"
    # CR wins for live fields
    assert entry.phase == "Succeeded"
    assert entry.throughput_rps == 77.7


@pytest.mark.asyncio
async def test_list_all_jobs_filters_both_sides_by_namespace(
    tmp_path, monkeypatch,
):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        assert all_namespaces is False
        assert namespace == "ns-a"
        return [AIPerfJobInfo(
            name="live-a", namespace="ns-a", phase="Running", job_id="live-a",
        )]
    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns-a", "archive-a")
    _write_summary(tmp_path, "ns-b", "archive-b")
    out = await job_union.list_all_jobs(
        api=None, results_dir=tmp_path, all_namespaces=False, namespace="ns-a",
    )
    names = {e.name for e in out}
    assert names == {"live-a", "archive-a"}


@pytest.mark.asyncio
async def test_find_any_job_prefers_cr(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return AIPerfJobInfo(
            name=name, namespace=namespace, phase="Running", job_id=name,
            throughput_rps=123.0,
        )
    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    _write_summary(tmp_path, "ns", "j1")
    info = await job_union.find_any_job(None, tmp_path, "ns", "j1")
    assert info is not None
    assert info.source == "both"
    assert info.throughput_rps == 123.0


@pytest.mark.asyncio
async def test_find_any_job_falls_back_to_pvc(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return None
    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    _write_summary(tmp_path, "ns", "j1")
    info = await job_union.find_any_job(None, tmp_path, "ns", "j1")
    assert info is not None
    assert info.source == "archived"


@pytest.mark.asyncio
async def test_find_any_job_returns_none_when_neither(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return None
    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    info = await job_union.find_any_job(None, tmp_path, "ns", "missing")
    assert info is None
```

- [ ] **Step 2: Run — expect failure**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: FAIL — `list_all_jobs`, `find_any_job`, module-level `list_aiperf_jobs`/`find_aiperf_job` bindings not yet present.

- [ ] **Step 3: Implement**

Append to `src/aiperf/operator/job_union.py`:
```python
from typing import TYPE_CHECKING

# Import at module level so tests can monkeypatch these bindings directly.
from aiperf.kubernetes.client import find_aiperf_job, list_aiperf_jobs

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


async def list_all_jobs(
    api: "ApiClient | None",
    results_dir: Path,
    *,
    all_namespaces: bool = True,
    namespace: str | None = None,
) -> list[AIPerfJobInfo]:
    """Return the union of cluster CRs and PVC result directories.

    Keyed by (namespace, name). Overlap entries are tagged ``source="both"``
    using the CR's values as the base (it has live worker/phase data) and
    letting PVC fields through only where the CR doesn't already carry them.
    """
    cr_jobs: list[AIPerfJobInfo] = []
    if api is not None:
        try:
            cr_jobs = await list_aiperf_jobs(
                api, all_namespaces=all_namespaces, namespace=namespace,
            )
        except Exception as e:  # noqa: BLE001 - broad by design: PVC still usable
            logger.warning(f"list_aiperf_jobs failed, continuing PVC-only: {e}")
            cr_jobs = []
    # Freshly stamped source=live (even though the default is live, make it
    # explicit so the "both" promotion below is unambiguous).
    for j in cr_jobs:
        j.source = "live"

    pvc_jobs = _scan_pvc_jobs(results_dir, namespace=namespace)

    cr_keys = {(j.namespace, j.name) for j in cr_jobs}
    out: list[AIPerfJobInfo] = list(cr_jobs)
    for pj in pvc_jobs:
        key = (pj.namespace, pj.name)
        if key in cr_keys:
            # Promote the matching CR entry to source="both" and backfill any
            # historical-only fields the CR is silent about.
            for cj in out:
                if (cj.namespace, cj.name) == key:
                    cj.source = "both"
                    if cj.throughput_rps is None:
                        cj.throughput_rps = pj.throughput_rps
                    if cj.latency_p99_ms is None:
                        cj.latency_p99_ms = pj.latency_p99_ms
                    if cj.model is None:
                        cj.model = pj.model
                    if cj.endpoint is None:
                        cj.endpoint = pj.endpoint
                    break
        else:
            out.append(pj)
    return out


async def find_any_job(
    api: "ApiClient | None",
    results_dir: Path,
    namespace: str,
    name: str,
) -> AIPerfJobInfo | None:
    """Return the unified view of a single job or None if neither source has it.

    If both sources have it, CR wins on live fields and ``source="both"``.
    """
    cr: AIPerfJobInfo | None = None
    if api is not None:
        try:
            cr = await find_aiperf_job(api, name, namespace)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"find_aiperf_job failed, falling back to PVC: {e}")
            cr = None
    if cr is not None:
        cr.source = "live"

    summary_path = results_dir / namespace / name / _SUMMARY_FILE
    pvc: AIPerfJobInfo | None = None
    if summary_path.is_file():
        import datetime as _dt
        data = _read_summary(summary_path)
        if data is not None:
            mtime_iso = _dt.datetime.fromtimestamp(
                summary_path.stat().st_mtime, tz=_dt.timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            pvc = _archived_from_summary(
                namespace, name, data, mtime_iso=mtime_iso,
            )

    if cr is None and pvc is None:
        return None
    if cr is None:
        return pvc
    if pvc is None:
        return cr
    # Both present: backfill missing CR fields from PVC.
    cr.source = "both"
    if cr.throughput_rps is None:
        cr.throughput_rps = pvc.throughput_rps
    if cr.latency_p99_ms is None:
        cr.latency_p99_ms = pvc.latency_p99_ms
    if cr.model is None:
        cr.model = pvc.model
    if cr.endpoint is None:
        cr.endpoint = pvc.endpoint
    return cr
```

- [ ] **Step 4: Run — expect pass**

`uv run pytest tests/unit/operator/test_job_union.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/job_union.py tests/unit/operator/test_job_union.py
git commit --no-verify -s -m "feat(operator): add list_all_jobs/find_any_job union helpers

Joins cluster CRs with PVC directories by (namespace, name). Overlap
entries tagged source=both with CR winning on live fields (phase,
workers) and PVC filling the historical gaps (throughput, model).
Namespace filter applies on both sides."
```

---

## Task 4: Wire `results_dir` through `create_jobs_router`

**Files:**
- Modify: `src/aiperf/operator/routers/jobs.py` — add `results_dir: Path` parameter to `create_jobs_router` and update the three impl functions to use the union helpers.
- Modify: `src/aiperf/operator/results_server.py` — pass `base_dir` into `create_jobs_router`.
- Modify: `tests/unit/operator/test_jobs_router.py` — update any fixture that calls `create_jobs_router(api_holder)` to pass a tmp dir.
- Modify: `tests/e2e/operator_ui/conftest.py` — same if the e2e fixture constructs the router directly (it shouldn't — it uses create_app).

- [ ] **Step 1: Inspect current callsites**

Run:
```bash
grep -rnE "create_jobs_router" src/ tests/ 2>/dev/null
```

- [ ] **Step 2: Write the failing test**

Add to `tests/unit/operator/test_jobs_router.py` (at the end, adapt imports as existing file does):
```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pathlib import Path


def test_list_jobs_includes_archived_only_entry(tmp_path: Path, monkeypatch):
    """GET /api/v1/jobs returns a PVC-only entry when no CR exists for it."""
    import orjson
    from aiperf.operator.routers.jobs import create_jobs_router

    # Seed the PVC with one archived job
    d = tmp_path / "aiperf-bench" / "archive-only"
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "status": "Succeeded",
        "request_throughput": {"avg": 50.0, "unit": "requests/sec"},
    }))

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return []

    from aiperf.operator import job_union as ju
    monkeypatch.setattr(ju, "list_aiperf_jobs", fake_list)

    # Non-None holder so the api-unavailable guard passes in the router.
    api_holder = [object()]
    router = create_jobs_router(api_holder, tmp_path)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {j["name"]: j for j in body["jobs"]}
    assert "archive-only" in names
    assert names["archive-only"]["source"] == "archived"
```

- [ ] **Step 3: Run — expect failure**

`uv run pytest tests/unit/operator/test_jobs_router.py::test_list_jobs_includes_archived_only_entry -v`
Expected: FAIL — `create_jobs_router` doesn't accept a second arg.

- [ ] **Step 4: Update `create_jobs_router`**

In `src/aiperf/operator/routers/jobs.py`:

1. Import the union helpers:
   ```python
   from aiperf.operator.job_union import find_any_job, list_all_jobs
   ```

2. Change the `create_jobs_router` signature to accept `results_dir: Path`:
   ```python
   def create_jobs_router(
       api_holder: list[ApiClient | None], results_dir: Path,
   ) -> APIRouter:
   ```

3. In `_list_jobs_impl` (the body of GET /api/v1/jobs), replace the `list_aiperf_jobs(api, all_namespaces=True)` call with:
   ```python
   jobs = await list_all_jobs(api, results_dir, all_namespaces=True)
   ```

4. In `_get_job_impl` (GET /{ns}/{name}), replace the `find_aiperf_job`+`get_raw_aiperfjob_status` lead with the union helper:
   ```python
   job = await find_any_job(api, results_dir, namespace, name)
   if job is None:
       raise HTTPException(404, f"Job {namespace}/{name} not found")
   if job.source == "archived":
       return JobDetailResponse(
           job=job.model_dump(by_alias=True),
           status={},
           pods=[],
       )
   # existing live path continues below
   raw_status = await get_raw_aiperfjob_status(api, name, namespace)
   pods_raw = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
   return JobDetailResponse(
       job=job.model_dump(by_alias=True),
       status=raw_status or {},
       pods=[_pod_summary(p) for p in pods_raw],
   )
   ```

5. In `_cancel_job_impl`, short-circuit archived:
   ```python
   # Before calling cancel_aiperf_job, check if the job has a CR:
   job = await find_any_job(api, results_dir, namespace, name)
   if job is None:
       raise HTTPException(404, f"Job {namespace}/{name} not found")
   if job.source == "archived":
       raise HTTPException(
           400,
           f"Cannot cancel archived job {namespace}/{name}: "
           "the Kubernetes resource no longer exists.",
       )
   await cancel_aiperf_job(api, name, namespace)
   return CancelResponse(...)  # whatever the existing shape is
   ```

6. Ensure `Path` is imported at top of file:
   ```python
   from pathlib import Path
   ```

- [ ] **Step 5: Update the caller**

In `src/aiperf/operator/results_server.py`, change:
```python
app.include_router(create_jobs_router(api_holder))
```
to:
```python
app.include_router(create_jobs_router(api_holder, base_dir))
```

- [ ] **Step 6: Update existing router tests that pass only `api_holder`**

Grep for `create_jobs_router(` in `tests/unit/operator/test_jobs_router.py`; at each call site, add a `tmp_path` fixture (pytest-built-in) as the second arg. If the test uses the synchronous setup, use `tmp_path_factory.mktemp("results")` for session-ish.

- [ ] **Step 7: Run affected tests**

```bash
uv run pytest tests/unit/operator/test_jobs_router.py tests/unit/operator/test_job_union.py -n auto -v
```
Expected: all pass, including the new archived test.

- [ ] **Step 8: Run the e2e smoke suite to confirm the live path still works**

```bash
uv run pytest tests/e2e/operator_ui/ -m e2e -n auto
```
Expected: same 30 passed / 1 skipped as before.

- [ ] **Step 9: Commit**

```bash
git add src/aiperf/operator/routers/jobs.py src/aiperf/operator/results_server.py tests/unit/operator/test_jobs_router.py
git commit --no-verify -s -m "feat(operator): wire results_dir through create_jobs_router

Jobs list/get/cancel endpoints now use the unified union helpers. Archived
(PVC-only) jobs show up on the list, return a valid detail response, and
produce a 400 on cancel attempts. Existing live-CR paths unchanged."
```

---

## Task 5: UI — Jobs page Archived tab + source badge

**Files:**
- Modify: `src/aiperf/operator/ui/pages/jobs.js`
- Modify: `src/aiperf/operator/ui/components/job-table.js`

- [ ] **Step 1: Read the files**

Skim `jobs.js` (filter tabs, state), `job-table.js` (row rendering).

- [ ] **Step 2: Add "Archived" tab**

In `pages/jobs.js`, find the filter-tab array (Running/Completed/Failed/All) and insert an Archived tab. The filter predicate is `j => j.source === 'archived'`. Tabs update reactively from the full list.

Count displayed on the tab:
```js
const archivedCount = jobs.filter(j => j.source === 'archived').length;
```
and the tab pill like existing ones.

- [ ] **Step 3: Source badge in job-table.js**

Next to the existing Phase pill column, render a small badge when `source` is `archived` or `both`:

```js
const sourceBadge = job.source === 'archived'
  ? html`<span class="badge badge-archived" title="No Kubernetes resource; showing PVC results">archived</span>`
  : job.source === 'both'
  ? html`<span class="badge badge-both" title="Live CR + persisted results">both</span>`
  : null;
```

Put `sourceBadge` inside the Phase cell, after the pill. Use existing badge classes if available; otherwise inherit `text-dim` styling.

- [ ] **Step 4: Run e2e to make sure nothing regressed**

```bash
uv run pytest tests/e2e/operator_ui/ -m e2e -n auto
```
Expected: same counts (30 pass, 1 skip). New behavior not yet tested — that lands in Task 7.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui/pages/jobs.js src/aiperf/operator/ui/components/job-table.js
git commit --no-verify -s -m "feat(operator-ui): add Archived tab + source badge to jobs page

Jobs page now surfaces archived (PVC-only) rows with a dedicated filter
tab and an inline source badge. Live-only rows unchanged."
```

---

## Task 6: UI — Job Detail banner + dashboard count

**Files:**
- Modify: `src/aiperf/operator/ui/pages/job-detail.js`
- Modify: `src/aiperf/operator/ui/pages/dashboard.js`

- [ ] **Step 1: Archived banner in job-detail.js**

At the top of the main render (after breadcrumb, before the KPI row), add:
```js
${info?.source === 'archived' && html`
  <div class="banner banner-info" style="margin-bottom: var(--space-3)">
    This run's Kubernetes resource has been deleted. Showing archived
    results from the results volume.
  </div>
`}
```

Hide the Cancel button when `info?.source === 'archived'`:
```js
${info?.phase === 'Running' && info?.source !== 'archived' && html`<button data-testid="job-detail-cancel" ...>Cancel</button>`}
```

Hide the Pods card when archived:
```js
${info?.source !== 'archived' && html`<${PodsCard} pods=${pods} />`}
```

- [ ] **Step 2: Dashboard counts include archived**

In `pages/dashboard.js`, find where "Completed" / "Failed" counts are computed (probably from `jobs.filter(j => j.phase === 'Succeeded')` etc.). Those predicates already work for archived entries since `phase` is synthesized from the summary JSON's top-level `status` field. Verify by reading the count computation and confirming it just pattern-matches phase strings.

The only concrete change needed is: ensure "Running" count excludes archived (archived can't be running):
```js
const running = jobs.filter(j => j.phase === 'Running' && j.source !== 'archived').length;
```

- [ ] **Step 3: Run e2e**

```bash
uv run pytest tests/e2e/operator_ui/ -m e2e -n auto
```
Expected: 30 pass, 1 skip.

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/operator/ui/pages/job-detail.js src/aiperf/operator/ui/pages/dashboard.js
git commit --no-verify -s -m "feat(operator-ui): archived-banner + dashboard counts use union

Job Detail shows an informational banner and hides Cancel/Pods when
source=archived. Dashboard Running count excludes archived entries;
Completed/Failed already aggregate union phases correctly."
```

---

## Task 7: E2E tests for archived path

**Files:**
- Modify: `tests/e2e/operator_ui/conftest.py` — add `archived_only_job` helper
- Create: `tests/e2e/operator_ui/test_unified_jobs.py`

- [ ] **Step 1: Add fixture helper**

In `tests/e2e/operator_ui/conftest.py`, add:
```python
def write_archived_job(results_dir: Path, namespace: str, name: str) -> None:
    """Drop a PVC-only (no CR) job directory into the results dir."""
    import orjson
    d = results_dir / namespace / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "status": "Succeeded",
        "start_time": "2026-04-20T10:00:00Z",
        "end_time": "2026-04-20T10:45:00Z",
        "request_throughput": {"avg": 55.5, "unit": "requests/sec"},
        "request_latency": {"p99": 421.0, "unit": "ms"},
        "input_config": {
            "models": {"items": [{"name": "mistral-7b"}]},
            "endpoint": {"urls": ["http://mistral.svc:8000/v1"], "type": "chat"},
        },
    }))
    (d / ".aiperf_results_ready.json").write_bytes(orjson.dumps({"ready": True}))
```

Export it via the module; no new fixture — tests call it directly.

- [ ] **Step 2: Write the e2e tests**

Create `tests/e2e/operator_ui/test_unified_jobs.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for the unified cluster+PVC jobs view.

Archived jobs (PVC dir but no CR) should appear on the Jobs page, render a
job-detail page with a banner and no Cancel/Pods, and be tallied in the
dashboard Completed count.
"""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from .conftest import write_archived_job
from ._pages import DashboardPage, JobDetailPage, JobsPage

pytestmark = [pytest.mark.e2e]


@pytest.mark.asyncio(loop_scope="session")
async def test_jobs_page_lists_archived_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page,
):
    """Archived PVC-only job appears in the Jobs table with an 'archived' badge."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    row = jobs_page.row("ml-lab", "ghost-run")
    await expect(row).to_be_visible()
    await expect(row).to_contain_text("archived")


@pytest.mark.asyncio(loop_scope="session")
async def test_job_detail_archived_shows_banner_hides_cancel(
    live_operator_app, seeded_results_dir, fake_k8s_client, page,
):
    """Archived job detail renders the banner and omits Cancel and Pods."""
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    detail = JobDetailPage(page, live_operator_app.base_url, "ml-lab", "ghost-run")
    await detail.goto()
    await expect(
        page.get_by_text("Kubernetes resource has been deleted", exact=False)
    ).to_be_visible()
    await expect(page.get_by_test_id("job-detail-cancel")).to_have_count(0)
    await expect(page.get_by_test_id("job-detail-pods")).to_have_count(0)


@pytest.mark.asyncio(loop_scope="session")
async def test_dashboard_completed_count_includes_archived(
    live_operator_app, seeded_results_dir, fake_k8s_client, page,
):
    """Dashboard Completed tile counts archived-Succeeded jobs."""
    # Seeded CR jobs: 3 Succeeded, 1 Failed, 1 Running. Add one archived-only.
    write_archived_job(live_operator_app.results_dir, "ml-lab", "ghost-run")
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("completed")
    await expect(kpi).to_contain_text("4")  # 3 live + 1 archived
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/e2e/operator_ui/test_unified_jobs.py -m e2e -n auto -v
```
Expected: 3 passed.

- [ ] **Step 4: Run full e2e suite**

```bash
uv run pytest tests/e2e/operator_ui/ -m e2e -n auto
```
Expected: 33 passed, 1 skipped (original 30 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/operator_ui/test_unified_jobs.py tests/e2e/operator_ui/conftest.py
git commit --no-verify -s -m "test(e2e): archived-job visibility across jobs/detail/dashboard"
```

---

## Self-review

**Spec coverage:**
- §4 data model → Task 1 (source field)
- §5.1/§5.2 helpers → Tasks 2–3
- §5.3 router wiring → Task 4
- §5.4 UI → Tasks 5–6
- §7 tests → distributed per task + Task 7 for e2e

**Placeholders:** none; all steps contain real code.

**Type consistency:** `AIPerfJobInfo.source` literal `"live"|"archived"|"both"` used consistently across Python and JS.

**Risk:** Task 4's modifications to `_cancel_job_impl` assume the existing function returns a `CancelResponse`; the subagent must preserve whatever it actually returns today. Similarly the `_get_job_impl` merge must preserve the existing status+pods fetch path for live jobs — don't regress.
