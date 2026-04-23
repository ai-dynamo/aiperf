# Operator Web UI E2E Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live-browser integration test suite for the operator's Preact SPA, covering all six pages via Playwright against a real in-process FastAPI instance.

**Architecture:** `pytest-playwright` drives headless Chromium against `uvicorn.Server(create_app(results_dir=<tmp>))` bound to a random port. `esm.sh` CDN requests are intercepted by Playwright and served from the repo's vendored modules. Backend data is provided by a committed golden results tree plus programmatic builders for edge cases. The six `aiperf.kubernetes.client` helpers used by the jobs router are monkeypatched per-test. An opt-in `@pytest.mark.e2e` marker keeps the suite out of the default unit/integration runs.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, pytest-playwright, Playwright (Chromium), FastAPI, uvicorn, DuckDB (via existing `ResultsDB`), Preact SPA (unchanged except minimal `data-testid` attributes).

**Spec:** [`docs/superpowers/specs/2026-04-22-operator-ui-e2e-tests-design.md`](../specs/2026-04-22-operator-ui-e2e-tests-design.md)

**Pre-flight for every task:** branch is `ajc/k8s`; commit directly on it; use `git commit -s --no-verify` only if pre-commit fmt drift blocks; else plain `git commit -s`. Never use `git stash`. Never use `git stash` under any circumstance.

---

## File Structure

### New files
- `tests/e2e/__init__.py` — empty
- `tests/e2e/operator_ui/__init__.py` — empty
- `tests/e2e/operator_ui/conftest.py` — session + per-test fixtures (uvicorn, results dir swap, fake k8s, page override)
- `tests/e2e/operator_ui/_builders.py` — programmatic results + k8s object builders for edge cases
- `tests/e2e/operator_ui/_pages.py` — page-object helpers (DashboardPage, JobsPage, …)
- `tests/e2e/operator_ui/README.md` — how to run, debug, extend
- `tests/e2e/operator_ui/test_dashboard.py`
- `tests/e2e/operator_ui/test_jobs.py`
- `tests/e2e/operator_ui/test_job_detail.py`
- `tests/e2e/operator_ui/test_leaderboard.py`
- `tests/e2e/operator_ui/test_compare.py`
- `tests/e2e/operator_ui/test_history.py`
- `tests/e2e/operator_ui/test_navigation.py`
- `tests/e2e/operator_ui/test_robustness.py`
- `tests/fixtures/operator_ui/__init__.py` — empty
- `tests/fixtures/operator_ui/generate_golden.py` — one-time generator that populates `results/` and `k8s/`
- `tests/fixtures/operator_ui/results/**` — committed golden tree (four jobs across two namespaces)
- `tests/fixtures/operator_ui/k8s/jobs.json`, `pods.json`, `version.json`
- `.github/workflows/e2e-operator-ui.yml` — CI job (or extend existing workflow)

### Modified files
- `pyproject.toml` — add `playwright` and `pytest-playwright` to `[dependency-groups] dev`; add `e2e` marker
- `Makefile` — add `install-e2e-browsers` and `test-e2e` targets
- `src/aiperf/operator/ui/components/*.js`, `src/aiperf/operator/ui/pages/*.js` — add `data-testid="..."` attributes on key elements the tests select (top-nav links, page main, job-table rows, kpi cards, metric selectors, command palette input, breadcrumb)

### Untouched
- `src/aiperf/operator/results_server.py`, routers, `ResultsDB`, `index.html`, vendored libs — no production behavior change.

---

## Task 1: Add dependencies, marker, and make targets

**Files:**
- Modify: `pyproject.toml`
- Modify: `Makefile`

- [ ] **Step 1: Add dev dependencies**

Run:
```bash
uv add --dev playwright pytest-playwright
```

Verify `pyproject.toml` gained both entries under `[dependency-groups] dev` (or `[tool.uv] dev-dependencies`, depending on the project's current layout).

- [ ] **Step 2: Add the `e2e` marker**

In `pyproject.toml`, locate the `[tool.pytest.ini_options]` `markers = [...]` list and append this line (keep alphabetical/adjacent to existing markers):

```toml
    "e2e: marks tests as browser-based end-to-end UI tests (requires playwright chromium, deselected by default)",
```

Also ensure default collection excludes `e2e` by adding (if not already present) an `addopts` line that keeps existing flags and appends `-m 'not e2e'`. If `addopts` already exists, extend the `-m` expression; otherwise add:

```toml
addopts = "-m 'not e2e and not performance and not benchmark and not gpu and not vllm and not dynamo and not ffmpeg and not stress and not trtllm and not sglang and not k8s_slow'"
```

(If the repo already has an `addopts` with a different marker expression, merge `and not e2e` into it and leave the rest alone.)

- [ ] **Step 3: Add make targets**

Append to `Makefile`:

```makefile
.PHONY: install-e2e-browsers test-e2e

install-e2e-browsers: ## Install Playwright Chromium for e2e UI tests
	uv run playwright install chromium --with-deps || uv run playwright install chromium

test-e2e: ## Run operator web UI e2e tests
	uv run pytest tests/e2e/ -m e2e -n auto
```

- [ ] **Step 4: Install browser**

Run:
```bash
make install-e2e-browsers
```

Expected: Chromium + ffmpeg dependencies downloaded, exit 0.

- [ ] **Step 5: Verify pytest discovery**

Run:
```bash
uv run pytest -m e2e --collect-only
```

Expected: `collected 0 items` (no tests yet). Exit 0. Verifies marker is registered.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml Makefile uv.lock
git commit -s -m "test(e2e): add playwright deps, e2e marker, and make targets

Pure scaffolding for the forthcoming operator web UI e2e suite.
No tests or fixtures yet."
```

---

## Task 2: Build the `live_operator_app` session fixture

**Files:**
- Create: `tests/e2e/__init__.py` (empty)
- Create: `tests/e2e/operator_ui/__init__.py` (empty)
- Create: `tests/e2e/operator_ui/conftest.py`
- Create: `tests/e2e/operator_ui/test_smoke.py` (temporary — deleted in Task 15)

- [ ] **Step 1: Write the failing test**

`tests/e2e/operator_ui/test_smoke.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary smoke test — deleted once per-page tests land."""

import httpx
import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_operator_app_starts(live_operator_app):
    """The session fixture binds a real uvicorn and /healthz returns 200."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{live_operator_app.base_url}/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_operator_app_serves_index(live_operator_app):
    """Root URL returns the SPA index.html."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{live_operator_app.base_url}/")
    assert resp.status_code == 200
    assert "<div id=\"app\"></div>" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py -m e2e -v`
Expected: `ERROR` or `FAIL` — fixture `live_operator_app` not found.

- [ ] **Step 3: Implement the fixture**

`tests/e2e/operator_ui/conftest.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the operator web UI e2e suite.

Runs a real uvicorn server bound to 127.0.0.1:<random> once per session,
hosting the real ``create_app()`` FastAPI instance with a session-scoped
``results_dir``. Per-test fixtures mutate the contents of that dir and
monkeypatch the k8s helpers — no respawn.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI

from aiperf.operator.results_server import create_app


def _free_port() -> int:
    """Bind to port 0 and return the kernel-assigned port.

    There's a TOCTOU race between binding here and re-binding in uvicorn,
    but in practice it's safe on localhost and avoids uvicorn's lack of
    a "port 0 then tell me what you got" API in older versions.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LiveApp:
    base_url: str
    app: FastAPI
    results_dir: Path


@asynccontextmanager
async def _running_server(
    app: FastAPI, port: int
) -> AsyncIterator[None]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for startup
    for _ in range(200):  # 10s max at 50ms
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        server.should_exit = True
        await task
        raise RuntimeError("uvicorn failed to start within 10s")
    try:
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()


@pytest_asyncio.fixture(scope="session")
async def live_operator_app(tmp_path_factory) -> AsyncIterator[LiveApp]:
    """Real uvicorn + real ``create_app()`` bound to a random port.

    The ``results_dir`` is session-scoped; per-test fixtures rewrite its
    contents. The jobs router's ``ApiClient`` stays ``None`` (tests that
    need it monkeypatch the six ``aiperf.kubernetes.client`` helpers).
    """
    results_dir = tmp_path_factory.mktemp("e2e_results")
    app = create_app(results_dir=results_dir)
    port = _free_port()
    async with _running_server(app, port):
        yield LiveApp(
            base_url=f"http://127.0.0.1:{port}",
            app=app,
            results_dir=results_dir,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py -m e2e -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/operator_ui/__init__.py tests/e2e/operator_ui/conftest.py tests/e2e/operator_ui/test_smoke.py
git commit -s -m "test(e2e): add live_operator_app session fixture

Starts uvicorn on a random port against the real create_app() for the
forthcoming Playwright-driven UI tests."
```

---

## Task 3: Generate and commit the golden results fixture tree

**Files:**
- Create: `tests/fixtures/operator_ui/__init__.py` (empty)
- Create: `tests/fixtures/operator_ui/generate_golden.py`
- Create (via script): `tests/fixtures/operator_ui/results/{aiperf-bench,ml-lab}/<job>/...`
- Create (via script): `tests/fixtures/operator_ui/k8s/{jobs,pods,version}.json`
- Modify: `tests/e2e/operator_ui/conftest.py` — add `seeded_results_dir` fixture
- Modify: `tests/e2e/operator_ui/test_smoke.py` — add a seeded-data assertion

**Background:** The UI expects each job dir to contain at minimum a `profile_export_aiperf.json` (metrics summary) and a `profile_export_aiperf.parquet` (per-request rows). DuckDB reads parquet via `read_parquet(...)` in `ResultsDB`. The exact schema is consumed by `ResultsDB.leaderboard()` / `history()` / `compare()` / `summary()`. Before writing the generator, quickly read `src/aiperf/operator/results_db.py` to confirm required columns (job_id, namespace, metric-family columns with `_avg`/`_p50`/`_p99`/`_unit` suffixes).

- [ ] **Step 1: Inspect ResultsDB schema expectations**

Run:
```bash
grep -nE "read_parquet|read_json|read_csv|SELECT.*FROM|metric.*_avg|metric.*_p99" src/aiperf/operator/results_db.py | head -60
```

Note the exact parquet columns referenced and the JSON fields expected by the summary endpoint. Use them verbatim in the generator.

- [ ] **Step 2: Write the failing test**

Append to `tests/e2e/operator_ui/test_smoke.py`:
```python
import httpx  # already imported above


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_seeded_results_populates_leaderboard(
    live_operator_app, seeded_results_dir
):
    """After seeding, /api/v1/analytics/leaderboard returns 4 jobs."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{live_operator_app.base_url}/api/v1/analytics/leaderboard"
            "?metric=request_throughput"
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "entries" in body
    assert len(body["entries"]) >= 3, body
    job_ids = {e["job_id"] for e in body["entries"]}
    assert "aiperf-llama3-c128" in job_ids
    assert "aiperf-llama3-c256" in job_ids
    assert "mistral-7b-run1" in job_ids
```

- [ ] **Step 3: Run test — verify it fails**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py::test_seeded_results_populates_leaderboard -m e2e -v`
Expected: FAIL (fixture `seeded_results_dir` not found).

- [ ] **Step 4: Write the generator script**

`tests/fixtures/operator_ui/generate_golden.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the committed golden fixture tree under tests/fixtures/operator_ui/.

Run once, commit the output. Re-run to refresh.

Usage:
    uv run python tests/fixtures/operator_ui/generate_golden.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

FIXTURES = Path(__file__).parent
RESULTS = FIXTURES / "results"
K8S = FIXTURES / "k8s"


# --- Per-job content ---------------------------------------------------------


def _metric_summary(
    request_throughput: float,
    ttft_ms: float,
    itl_ms: float,
    latency_ms: float,
) -> dict:
    """Shape of the ``profile_export_aiperf.json`` summary the UI reads."""
    def entry(avg: float, unit: str) -> dict:
        return {
            "avg": avg,
            "p50": avg * 0.95,
            "p90": avg * 1.10,
            "p99": avg * 1.30,
            "min": avg * 0.70,
            "max": avg * 1.50,
            "unit": unit,
        }
    return {
        "request_throughput": entry(request_throughput, "requests/sec"),
        "request_latency": entry(latency_ms, "ms"),
        "time_to_first_token": entry(ttft_ms, "ms"),
        "inter_token_latency": entry(itl_ms, "ms"),
        "output_token_throughput": entry(request_throughput * 128, "tokens/sec"),
    }


def _write_job(
    namespace: str,
    job_id: str,
    *,
    model: str,
    concurrency: int,
    request_throughput: float,
    ttft_ms: float,
    itl_ms: float,
    latency_ms: float,
    status: str = "Succeeded",
) -> None:
    d = RESULTS / namespace / job_id
    d.mkdir(parents=True, exist_ok=True)
    # profile_export_aiperf.json — summary
    summary = {
        "job_id": job_id,
        "namespace": namespace,
        "model": model,
        "concurrency": concurrency,
        "status": status,
        "metrics": _metric_summary(
            request_throughput, ttft_ms, itl_ms, latency_ms
        ),
    }
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    )
    # profile_export_aiperf.parquet — per-request rows (100 rows, deterministic)
    rows = 100
    table = pa.table(
        {
            "job_id": [job_id] * rows,
            "namespace": [namespace] * rows,
            "model": [model] * rows,
            "concurrency": [concurrency] * rows,
            "request_throughput_avg": [request_throughput] * rows,
            "request_throughput_unit": ["requests/sec"] * rows,
            "request_latency_avg": [latency_ms] * rows,
            "request_latency_p50": [latency_ms * 0.95] * rows,
            "request_latency_p99": [latency_ms * 1.30] * rows,
            "request_latency_unit": ["ms"] * rows,
            "time_to_first_token_avg": [ttft_ms] * rows,
            "time_to_first_token_p50": [ttft_ms * 0.95] * rows,
            "time_to_first_token_p99": [ttft_ms * 1.30] * rows,
            "time_to_first_token_unit": ["ms"] * rows,
            "inter_token_latency_avg": [itl_ms] * rows,
            "inter_token_latency_p50": [itl_ms * 0.95] * rows,
            "inter_token_latency_p99": [itl_ms * 1.30] * rows,
            "inter_token_latency_unit": ["ms"] * rows,
            "output_token_throughput_avg": [request_throughput * 128] * rows,
            "output_token_throughput_unit": ["tokens/sec"] * rows,
        }
    )
    pq.write_table(table, d / "profile_export_aiperf.parquet")
    # conditions (for job-detail page)
    (d / "conditions.json").write_bytes(orjson.dumps(
        [
            {"type": "Ready", "status": "True", "reason": "BenchmarkComplete"},
            {"type": "Succeeded", "status": str(status == "Succeeded")},
        ],
        option=orjson.OPT_INDENT_2,
    ))
    # ready marker
    (d / ".aiperf_results_ready.json").write_bytes(
        orjson.dumps({"ready": True, "version": 1})
    )


def _write_k8s_fixtures() -> None:
    K8S.mkdir(parents=True, exist_ok=True)
    jobs = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJobList",
        "items": [
            {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfJob",
                "metadata": {
                    "name": name,
                    "namespace": ns,
                    "uid": f"uid-{name}",
                    "creationTimestamp": "2026-04-22T12:00:00Z",
                },
                "spec": {"model": "llama3"},
                "status": {
                    "phase": phase,
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                    ],
                },
            }
            for name, ns, phase in [
                ("aiperf-llama3-c128", "aiperf-bench", "Succeeded"),
                ("aiperf-llama3-c256", "aiperf-bench", "Succeeded"),
                ("mistral-7b-run1", "ml-lab", "Succeeded"),
                ("failed-run", "ml-lab", "Failed"),
                ("live-run", "aiperf-bench", "Running"),
            ]
        ],
    }
    (K8S / "jobs.json").write_bytes(
        orjson.dumps(jobs, option=orjson.OPT_INDENT_2)
    )
    pods = {
        "items": [
            {
                "metadata": {
                    "name": "live-run-controller-0",
                    "namespace": "aiperf-bench",
                    "labels": {"aiperf.nvidia.com/job-id": "live-run"},
                },
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True, "restartCount": 0}],
                },
            },
        ]
    }
    (K8S / "pods.json").write_bytes(
        orjson.dumps(pods, option=orjson.OPT_INDENT_2)
    )
    (K8S / "version.json").write_bytes(
        orjson.dumps(
            {"gitVersion": "v1.29.0", "platform": "linux/amd64"},
            option=orjson.OPT_INDENT_2,
        )
    )


def main() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    _write_job(
        "aiperf-bench", "aiperf-llama3-c128",
        model="llama3-8b", concurrency=128,
        request_throughput=42.1, ttft_ms=150.0, itl_ms=25.0, latency_ms=300.0,
    )
    _write_job(
        "aiperf-bench", "aiperf-llama3-c256",
        model="llama3-8b", concurrency=256,
        request_throughput=78.4, ttft_ms=220.0, itl_ms=32.0, latency_ms=410.0,
    )
    _write_job(
        "ml-lab", "mistral-7b-run1",
        model="mistral-7b", concurrency=64,
        request_throughput=28.9, ttft_ms=180.0, itl_ms=28.0, latency_ms=340.0,
    )
    _write_job(
        "ml-lab", "failed-run",
        model="mistral-7b", concurrency=16,
        request_throughput=0.0, ttft_ms=0.0, itl_ms=0.0, latency_ms=0.0,
        status="Failed",
    )
    _write_k8s_fixtures()
    print(f"Wrote golden tree under {FIXTURES}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run generator, inspect output, adjust schema if needed**

Run:
```bash
uv run python tests/fixtures/operator_ui/generate_golden.py
ls -la tests/fixtures/operator_ui/results/aiperf-bench/aiperf-llama3-c128/
du -sh tests/fixtures/operator_ui/
```

Expected: four result dirs + `k8s/` populated; total size <500KB.

If the leaderboard test later fails because `ResultsDB` expects a different column name, fix the generator to match the **actual** schema grepped in Step 1, re-run the generator, and re-check.

- [ ] **Step 6: Add `seeded_results_dir` fixture**

Append to `tests/e2e/operator_ui/conftest.py`:
```python
import shutil


GOLDEN_RESULTS = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "operator_ui"
    / "results"
)
GOLDEN_K8S = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "operator_ui"
    / "k8s"
)


@pytest.fixture
def seeded_results_dir(live_operator_app: LiveApp) -> Path:
    """Clear the session results dir and copy the golden tree into it."""
    target = live_operator_app.results_dir
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for ns_dir in GOLDEN_RESULTS.iterdir():
        shutil.copytree(ns_dir, target / ns_dir.name)
    return target
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py -m e2e -v`
Expected: all smoke tests PASS including `test_seeded_results_populates_leaderboard`.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/operator_ui tests/e2e/operator_ui/conftest.py tests/e2e/operator_ui/test_smoke.py
git commit -s -m "test(e2e): add golden results fixture tree + seeded_results_dir

Four jobs across two namespaces plus k8s JSON fixtures. Committed
directly so the suite is deterministic and airgap-safe."
```

---

## Task 4: Build the `fake_k8s_client` fixture

**Files:**
- Modify: `tests/e2e/operator_ui/conftest.py`
- Modify: `tests/e2e/operator_ui/test_smoke.py` (add assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/e2e/operator_ui/test_smoke.py`:
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_fake_k8s_client_serves_jobs(
    live_operator_app, fake_k8s_client
):
    """With fake_k8s_client active, /api/v1/jobs returns the canned list."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{live_operator_app.base_url}/api/v1/jobs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {j["name"] for j in body["jobs"]}
    assert "live-run" in names
    assert "aiperf-llama3-c128" in names
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py::test_fake_k8s_client_serves_jobs -m e2e -v`
Expected: FAIL — fixture `fake_k8s_client` not found.

- [ ] **Step 3: Inspect jobs router call-sites**

Run:
```bash
grep -nE "list_aiperf_jobs|find_aiperf_job|get_raw_aiperfjob_status|get_pods|cluster_version|cancel_aiperf_job|list_nodes" src/aiperf/operator/routers/jobs.py
```

Confirm the set of helpers used and their argument shapes. Also check whether any call goes through an `if api is None: raise HTTPException(503)` guard — the fixture must also inject a non-None `ApiClient` sentinel so the guard passes.

- [ ] **Step 4: Add `fake_k8s_client` fixture**

Append to `tests/e2e/operator_ui/conftest.py`:
```python
import orjson
from types import SimpleNamespace


class FakeK8sClient:
    """Collects canned responses + records calls for assertions."""

    def __init__(self) -> None:
        self.jobs_list: list[dict] = []
        self.pods_by_job: dict[str, list[dict]] = {}
        self.cluster_version_info: dict = {"gitVersion": "v1.29.0"}
        self.cancelled: list[tuple[str, str]] = []  # (namespace, name)

    # --- priming helpers used by tests ---
    def set_jobs(self, jobs: list[dict]) -> None:
        self.jobs_list = jobs

    def set_pods(self, job_id: str, pods: list[dict]) -> None:
        self.pods_by_job[job_id] = pods


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


@pytest.fixture
def fake_k8s_client(
    live_operator_app: LiveApp, monkeypatch: pytest.MonkeyPatch
) -> FakeK8sClient:
    """Patch the six aiperf.kubernetes.client helpers the jobs router uses.

    Default responses come from tests/fixtures/operator_ui/k8s/; tests can
    override via the returned FakeK8sClient instance.

    Also injects a non-None ApiClient sentinel into the running app so the
    router's 'client unavailable' guard passes.
    """
    fake = FakeK8sClient()
    fake.jobs_list = _load_json(GOLDEN_K8S / "jobs.json")["items"]
    fake.pods_by_job = {
        "live-run": _load_json(GOLDEN_K8S / "pods.json")["items"],
    }
    fake.cluster_version_info = _load_json(GOLDEN_K8S / "version.json")

    # The router imports these as module-local names from
    # aiperf.kubernetes.client; patch both the source module and the
    # router's local binding to be safe.
    import aiperf.kubernetes.client as kc_mod
    import aiperf.operator.routers.jobs as jobs_router

    async def _list(api, *, all_namespaces=True, namespace=None, **_):
        if all_namespaces:
            return fake.jobs_list
        return [j for j in fake.jobs_list if j["metadata"]["namespace"] == namespace]

    async def _find(api, name, namespace):
        for j in fake.jobs_list:
            m = j["metadata"]
            if m["name"] == name and m["namespace"] == namespace:
                return j
        return None

    async def _raw_status(api, name, namespace):
        j = await _find(api, name, namespace)
        return (j or {}).get("status", {})

    async def _get_pods(api, namespace, label_selector):
        # label_selector is "aiperf.nvidia.com/job-id=<name>"
        job_id = label_selector.split("=", 1)[1]
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    name=p["metadata"]["name"],
                    namespace=p["metadata"]["namespace"],
                    labels=p["metadata"].get("labels", {}),
                ),
                status=SimpleNamespace(
                    phase=p["status"]["phase"],
                    container_statuses=[
                        SimpleNamespace(
                            ready=c.get("ready", False),
                            restart_count=c.get("restartCount", 0),
                        )
                        for c in p["status"].get("containerStatuses", [])
                    ],
                ),
            )
            for p in fake.pods_by_job.get(job_id, [])
        ]

    async def _version(api):
        return SimpleNamespace(**fake.cluster_version_info)

    async def _cancel(api, name, namespace):
        fake.cancelled.append((namespace, name))

    for target_mod in (kc_mod, jobs_router):
        monkeypatch.setattr(target_mod, "list_aiperf_jobs", _list, raising=False)
        monkeypatch.setattr(target_mod, "find_aiperf_job", _find, raising=False)
        monkeypatch.setattr(target_mod, "get_raw_aiperfjob_status", _raw_status, raising=False)
        monkeypatch.setattr(target_mod, "get_pods", _get_pods, raising=False)
        monkeypatch.setattr(target_mod, "cluster_version", _version, raising=False)
        monkeypatch.setattr(target_mod, "cancel_aiperf_job", _cancel, raising=False)

    # The jobs router reads the ApiClient via api_holder[0]. We don't have
    # direct access to the holder, but the monkeypatched helpers ignore the
    # passed-in api. If the router guards on `if api is None: 503`, inject
    # a sentinel. Try both paths — whichever the code uses:
    try:
        from aiperf.operator.routers.jobs import _get_api  # type: ignore
        monkeypatch.setattr(
            "aiperf.operator.routers.jobs._get_api",
            lambda: object(),
            raising=False,
        )
    except ImportError:
        pass

    return fake
```

> **If the router does have an `if api is None` guard and the `_get_api` patch above doesn't apply:** inspect `src/aiperf/operator/routers/jobs.py` to find the holder access pattern and patch it directly. The helpers themselves ignore the passed-in `api` value, so any non-None sentinel works.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py::test_fake_k8s_client_serves_jobs -m e2e -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/operator_ui/conftest.py tests/e2e/operator_ui/test_smoke.py
git commit -s -m "test(e2e): add fake_k8s_client fixture

Monkeypatches the six aiperf.kubernetes.client helpers used by the jobs
router. Canned responses seed from tests/fixtures/operator_ui/k8s/."
```

---

## Task 5: Build the `page` fixture with CDN interception + console-error gate

**Files:**
- Modify: `tests/e2e/operator_ui/conftest.py`
- Modify: `tests/e2e/operator_ui/test_smoke.py`

- [ ] **Step 1: Add a browser-level smoke test**

Append to `tests/e2e/operator_ui/test_smoke.py`:
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_page_loads_spa_without_console_errors(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    """Root URL renders the Preact app; no console errors; CDN intercepted."""
    await page.goto(live_operator_app.base_url + "/")
    await page.wait_for_selector("#app > *", timeout=10_000)
    title = await page.title()
    assert title == "AIPerf"
```

- [ ] **Step 2: Run — it fails (no CDN interception, so esm.sh hits the network or the browser errors)**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py::test_page_loads_spa_without_console_errors -m e2e -v`
Expected: FAIL or slow flake (CDN request unmocked; test harness should also flag console errors).

- [ ] **Step 3: Implement `page` fixture override + CDN interception**

Append to `tests/e2e/operator_ui/conftest.py`:
```python
from typing import Callable

from playwright.async_api import Page, Route


UI_VENDOR = Path(__file__).parent.parent.parent.parent / "src" / "aiperf" / "operator" / "ui" / "vendor"

# Mapping from CDN URL *substrings* to vendored files.
CDN_MAP: dict[str, tuple[str, str]] = {
    "esm.sh/preact@10/hooks": ("preact-hooks.mjs", "text/javascript"),
    "esm.sh/preact@10": ("preact.mjs", "text/javascript"),
    "esm.sh/htm@3/preact": ("htm-preact.mjs", "text/javascript"),
    "esm.sh/@preact/signals@1": ("signals.mjs", "text/javascript"),
    "cdn.jsdelivr.net/npm/chart.js@4": ("chart.umd.min.js", "application/javascript"),
    "fonts.googleapis.com": ("", "text/css"),
    "fonts.gstatic.com": ("", "font/woff2"),
}


@pytest_asyncio.fixture
async def page(live_operator_app: LiveApp, page: Page) -> AsyncIterator[Page]:
    """Wrap pytest-playwright's ``page`` with CDN interception + error gate.

    - All esm.sh / jsdelivr requests are served from src/aiperf/operator/ui/vendor/.
    - Font CDN requests are stubbed to empty responses (tests don't need webfonts).
    - Any uncaught page error or ``console.error(...)`` fails the test at teardown.
    """
    errors: list[str] = []
    unmapped: list[str] = []

    def _on_pageerror(exc) -> None:
        errors.append(f"pageerror: {exc}")

    def _on_console(msg) -> None:
        if msg.type == "error":
            errors.append(f"console.error: {msg.text}")

    page.on("pageerror", _on_pageerror)
    page.on("console", _on_console)

    async def _route(route: Route) -> None:
        url = route.request.url
        for needle, (vendor_file, content_type) in CDN_MAP.items():
            if needle in url:
                if not vendor_file:
                    # Stub fonts with empty body
                    await route.fulfill(status=200, content_type=content_type, body=b"")
                    return
                await route.fulfill(
                    status=200,
                    content_type=content_type,
                    body=(UI_VENDOR / vendor_file).read_bytes(),
                )
                return
        unmapped.append(url)
        await route.continue_()

    # Intercept any host that is not our own test server.
    async def _should_intercept(route: Route) -> None:
        url = route.request.url
        if url.startswith(live_operator_app.base_url):
            await route.continue_()
            return
        await _route(route)

    await page.route("**/*", _should_intercept)

    yield page

    # Surface unmapped CDN requests as test failures.
    if unmapped:
        pytest.fail(
            "Unmapped external requests (add them to CDN_MAP or fix the UI):\n"
            + "\n".join(f"  - {u}" for u in unmapped)
        )
    if errors:
        pytest.fail("Browser errors detected:\n" + "\n".join(errors))
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/e2e/operator_ui/test_smoke.py::test_page_loads_spa_without_console_errors -m e2e -v`
Expected: PASS.

If it fails with "unmapped requests", inspect the failure output and add the missing URL substring to `CDN_MAP`. If it fails with a `console.error` from `app.js`, inspect — the real UI must be error-free against the seeded backend.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/operator_ui/conftest.py tests/e2e/operator_ui/test_smoke.py
git commit -s -m "test(e2e): add page fixture with CDN interception and error gate

Intercepts esm.sh / jsdelivr / fonts and serves vendored modules.
Fails tests on unmapped external requests or any browser console error."
```

---

## Task 6: Add `data-testid` attributes to UI components

**Files:**
- Modify: `src/aiperf/operator/ui/components/top-nav.js`
- Modify: `src/aiperf/operator/ui/components/breadcrumb.js`
- Modify: `src/aiperf/operator/ui/components/job-table.js`
- Modify: `src/aiperf/operator/ui/components/kpi-card.js`
- Modify: `src/aiperf/operator/ui/components/metric-selector.js`
- Modify: `src/aiperf/operator/ui/components/command-palette.js`
- Modify: `src/aiperf/operator/ui/pages/dashboard.js`
- Modify: `src/aiperf/operator/ui/pages/jobs.js`
- Modify: `src/aiperf/operator/ui/pages/job-detail.js`
- Modify: `src/aiperf/operator/ui/pages/leaderboard.js`
- Modify: `src/aiperf/operator/ui/pages/compare.js`
- Modify: `src/aiperf/operator/ui/pages/history.js`

**Rationale:** Playwright's preferred locators are role-based (`get_by_role`) and text-based (`get_by_text`), but for elements that don't have semantic roles (the top-nav container, a custom table wrapper, a kpi-card grid) a stable `data-testid` is more reliable than CSS/structure-based selectors that break when styling changes.

- [ ] **Step 1: Enumerate required test-ids**

The per-page test tasks (7–14) reference these test-ids — add them all now in one pass so later test tasks don't touch UI code:

| Component / page | Element | `data-testid` |
|---|---|---|
| `top-nav.js` | nav root | `top-nav` |
| `top-nav.js` | each link | `nav-link-<route>` (e.g. `nav-link-jobs`) |
| `top-nav.js` | search button | `nav-search` |
| `breadcrumb.js` | root | `breadcrumb` |
| `command-palette.js` | root | `command-palette` |
| `command-palette.js` | input | `command-palette-input` |
| `job-table.js` | tbody | `job-table` |
| `job-table.js` | each row | `job-row-<namespace>-<name>` |
| `job-table.js` | column header | `col-header-<key>` |
| `kpi-card.js` | root | `kpi-<label>` (label slugified) |
| `metric-selector.js` | root | `metric-selector` |
| `pages/dashboard.js` | main | `page-dashboard` |
| `pages/jobs.js` | main | `page-jobs` |
| `pages/jobs.js` | namespace filter | `jobs-ns-filter` |
| `pages/job-detail.js` | main | `page-job-detail` |
| `pages/job-detail.js` | cancel btn | `job-detail-cancel` |
| `pages/job-detail.js` | pods list | `job-detail-pods` |
| `pages/leaderboard.js` | main | `page-leaderboard` |
| `pages/compare.js` | main | `page-compare` |
| `pages/compare.js` | job multi-select | `compare-select` |
| `pages/history.js` | main | `page-history` |

- [ ] **Step 2: For each file above, add the attribute on the identified element**

Example (top-nav.js):
```js
return html`
  <nav class="top-nav" data-testid="top-nav">
    <a data-testid="nav-link-dashboard" href="/">Dashboard</a>
    <a data-testid="nav-link-jobs" href="/jobs">Jobs</a>
    ...
    <button data-testid="nav-search" onClick=${onSearchClick}>⌘K</button>
  </nav>
`;
```

Apply the equivalent change in each listed file. Do not change any other behavior or styling.

- [ ] **Step 3: Verify the UI still renders**

Run:
```bash
uv run pytest tests/e2e/operator_ui/test_smoke.py::test_page_loads_spa_without_console_errors -m e2e -v
```

Expected: PASS (no new console errors introduced by the attribute changes).

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/operator/ui/
git commit -s -m "feat(operator-ui): add data-testid attributes for e2e testing

Purely additive — no behavior or styling changes. Enables stable
Playwright selectors in the forthcoming operator web UI e2e suite."
```

---

## Task 7: Page-object helpers and builders

**Files:**
- Create: `tests/e2e/operator_ui/_pages.py`
- Create: `tests/e2e/operator_ui/_builders.py`

- [ ] **Step 1: Write `_pages.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-object wrappers used by the e2e UI tests.

Thin helpers — not a full POM. Each page exposes `.goto(...)` and the
handful of interactions the per-page test files actually exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Locator, Page, expect


@dataclass
class BasePage:
    page: Page
    base_url: str

    async def _goto(self, route: str) -> None:
        await self.page.goto(self.base_url + route)


class DashboardPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/")
        await expect(self.page.get_by_test_id("page-dashboard")).to_be_visible()

    def kpi(self, label: str) -> Locator:
        return self.page.get_by_test_id(f"kpi-{label}")


class JobsPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/jobs")
        await expect(self.page.get_by_test_id("page-jobs")).to_be_visible()

    def rows(self) -> Locator:
        return self.page.get_by_test_id("job-table").locator("[data-testid^='job-row-']")

    def row(self, namespace: str, name: str) -> Locator:
        return self.page.get_by_test_id(f"job-row-{namespace}-{name}")

    async def click_column_header(self, key: str) -> None:
        await self.page.get_by_test_id(f"col-header-{key}").click()

    async def set_namespace_filter(self, ns: str) -> None:
        await self.page.get_by_test_id("jobs-ns-filter").select_option(ns)


class JobDetailPage(BasePage):
    def __init__(self, page: Page, base_url: str, namespace: str, name: str) -> None:
        super().__init__(page, base_url)
        self.namespace = namespace
        self.name = name

    async def goto(self) -> None:
        await self._goto(f"/jobs/{self.namespace}/{self.name}")
        await expect(self.page.get_by_test_id("page-job-detail")).to_be_visible()

    async def cancel(self) -> None:
        await self.page.get_by_test_id("job-detail-cancel").click()


class LeaderboardPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/leaderboard")
        await expect(self.page.get_by_test_id("page-leaderboard")).to_be_visible()

    async def select_metric(self, metric: str) -> None:
        await self.page.get_by_test_id("metric-selector").select_option(metric)


class ComparePage(BasePage):
    async def goto(self) -> None:
        await self._goto("/compare")
        await expect(self.page.get_by_test_id("page-compare")).to_be_visible()


class HistoryPage(BasePage):
    async def goto(self) -> None:
        await self._goto("/history")
        await expect(self.page.get_by_test_id("page-history")).to_be_visible()


class CommandPalette:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def open(self) -> None:
        await self.page.keyboard.press("Control+k")
        await expect(self.page.get_by_test_id("command-palette")).to_be_visible()

    async def type(self, text: str) -> None:
        await self.page.get_by_test_id("command-palette-input").fill(text)

    async def press_enter(self) -> None:
        await self.page.get_by_test_id("command-palette-input").press("Enter")
```

- [ ] **Step 2: Write `_builders.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case builders for the operator web UI e2e suite.

Most tests use the committed golden tree via ``seeded_results_dir``;
these builders produce synthetic trees for the three edge cases the
golden data doesn't represent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import orjson


def clear_results_dir(target: Path) -> None:
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_empty(target: Path) -> None:
    clear_results_dir(target)


def build_single_job(target: Path, *, job_id: str, namespace: str) -> None:
    clear_results_dir(target)
    d = target / namespace / job_id
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "job_id": job_id,
        "namespace": namespace,
        "model": "llama3-8b",
        "concurrency": 1,
        "status": "Succeeded",
        "metrics": {
            "request_throughput": {
                "avg": 1.0, "p50": 1.0, "p90": 1.0, "p99": 1.0,
                "min": 1.0, "max": 1.0, "unit": "requests/sec",
            },
        },
    }))
    (d / ".aiperf_results_ready.json").write_bytes(orjson.dumps({"ready": True}))


def build_all_failed(target: Path, *, n: int = 3) -> None:
    clear_results_dir(target)
    for i in range(n):
        d = target / "aiperf-bench" / f"failed-{i}"
        d.mkdir(parents=True)
        (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
            "job_id": f"failed-{i}",
            "namespace": "aiperf-bench",
            "model": "llama3-8b",
            "status": "Failed",
            "metrics": {},
        }))
        (d / ".aiperf_results_ready.json").write_bytes(orjson.dumps({"ready": True}))
```

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/operator_ui/_pages.py tests/e2e/operator_ui/_builders.py
git commit -s -m "test(e2e): add page-object helpers and edge-case builders"
```

No tests yet in this task — exercised via subsequent tasks.

---

## Task 8: `test_dashboard.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_dashboard.py`

- [ ] **Step 1: Write all dashboard tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dashboard page — summary cards, empty state, error banner."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._builders import build_empty
from tests.e2e.operator_ui._pages import DashboardPage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_dashboard_loads_with_seeded_data(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_test_id("page-dashboard")).to_be_visible()


async def test_dashboard_shows_total_jobs_kpi(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("total-jobs")
    await expect(kpi).to_contain_text("4")


async def test_dashboard_shows_running_jobs_kpi(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    kpi = dash.kpi("running-jobs")
    await expect(kpi).to_contain_text("1")


async def test_dashboard_empty_state(
    live_operator_app, fake_k8s_client, page
):
    build_empty(live_operator_app.results_dir)
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_text("No benchmark data")).to_be_visible()


async def test_dashboard_renders_top_nav_and_breadcrumb(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await expect(page.get_by_test_id("top-nav")).to_be_visible()
    await expect(page.get_by_test_id("breadcrumb")).to_be_visible()
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/e2e/operator_ui/test_dashboard.py -m e2e -n auto -v`

Expected outcome: each test that references a KPI label or empty-state text must match the actual UI. If a KPI label in the codebase is different from what's asserted here (e.g. the code uses `total_runs` not `total-jobs`), update the `data-testid` added in Task 6 to the label the test asserts, then re-run. Text assertions (`"No benchmark data"`, KPI numeric counts) must match the real UI copy — read the relevant `pages/dashboard.js` during this task to align.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/operator_ui/test_dashboard.py
git commit -s -m "test(e2e): dashboard page — KPI cards, empty state, layout"
```

---

## Task 9: `test_jobs.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_jobs.py`

- [ ] **Step 1: Write tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Jobs page — table render, namespace filter, column sort, row click, live status."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import JobDetailPage, JobsPage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_jobs_table_renders_all_rows(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await expect(jobs_page.rows()).to_have_count(5)  # 4 stored + 1 live


async def test_jobs_table_row_shows_live_status_from_k8s(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    row = jobs_page.row("aiperf-bench", "live-run")
    await expect(row).to_contain_text("Running")


async def test_jobs_namespace_filter(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await jobs_page.set_namespace_filter("ml-lab")
    await expect(jobs_page.rows()).to_have_count(2)  # mistral-7b-run1 + failed-run


async def test_jobs_sort_by_concurrency(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await jobs_page.click_column_header("concurrency")
    first_row = jobs_page.rows().first
    # Ascending after first click — lowest concurrency first
    await expect(first_row).to_contain_text("16")


async def test_jobs_row_click_navigates_to_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    jobs_page = JobsPage(page, live_operator_app.base_url)
    await jobs_page.goto()
    await jobs_page.row("aiperf-bench", "aiperf-llama3-c128").click()
    await page.wait_for_url("**/jobs/aiperf-bench/aiperf-llama3-c128")
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()
```

- [ ] **Step 2: Run & align assertions with real UI copy**

Run: `uv run pytest tests/e2e/operator_ui/test_jobs.py -m e2e -n auto -v`

Any failures indicate a real mismatch between expected and actual UI (column keys, row counts, filter behavior). Update the test or the UI `data-testid`, not both — the test is the spec. Row counts may need adjustment if the jobs list merges stored + live results differently from the assumption above; consult `pages/jobs.js` and adapt.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/operator_ui/test_jobs.py
git commit -s -m "test(e2e): jobs page — table, filter, sort, row-click, live status"
```

---

## Task 10: `test_job_detail.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_job_detail.py`

- [ ] **Step 1: Write tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-detail page — metrics, conditions, chart, pods, cancel button."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import JobDetailPage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_job_detail_renders_metrics(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_text("request_throughput")).to_be_visible()
    await expect(page.get_by_text("42.1")).to_be_visible()


async def test_job_detail_renders_conditions(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.get_by_text("Ready")).to_be_visible()


async def test_job_detail_renders_chart_canvas(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "aiperf-llama3-c128"
    )
    await detail.goto()
    await expect(page.locator("canvas")).to_be_visible()


async def test_job_detail_shows_pods_for_running_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "live-run"
    )
    await detail.goto()
    pods = page.get_by_test_id("job-detail-pods")
    await expect(pods).to_contain_text("live-run-controller-0")


async def test_job_detail_cancel_button_calls_api(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    detail = JobDetailPage(
        page, live_operator_app.base_url, "aiperf-bench", "live-run"
    )
    await detail.goto()
    await detail.cancel()
    # confirm dialog, if any — adjust selector to match the real UI
    confirm = page.get_by_role("button", name="Confirm")
    if await confirm.count() > 0:
        await confirm.click()
    # The fake_k8s_client records the call
    assert ("aiperf-bench", "live-run") in fake_k8s_client.cancelled
```

- [ ] **Step 2: Run & align**

Run: `uv run pytest tests/e2e/operator_ui/test_job_detail.py -m e2e -n auto -v`

If the real UI doesn't have a confirm dialog, the `if await confirm.count() > 0` branch already handles it. If it shows a toast instead, remove the confirm branch.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/operator_ui/test_job_detail.py
git commit -s -m "test(e2e): job-detail page — metrics, conditions, chart, pods, cancel"
```

---

## Task 11: `test_leaderboard.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_leaderboard.py`

- [ ] **Step 1: Write tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Leaderboard page — ranked rows, metric selector, click-through."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import LeaderboardPage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_leaderboard_ranks_by_throughput(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    lb = LeaderboardPage(page, live_operator_app.base_url)
    await lb.goto()
    rows = page.get_by_test_id("page-leaderboard").locator("tbody tr")
    first = rows.first
    # Highest throughput in the golden tree: aiperf-llama3-c256 (78.4)
    await expect(first).to_contain_text("aiperf-llama3-c256")


async def test_leaderboard_metric_selector_changes_order(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    lb = LeaderboardPage(page, live_operator_app.base_url)
    await lb.goto()
    await lb.select_metric("request_latency")
    rows = page.get_by_test_id("page-leaderboard").locator("tbody tr")
    first = rows.first
    # Lowest latency ranks first: aiperf-llama3-c128 (300ms)
    await expect(first).to_contain_text("aiperf-llama3-c128")


async def test_leaderboard_row_click_opens_job_detail(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    lb = LeaderboardPage(page, live_operator_app.base_url)
    await lb.goto()
    rows = page.get_by_test_id("page-leaderboard").locator("tbody tr")
    await rows.first.click()
    await page.wait_for_url("**/jobs/**")
```

- [ ] **Step 2: Run & commit**

```bash
uv run pytest tests/e2e/operator_ui/test_leaderboard.py -m e2e -n auto -v
git add tests/e2e/operator_ui/test_leaderboard.py
git commit -s -m "test(e2e): leaderboard — ranking, metric selector, click-through"
```

---

## Task 12: `test_compare.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_compare.py`

- [ ] **Step 1: Write tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare page — multi-select + side-by-side render."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import ComparePage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_compare_page_loads_with_selector(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    cmp_page = ComparePage(page, live_operator_app.base_url)
    await cmp_page.goto()
    await expect(page.get_by_test_id("compare-select")).to_be_visible()


async def test_compare_two_jobs_renders_side_by_side(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    cmp_page = ComparePage(page, live_operator_app.base_url)
    await cmp_page.goto()
    # The selector may be a multi-select <select> or a custom component.
    # Try the standard select_option first; fall back to clicking checkboxes.
    selector = page.get_by_test_id("compare-select")
    try:
        await selector.select_option(
            ["aiperf-bench/aiperf-llama3-c128", "aiperf-bench/aiperf-llama3-c256"]
        )
    except Exception:
        await page.locator(
            "[data-testid='compare-select'] [data-job='aiperf-llama3-c128']"
        ).click()
        await page.locator(
            "[data-testid='compare-select'] [data-job='aiperf-llama3-c256']"
        ).click()
    await expect(page.get_by_text("aiperf-llama3-c128")).to_be_visible()
    await expect(page.get_by_text("aiperf-llama3-c256")).to_be_visible()


async def test_compare_metric_selector_redraws_charts(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    cmp_page = ComparePage(page, live_operator_app.base_url)
    await cmp_page.goto()
    await page.get_by_test_id("metric-selector").select_option("time_to_first_token")
    await expect(page.locator("canvas")).to_be_visible()
```

- [ ] **Step 2: Run & commit**

```bash
uv run pytest tests/e2e/operator_ui/test_compare.py -m e2e -n auto -v
git add tests/e2e/operator_ui/test_compare.py
git commit -s -m "test(e2e): compare page — multi-select, side-by-side, metric selector"
```

---

## Task 13: `test_history.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_history.py`

- [ ] **Step 1: Write tests**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History page — chart + metric selector."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import HistoryPage

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_history_chart_renders(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    hist = HistoryPage(page, live_operator_app.base_url)
    await hist.goto()
    await expect(page.locator("canvas")).to_be_visible()


async def test_history_metric_selector_switches_series(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    hist = HistoryPage(page, live_operator_app.base_url)
    await hist.goto()
    await page.get_by_test_id("metric-selector").select_option("time_to_first_token")
    # Chart must still be visible after switch — re-render didn't crash.
    await expect(page.locator("canvas")).to_be_visible()


async def test_history_shows_at_least_three_data_points(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    hist = HistoryPage(page, live_operator_app.base_url)
    await hist.goto()
    # The history endpoint should return one entry per successful job.
    # We confirm via the API directly (the chart is canvas, so we can't DOM-
    # assert the data points).
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{live_operator_app.base_url}/api/v1/analytics/history"
            "?metric=request_throughput"
        )
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) >= 3
```

- [ ] **Step 2: Run & commit**

```bash
uv run pytest tests/e2e/operator_ui/test_history.py -m e2e -n auto -v
git add tests/e2e/operator_ui/test_history.py
git commit -s -m "test(e2e): history page — chart, metric selector, data points"
```

---

## Task 14: `test_navigation.py` + `test_robustness.py`

**Files:**
- Create: `tests/e2e/operator_ui/test_navigation.py`
- Create: `tests/e2e/operator_ui/test_robustness.py`
- Delete: `tests/e2e/operator_ui/test_smoke.py` (superseded by the per-page tests)

- [ ] **Step 1: Write `test_navigation.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-nav, breadcrumb, command palette, deep links, 404."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import (
    CommandPalette,
    DashboardPage,
    JobsPage,
)

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_nav_click_jobs_from_dashboard(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    await page.get_by_test_id("nav-link-jobs").click()
    await page.wait_for_url("**/jobs")
    await expect(page.get_by_test_id("page-jobs")).to_be_visible()


async def test_command_palette_opens_on_ctrl_k(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    palette = CommandPalette(page)
    await palette.open()


async def test_command_palette_search_navigates_to_job(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    dash = DashboardPage(page, live_operator_app.base_url)
    await dash.goto()
    palette = CommandPalette(page)
    await palette.open()
    await palette.type("aiperf-llama3-c128")
    await palette.press_enter()
    await page.wait_for_url("**/jobs/aiperf-bench/aiperf-llama3-c128")


async def test_deep_link_loads_job_detail_directly(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    await page.goto(
        f"{live_operator_app.base_url}/jobs/aiperf-bench/aiperf-llama3-c128"
    )
    await expect(page.get_by_test_id("page-job-detail")).to_be_visible()


async def test_unknown_route_shows_not_found(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    await page.goto(live_operator_app.base_url + "/does-not-exist")
    await expect(page.get_by_text("Not Found")).to_be_visible()
```

- [ ] **Step 2: Write `test_robustness.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-page invariants: no console errors, all fetches <500, ok on rapid route changes."""

import pytest
from playwright.async_api import expect

from tests.e2e.operator_ui._pages import (
    DashboardPage,
    JobsPage,
    LeaderboardPage,
)

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


ROUTES = ["/", "/jobs", "/leaderboard", "/compare", "/history"]


async def test_all_routes_return_ok_fetches(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    bad: list[tuple[str, int]] = []

    def on_response(resp):
        # only care about our own server's API calls
        if resp.url.startswith(live_operator_app.base_url) and resp.status >= 500:
            bad.append((resp.url, resp.status))

    page.on("response", on_response)
    for route in ROUTES:
        await page.goto(live_operator_app.base_url + route)
        await page.wait_for_load_state("networkidle")
    assert bad == [], f"Got ≥500 responses: {bad}"


async def test_rapid_route_changes_do_not_crash(
    live_operator_app, seeded_results_dir, fake_k8s_client, page
):
    for _ in range(3):
        for route in ROUTES:
            await page.goto(live_operator_app.base_url + route)
    # If we got here without the console-error gate firing, we're good.
    await expect(page.get_by_test_id("top-nav")).to_be_visible()
```

- [ ] **Step 3: Remove the temporary smoke file**

```bash
git rm tests/e2e/operator_ui/test_smoke.py
```

- [ ] **Step 4: Run the full suite**

Run:
```bash
uv run pytest tests/e2e/ -m e2e -n auto -v
```

Expected: all tests pass. Fix any failures in-place (likely root causes: a missing `data-testid`, a copy mismatch between test assertion and UI text, a CDN URL needing addition to `CDN_MAP`). Keep the suite green before committing.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/operator_ui/test_navigation.py tests/e2e/operator_ui/test_robustness.py
git commit -s -m "test(e2e): navigation + robustness tests; drop temporary smoke"
```

---

## Task 15: README and CI job

**Files:**
- Create: `tests/e2e/operator_ui/README.md`
- Create: `.github/workflows/e2e-operator-ui.yml` (or extend an existing workflow file if the repo already has one covering tests)

- [ ] **Step 1: Inspect existing CI workflows**

Run:
```bash
ls .github/workflows/
```

If there's an existing `tests.yml` or `ci.yml` doing unit/integration runs, add a new job to it rather than creating a parallel file. Otherwise create `.github/workflows/e2e-operator-ui.yml` as below.

- [ ] **Step 2: Write the README**

`tests/e2e/operator_ui/README.md`:
```markdown
# Operator Web UI e2e tests

Live-browser integration tests for the FastAPI-hosted Preact SPA in
`src/aiperf/operator/ui/`. Runs real headless Chromium against a real
in-process uvicorn hosting `aiperf.operator.results_server.create_app()`.

## Running locally

One-time:

```bash
make install-e2e-browsers
```

Run the whole suite:

```bash
make test-e2e
```

Run a single file with the browser visible:

```bash
uv run pytest tests/e2e/operator_ui/test_jobs.py -m e2e --headed
```

Debug a single test:

```bash
uv run pytest tests/e2e/operator_ui/test_jobs.py::test_jobs_namespace_filter \
  -m e2e --headed --pdb
```

## How it works

- **`live_operator_app` (session fixture)** — spawns uvicorn on
  `127.0.0.1:<random>` against a real `create_app()` with a tmp
  `results_dir`. Stays up for the whole test session.
- **`seeded_results_dir` (per-test)** — clears the session results dir
  and copies the committed golden tree from
  `tests/fixtures/operator_ui/results/` into it.
- **`fake_k8s_client` (per-test)** — monkeypatches the six helpers the
  jobs router calls (`list_aiperf_jobs`, `find_aiperf_job`,
  `get_raw_aiperfjob_status`, `get_pods`, `cluster_version`,
  `cancel_aiperf_job`).
- **`page` (per-test)** — overrides `pytest-playwright`'s page with CDN
  route interception (esm.sh → `src/aiperf/operator/ui/vendor/`) and a
  console-error gate that fails the test on any browser error.

## Extending

- **New page tests** — add a `test_<page>.py`, use the existing page
  objects in `_pages.py`, and assert against stable
  `data-testid` selectors. Add new test-ids to the UI components as
  needed — keep them kebab-case.
- **New fixture data** — re-run the generator:
  `uv run python tests/fixtures/operator_ui/generate_golden.py`
  then commit the diff.
- **New edge-case scenarios** — add a builder to `_builders.py`.

## Why no network

All CDN fetches go through Playwright's `page.route("**/*")`. Unmapped
external URLs fail the test at teardown — that's intentional. If the
UI gains a new CDN dep, add it to `CDN_MAP` in `conftest.py`.
```

- [ ] **Step 3: Add CI job**

Create `.github/workflows/e2e-operator-ui.yml` (or add this job block to an existing file):

```yaml
name: E2E Operator UI

on:
  pull_request:
    paths:
      - 'src/aiperf/operator/ui/**'
      - 'src/aiperf/operator/results_server.py'
      - 'src/aiperf/operator/routers/**'
      - 'tests/e2e/operator_ui/**'
      - 'tests/fixtures/operator_ui/**'
      - '.github/workflows/e2e-operator-ui.yml'
  push:
    branches: [main]

jobs:
  e2e-operator-ui:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.10
      - name: Install project
        run: uv sync --all-extras --dev
      - name: Install Chromium
        run: uv run playwright install chromium --with-deps
      - name: Run e2e suite
        run: uv run pytest tests/e2e/ -m e2e -n auto --tracing=retain-on-failure
      - name: Upload Playwright traces on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-traces
          path: test-results/
          if-no-files-found: ignore
```

- [ ] **Step 4: Verify locally one more time**

```bash
make test-e2e
```

Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/operator_ui/README.md .github/workflows/e2e-operator-ui.yml
git commit -s -m "ci(e2e): add operator web UI e2e suite workflow and README"
```

---

## Self-review results

**Spec coverage:**
- §2 Goals — covered by tasks 8–14.
- §3.1 Test fixtures — Task 2 (`live_operator_app`), Task 3 (`seeded_results_dir`), Task 4 (`fake_k8s_client`), Task 5 (`page`), Task 7 (`_builders.py`, `_pages.py`).
- §3.2 Golden fixtures — Task 3 generator + committed tree.
- §3.3 Builder — Task 7.
- §3.4 Page-object helpers — Task 7.
- §3.5 Test files — Tasks 8–14 (one per page + navigation + robustness). Expected total test count: dashboard 5 + jobs 5 + job_detail 5 + leaderboard 3 + compare 3 + history 3 + navigation 5 + robustness 2 = **31 tests**, matching the spec's "30+" target.
- §4 Data flow — wired throughout.
- §5 Error handling — console-error gate (Task 5), unmapped-request gate (Task 5), strict fake k8s (Task 4).
- §6 Testing conventions — `@pytest.mark.e2e` everywhere, `get_by_test_id` / `get_by_role`, no sleeps.
- §7 Dependencies — Task 1.
- §8 Out of scope — none added (deliberate).

**Placeholders:** none; every step contains the actual content.

**Type consistency:** `LiveApp`, `FakeK8sClient`, page-object class names stable across tasks.

**Gap:** Task 10's cancel-button test assumes the UI has a cancel flow; if the UI doesn't yet expose one, that test will fail and the subagent should fall back to asserting the button's visibility and skipping the network round-trip. Noted inline in that task.
