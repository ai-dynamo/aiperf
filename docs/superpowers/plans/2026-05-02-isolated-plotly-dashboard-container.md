# Isolated Plotly Dashboard Container — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the in-process Plotly Dash mount out of `results-server` into a dedicated, opt-in, third sidecar container in the operator Pod. Single externally-visible URL (proxy via `results-server`), refresh on completion, default off.

**Architecture:** New container runs `python -m aiperf.operator.dashboard_server` — a tiny FastAPI app exposing `/healthz`, `/dashboard/*` (WSGI-mounted Dash), and `POST /admin/refresh`. `results-server` reverse-proxies `/dashboard/*` to `localhost:8082` via httpx-stream. `try_claim_completion` fires a best-effort POST to `/admin/refresh` after claiming a job. SPA's `/api/v1/config/features` flag gates the top-nav "Plots ↗" link.

**Tech Stack:** FastAPI, uvicorn, httpx (stream proxy), Dash 3.1, pydantic-settings, htm/preact (SPA), Helm.

**Spec:** `docs/superpowers/specs/2026-05-02-isolated-plotly-dashboard-container-design.md`

---

## File Structure

**Created:**
- `src/aiperf/operator/dashboard_server.py` — new container's entry point and FastAPI app
- `src/aiperf/operator/routers/dashboard_proxy.py` — httpx-streamed reverse proxy mounted into `results-server`
- `tests/unit/operator/test_dashboard_server.py` — startup, refresh, idempotency
- `tests/unit/operator/test_dashboard_proxy_router.py` — proxy forwarding, 503 paths
- `tests/unit/operator/test_completion_refresh_post.py` — `try_claim_completion` POSTs refresh

**Modified:**
- `src/aiperf/operator/environment.py` — new `_DashboardSettings` block on `_OperatorEnvironment`
- `src/aiperf/operator/results_server.py` — drop `_mount_dashboard`, mount the new proxy router
- `src/aiperf/operator/client_cache.py::try_claim_completion` — add fire-and-forget POST
- `src/aiperf/operator/routers/config.py` — add `/features` endpoint with `dashboard_enabled`
- `src/aiperf/operator/ui-v1/app.js` — fetch `/api/v1/config/features` at boot, pass to TopNav
- `src/aiperf/operator/ui-v1/components/top-nav.js` — conditional `Plots ↗` external link
- `deploy/helm/aiperf-operator/values.yaml` — new `dashboard:` block
- `deploy/helm/aiperf-operator/templates/deployment.yaml` — third container + env wiring
- `deploy/helm/aiperf-operator/templates/_helpers.tpl` — helper for dashboard env block
- `tests/kubernetes/test_helm_dashboard.py` — chart-render assertions (new file, not in `test_helm.py` because that's an in-cluster integration suite)
- `tests/unit/api/test_config_router.py` — `dashboard_enabled` reflects env
- `docs/kubernetes/dashboard-ui.md` — opt-in dashboard sidecar section
- `docs/kubernetes/configuration.md` — values reference
- `docs/kubernetes/sidecars.md` — add the new container
- `docs/superpowers/plans/2026-05-02-isolated-plotly-dashboard-container.md` — this doc

---

## Task 1: Add `_DashboardSettings` to operator environment

**Files:**
- Modify: `src/aiperf/operator/environment.py`
- Test: `tests/unit/operator/test_environment_dashboard_settings.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/operator/test_environment_dashboard_settings.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DASHBOARD nested settings on _OperatorEnvironment."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("env", "expected_port", "expected_enabled"),
    [
        ({}, 0, False),
        ({"AIPERF_DASHBOARD_PORT": "8082"}, 8082, False),
        (
            {"AIPERF_DASHBOARD_PORT": "8082", "AIPERF_DASHBOARD_PROXY_ENABLED": "1"},
            8082,
            True,
        ),
        (
            {"AIPERF_DASHBOARD_PROXY_ENABLED": "true"},
            0,
            True,
        ),
    ],
)
def test_dashboard_settings_load_from_env(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected_port: int,
    expected_enabled: bool,
) -> None:
    for k in ("AIPERF_DASHBOARD_PORT", "AIPERF_DASHBOARD_PROXY_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from aiperf.operator import environment as mod

    importlib.reload(mod)
    assert mod.OperatorEnvironment.DASHBOARD.PORT == expected_port
    assert mod.OperatorEnvironment.DASHBOARD.PROXY_ENABLED is expected_enabled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_environment_dashboard_settings.py -n auto -v`
Expected: FAIL — `AttributeError: 'OperatorEnvironment' object has no attribute 'DASHBOARD'`

- [ ] **Step 3: Implement `_DashboardSettings` and wire into root**

Edit `src/aiperf/operator/environment.py`. Add a new settings class above `_OperatorEnvironment` (place it next to `_SweepControllerSettings`):

```python
class _DashboardSettings(BaseSettings):
    """Plotly Dashboard sidecar wiring (operator + results-server).

    The dashboard is an opt-in third container in the operator Pod;
    these settings let other containers locate it.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DASHBOARD_",
    )

    PORT: int = Field(
        default=0,
        ge=0,
        le=65535,
        description="Pod-local HTTP port the dashboard sidecar listens on. "
        "0 means the sidecar is disabled / absent. results-server uses this "
        "to reverse-proxy /dashboard/*; the operator uses it to fire "
        "fire-and-forget refresh POSTs after a benchmark completion claim.",
    )
    PROXY_ENABLED: bool = Field(
        default=False,
        description="When true, results-server forwards /dashboard/* to the "
        "sidecar at localhost:PORT and the SPA shows the 'Plots ↗' top-nav "
        "entry. When false, /dashboard/* returns 503 and the link is hidden. "
        "Set independently from PORT so a misconfigured chart fails closed.",
    )
```

Then in `_OperatorEnvironment`, append a new field:

```python
    DASHBOARD: _DashboardSettings = Field(
        default_factory=_DashboardSettings,
        description="Plotly Dashboard sidecar wiring.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_environment_dashboard_settings.py -n auto -v`
Expected: PASS — all 4 parametrized cases.

- [ ] **Step 5: Run pre-commit + full unit suite**

Run:
```bash
ruff format . && ruff check --fix .
uv run pytest -n auto tests/unit/
```
Expected: PASS, no new failures vs `origin/main` baseline.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/environment.py tests/unit/operator/test_environment_dashboard_settings.py
git commit -s -m "feat(operator): add _DashboardSettings (PORT, PROXY_ENABLED) to OperatorEnvironment"
```

---

## Task 2: Create `dashboard_server.py` module shell with `/healthz`

**Files:**
- Create: `src/aiperf/operator/dashboard_server.py`
- Test: `tests/unit/operator/test_dashboard_server.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/operator/test_dashboard_server.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone dashboard sidecar HTTP app."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_healthz_returns_200(tmp_path: Path) -> None:
    from aiperf.operator.dashboard_server import create_app

    app = create_app(results_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiperf.operator.dashboard_server'`.

- [ ] **Step 3: Create the module shell**

```python
# src/aiperf/operator/dashboard_server.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Plotly Dash sidecar for the operator Pod.

Lives as a third container alongside the kopf operator and the
``results-server`` sidecar. Exposes:

    GET  /healthz          - liveness + readiness target
    GET  /dashboard/*      - WSGI-mounted Dash app (mounted in Task 3)
    POST /admin/refresh    - hot-swap rebuild trigger (mounted in Task 4)

results-server reverse-proxies /dashboard/* to localhost:<PORT> so the
external request path stays single-origin.

Run: ``python -m aiperf.operator.dashboard_server``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/data"))


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the dashboard sidecar FastAPI app.

    Args:
        results_dir: Root of the results PVC. Defaults to ``RESULTS_DIR``.
    """
    base_dir = results_dir or RESULTS_DIR
    app = FastAPI(
        title="AIPerf Dashboard Sidecar",
        description="Hosts the Plotly Dash app at /dashboard/.",
        version="1.0.0",
    )

    # Used by later tasks (mount + refresh).
    app.state.results_dir = base_dir

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    """Run the dashboard sidecar."""
    from aiperf.operator.environment import OperatorEnvironment

    port = OperatorEnvironment.DASHBOARD.PORT or 8082
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/dashboard_server.py tests/unit/operator/test_dashboard_server.py
git commit -s -m "feat(operator): add dashboard_server module shell with /healthz"
```

---

## Task 3: Mount Plotly Dash inside the dashboard sidecar

**Files:**
- Modify: `src/aiperf/operator/dashboard_server.py`
- Test: `tests/unit/operator/test_dashboard_server.py`

- [ ] **Step 1: Add failing test for `/dashboard/` placeholder**

Append to `tests/unit/operator/test_dashboard_server.py`:

```python
def test_dashboard_route_serves_placeholder_when_pvc_empty(tmp_path: Path) -> None:
    from aiperf.operator.dashboard_server import create_app

    app = create_app(results_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 503
    assert b"Dashboard" in resp.content


def test_dashboard_route_serves_real_app_when_runs_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hot-swap path: when the build returns a Dash app, requests reach it."""
    from aiperf.operator import dashboard_server

    class _StubDashServer:
        def __call__(self, environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"dashboard-served"]

    class _StubDash:
        server = _StubDashServer()

    def _fake_build(results_dir):
        return _StubDash(), 1

    monkeypatch.setattr(dashboard_server, "build_dashboard", _fake_build)

    app = dashboard_server.create_app(results_dir=tmp_path)
    # Trigger the startup task synchronously by entering the TestClient context
    # then waiting for the swap. The swap is in-process and fast in tests.
    with TestClient(app) as client:
        # Drain the startup task before issuing the request.
        import asyncio
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))
        resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert resp.content == b"dashboard-served"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v -k dashboard_route`
Expected: FAIL — 404 (route not mounted).

- [ ] **Step 3: Mount the Dash WSGI app + placeholder + startup-build**

Replace the body of `create_app` and add helpers in `src/aiperf/operator/dashboard_server.py`:

```python
import asyncio
from typing import TYPE_CHECKING

from fastapi.middleware.wsgi import WSGIMiddleware

from aiperf.operator.dashboard_mount import DashboardProxy, build_dashboard

if TYPE_CHECKING:
    from collections.abc import Iterable


def _pending_dashboard_app(message: bytes):
    """WSGI stub returning 503 with a friendly body until the build lands."""

    def _app(environ, start_response):
        start_response(
            "503 Service Unavailable",
            [("Content-Type", "text/plain; charset=utf-8")],
        )
        return [message]

    return _app


def _initial_dashboard_proxy() -> DashboardProxy:
    return DashboardProxy(
        _pending_dashboard_app(b"Dashboard is initializing; retry shortly.")
    )


async def _build_and_swap(proxy: DashboardProxy, base_dir: Path) -> None:
    try:
        dash_app, run_count = await asyncio.to_thread(build_dashboard, base_dir)
    except OSError as exc:
        logger.warning(
            "Dashboard init failed (likely read-only rootfs): %s", exc
        )
        proxy.app = _pending_dashboard_app(
            b"Dashboard unavailable: read-only filesystem blocked plot config."
        )
        return
    except Exception:
        logger.exception("Dashboard init failed; keeping placeholder mounted")
        proxy.app = _pending_dashboard_app(
            b"Dashboard unavailable: initialization failed."
        )
        return

    if dash_app is None:
        logger.info("No runs on PVC yet; /dashboard/ returns 503 until runs exist")
        proxy.app = _pending_dashboard_app(
            b"Dashboard not yet available: no completed runs on PVC."
        )
        return

    logger.info("Mounting Plotly Dash dashboard with %d runs", run_count)
    proxy.app = dash_app.server


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the dashboard sidecar FastAPI app."""
    base_dir = results_dir or RESULTS_DIR
    app = FastAPI(
        title="AIPerf Dashboard Sidecar",
        description="Hosts the Plotly Dash app at /dashboard/.",
        version="1.0.0",
    )
    app.state.results_dir = base_dir
    proxy = _initial_dashboard_proxy()
    app.state.dashboard_proxy = proxy
    app.mount("/dashboard", WSGIMiddleware(proxy))

    @app.on_event("startup")
    async def _start_initial_build() -> None:
        asyncio.create_task(_build_and_swap(proxy, base_dir))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v`
Expected: PASS for the placeholder test. The hot-swap test races the startup task — if flaky in xdist, mark it `@pytest.mark.flaky` or rewrite to call `_build_and_swap(proxy, tmp_path)` directly without the TestClient lifespan. Prefer the direct call:

```python
def test_dashboard_route_serves_real_app_when_runs_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from aiperf.operator import dashboard_server

    class _StubDashServer:
        def __call__(self, environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"dashboard-served"]

    class _StubDash:
        server = _StubDashServer()

    monkeypatch.setattr(
        dashboard_server, "build_dashboard", lambda _d: (_StubDash(), 1)
    )

    app = dashboard_server.create_app(results_dir=tmp_path)
    proxy = app.state.dashboard_proxy
    asyncio.run(dashboard_server._build_and_swap(proxy, tmp_path))

    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert resp.content == b"dashboard-served"
```

Re-run: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/dashboard_server.py tests/unit/operator/test_dashboard_server.py
git commit -s -m "feat(operator): mount Plotly Dash + initial build in dashboard sidecar"
```

---

## Task 4: Add `POST /admin/refresh` with idempotent rebuild

**Files:**
- Modify: `src/aiperf/operator/dashboard_server.py`
- Test: `tests/unit/operator/test_dashboard_server.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/operator/test_dashboard_server.py`:

```python
def test_refresh_returns_202_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First refresh kicks off a rebuild and returns 202."""
    from aiperf.operator import dashboard_server

    monkeypatch.setattr(
        dashboard_server, "build_dashboard", lambda _d: (None, 0)
    )
    app = dashboard_server.create_app(results_dir=tmp_path)
    app.state.dashboard_refresh_inflight = False

    with TestClient(app) as client:
        resp = client.post("/admin/refresh")
    assert resp.status_code == 202
    assert resp.json() == {"status": "rebuilding"}


def test_refresh_short_circuits_when_inflight(tmp_path: Path) -> None:
    """A second concurrent refresh sees inflight=True and returns 200."""
    from aiperf.operator import dashboard_server

    app = dashboard_server.create_app(results_dir=tmp_path)
    # Simulate a rebuild in progress.
    app.state.dashboard_refresh_inflight = True

    with TestClient(app) as client:
        resp = client.post("/admin/refresh")
    assert resp.status_code == 200
    assert resp.json() == {"status": "already_rebuilding"}


def test_refresh_swaps_proxy_app_after_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct invocation of _build_and_swap installs the new Dash app."""
    import asyncio

    from aiperf.operator import dashboard_server

    class _StubDashServer:
        def __call__(self, environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"refreshed"]

    class _StubDash:
        server = _StubDashServer()

    monkeypatch.setattr(
        dashboard_server, "build_dashboard", lambda _d: (_StubDash(), 1)
    )

    app = dashboard_server.create_app(results_dir=tmp_path)
    proxy = app.state.dashboard_proxy
    asyncio.run(dashboard_server._build_and_swap(proxy, tmp_path))

    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert resp.content == b"refreshed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v -k refresh`
Expected: FAIL — `/admin/refresh` returns 405 (route not registered).

- [ ] **Step 3: Implement the refresh route**

In `src/aiperf/operator/dashboard_server.py`, modify `create_app` to set up an inflight flag and register the route:

```python
def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the dashboard sidecar FastAPI app."""
    base_dir = results_dir or RESULTS_DIR
    app = FastAPI(
        title="AIPerf Dashboard Sidecar",
        description="Hosts the Plotly Dash app at /dashboard/.",
        version="1.0.0",
    )
    app.state.results_dir = base_dir
    app.state.dashboard_refresh_inflight = False
    proxy = _initial_dashboard_proxy()
    app.state.dashboard_proxy = proxy
    app.mount("/dashboard", WSGIMiddleware(proxy))

    @app.on_event("startup")
    async def _start_initial_build() -> None:
        asyncio.create_task(_build_and_swap(proxy, base_dir))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/admin/refresh")
    async def refresh() -> JSONResponse:
        if app.state.dashboard_refresh_inflight:
            return JSONResponse(
                status_code=200, content={"status": "already_rebuilding"}
            )
        app.state.dashboard_refresh_inflight = True

        async def _refresh_task() -> None:
            try:
                await _build_and_swap(proxy, base_dir)
            finally:
                app.state.dashboard_refresh_inflight = False

        asyncio.create_task(_refresh_task())
        return JSONResponse(status_code=202, content={"status": "rebuilding"})

    return app
```

Add the import at the top of the module:

```python
from fastapi.responses import JSONResponse
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_dashboard_server.py -n auto -v`
Expected: PASS. (If the idempotency test still flakes due to the synchronous `asyncio.run` inside the monkeypatched `build_dashboard`, swap to `time.sleep(0.5)` on the build path — the assertion is solely about the second call returning `already_rebuilding` while the first is in flight.)

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/dashboard_server.py tests/unit/operator/test_dashboard_server.py
git commit -s -m "feat(operator): add idempotent /admin/refresh to dashboard sidecar"
```

---

## Task 5: Wire dashboard refresh trigger into `try_claim_completion`

**Files:**
- Modify: `src/aiperf/operator/client_cache.py`
- Test: `tests/unit/operator/test_completion_refresh_post.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/operator/test_completion_refresh_post.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dashboard-refresh fire-and-forget call from try_claim_completion."""

from __future__ import annotations

import importlib
from typing import Any

import httpx
import pytest


@pytest.fixture
def reload_env_with_dashboard_port(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", "8082")
    monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    yield env_mod


@pytest.mark.asyncio
async def test_post_dashboard_refresh_succeeds(
    monkeypatch: pytest.MonkeyPatch, reload_env_with_dashboard_port
) -> None:
    """When PORT is set, post_dashboard_refresh hits localhost:<port>/admin/refresh."""
    from aiperf.operator import client_cache

    seen: dict[str, str] = {}

    async def _fake_post(self, url, **_kwargs):
        seen["url"] = url

        class _Resp:
            status_code = 202

        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    await client_cache._post_dashboard_refresh()
    assert seen["url"] == "http://localhost:8082/admin/refresh"


@pytest.mark.asyncio
async def test_post_dashboard_refresh_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, reload_env_with_dashboard_port
) -> None:
    """httpx errors must not propagate out of the helper."""
    from aiperf.operator import client_cache

    async def _broken_post(self, *_args, **_kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "post", _broken_post)

    # Must not raise.
    await client_cache._post_dashboard_refresh()


@pytest.mark.asyncio
async def test_post_dashboard_refresh_skipped_when_port_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIPERF_DASHBOARD_PORT", raising=False)
    monkeypatch.delenv("AIPERF_DASHBOARD_PROXY_ENABLED", raising=False)

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)

    from aiperf.operator import client_cache

    posted = False

    async def _fake_post(self, *_args, **_kwargs):
        nonlocal posted
        posted = True

        class _Resp:
            status_code = 202

        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    await client_cache._post_dashboard_refresh()
    assert posted is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_completion_refresh_post.py -n auto -v`
Expected: FAIL — `AttributeError: module 'aiperf.operator.client_cache' has no attribute '_post_dashboard_refresh'`.

- [ ] **Step 3: Implement helper + wire into `try_claim_completion`**

Add to top of `src/aiperf/operator/client_cache.py` (alongside existing imports):

```python
import httpx
```

Add a new private helper near the bottom (above `_submit_claim_patch`):

```python
async def _post_dashboard_refresh() -> None:
    """Fire-and-forget POST to the dashboard sidecar's /admin/refresh.

    Called after a successful completion claim so the Plotly Dash view
    picks up the new run on the PVC. Best-effort: failures (sidecar off,
    dashboard disabled, port unreachable) are logged at debug and
    swallowed — refresh is not load-bearing.
    """
    from aiperf.operator.environment import OperatorEnvironment

    port = OperatorEnvironment.DASHBOARD.PORT
    if port <= 0:
        return
    url = f"http://localhost:{port}/admin/refresh"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(url)
    except (httpx.HTTPError, OSError) as exc:
        logger.debug("dashboard refresh skipped: %s", exc)
```

Then in `try_claim_completion`, wire it in on the win path. Modify the section:

```python
    claimed = await _submit_claim_patch(namespace, name, patch_ops)
    if claimed is True:
        _shutdown_sent.add(key)
        await _post_dashboard_refresh()
        return True
```

(Ensure `logger` is already defined at module level — it is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_completion_refresh_post.py -n auto -v`
Expected: PASS for all 3 cases.

Run: `uv run pytest tests/unit/operator/test_client_cache.py -n auto -v` (existing tests must still pass — the helper is opt-in via env).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/client_cache.py tests/unit/operator/test_completion_refresh_post.py
git commit -s -m "feat(operator): post dashboard /admin/refresh on completion claim"
```

---

## Task 6: Add `dashboard_proxy` router; remove `_mount_dashboard` from results_server

**Files:**
- Create: `src/aiperf/operator/routers/dashboard_proxy.py`
- Modify: `src/aiperf/operator/results_server.py`
- Test: `tests/unit/operator/test_dashboard_proxy_router.py` (new)
- Test (modify): `tests/unit/operator/test_results_server.py`

- [ ] **Step 1: Write the failing test for the new router**

```python
# tests/unit/operator/test_dashboard_proxy_router.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dashboard-proxy reverse-proxy router mounted in results-server."""

from __future__ import annotations

import importlib

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, port: int) -> FastAPI:
    if enabled:
        monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    else:
        monkeypatch.delenv("AIPERF_DASHBOARD_PROXY_ENABLED", raising=False)
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", str(port))

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import dashboard_proxy

    importlib.reload(dashboard_proxy)

    app = FastAPI()
    app.include_router(dashboard_proxy.create_dashboard_proxy_router())
    return app


def test_proxy_returns_503_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=False, port=0)
    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 503
    assert b"disabled" in resp.content.lower()


def test_proxy_forwards_to_localhost_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxy passes method, path, and body to localhost:<port> and streams the response back."""
    app = _make_app(monkeypatch, enabled=True, port=8082)

    captured: dict[str, object] = {}

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aiter_raw(self):
            yield b'{"hello": "world"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

    class _FakeStream:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            captured["entered"] = True
            return _FakeResp()

        async def __aexit__(self, *_a):
            return None

    def _fake_stream(self, method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        return _FakeStream()

    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream)

    with TestClient(app) as client:
        resp = client.get("/dashboard/foo/bar?x=1")

    assert resp.status_code == 200
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8082/dashboard/foo/bar?x=1"
    assert b'"hello": "world"' in resp.content


def test_proxy_returns_503_when_upstream_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, enabled=True, port=8082)

    def _fake_stream(self, *_a, **_kw):
        raise httpx.ConnectError("upstream down")

    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream)

    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 503
    assert b"unreachable" in resp.content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/operator/test_dashboard_proxy_router.py -n auto -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiperf.operator.routers.dashboard_proxy'`.

- [ ] **Step 3: Implement the proxy router**

```python
# src/aiperf/operator/routers/dashboard_proxy.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reverse proxy from results-server's /dashboard/* to the dashboard sidecar.

The proxy is mounted on the results-server FastAPI app. It forwards
method, path, query, body, and most headers (drops ``host`` and lets
httpx re-set ``content-length``) to ``http://localhost:<PORT>/dashboard/...``
and streams the upstream response back.

When the toggle is off (``AIPERF_DASHBOARD_PROXY_ENABLED`` falsy), the
route returns 503 with a friendly body so the SPA's "Plots ↗" link
fails clearly instead of 404'ing.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# Hop-by-hop and otherwise-unsafe headers we don't forward upstream.
_FORWARD_REQUEST_HEADER_DROP = frozenset(
    {"host", "content-length", "connection", "transfer-encoding"}
)
_FORWARD_RESPONSE_HEADER_DROP = frozenset(
    {"content-encoding", "transfer-encoding", "connection"}
)


def create_dashboard_proxy_router() -> APIRouter:
    """Create the ``/dashboard/{path:path}`` proxy router.

    Reads ``OperatorEnvironment.DASHBOARD`` at request time so a toggle
    flip does not require a reload (env reload is the test concern,
    not prod — but reading-on-each-request is cheap).
    """
    from aiperf.operator.environment import OperatorEnvironment

    router = APIRouter(tags=["dashboard"])

    @router.api_route(
        "/dashboard/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def proxy(path: str, request: Request) -> Response:
        if not OperatorEnvironment.DASHBOARD.PROXY_ENABLED:
            return Response(
                content=b"Dashboard is disabled on this cluster.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

        port = OperatorEnvironment.DASHBOARD.PORT
        if port <= 0:
            return Response(
                content=b"Dashboard is disabled on this cluster.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

        upstream_url = f"http://localhost:{port}/dashboard/{path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _FORWARD_REQUEST_HEADER_DROP
        }
        body = await request.body()

        try:
            client = httpx.AsyncClient(timeout=30.0)
            stream_ctx = client.stream(
                request.method,
                upstream_url,
                headers=forward_headers,
                content=body,
            )
            async with stream_ctx as upstream:
                response_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in _FORWARD_RESPONSE_HEADER_DROP
                }
                content = b""
                async for chunk in upstream.aiter_raw():
                    content += chunk
            await client.aclose()
            return Response(
                content=content,
                status_code=upstream.status_code,
                headers=response_headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("dashboard upstream unreachable: %s", exc)
            return Response(
                content=b"Dashboard sidecar is unreachable.",
                status_code=503,
                media_type="text/plain; charset=utf-8",
            )

    return router
```

(Buffered-then-returned response is fine for v1 — Dash callbacks are JSON, small, and async-streaming a `StreamingResponse` over `httpx.stream` adds bug surface around lifecycle. If profiling later shows large payloads being slow, swap to `StreamingResponse` with a generator that closes the client on completion.)

- [ ] **Step 4: Run proxy tests to verify they pass**

Run: `uv run pytest tests/unit/operator/test_dashboard_proxy_router.py -n auto -v`
Expected: PASS for all 3 cases.

- [ ] **Step 5: Replace `_mount_dashboard` with the new router in results-server**

Edit `src/aiperf/operator/results_server.py`:

1. Remove the `_mount_dashboard` function (the entire block from `def _pending_dashboard_app` through the end of `_mount_dashboard`).
2. Remove the WSGI middleware import: `from fastapi.middleware.wsgi import WSGIMiddleware`.
3. Remove the `dashboard_mount` re-export — the rest of the codebase no longer imports it from here, and the new sidecar imports it directly:

```python
# DELETE these lines:
from aiperf.operator.dashboard_mount import DashboardProxy, build_dashboard
# ...and from __all__:
#     "DashboardProxy",
#     "build_dashboard",
```

4. In `create_app`, replace the `_mount_dashboard(app, base_dir)` call with mounting the new router:

```python
    from aiperf.operator.routers.dashboard_proxy import (
        create_dashboard_proxy_router,
    )

    app.include_router(create_dashboard_proxy_router())
```

- [ ] **Step 6: Update `test_results_server.py`**

Open `tests/unit/operator/test_results_server.py` and find the `monkeypatch.setattr(results_server, "build_dashboard", slow_build)` test (line ~262). Replace its body to assert the proxy path instead — the dashboard is no longer in-process inside results_server, so the test must verify the route registration:

```python
def test_results_server_mounts_dashboard_proxy_route(tmp_path: Path) -> None:
    """results-server registers the /dashboard/{path:path} proxy route."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/dashboard/{path:path}" in paths
```

Remove any `slow_build` / `monkeypatch.setattr(results_server, "build_dashboard", ...)` setup that's now dead.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/unit/operator/test_results_server.py tests/unit/operator/test_dashboard_proxy_router.py -n auto -v`
Expected: PASS.

Run: `uv run pytest -n auto tests/unit/`
Expected: PASS, no new failures.

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/operator/routers/dashboard_proxy.py \
        src/aiperf/operator/results_server.py \
        tests/unit/operator/test_dashboard_proxy_router.py \
        tests/unit/operator/test_results_server.py
git commit -s -m "refactor(operator): replace in-process Dash mount with proxy to sidecar"
```

---

## Task 7: Expose `dashboard_enabled` via `/api/v1/config/features`

**Files:**
- Modify: `src/aiperf/operator/routers/config.py`
- Test: `tests/unit/api/test_config_router.py`

- [ ] **Step 1: Find existing test file & write failing test**

Locate the existing config-router test (it may not exist yet — `find tests -name 'test_config_router*'`). If absent, create:

```python
# tests/unit/api/test_config_router.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator's /api/v1/config/* endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch: pytest.MonkeyPatch, **env: str) -> TestClient:
    for k in ("AIPERF_DASHBOARD_PROXY_ENABLED", "AIPERF_DASHBOARD_PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import config as config_mod

    importlib.reload(config_mod)
    app = FastAPI()
    app.include_router(config_mod.create_config_router())
    return TestClient(app)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"AIPERF_DASHBOARD_PROXY_ENABLED": "1"}, True),
        ({"AIPERF_DASHBOARD_PROXY_ENABLED": "0"}, False),
    ],
)
def test_features_endpoint_reflects_dashboard_proxy_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    client = _client(monkeypatch, **env)
    resp = client.get("/api/v1/config/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dashboard_enabled"] is expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_config_router.py -n auto -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Add the `/features` endpoint**

Edit `src/aiperf/operator/routers/config.py`. Append a new model and route:

```python
class FeaturesResponse(BaseModel):
    """Boot-time feature flags the SPA needs to gate top-nav entries."""

    dashboard_enabled: bool = Field(
        description="Whether the Plotly Dash sidecar is wired up. When true, "
        "the SPA shows the 'Plots ↗' top-nav link pointing at /dashboard/. "
        "Reflects AIPERF_DASHBOARD_PROXY_ENABLED on the results-server "
        "container, which Helm sets only when dashboard.enabled=true."
    )
```

In `create_config_router`, add inside the function body:

```python
    @router.get("/features", response_model=FeaturesResponse)
    async def get_features() -> FeaturesResponse:
        return FeaturesResponse(
            dashboard_enabled=OperatorEnvironment.DASHBOARD.PROXY_ENABLED,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/api/test_config_router.py -n auto -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/routers/config.py tests/unit/api/test_config_router.py
git commit -s -m "feat(operator): expose dashboard_enabled via /api/v1/config/features"
```

---

## Task 8: SPA — fetch features at boot, render conditional `Plots ↗`

**Files:**
- Modify: `src/aiperf/operator/ui-v1/app.js`
- Modify: `src/aiperf/operator/ui-v1/components/top-nav.js`

(No new TDD test; SPA changes are exercised by the e2e in Task 11. We do a quick unit-style check via the existing js tests if present — `tests/unit/api/test_dashboard_js.py` may be applicable.)

- [ ] **Step 1: Plumb features through TopNav**

Edit `src/aiperf/operator/ui-v1/components/top-nav.js` — replace the static `NAV_GROUPS` const with a function that takes a `features` object, and accept a `features` prop on `TopNav`:

```javascript
import { html } from 'htm/preact';
import { route, navigate } from '../lib/router.js';

const PRIMARY_GROUP = {
  items: [
    { path: '/', label: 'Dashboard' },
    { path: '/jobs', label: 'Jobs' },
    { path: '/sweeps', label: 'Sweeps' },
    { path: '/launch', label: 'Launch' },
  ],
};

const ANALYTICS_GROUP = {
  items: [
    { path: '/leaderboard', label: 'Leaderboard' },
    { path: '/compare', label: 'Compare' },
    { path: '/history', label: 'History' },
  ],
};

function buildNavGroups(features) {
  const groups = [PRIMARY_GROUP, ANALYTICS_GROUP];
  if (features && features.dashboard_enabled) {
    groups.push({
      items: [
        {
          path: '/dashboard/',
          label: 'Plots ↗',
          external: true,
          testId: 'nav-link-plots',
        },
      ],
    });
  }
  return groups;
}

function isActive(itemPath, currentRoute) {
  if (itemPath === '/') return currentRoute === '/' || currentRoute === '';
  return currentRoute.startsWith(itemPath);
}

function routeSlug(path) {
  if (path === '/' || path === '') return 'dashboard';
  return path.replace(/^\//, '').replace(/\/$/, '').replace(/\//g, '-');
}

export function TopNav({ onSearchClick, features }) {
  const currentRoute = route.value;
  const navGroups = buildNavGroups(features);

  return html`
    <header class="topbar" data-testid="top-nav">
      <div class="topbar-left">
        <div class="logo">
          <div class="logo-icon">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          AIPerf Operator
        </div>
        <nav class="nav" aria-label="Main navigation">
          ${navGroups.map((group, gi) => html`
            ${gi > 0 && html`<span class="nav-sep" />`}
            ${group.items.map((item) => item.external ? html`
              <a
                key=${item.path}
                href=${item.path}
                target="_blank"
                rel="noopener"
                class="nav-tab"
                data-testid=${item.testId || ('nav-link-' + routeSlug(item.path))}
              >
                ${item.label}
              </a>
            ` : html`
              <button
                key=${item.path}
                class=${'nav-tab' + (isActive(item.path, currentRoute) ? ' active' : '')}
                onclick=${() => navigate(item.path)}
                aria-current=${isActive(item.path, currentRoute) ? 'page' : undefined}
                data-testid=${item.testId || ('nav-link-' + routeSlug(item.path))}
              >
                ${item.label}
              </button>
            `)}
          `)}
        </nav>
      </div>
      <div class="topbar-right">
        <button
          class="search-btn"
          onclick=${onSearchClick}
          title="Search (Ctrl+K)"
          aria-label="Open search"
          data-testid="nav-search"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Search
          <kbd>Ctrl+K</kbd>
        </button>
      </div>
    </header>
  `;
}
```

- [ ] **Step 2: Fetch features in App, pass to TopNav**

Edit `src/aiperf/operator/ui-v1/app.js`. After the existing `useEffect` for the keyboard handler, add a features-fetch effect:

```javascript
function App() {
  const [showPalette, setShowPalette] = useState(false);
  const [features, setFeatures] = useState({ dashboard_enabled: false });
  const currentRoute = route.value;
  const error = globalError.value;

  useEffect(() => {
    function handleKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setShowPalette((v) => !v);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/v1/config/features')
      .then((r) => (r.ok ? r.json() : null))
      .then((f) => { if (!cancelled && f) setFeatures(f); })
      .catch(() => { /* features stay default — no Plots link */ });
    return () => { cancelled = true; };
  }, []);

  // ...rest unchanged, but update the TopNav call:
```

And in the JSX block, pass `features`:

```javascript
      <${TopNav} onSearchClick=${() => setShowPalette(true)} features=${features} />
```

- [ ] **Step 3: Smoke-test in dev (optional but cheap)**

Run: `uv run python -m aiperf.operator.results_server &` from a checkout with `AIPERF_DASHBOARD_PROXY_ENABLED=1 AIPERF_DASHBOARD_PORT=8082` set, and `curl localhost:8081/api/v1/config/features`.
Expected: `{"dashboard_enabled": true}`.
(Stop the server.)

- [ ] **Step 4: Commit**

```bash
git add src/aiperf/operator/ui-v1/app.js src/aiperf/operator/ui-v1/components/top-nav.js
git commit -s -m "feat(ui-v1): conditional Plots ↗ top-nav link gated by /api/v1/config/features"
```

---

## Task 9: Helm — `dashboard:` block in values.yaml

**Files:**
- Modify: `deploy/helm/aiperf-operator/values.yaml`

- [ ] **Step 1: Inspect current values block & insert new section**

Open `deploy/helm/aiperf-operator/values.yaml`. After the `resultsServer:` block (ends around line 158), insert:

```yaml
# dashboard is the optional Plotly Dash sidecar for the operator Pod.
# When enabled, a third container is added that serves /dashboard/ —
# the heavy plot-building runs there, isolated from results-server so
# a memory blow-up cannot OOM-kill the API surface. results-server
# reverse-proxies /dashboard/* to localhost:<dashboard.port>.
dashboard:
  # dashboard.enabled toggles the sidecar. Off by default.
  enabled: false

  # dashboard.port is the Pod-local HTTP port the dashboard listens on.
  # Not exposed via the Service — the request path is:
  #   client -> Ingress/PF -> results-server :8081 -> localhost:<port>
  port: 8082

  # dashboard.resources caps the dashboard container's CPU / memory.
  # Default: 1Gi requested, no limit. Set resources.limits.memory to
  # enforce a ceiling on shared clusters; the dashboard will be
  # OOMKilled on its own without affecting the operator or results-server.
  resources:
    requests:
      cpu: 100m
      memory: 1Gi
    limits: {}
```

- [ ] **Step 2: Validate chart**

Run: `helm lint deploy/helm/aiperf-operator/`
Expected: no errors.

Run: `helm template deploy/helm/aiperf-operator/ | grep -A2 'dashboard'` — should show no extra container in the rendered output (Task 10 wires it).
Expected: no `name: dashboard` container line yet.

- [ ] **Step 3: Commit**

```bash
git add deploy/helm/aiperf-operator/values.yaml
git commit -s -m "feat(helm): add dashboard.enabled / port / resources to values.yaml"
```

---

## Task 10: Helm — third container in deployment.yaml + env wiring

**Files:**
- Modify: `deploy/helm/aiperf-operator/templates/deployment.yaml`

- [ ] **Step 1: Add `AIPERF_DASHBOARD_*` env to operator + results-server containers**

In `deploy/helm/aiperf-operator/templates/deployment.yaml`, find the `operator` container's `env:` block (after `AIPERF_K8S_SHARE_PROCESS_NAMESPACE`, before the optional chaos overrides). Append:

```yaml
        - name: AIPERF_DASHBOARD_PORT
          value: {{ if .Values.dashboard.enabled }}{{ .Values.dashboard.port | quote }}{{ else }}"0"{{ end }}
```

Find the `results-server` container's `env:` block (after `MPLCONFIGDIR`). Append:

```yaml
        - name: AIPERF_DASHBOARD_PORT
          value: {{ if .Values.dashboard.enabled }}{{ .Values.dashboard.port | quote }}{{ else }}"0"{{ end }}
        - name: AIPERF_DASHBOARD_PROXY_ENABLED
          value: {{ if .Values.dashboard.enabled }}"1"{{ else }}"0"{{ end }}
```

- [ ] **Step 2: Add the third container, wrapped in `if .Values.dashboard.enabled`**

After the `results-server` container's volumeMounts (around line 180), but **before** the `volumes:` block, insert:

```yaml
      {{- if .Values.dashboard.enabled }}
      - name: dashboard
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        command:
        - python
        - -m
        - aiperf.operator.dashboard_server
        ports:
        - name: dashboard
          containerPort: {{ .Values.dashboard.port }}
          protocol: TCP
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: AIPERF_RESULTS_DIR
          value: {{ .Values.storage.mountPath | quote }}
        - name: AIPERF_DASHBOARD_PORT
          value: {{ .Values.dashboard.port | quote }}
        - name: TMPDIR
          value: "/tmp"
        - name: MPLCONFIGDIR
          value: "/tmp/matplotlib"
        livenessProbe:
          httpGet:
            path: /healthz
            port: {{ .Values.dashboard.port }}
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 5
        readinessProbe:
          httpGet:
            path: /healthz
            port: {{ .Values.dashboard.port }}
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 6
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL
        resources:
          {{- toYaml .Values.dashboard.resources | nindent 10 }}
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: results
          mountPath: {{ .Values.storage.mountPath | quote }}
          readOnly: true
      {{- end }}
```

- [ ] **Step 3: Render and verify**

Run:
```bash
helm template deploy/helm/aiperf-operator/ --set dashboard.enabled=true | grep -E 'name: dashboard|AIPERF_DASHBOARD'
```
Expected output should include the three env vars and the new container's `name: dashboard` and `containerPort: 8082`.

Run:
```bash
helm template deploy/helm/aiperf-operator/ | grep -E 'name: dashboard|AIPERF_DASHBOARD'
```
Expected: `AIPERF_DASHBOARD_PORT: "0"`, `AIPERF_DASHBOARD_PROXY_ENABLED: "0"`, **no** `name: dashboard` container line.

Run: `helm lint deploy/helm/aiperf-operator/`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add deploy/helm/aiperf-operator/templates/deployment.yaml
git commit -s -m "feat(helm): wire dashboard sidecar container + env behind dashboard.enabled"
```

---

## Task 11: Chart-render unit tests

**Files:**
- Create: `tests/kubernetes/test_helm_dashboard.py`

(Note: this is a chart-render test, not an in-cluster test. It does **not** belong in `tests/kubernetes/test_helm.py` which is the in-cluster integration suite. It runs `helm template` and parses YAML.)

- [ ] **Step 1: Write the failing test**

```python
# tests/kubernetes/test_helm_dashboard.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chart-render assertions for the optional Plotly Dashboard sidecar."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART = Path(__file__).resolve().parents[2] / "deploy" / "helm" / "aiperf-operator"


def _render(*set_args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm CLI not available")
    cmd = ["helm", "template", "test", str(CHART)]
    for s in set_args:
        cmd.extend(["--set", s])
    out = subprocess.check_output(cmd, text=True)
    return [d for d in yaml.safe_load_all(out) if d]


def _operator_deployment(docs: list[dict]) -> dict:
    for d in docs:
        if d.get("kind") == "Deployment" and "operator" in d["metadata"]["name"]:
            return d
    raise AssertionError("operator Deployment not in render")


def _container(deploy: dict, name: str) -> dict | None:
    return next(
        (c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == name),
        None,
    )


def _env(container: dict, name: str) -> str | None:
    for e in container.get("env", []):
        if e["name"] == name:
            return e.get("value")
    return None


def test_dashboard_container_absent_by_default() -> None:
    docs = _render()
    deploy = _operator_deployment(docs)
    assert _container(deploy, "dashboard") is None
    operator = _container(deploy, "operator")
    results = _container(deploy, "results-server")
    assert _env(operator, "AIPERF_DASHBOARD_PORT") == "0"
    assert _env(results, "AIPERF_DASHBOARD_PROXY_ENABLED") == "0"


def test_dashboard_container_present_when_enabled() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash is not None
    assert dash["command"] == [
        "python",
        "-m",
        "aiperf.operator.dashboard_server",
    ]
    port = next(p["containerPort"] for p in dash["ports"] if p["name"] == "dashboard")
    assert port == 8082
    operator = _container(deploy, "operator")
    results = _container(deploy, "results-server")
    assert _env(operator, "AIPERF_DASHBOARD_PORT") == "8082"
    assert _env(results, "AIPERF_DASHBOARD_PROXY_ENABLED") == "1"
    assert _env(results, "AIPERF_DASHBOARD_PORT") == "8082"


def test_dashboard_limits_omitted_when_empty() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash["resources"]["requests"]["memory"] == "1Gi"
    assert dash["resources"].get("limits") in (None, {})


def test_dashboard_limits_respected_when_set() -> None:
    docs = _render(
        "dashboard.enabled=true",
        "dashboard.resources.limits.memory=4Gi",
    )
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    assert dash["resources"]["limits"]["memory"] == "4Gi"


def test_dashboard_pvc_mount_is_readonly() -> None:
    docs = _render("dashboard.enabled=true")
    deploy = _operator_deployment(docs)
    dash = _container(deploy, "dashboard")
    results_mount = next(m for m in dash["volumeMounts"] if m["name"] == "results")
    assert results_mount.get("readOnly") is True
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/kubernetes/test_helm_dashboard.py -v`
Expected: PASS for all 5 cases. (If `helm` is not installed, all skip cleanly.)

- [ ] **Step 3: Commit**

```bash
git add tests/kubernetes/test_helm_dashboard.py
git commit -s -m "test(helm): chart-render assertions for dashboard sidecar"
```

---

## Task 12: E2E navigation test for the Plots link

**Files:**
- Modify: `tests/e2e/operator_ui/test_navigation.py`

(The existing e2e suite uses the Page-Object pattern with a `_pages.py` helper. We add a thin assertion that the link appears with `target="_blank"` when `dashboard_enabled` is true and is absent when false. This is a parametrized test that mocks the `/api/v1/config/features` endpoint at the network layer or relies on the operator under test having the env set. The simpler path: drive via the same operator process with the env toggled.)

- [ ] **Step 1: Inspect the existing test file**

Run: `grep -n "test_unknown_route\|def test_" tests/e2e/operator_ui/test_navigation.py | head -20`

- [ ] **Step 2: Add new tests**

Append to `tests/e2e/operator_ui/test_navigation.py`:

```python
def test_plots_link_hidden_when_dashboard_disabled(page) -> None:
    """With dashboard_enabled=false, the Plots ↗ top-nav link is absent."""
    page.goto(BASE_URL)  # BASE_URL fixture / convention from this file
    page.wait_for_selector('[data-testid="top-nav"]')
    assert page.locator('[data-testid="nav-link-plots"]').count() == 0


def test_plots_link_opens_in_new_tab_when_enabled(page, dashboard_enabled_env) -> None:
    """With dashboard_enabled=true, the Plots ↗ link uses target=_blank."""
    page.goto(BASE_URL)
    page.wait_for_selector('[data-testid="top-nav"]')
    link = page.locator('[data-testid="nav-link-plots"]')
    assert link.count() == 1
    assert link.get_attribute("target") == "_blank"
    assert link.get_attribute("href") == "/dashboard/"
```

The `dashboard_enabled_env` fixture flips the operator's env. Add it to `tests/e2e/operator_ui/conftest.py` (or wherever the existing fixtures live):

```python
@pytest.fixture
def dashboard_enabled_env(monkeypatch: pytest.MonkeyPatch):
    """Flip the operator-under-test to expose dashboard_enabled=true.

    Note: the e2e harness restarts the operator's results-server between
    tests carrying this fixture so the SPA bundle re-fetches /api/v1/config/features.
    """
    monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", "8082")
    yield
```

(If your e2e harness does not restart between tests, instead route `/api/v1/config/features` through a Playwright route handler that returns the desired body — see existing patterns in `test_jobs.py` for `page.route(...)`.)

- [ ] **Step 3: Run e2e**

Run: `uv run pytest tests/e2e/operator_ui/test_navigation.py -v`
Expected: PASS (if e2e dependencies installed; SKIP otherwise — unblock manually).

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/operator_ui/test_navigation.py tests/e2e/operator_ui/conftest.py
git commit -s -m "test(e2e): assert Plots ↗ link visibility under dashboard toggle"
```

---

## Task 13: Documentation

**Files:**
- Modify: `docs/kubernetes/dashboard-ui.md`
- Modify: `docs/kubernetes/configuration.md`
- Modify: `docs/kubernetes/sidecars.md`
- Modify: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` (four-file sync rule — only if a new pattern needs documenting)

- [ ] **Step 1: `docs/kubernetes/dashboard-ui.md` — add a section**

Append a new section near the bottom titled "Isolated Plotly Dashboard Sidecar (opt-in)":

```markdown
## Isolated Plotly Dashboard Sidecar (opt-in)

The Plotly Dash plot-building runs in its own container in the operator
Pod, behind the `dashboard.enabled` Helm value (default `false`). When
enabled:

- The operator Pod runs three containers: `aiperf-operator`,
  `results-server`, and `dashboard`.
- `results-server` reverse-proxies `/dashboard/*` to
  `localhost:<dashboard.port>` so external callers still hit one URL
  (the existing `results-server.port`, default 8081).
- The SPA's "Plots ↗" top-nav link appears, opening `/dashboard/` in
  a new tab. The link is gated by `/api/v1/config/features`'s
  `dashboard_enabled` field so a misconfigured chart fails closed.
- After every benchmark completion, the operator fires a
  fire-and-forget `POST /admin/refresh` against the dashboard sidecar
  so the next `/dashboard/` view sees the new run.

### Memory budgeting

By default the dashboard container has `requests: 1Gi` and **no
memory limit** — it can burst to whatever the node has free. This
matches the original in-process behaviour but isolates blast radius
to a single container. To enforce a ceiling on shared clusters:

```yaml
dashboard:
  enabled: true
  resources:
    limits:
      memory: 4Gi
```

When the limit is exceeded, only the dashboard container is
OOMKilled — `results-server` (API, jobs router, WS) and the operator
keep running.

### Disabling

```bash
helm upgrade ... --set dashboard.enabled=false
```

When off, the `/dashboard/*` route returns 503 with a friendly body
and the SPA hides the "Plots ↗" link.
```

- [ ] **Step 2: `docs/kubernetes/configuration.md` — values reference**

Find the existing `resultsServer:` documentation. Add a new subsection below it:

```markdown
### `dashboard`

Optional Plotly Dash sidecar for the operator Pod. Default off.

| Key                              | Default      | Description                                                                   |
|----------------------------------|--------------|-------------------------------------------------------------------------------|
| `dashboard.enabled`              | `false`      | Whether to add the dashboard container and surface the "Plots ↗" SPA link.   |
| `dashboard.port`                 | `8082`       | Pod-local HTTP port. `results-server` reverse-proxies `/dashboard/*` here.    |
| `dashboard.resources.requests`   | `cpu: 100m, memory: 1Gi` | Resource requests. Leave generous so the build has memory.        |
| `dashboard.resources.limits`     | `{}`         | Empty by default = no limit. Set `memory:` to enforce a ceiling.             |

See [`dashboard-ui.md`](dashboard-ui.md#isolated-plotly-dashboard-sidecar-opt-in) for the full architecture.
```

- [ ] **Step 3: `docs/kubernetes/sidecars.md` — inventory**

Add the dashboard container to the sidecar inventory section:

```markdown
### `dashboard` (optional)

- **Image:** same as `aiperf-operator`.
- **Command:** `python -m aiperf.operator.dashboard_server`
- **Port:** `dashboard.port` (Pod-local; not in the Service).
- **Mounts:** results PVC, **read-only**.
- **Trigger:** opt-in via `dashboard.enabled`.
- **Refresh:** the operator fires `POST /admin/refresh` after each
  successful completion claim so the Dash app picks up new runs.
```

- [ ] **Step 4: Run docs index check (only if a new doc file was added)**

We modified existing files only, so no new entry in `docs/index.yml` is required. Skip the check unless you added a new path under `docs/`.

- [ ] **Step 5: Commit**

```bash
git add docs/kubernetes/dashboard-ui.md docs/kubernetes/configuration.md docs/kubernetes/sidecars.md
git commit -s -m "docs(kubernetes): document optional Plotly Dash sidecar + memory budgeting"
```

---

## Task 14: Final integration + cleanup pass

**Files:**
- Verify all touched files pass mechanical checks.

- [ ] **Step 1: Run full mechanical sweep**

```bash
ruff format . && ruff check --fix .
make check-ergonomics
make check-ruff-baselined
make check-agent-files-sync
uv run pytest -n auto tests/unit/
helm lint deploy/helm/aiperf-operator/
```

Expected: all green. No new entries in baselines.

- [ ] **Step 2: Smoke-test on a kind cluster (optional but recommended)**

```bash
kind create cluster --name aiperf-dash
helm install aiperf deploy/helm/aiperf-operator/ \
  --set dashboard.enabled=true \
  --kube-context kind-aiperf-dash
kubectl --context kind-aiperf-dash port-forward svc/aiperf-aiperf-operator 8081:8081
```

Open `http://localhost:8081/v1/`. Verify "Plots ↗" link is visible. Click it; Dash placeholder appears. (Without runs on the PVC, you'll see the "no completed runs" 503.)

```bash
helm upgrade aiperf deploy/helm/aiperf-operator/ \
  --set dashboard.enabled=false \
  --kube-context kind-aiperf-dash
```

Verify link disappears, `/dashboard/` returns 503 with friendly body.

```bash
kind delete cluster --name aiperf-dash
```

- [ ] **Step 3: Branch sanity**

Run: `git log --oneline origin/main..HEAD`
Expected: ~13 commits, one per task.

- [ ] **Step 4: No final commit needed unless mechanical sweep changed anything.**

If `ruff format` modified files, run:

```bash
git add -A
git commit -s -m "chore: ruff format pass after dashboard isolation"
```

---

## Notes for the implementer

- **Follow the v1-import-leak rule** — none of the new files should import from `aiperf.config.v1.*`. Dashboard wiring is K8s + plot-side; v1 doesn't enter.
- **Lambda-form debug logs** per CLAUDE.md: any new `logger.debug(...)` with f-string interpolation must be `self.debug(lambda: f"...")` if inside a `BaseComponentService` subclass; module-level functions can use the cheap form.
- **No emojis** in code or comments.
- **No `Optional[X]`** — use `X | None`.
- **Pydantic fields need `Field(description=...)`** — the new `_DashboardSettings` and `FeaturesResponse` models comply.
- **Pre-commit hooks** will run on every commit. If `generate-cli-docs` or `generate-env-vars-docs` reflows `docs/environment-variables.md` because of the new `_DashboardSettings`, stage the regenerated doc with the same commit (not in a separate one).
