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
    """Hot-swap path: when build_dashboard returns an app, requests reach it."""
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


def test_refresh_returns_202_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First refresh kicks off a rebuild and returns 202."""
    from aiperf.operator import dashboard_server

    monkeypatch.setattr(dashboard_server, "build_dashboard", lambda _d: (None, 0))
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
