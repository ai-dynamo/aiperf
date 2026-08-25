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

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse

from aiperf.operator.dashboard_mount import DashboardProxy, build_dashboard

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/data"))


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
    """Build the Dash app on the PVC and swap it into the WSGI proxy.

    Failures (read-only rootfs, no runs yet, anything else) leave a
    placeholder app mounted so /dashboard/ degrades to a friendly 503
    rather than 500.
    """
    try:
        dash_app, run_count = await asyncio.to_thread(build_dashboard, base_dir)
    except OSError as exc:
        logger.warning("Dashboard init failed (likely read-only rootfs): %s", exc)
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
    app.state.results_dir = base_dir
    app.state.dashboard_refresh_inflight = False
    # Strong references to in-flight refresh tasks: asyncio keeps only a weak
    # one, and a collected task never runs its finally-block, so the inflight
    # flag would stay True and every later refresh return already_rebuilding.
    app.state.dashboard_refresh_tasks = set()
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

        task = asyncio.create_task(_refresh_task())
        app.state.dashboard_refresh_tasks.add(task)
        task.add_done_callback(app.state.dashboard_refresh_tasks.discard)
        return JSONResponse(status_code=202, content={"status": "rebuilding"})

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
