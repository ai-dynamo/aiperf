# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP server for serving stored benchmark results from the operator PVC.

Runs as a sidecar container alongside the kopf operator, sharing the results
PVC volume. Provides two layers:

1. **File serving** - download raw result files with zstd content negotiation
2. **Analytics** - DuckDB-powered query endpoints for leaderboards, history,
   and cross-job comparisons (reads result files directly, no ETL)

Endpoints:
    GET /healthz                                        - health check

    File serving (``aiperf.operator.routers.results_files``):
    GET /api/v1/results                                 - list all jobs
    GET /api/v1/results/{namespace}/{job_id}             - list files for a job
    GET /api/v1/results/{namespace}/{job_id}/{filename}  - download a file

    Analytics (``aiperf.operator.routers.results_analytics``):
    GET /api/v1/analytics/leaderboard                   - rank runs by metric
    GET /api/v1/analytics/history                        - metric over time
    GET /api/v1/analytics/compare                       - compare specific jobs
    GET /api/v1/analytics/summary/{namespace}/{job_id}   - full summary for a job

Run: python -m aiperf.operator.results_server
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from kubernetes_asyncio.client.exceptions import ApiException

# Re-exported for tests and downstream callers.
from aiperf.operator.dashboard_mount import DashboardProxy, build_dashboard
from aiperf.operator.routers.results_files import (
    _display_name,
    _safe_resolve,
    create_results_files_router,
)

__all__ = [
    "DashboardProxy",
    "_display_name",
    "_safe_resolve",
    "build_dashboard",
    "create_app",
    "main",
]

logger = logging.getLogger(__name__)

# Configured via environment variable, matching the operator's AIPERF_RESULTS_DIR
RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/data"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI application with results and analytics routes.

    Args:
        results_dir: Base directory for stored results. Defaults to RESULTS_DIR.
    """
    from aiperf.operator.results_db import ResultsDB

    base_dir = results_dir or RESULTS_DIR
    db: ResultsDB | None = None

    # Mutable holder for Kubernetes ApiClient - populated during lifespan,
    # read by router. Typed as list to match create_jobs_router signature.
    from kubernetes_asyncio.client import ApiClient

    api_holder: list[ApiClient | None] = [None]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal db
        db = ResultsDB(base_dir)
        logger.info(f"DuckDB analytics engine initialized (results_dir={base_dir})")

        # Initialize kubernetes_asyncio client for live job/cluster endpoints.
        # Load config (in-cluster first, kubeconfig fallback) and build an
        # ApiClient that lives for the process lifetime.
        try:
            from kubernetes_asyncio import config

            from aiperf.common.noisy_loggers import suppress_noisy_http_loggers

            suppress_noisy_http_loggers()
            try:
                config.load_incluster_config()
            except config.ConfigException:
                await config.load_kube_config()
            api_holder[0] = ApiClient()
            logger.info("kubernetes_asyncio client initialized for UI endpoints")
        except (
            config.ConfigException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
        ) as e:
            logger.warning(
                f"Kubernetes client unavailable, live job endpoints disabled: {e}"
            )

        yield

        db.close()
        logger.info("DuckDB analytics engine closed")

        api = api_holder[0]
        if api is not None:
            try:
                await api.close()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                logger.warning(f"Error closing kubernetes_asyncio client: {e}")
            api_holder[0] = None

    app = FastAPI(
        title="AIPerf Operator Results API",
        description="Serves benchmark results and analytics from the operator PVC.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Surface kubernetes_asyncio errors verbatim instead of masking them as 500s;
    # router docstrings document this contract (e.g. RBAC 403 stays 403).
    @app.exception_handler(ApiException)
    async def _k8s_api_exception_handler(
        request: Request, exc: ApiException
    ) -> JSONResponse:
        logger.warning(
            f"Kubernetes API error on {request.method} {request.url.path}: "
            f"status={exc.status} reason={exc.reason}"
        )
        return JSONResponse(
            status_code=exc.status or 500,
            content={"detail": str(exc.body or exc.reason or "Kubernetes API error")},
        )

    # Register live jobs/cluster router (client populated during lifespan)
    from aiperf.operator.routers.jobs import create_jobs_router
    from aiperf.operator.routers.results_analytics import (
        create_results_analytics_router,
    )

    app.include_router(create_jobs_router(api_holder))
    app.include_router(create_results_files_router(base_dir))

    def _get_db() -> ResultsDB:
        if db is None:
            raise HTTPException(503, "Analytics engine not initialized")
        return db

    app.include_router(create_results_analytics_router(_get_db, base_dir))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # Mount the Plotly Dash dashboard at /dashboard/. DashboardProxy is mutable
    # so the inner WSGI app can be hot-swapped when new runs land on the PVC
    # without disturbing the outer route. If no runs exist yet, mount a
    # placeholder that returns 503 until build_dashboard succeeds.
    dash_app, run_count = build_dashboard(base_dir)
    if dash_app is not None:
        logger.info(
            f"Mounting Plotly Dash dashboard with {run_count} runs at /dashboard/"
        )
        dashboard_proxy = DashboardProxy(dash_app.server)
    else:
        logger.info("No runs on PVC yet; /dashboard/ returns 503 until runs exist")

        def _pending_app(environ, start_response):
            start_response(
                "503 Service Unavailable",
                [("Content-Type", "text/plain; charset=utf-8")],
            )
            return [b"Dashboard not yet available: no completed runs on PVC."]

        dashboard_proxy = DashboardProxy(_pending_app)

    app.mount("/dashboard", WSGIMiddleware(dashboard_proxy))

    # Mount UI static files last (catch-all for SPA routing)
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


def main() -> None:
    """Run the results server as a standalone process."""
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
