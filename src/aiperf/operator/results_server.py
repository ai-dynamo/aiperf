# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP server for serving stored benchmark results from the operator PVC.

Runs as a sidecar container alongside the kopf operator, sharing the results
PVC volume.

**This app is the operator Pod's only FastAPI app.** Every ``/api/v1/*`` router
-- jobs, sweeps, results, config, admin, analytics, dashboard_proxy -- mounts
here, not on the operator container, which stays pure kopf and exposes only
kopf's own ``/healthz`` plus the Prometheus metrics port. New HTTP surface goes
here by default.

Honest note on provenance: that split is enforced by the chart and restated in
the agent instruction files, but no commit, spec, or design doc in the
subsystem's history ever argues for it -- it is only ever stated. The mechanism
visible in the code is that kopf owns the operator container's event loop and
its liveness endpoint, so a second ASGI server there would contend with it;
treat that as the likely driver rather than the recorded one. Anyone with cause
to revisit the split should know they are not overturning a written decision.

Provides two layers:

1. **File serving** - download raw result files with zstd content negotiation
2. **Analytics** - SQLite-backed query endpoints (via ``runs_index``) for
   leaderboards, history, and cross-job comparisons.

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

    Config (``aiperf.operator.routers.config``):
    GET /api/v1/config/retention                         - current retention policy

Run: python -m aiperf.operator.results_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from kubernetes_asyncio.client.exceptions import ApiException
from starlette.middleware.gzip import GZipMiddleware

from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.routers.results_files import (
    _display_name,
    _safe_resolve,
    create_results_files_router,
)

__all__ = [
    "_display_name",
    "_safe_resolve",
    "create_app",
    "main",
]

logger = logging.getLogger(__name__)

# Configured via environment variable, matching the operator's AIPERF_RESULTS_DIR
RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/data"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))


def _build_lifespan(base_dir: Path, api_holder: list, db_holder: list):
    """Build the FastAPI lifespan context manager for DB + k8s client setup/teardown."""
    from aiperf.operator import runs_index
    from aiperf.operator.results_db import ResultsDB

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The results-server sidecar serves from a read-only PVC mount. The
        # kopf operator process is the single SQLite writer and creates/migrates
        # the index during operator startup; this process only opens a read-only
        # connection for normal API serving.
        index_owned_here = False
        if not runs_index.is_open():
            try:
                await runs_index.open_readonly(base_dir / ".aiperf_index.sqlite")
            except (RuntimeError, OSError, sqlite3.Error) as exc:
                logger.warning("runs_index read-only open skipped: %s", exc)
            else:
                index_owned_here = True
        db_holder[0] = ResultsDB(base_dir)
        logger.info(f"runs_index analytics facade initialized (results_dir={base_dir})")

        # The API client must outlive a single request, so the shared
        # ``k8s_client()`` context manager is held open for the whole app
        # lifespan via an exit stack rather than instantiating ``ApiClient``
        # directly (which would bypass the in-cluster/kubeconfig fallback).
        # Kubernetes is OPTIONAL for this process. Only the live-job endpoints
        # need it; every results, sweeps, and artifact route reads the PVC. So
        # this init is bounded and broadly guarded -- a results-server that
        # refuses to start because a cluster is unreachable cannot serve the
        # disk it exists to serve.
        #
        # Both failure modes below were hit in practice:
        #   * HANG -- `k8s_client()` resolving a proxied/unreachable apiserver
        #     never returned, so startup never completed and the process
        #     answered nothing, with no error to explain it. Hence the timeout.
        #   * TypeError -- kubernetes_asyncio's kubeconfig loader raised
        #     `'NoneType' object does not support item assignment` on a config
        #     it could not merge. That escaped the old except clause and killed
        #     startup outright ("Application startup failed. Exiting.").
        # Config parsing raises ValueError/KeyError on other malformed inputs,
        # so those are caught too. Anything unexpected still propagates.
        k8s_stack = AsyncExitStack()
        try:
            from kubernetes_asyncio import config

            from aiperf.kubernetes.client import k8s_client

            async with asyncio.timeout(
                OperatorEnvironment.RESULTS.K8S_INIT_TIMEOUT_SEC
            ):
                api_holder[0] = await k8s_stack.enter_async_context(k8s_client())
            logger.info("kubernetes_asyncio client initialized for UI endpoints")
        except (
            TimeoutError,
            config.ConfigException,
            aiohttp.ClientError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ) as e:
            logger.warning(
                "Kubernetes client unavailable, live job endpoints disabled; "
                "PVC-backed results, sweeps and artifact routes still serve. "
                f"({type(e).__name__}: {e})"
            )

        yield

        db_holder[0].close()
        if index_owned_here:
            await runs_index.close()
        logger.info("runs_index analytics facade closed")

        if api_holder[0] is not None:
            try:
                await k8s_stack.aclose()
            except (TimeoutError, aiohttp.ClientError, OSError) as e:
                logger.warning(f"Error closing kubernetes_asyncio client: {e}")
            api_holder[0] = None

    return lifespan


def _register_k8s_exception_handler(app: FastAPI) -> None:
    """Surface kubernetes_asyncio errors verbatim instead of masking them as 500s."""

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

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # Analytics endpoints validate metric/stat identifiers via
        # runs_index._validate_identifier — adversarial input raises
        # ValueError, which we surface as 422 instead of 500.
        logger.info(
            f"Rejected invalid input on {request.method} {request.url.path}: {exc}"
        )
        return JSONResponse(status_code=422, content={"detail": str(exc)})


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI application with results and analytics routes.

    Args:
        results_dir: Base directory for stored results. Defaults to RESULTS_DIR.
    """
    from kubernetes_asyncio.client import ApiClient

    from aiperf.operator.results_db import ResultsDB
    from aiperf.operator.routers.admin import create_admin_router
    from aiperf.operator.routers.config import create_config_router
    from aiperf.operator.routers.jobs import create_jobs_router
    from aiperf.operator.routers.jobs_ws import create_jobs_ws_router
    from aiperf.operator.routers.mutating_auth import mutating_route_dependencies
    from aiperf.operator.routers.results_analytics import (
        create_results_analytics_router,
    )
    from aiperf.operator.routers.sweeps import create_sweeps_router
    from aiperf.operator.routers.validate import create_validate_router

    base_dir = results_dir or RESULTS_DIR
    api_holder: list[ApiClient | None] = [None]
    db_holder: list[ResultsDB | None] = [None]

    app = FastAPI(
        title="AIPerf Operator Results API",
        description="Serves benchmark results and analytics from the operator PVC.",
        version="1.0.0",
        lifespan=_build_lifespan(base_dir, api_holder, db_holder),
    )

    # Analytics and job-listing payloads are highly repetitive JSON, so the
    # operator UI pays a large transfer cost without compression. The 500-byte
    # floor keeps /healthz and other tiny responses uncompressed.
    # compresslevel=6 is the standard "sweet spot": level 9 costs 2-4x CPU
    # for under 2% additional size reduction, which is pure event-loop time.
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)

    _register_k8s_exception_handler(app)

    mutating_dependencies = mutating_route_dependencies()

    app.include_router(create_validate_router())
    app.include_router(create_jobs_router(api_holder, base_dir, mutating_dependencies))
    app.include_router(create_jobs_ws_router(api_holder))
    app.include_router(
        create_sweeps_router(api_holder, base_dir, mutating_dependencies)
    )
    app.include_router(create_results_files_router(base_dir))
    app.include_router(create_config_router())
    app.include_router(
        create_admin_router(
            base_dir,
            base_dir / ".aiperf_index.sqlite",
            mutating_dependencies,
            allow_rebuild=False,
        )
    )

    def _get_db() -> ResultsDB:
        db = db_holder[0]
        if db is None:
            raise HTTPException(503, "Analytics engine not initialized")
        return db

    app.include_router(create_results_analytics_router(_get_db, base_dir, api_holder))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    from aiperf.operator.routers.dashboard_proxy import (
        create_dashboard_proxy_router,
    )

    app.include_router(create_dashboard_proxy_router())

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
