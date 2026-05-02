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
