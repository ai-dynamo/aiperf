# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal results sidecar for controller pods.

Serves exported files from the controller pod's shared ``/results`` volume so
the operator can recover artifacts even if the main controller container exits
after export. Files are hidden until a ready marker is written by the
controller, preventing consumers from downloading partial exports.

Listing walks the volume recursively so nested AIPerfSweep harvests
(``/results/<ns>/sweeps/<sweep>/<epoch>/...``) surface too; the enumeration,
path-safety and readiness rules themselves are shared with the in-process
results router via ``aiperf.api.results_files``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from aiperf.api.models.results import ResultsListResponse
from aiperf.api.results_files import (
    build_result_file_response,
    list_result_files,
    resolve_result_file_or_404,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/results"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SIDECAR_PORT", "9091"))


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI app for serving controller-side results."""
    base_dir = results_dir or RESULTS_DIR
    app = FastAPI(
        title="AIPerf Controller Results Sidecar",
        version="1.0.0",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/results/list", response_model=ResultsListResponse)
    async def list_results() -> ResultsListResponse:
        return await list_result_files(base_dir, recursive=True)

    @app.get("/api/results/files/{filename:path}")
    async def get_result_file(filename: str, request: Request) -> StreamingResponse:
        file_path = await resolve_result_file_or_404(base_dir, filename)
        return build_result_file_response(file_path, request)

    return app


def main() -> None:
    """Run the sidecar HTTP server."""
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=SERVER_PORT,
        access_log=False,
        log_level=os.environ.get("AIPERF_RESULTS_SIDECAR_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
