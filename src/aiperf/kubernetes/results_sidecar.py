# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal results sidecar for controller pods.

Serves exported files from the controller pod's shared ``/results`` volume so
the operator can recover artifacts even if the main controller container exits
after export. Files are hidden until a ready marker is written by the
controller, preventing consumers from downloading partial exports.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from aiofiles import os as aio_os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.api.models.results import ResultFileInfo, ResultsListResponse
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.results_markers import (
    _RESERVED_MARKER_NAMES,
    READY_MARKER_NAME,
    _is_checkpoint_path,
    _is_processing,
    _is_ready,
    _safe_resolve,
    _safe_size,
    checkpoints_dir,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/results"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SIDECAR_PORT", "9091"))

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}


def _collect_result_files(base_dir: Path) -> list[ResultFileInfo]:
    """Enumerate every artifact under ``base_dir`` once the ready marker is set.

    Walks recursively so the AIPerfSweep harvest path (``/results/<ns>/sweeps/
    <sweep>/<epoch>/aggregate.json``, ``children.json``, ``aggregate/
    profile_export_aiperf_aggregate.{json,csv}``) and any future nested layout
    surface in the listing. Both transaction markers are excluded because they
    are sidecar-internal state, not downloadable artifacts.

    Checkpoint files (under ``checkpoints/``) are surfaced unconditionally
    (even before the marker) so an AIPerfJob's iterative checkpoint stream
    is fetchable mid-run; everything else is gated on
    ``.aiperf_results_ready.json``.
    """
    files: list[ResultFileInfo] = []
    ready = _is_ready(base_dir)
    cp_dir = checkpoints_dir(base_dir)

    if ready:
        for entry in base_dir.rglob("*"):
            if (
                not entry.is_file()
                or entry.name in _RESERVED_MARKER_NAMES
                or cp_dir in entry.parents
            ):
                continue
            size = _safe_size(entry)
            if size is not None:
                files.append(
                    ResultFileInfo(
                        name=entry.relative_to(base_dir).as_posix(),
                        size=size,
                    )
                )

    if cp_dir.is_dir():
        for entry in cp_dir.rglob("*"):
            if not entry.is_file():
                continue
            size = _safe_size(entry)
            if size is not None:
                files.append(
                    ResultFileInfo(
                        name=entry.relative_to(base_dir).as_posix(),
                        size=size,
                    )
                )

    return sorted(files, key=lambda item: item.name)


async def _list_results(base_dir: Path) -> ResultsListResponse:
    if not await aio_os.path.isdir(base_dir):
        return ResultsListResponse()
    files = await asyncio.to_thread(_collect_result_files, base_dir)
    return ResultsListResponse(
        files=files,
        ready=_is_ready(base_dir),
        processing=_is_processing(base_dir),
    )


async def _resolve_result_file(base_dir: Path, filename: str) -> Path:
    """Validate, locate, and return the result file path or raise HTTPException."""
    file_path = _safe_resolve(base_dir, filename)
    if file_path is None or file_path.name in _RESERVED_MARKER_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename {filename!r}: path traversal or reserved marker name",
        )
    if not _is_ready(base_dir) and not _is_checkpoint_path(
        base_dir.resolve(), file_path
    ):
        processing_detail = (
            " export still processing;" if _is_processing(base_dir) else ""
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results not ready for {base_dir.name};{processing_detail} "
                f"marker file {READY_MARKER_NAME} not present — retry after completion"
            ),
        )
    if not await aio_os.path.isfile(file_path):
        raise HTTPException(
            status_code=404, detail=f"Result file not found: {filename}"
        )
    return file_path


def _build_file_response(file_path: Path, request: Request) -> StreamingResponse:
    accept_encoding = request.headers.get("accept-encoding")
    encoding = select_encoding(accept_encoding, default=CompressionEncoding.IDENTITY)
    content_type = _CONTENT_TYPES.get(
        file_path.suffix.lower(), "application/octet-stream"
    )

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{file_path.name}"',
        "X-Filename": file_path.name,
    }
    if encoding != CompressionEncoding.IDENTITY:
        headers["Content-Encoding"] = encoding

    return StreamingResponse(
        stream_file_compressed(file_path, encoding),
        media_type=content_type,
        headers=headers,
    )


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
        return await _list_results(base_dir)

    @app.get("/api/results/files/{filename:path}")
    async def get_result_file(filename: str, request: Request) -> StreamingResponse:
        file_path = await _resolve_result_file(base_dir, filename)
        return _build_file_response(file_path, request)

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
