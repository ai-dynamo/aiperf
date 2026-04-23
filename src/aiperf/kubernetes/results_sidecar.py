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

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/results"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SIDECAR_PORT", "9091"))
READY_MARKER_NAME = ".aiperf_results_ready.json"
CHECKPOINTS_DIR_NAME = "checkpoints"

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}


def ready_marker_path(base_dir: Path) -> Path:
    """Return the sidecar readiness marker path."""
    return base_dir / READY_MARKER_NAME


def checkpoints_dir(base_dir: Path) -> Path:
    """Return the checkpoint directory under the results directory."""
    return base_dir / CHECKPOINTS_DIR_NAME


def write_ready_marker(base_dir: Path, *, was_cancelled: bool = False) -> Path:
    """Write the readiness marker after exports complete."""
    import orjson

    marker = ready_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(
        orjson.dumps(
            {
                "ready": True,
                "was_cancelled": was_cancelled,
            }
        )
    )
    return marker


def _safe_resolve(base_dir: Path, filename: str) -> Path | None:
    """Resolve a path under base_dir, rejecting traversal."""
    try:
        resolved = (base_dir / filename).resolve()
        resolved.relative_to(base_dir.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _is_ready(base_dir: Path) -> bool:
    """Whether the controller has finished exporting results."""
    return ready_marker_path(base_dir).is_file()


def _is_checkpoint_path(base_dir: Path, path: Path) -> bool:
    """Whether a path points at a checkpoint artifact under the results dir."""
    try:
        relative = path.relative_to(base_dir.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == CHECKPOINTS_DIR_NAME


def _collect_result_files(base_dir: Path) -> list[ResultFileInfo]:
    """Enumerate ready top-level exports and all checkpoint artifacts."""
    files: list[ResultFileInfo] = []

    if _is_ready(base_dir):
        files.extend(
            ResultFileInfo(name=entry.name, size=entry.stat().st_size)
            for entry in base_dir.iterdir()
            if entry.is_file() and entry.name != READY_MARKER_NAME
        )

    cp_dir = checkpoints_dir(base_dir)
    if cp_dir.is_dir():
        files.extend(
            ResultFileInfo(
                name=entry.relative_to(base_dir).as_posix(),
                size=entry.stat().st_size,
            )
            for entry in cp_dir.rglob("*")
            if entry.is_file()
        )

    return sorted(files, key=lambda item: item.name)


async def _list_results(base_dir: Path) -> ResultsListResponse:
    if not await aio_os.path.isdir(base_dir):
        return ResultsListResponse()
    files = await asyncio.to_thread(_collect_result_files, base_dir)
    return ResultsListResponse(files=files)


async def _resolve_result_file(base_dir: Path, filename: str) -> Path:
    """Validate, locate, and return the result file path or raise HTTPException."""
    file_path = _safe_resolve(base_dir, filename)
    if file_path is None or file_path.name == READY_MARKER_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename {filename!r}: path traversal or reserved marker name",
        )
    if not _is_ready(base_dir) and not _is_checkpoint_path(
        base_dir.resolve(), file_path
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Results not ready for {base_dir.name}; marker file {READY_MARKER_NAME} not present — retry after completion",
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
