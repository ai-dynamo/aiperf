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
PROCESSING_MARKER_NAME = ".aiperf_results_processing.json"
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


def processing_marker_path(base_dir: Path) -> Path:
    """Return the sidecar processing marker path."""
    return base_dir / PROCESSING_MARKER_NAME


def write_processing_marker(base_dir: Path) -> Path:
    """Write the processing marker before export starts."""
    import orjson

    marker = processing_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(orjson.dumps({"processing": True}))
    return marker


def clear_processing_marker(base_dir: Path) -> None:
    """Remove the processing marker once final exports are stable."""
    processing_marker_path(base_dir).unlink(missing_ok=True)


def write_ready_marker(base_dir: Path, *, was_cancelled: bool = False) -> Path:
    """Write the readiness marker after exports complete."""
    import orjson

    marker = ready_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    clear_processing_marker(base_dir)
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


def _is_processing(base_dir: Path) -> bool:
    """Whether the controller is still exporting results."""
    return processing_marker_path(base_dir).is_file()


def _is_checkpoint_path(base_dir: Path, path: Path) -> bool:
    """Whether a path points at a checkpoint artifact under the results dir."""
    try:
        relative = path.relative_to(base_dir.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == CHECKPOINTS_DIR_NAME


def _safe_size(entry: Path) -> int | None:
    """Return the file size, or ``None`` if the file vanished mid-listing.

    A checkpoint parquet can be rotated/unlinked by the writer between
    ``rglob`` enumeration and ``stat``; treat the race as "skip this entry"
    rather than 500-ing the whole listing.
    """
    try:
        return entry.stat().st_size
    except OSError:
        return None


def _collect_result_files(base_dir: Path) -> list[ResultFileInfo]:
    """Enumerate every artifact under ``base_dir`` once the ready marker is set.

    Walks recursively so the AIPerfSweep harvest path (``/results/<ns>/sweeps/
    <sweep>/<epoch>/aggregate.json``, ``children.json``, ``aggregate/
    profile_export_aiperf_aggregate.{json,csv}``) and any future nested layout
    surface in the listing. The marker file itself is excluded — it is a
    sidecar-internal gate, not a downloadable artifact.

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
                or entry.name == READY_MARKER_NAME
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
    if file_path is None or file_path.name == READY_MARKER_NAME:
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
