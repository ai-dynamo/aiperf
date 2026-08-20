# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Results router component -- owns final results state and /api/results endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from pathlib import Path
from typing import Annotated, Any

import aiofiles
from aiofiles import os as aio_os
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from aiperf.api.models.responses import BenchmarkResultsResponse, BenchmarkStatus
from aiperf.api.models.results import ResultFileInfo, ResultsListResponse
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import ProcessAllResultsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.common.results_markers import (
    CHECKPOINTS_DIR_NAME,
    READY_MARKER_NAME,
    _is_processing,
    ready_marker_path,
)
from aiperf.config.defaults import OutputDefaults

ResultsDep = Annotated["ResultsRouter", component_dependency("results")]

results_router = APIRouter(tags=["Results"])


def _commit_uploaded_file(temporary_path: Path, destination_path: Path) -> None:
    """Fsync a completed upload, atomically publish it, then fsync its directory."""
    if IS_WINDOWS:
        os.replace(temporary_path, destination_path)
        return

    file_descriptor = os.open(temporary_path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.replace(temporary_path, destination_path)

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_descriptor = os.open(destination_path.parent, directory_flags)
    except OSError:
        return
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}


class ResultsRouter(MessageBusClientMixin, BaseRouter):
    """Owns final benchmark results and exposes /api/results endpoints."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._final_results: ProcessRecordsResult | None = None
        self._benchmark_complete: bool = False

    def get_router(self) -> APIRouter:
        return results_router

    @on_message(MessageType.PROCESS_ALL_RESULTS)
    async def _on_process_all_results(self, message: ProcessAllResultsMessage) -> None:
        self._final_results = message.results

    @on_message(MessageType.BENCHMARK_COMPLETE)
    async def _on_benchmark_complete(self, message: Any) -> None:
        # Results files have been exported to disk by the time this message
        # arrives (the controller exports BEFORE publishing this message).
        # Only now do we report "complete" to external consumers so they
        # can safely fetch all result files.
        self._benchmark_complete = True


@results_router.get("/api/results", response_model=BenchmarkResultsResponse)
async def get_results(component: ResultsDep) -> BenchmarkResultsResponse:
    """Get final benchmark results."""
    if not component._benchmark_complete or component._final_results is None:
        return BenchmarkResultsResponse(status=BenchmarkStatus.RUNNING)

    status = (
        BenchmarkStatus.CANCELLED
        if component._final_results.results.was_cancelled
        else BenchmarkStatus.COMPLETE
    )
    return BenchmarkResultsResponse(status=status, results=component._final_results)


@results_router.get("/api/results/list", response_model=ResultsListResponse)
async def list_results(component: ResultsDep) -> ResultsListResponse:
    """List available result files in the artifacts directory.

    Mirrors the readiness gate enforced by the controller's results sidecar:
    until the ``.aiperf_results_ready.json`` marker is written, only
    ``checkpoints/`` artifacts are listed. This prevents the operator from
    fetching partial top-level exports during sub-second benchmarks where the
    fetch can race the controller's export pipeline.
    """
    results_dir = component.run.cfg.artifacts.artifact_directory
    if not await aio_os.path.exists(results_dir):
        return ResultsListResponse()

    def _list_files() -> list[ResultFileInfo]:
        files: list[ResultFileInfo] = []

        if ready_marker_path(results_dir).is_file():
            files.extend(
                ResultFileInfo(name=entry.name, size=entry.stat().st_size)
                for entry in results_dir.iterdir()
                if entry.is_file() and entry.name != READY_MARKER_NAME
            )

        cp_dir = results_dir / CHECKPOINTS_DIR_NAME
        if cp_dir.is_dir():
            files.extend(
                ResultFileInfo(
                    name=entry.relative_to(results_dir).as_posix(),
                    size=entry.stat().st_size,
                )
                for entry in cp_dir.rglob("*")
                if entry.is_file()
            )

        return sorted(files, key=lambda f: f.name)

    files = await asyncio.to_thread(_list_files)
    return ResultsListResponse(
        files=files,
        ready=ready_marker_path(results_dir).is_file(),
        processing=_is_processing(results_dir),
    )


@results_router.get("/api/results/files/{filename:path}")
async def get_result_file(
    component: ResultsDep, request: Request, filename: str
) -> StreamingResponse:
    """Download a result file by name.

    Until the readiness marker is present, only files under ``checkpoints/``
    are downloadable. This mirrors the sidecar's gate so the primary endpoint
    cannot serve partial exports during the controller's sub-second export race.
    """
    artifact_dir = component.run.cfg.artifacts.artifact_directory
    file_path = (artifact_dir / filename).resolve()
    artifact_dir_resolved = artifact_dir.resolve()

    if not file_path.is_relative_to(artifact_dir_resolved):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if file_path.name == READY_MARKER_NAME:
        raise HTTPException(
            status_code=404, detail=f"Result file not found: {filename}"
        )

    is_checkpoint = file_path.relative_to(artifact_dir_resolved).parts[:1] == (
        CHECKPOINTS_DIR_NAME,
    )
    if not ready_marker_path(artifact_dir).is_file() and not is_checkpoint:
        processing_detail = (
            " export still processing;" if _is_processing(artifact_dir) else ""
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results not ready for {artifact_dir.name};{processing_detail} "
                f"marker file {READY_MARKER_NAME} not present — retry after completion"
            ),
        )

    if not await aio_os.path.isfile(file_path):
        raise HTTPException(
            status_code=404, detail=f"Result file not found: {filename}"
        )

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


@results_router.post("/api/results/upload/{filename:path}", status_code=201)
async def upload_result_file(
    component: ResultsDep, filename: str, file: UploadFile
) -> dict[str, str]:
    """Upload a result file (used by worker pods to send raw records to controller).

    Files are saved to the raw_records subdirectory of the artifact directory.
    Only .jsonl files with the raw_records_ prefix are accepted.
    """
    if not filename.startswith("raw_records_") or not filename.endswith(".jsonl"):
        raise HTTPException(
            status_code=400,
            detail="Only raw_records_*.jsonl files are accepted",
        )

    artifact_dir = component.run.cfg.artifacts.artifact_directory
    raw_records_dir = artifact_dir / OutputDefaults.RAW_RECORDS_FOLDER
    dest_path = (raw_records_dir / filename).resolve()

    if not dest_path.is_relative_to(raw_records_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    await asyncio.to_thread(raw_records_dir.mkdir, parents=True, exist_ok=True)

    temporary_path = raw_records_dir / f".{filename}.{uuid.uuid4().hex}.uploading"
    try:
        async with aiofiles.open(temporary_path, "wb") as f:
            while chunk := await file.read(64 * 1024):
                await f.write(chunk)
            await f.flush()
        await asyncio.to_thread(_commit_uploaded_file, temporary_path, dest_path)
    except BaseException:
        with contextlib.suppress(OSError):
            await aio_os.remove(temporary_path)
        raise

    size = (await aio_os.stat(dest_path)).st_size
    return {"filename": filename, "size": str(size)}
