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

from aiperf.api.models.results import (
    BenchmarkResultsResponse,
    BenchmarkStatus,
    ResultsListResponse,
)
from aiperf.api.results_files import (
    build_result_file_response,
    list_result_files,
    resolve_result_file_or_404,
)
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_message
from aiperf.common.messages import ProcessAllResultsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.config.artifacts import OutputDefaults

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

    Fail-closed on the readiness marker: top-level files are withheld until
    ``write_ready_marker`` commits, so a partial export never reads as a
    result set. Checkpoint artifacts under ``checkpoints/`` are listed
    recursively regardless of readiness because they are written
    incrementally by design. Listing is top-level only (plus checkpoints)
    because this artifact directory also holds non-result subdirectories
    such as the worker upload staging area.
    """
    return await list_result_files(
        component.run.cfg.artifacts.artifact_directory, recursive=False
    )


@results_router.get("/api/results/files/{filename:path}")
async def get_result_file(
    component: ResultsDep, request: Request, filename: str
) -> StreamingResponse:
    """Download a result file by name.

    Paths escaping the artifact directory and marker names are rejected with
    400. Top-level files 404 until the readiness marker commits, so consumers
    cannot read a half-written export; checkpoint artifacts under
    ``checkpoints/`` bypass that gate.
    """
    file_path = await resolve_result_file_or_404(
        component.run.cfg.artifacts.artifact_directory, filename
    )
    return build_result_file_response(file_path, request)


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
            while chunk := await file.read(Environment.COMPRESSION.CHUNK_SIZE):
                await f.write(chunk)
            await f.flush()
        await asyncio.to_thread(_commit_uploaded_file, temporary_path, dest_path)
    except BaseException:
        with contextlib.suppress(OSError):
            await aio_os.remove(temporary_path)
        raise

    size = (await aio_os.stat(dest_path)).st_size
    return {"filename": filename, "size": str(size)}
