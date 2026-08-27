# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Results router component -- owns final results state and /api/results endpoints."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

from aiofiles import os as aio_os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.api.models.responses import BenchmarkResultsResponse, BenchmarkStatus
from aiperf.api.models.results import ResultFileInfo, ResultsListResponse
from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import BenchmarkCompleteMessage, ProcessAllResultsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.record_models import ProcessRecordsResult

ResultsDep = Annotated["ResultsRouter", component_dependency("results")]

results_router = APIRouter(tags=["Results"])

# Grace period after BENCHMARK_COMPLETE for the unified ProcessAllResultsMessage
# to arrive before we stop waiting on it. RecordsManager publishes its per-stream
# results first (the actual shutdown trigger the controller reacts to) and only
# afterwards, as a separate step, publishes ProcessAllResultsMessage. If
# RecordsManager dies/is reaped between those two steps, that message never
# arrives even though SystemController still announces BENCHMARK_COMPLETE
# unconditionally during teardown. Without this bound, /api/results would
# report RUNNING forever.
_FINAL_RESULTS_GRACE_SEC: float = 10.0


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
        self._benchmark_complete_was_cancelled: bool = False
        self._benchmark_complete_at: float | None = None

    def get_router(self) -> APIRouter:
        return results_router

    @on_message(MessageType.PROCESS_ALL_RESULTS)
    async def _on_process_all_results(self, message: ProcessAllResultsMessage) -> None:
        self._final_results = message.results

    @on_message(MessageType.BENCHMARK_COMPLETE)
    async def _on_benchmark_complete(self, message: BenchmarkCompleteMessage) -> None:
        # Results files have been exported to disk by the time this message
        # arrives (the controller exports BEFORE publishing this message).
        # Only now do we report "complete" to external consumers so they
        # can safely fetch all result files.
        self._benchmark_complete = True
        self._benchmark_complete_was_cancelled = message.was_cancelled
        self._benchmark_complete_at = time.monotonic()


@results_router.get("/api/results", response_model=BenchmarkResultsResponse)
async def get_results(component: ResultsDep) -> BenchmarkResultsResponse:
    """Get final benchmark results."""
    if component._final_results is not None:
        status = (
            BenchmarkStatus.CANCELLED
            if component._final_results.results.was_cancelled
            else BenchmarkStatus.COMPLETE
        )
        return BenchmarkResultsResponse(status=status, results=component._final_results)

    if not component._benchmark_complete:
        return BenchmarkResultsResponse(status=BenchmarkStatus.RUNNING)

    elapsed = time.monotonic() - (component._benchmark_complete_at or 0.0)
    if elapsed < _FINAL_RESULTS_GRACE_SEC:
        # BENCHMARK_COMPLETE and PROCESS_ALL_RESULTS are published by
        # different services with no ordering guarantee between them; give
        # the unified results message a bounded window to still show up.
        return BenchmarkResultsResponse(status=BenchmarkStatus.RUNNING)

    # The benchmark was announced complete, but RecordsManager never
    # published the unified results (most likely it died/was reaped between
    # its per-stream publish and ProcessAllResultsMessage). A cancelled run
    # is still reported as cancelled; anything else is INCOMPLETE rather than
    # COMPLETE, so callers are never told a resultless run finished cleanly.
    status = (
        BenchmarkStatus.CANCELLED
        if component._benchmark_complete_was_cancelled
        else BenchmarkStatus.INCOMPLETE
    )
    return BenchmarkResultsResponse(status=status, results=None)


@results_router.get("/api/results/list", response_model=ResultsListResponse)
async def list_results(component: ResultsDep) -> ResultsListResponse:
    """List available result files in the artifacts directory.

    Lists only regular files at the top level of the artifact directory,
    sorted by name. Files are listed as soon as they exist — there is no
    readiness gate, so partial exports may appear while a run is still
    writing them.
    """
    results_dir = component.run.cfg.artifacts.artifact_directory
    if not await aio_os.path.exists(results_dir):
        return ResultsListResponse()

    def _list_files() -> list[ResultFileInfo]:
        files: list[ResultFileInfo] = []

        files.extend(
            ResultFileInfo(name=entry.name, size=entry.stat().st_size)
            for entry in results_dir.iterdir()
            if entry.is_file()
        )

        return sorted(files, key=lambda f: f.name)

    files = await asyncio.to_thread(_list_files)
    return ResultsListResponse(files=files)


@results_router.get("/api/results/files/{filename:path}")
async def get_result_file(
    component: ResultsDep, request: Request, filename: str
) -> StreamingResponse:
    """Download a result file by name.

    Any file inside the artifact directory is downloadable as soon as it
    exists; paths escaping the artifact directory are rejected with 400.
    """
    artifact_dir = component.run.cfg.artifacts.artifact_directory
    file_path = (artifact_dir / filename).resolve()
    artifact_dir_resolved = artifact_dir.resolve()

    if not file_path.is_relative_to(artifact_dir_resolved):
        raise HTTPException(status_code=400, detail="Invalid filename")
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
