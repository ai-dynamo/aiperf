# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Results router component -- owns final results state and /api/results endpoints."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from aiofiles import os as aio_os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.enums import CaseInsensitiveStrEnum, MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.record_models import ProcessRecordsResult

ResultsDep = Annotated["ResultsRouter", component_dependency("results")]

results_router = APIRouter(tags=["Results"])


class BenchmarkStatus(CaseInsensitiveStrEnum):
    """Status of a benchmark run."""

    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class BenchmarkResultsResponse(AIPerfBaseModel):
    """Final benchmark results response."""

    status: BenchmarkStatus = Field(
        description="Benchmark status: running, complete, or cancelled"
    )
    results: ProcessRecordsResult | None = Field(
        default=None, description="Final benchmark results if complete"
    )


class ResultFileInfo(AIPerfBaseModel):
    """Metadata for a single result file."""

    name: str = Field(description="Filename of the result artifact")
    size: int = Field(description="File size in bytes")


class ResultsListResponse(AIPerfBaseModel):
    """Response for listing available result files."""

    files: list[ResultFileInfo] = Field(
        default_factory=list, description="Available result files"
    )


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

    @on_message(MessageType.PROCESS_RECORDS_RESULT)
    async def _on_process_records_result(
        self, message: ProcessRecordsResultMessage
    ) -> None:
        self._final_results = message.results
        self._benchmark_complete = True


@results_router.get("/api/results", response_model=BenchmarkResultsResponse)
async def get_results(component: ResultsDep) -> BenchmarkResultsResponse:
    """Get final benchmark results."""
    if component._final_results is None:
        return BenchmarkResultsResponse(status=BenchmarkStatus.RUNNING)

    status = (
        BenchmarkStatus.CANCELLED
        if component._final_results.results.was_cancelled
        else BenchmarkStatus.COMPLETE
    )
    return BenchmarkResultsResponse(status=status, results=component._final_results)


@results_router.get("/api/results/list", response_model=ResultsListResponse)
async def list_results(component: ResultsDep) -> ResultsListResponse:
    """List all available result files in the artifacts directory."""
    results_dir = component.user_config.output.artifact_directory
    if not await aio_os.path.exists(results_dir):
        return ResultsListResponse()

    def _list_files() -> list[ResultFileInfo]:
        return sorted(
            (
                ResultFileInfo(name=e.name, size=e.stat().st_size)
                for e in results_dir.iterdir()
                if e.is_file()
            ),
            key=lambda f: f.name,
        )

    files = await asyncio.to_thread(_list_files)
    return ResultsListResponse(files=files)


@results_router.get("/api/results/files/{filename:path}")
async def get_result_file(
    component: ResultsDep, request: Request, filename: str
) -> StreamingResponse:
    """Download a result file by name."""
    artifact_dir = component.user_config.output.artifact_directory
    file_path = (artifact_dir / filename).resolve()

    if not file_path.is_relative_to(artifact_dir.resolve()):
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
