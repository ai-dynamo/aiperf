# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI adapters over the fail-closed results-marker policy.

Thin translation layer only: the enumeration, path-safety and readiness rules
live in ``aiperf.common.results_markers`` (stdlib-only so non-serving
processes can import them). Both results surfaces -- the in-process router
(``aiperf.api.routers.results``) and the controller-side sidecar
(``aiperf.kubernetes.results_sidecar``) -- serve identical bytes, headers and
error payloads by going through here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiofiles import os as aio_os
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from aiperf.api.models.results import ResultFileInfo, ResultsListResponse
from aiperf.common.compression import (
    CompressionEncoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.results_markers import (
    ResultFileUnavailable,
    _is_processing,
    _is_ready,
    collect_ready_artifacts,
    content_type_for,
    read_ready_marker_info,
    resolve_result_file,
)


async def list_result_files(base_dir: Path, *, recursive: bool) -> ResultsListResponse:
    """Build the ``/api/results/list`` payload for a results directory."""
    if not await aio_os.path.isdir(base_dir):
        return ResultsListResponse()

    entries = await asyncio.to_thread(
        collect_ready_artifacts, base_dir, recursive=recursive
    )
    partial, failed_exporters = await asyncio.to_thread(
        read_ready_marker_info, base_dir
    )
    return ResultsListResponse(
        files=[ResultFileInfo(name=name, size=size) for name, size in entries],
        ready=_is_ready(base_dir),
        processing=_is_processing(base_dir),
        partial=partial,
        failed_exporters=failed_exporters,
    )


async def resolve_result_file_or_404(base_dir: Path, filename: str) -> Path:
    """Resolve a downloadable result file, raising the HTTP error on refusal."""
    try:
        file_path = resolve_result_file(base_dir, filename)
    except ResultFileUnavailable as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if not await aio_os.path.isfile(file_path):
        raise HTTPException(
            status_code=404, detail=f"Result file not found: {filename}"
        )
    return file_path


def build_result_file_response(file_path: Path, request: Request) -> StreamingResponse:
    """Stream a result file with content negotiation and download headers."""
    encoding = select_encoding(
        request.headers.get("accept-encoding"), default=CompressionEncoding.IDENTITY
    )

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{file_path.name}"',
        "X-Filename": file_path.name,
    }
    if encoding != CompressionEncoding.IDENTITY:
        headers["Content-Encoding"] = encoding

    return StreamingResponse(
        stream_file_compressed(file_path, encoding),
        media_type=content_type_for(file_path),
        headers=headers,
    )
