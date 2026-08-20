# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-serving routes for the operator results API.

Lists and downloads raw benchmark result files from the operator PVC,
with zstd/gzip/identity content negotiation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from aiperf.common.results_markers import READY_MARKER_NAME, ready_marker_path
from aiperf.operator.artifact_names import (
    find_summary_path,
    key_export_names_from_run_dir,
)
from aiperf.operator.results_layout import EPOCH_RE, resolve_run_dir
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers.results_files_io import (
    PROFILE_EXPORT_FILENAME,
    _display_name,
    _read_profile_export_bytes,
    _safe_resolve,
    _scan_job_dirs,
    _serve_job_file,
    _stream_job_bundle,
    list_job_files_with_readiness,
)
from aiperf.operator.routers.results_schemas import (
    FileListResponse,
    ResultsHistoryListResponse,
    RunHistoryEntry,
    RunHistoryListResponse,
)

__all__ = [
    "PROFILE_EXPORT_FILENAME",
    "_display_name",
    "_safe_resolve",
    "create_results_files_router",
]


def _resolve_job_dir(
    base_dir: Path,
    namespace: str,
    job_id: str,
    epoch: str | None = None,
) -> Path:
    """Resolve a run dir under ``<base>/<ns>/<name>/``.

    Callers serving concrete result files must pass an explicit epoch so the
    UI/API cannot silently drift to a different run via ``latest.txt``.

    Rejects with 400 any ``namespace``/``job_id`` that is not a valid
    Kubernetes name BEFORE the path join — a decoded ``..%2F..`` traversal
    segment must never reach ``resolve_run_dir``.
    """
    validate_results_path_params(namespace, job_id)
    resolved = resolve_run_dir(base_dir, namespace, job_id, epoch=epoch)
    if resolved is None:
        target = f"{namespace}/{job_id}" + (f"/runs/{epoch}" if epoch else "")
        raise HTTPException(404, f"No results for {target}")
    return resolved


def _require_epoch_for_results(namespace: str, job_id: str) -> None:
    """Reject ambiguous non-epoch result lookups.

    Final artifacts are run-scoped, not job-scoped. Requiring
    ``/runs/<epoch>`` prevents callers from mixing a live job status with the
    latest persisted run's files.
    """
    raise HTTPException(
        409,
        f"Run epoch required; use /api/v1/results/{namespace}/{job_id}/runs/<epoch>",
    )


def _validate_epoch(epoch: str) -> None:
    """Raise 422 if ``epoch`` does not match the EPOCH_RE allowlist."""
    if not EPOCH_RE.match(epoch):
        raise HTTPException(422, f"Invalid epoch: {epoch}")


def _require_run_ready(job_dir: Path) -> None:
    """Reject final artifact access before the sidecar readiness marker exists."""
    if ready_marker_path(job_dir).is_file():
        return
    raise HTTPException(
        404,
        (
            f"Results not ready for {job_dir.name}; marker file "
            f"{READY_MARKER_NAME} not present — retry after completion"
        ),
    )


def _bundle_response(job_dir: Path, bundle_name: str) -> StreamingResponse:
    """Stream a zip bundle of ``job_dir`` with Content-Disposition set."""
    _require_run_ready(job_dir)
    return StreamingResponse(
        _stream_job_bundle(job_dir),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{bundle_name}"',
            "X-Filename": bundle_name,
            # Zip is already a compressed format. Setting Content-Encoding:
            # identity prevents GZipMiddleware from re-compressing it, which
            # would corrupt the archive and strip Content-Length (breaking
            # browser download-progress indicators).
            "Content-Encoding": "identity",
        },
    )


async def _build_run_history_response(
    base_dir: Path, namespace: str, job_id: str
) -> RunHistoryListResponse:
    """Resolve every run dir for a job, raising 404 when none exist."""
    from aiperf.operator.results_layout import list_runs_async

    validate_results_path_params(namespace, job_id)
    runs = await list_runs_async(base_dir, namespace, job_id)
    if not runs:
        raise HTTPException(404, f"No runs for {namespace}/{job_id}")
    latest = next((r.epoch for r in runs if r.is_latest), None)
    return RunHistoryListResponse(
        namespace=namespace,
        job_id=job_id,
        latest_epoch=latest,
        runs=[
            RunHistoryEntry(
                epoch=r.epoch,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
                total_size_bytes=r.total_size_bytes,
                is_latest=r.is_latest,
            )
            for r in runs
        ],
    )


async def _build_jobs_response(base_dir: Path) -> ResultsHistoryListResponse:
    """Scan ``base_dir`` for jobs with stored results, returning empty on miss."""
    if not base_dir.exists():
        return ResultsHistoryListResponse()
    jobs = await asyncio.to_thread(_scan_job_dirs, base_dir)
    return ResultsHistoryListResponse(jobs=jobs)


async def _build_file_list_response(
    base_dir: Path, namespace: str, job_id: str, epoch: str | None = None
) -> FileListResponse:
    """Resolve a job's run dir and enumerate its files."""
    job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
    files, ready = await asyncio.to_thread(list_job_files_with_readiness, job_dir)
    names = key_export_names_from_run_dir(job_dir)
    file_names = {file_info["name"] for file_info in files}
    return FileListResponse(
        namespace=namespace,
        job_id=job_id,
        ready=ready,
        summary_available=ready and find_summary_path(job_dir) is not None,
        per_record_filename=(
            names.jsonl_name if names.jsonl_name in file_names else None
        ),
        server_metrics_filename=(
            names.server_metrics_json_name
            if names.server_metrics_json_name in file_names
            else None
        ),
        files=files,
    )


def _epoch_bundle_response(
    base_dir: Path, namespace: str, job_id: str, epoch: str
) -> StreamingResponse:
    """Validate ``epoch`` and return a bundle for the matching run dir."""
    _validate_epoch(epoch)
    job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
    return _bundle_response(job_dir, f"{namespace}__{job_id}__{epoch}.zip")


async def _profile_export_quick_response(
    base_dir: Path, namespace: str, job_id: str, epoch: str
) -> Response:
    """Read the run-specific summary JSON for one run.

    Mirrors the per-file route but skips the directory-listing roundtrip
    the artifacts table normally performs, transparently decompressing the
    ``.zst`` companion when the uncompressed file is absent. Returns
    ``application/json`` with ``Content-Disposition: attachment;
    filename selected by ``artifacts.prefix``. Raises 404 if the artifact is
    absent (run still warming up, sidecar's ready marker has gated the
    directory upstream, or this run type doesn't produce a profile export).
    """
    _validate_epoch(epoch)
    job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
    _require_run_ready(job_dir)
    filename = key_export_names_from_run_dir(job_dir).json_name
    try:
        payload = await asyncio.to_thread(_read_profile_export_bytes, job_dir)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {filename}") from None
    return Response(
        content=payload,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Filename": filename,
        },
    )


def _register_listing_routes(router: APIRouter, base_dir: Path) -> None:
    """Register top-level listing routes and rejection aliases.

    Registers in the same relative order as the original factory: the
    job-id ``.zip`` rejection comes before the job-id listing rejection,
    and the epoch ``.zip`` route comes before the bare ``{epoch}`` listing
    so that ``runs/<epoch>.zip`` doesn't get captured by ``{epoch}``.
    """

    @router.get("/results", response_model=ResultsHistoryListResponse)
    async def list_jobs() -> ResultsHistoryListResponse:
        """List all namespaces and jobs with stored results."""
        return await _build_jobs_response(base_dir)

    @router.get("/results/{namespace}/{job_id}.zip")
    async def download_bundle(namespace: str, job_id: str) -> StreamingResponse:
        """Reject non-epoch zip downloads; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)

    @router.get("/results/{namespace}/{job_id}", response_model=FileListResponse)
    async def list_job_files(namespace: str, job_id: str) -> FileListResponse:
        """Reject non-epoch file listings; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)

    @router.get(
        "/results/{namespace}/{job_id}/runs",
        response_model=RunHistoryListResponse,
    )
    async def list_runs_endpoint(namespace: str, job_id: str) -> RunHistoryListResponse:
        """List every run dir for a job, newest first, with summary metadata."""
        return await _build_run_history_response(base_dir, namespace, job_id)


def _register_run_routes(router: APIRouter, base_dir: Path) -> None:
    """Register epoch-pinned routes (zip, listing, quick-export, files).

    Order matters: ``.zip`` and ``profile_export`` register before the bare
    ``{epoch}`` listing and the catchall ``{filename:path}`` so request
    paths route to the most-specific endpoint.
    """

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}.zip")
    async def download_historical_bundle(
        namespace: str, job_id: str, epoch: str
    ) -> StreamingResponse:
        return _epoch_bundle_response(base_dir, namespace, job_id, epoch)

    @router.get(
        "/results/{namespace}/{job_id}/runs/{epoch}",
        response_model=FileListResponse,
    )
    async def list_historical_files(
        namespace: str, job_id: str, epoch: str
    ) -> FileListResponse:
        _validate_epoch(epoch)
        return await _build_file_list_response(base_dir, namespace, job_id, epoch)

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}/profile_export")
    async def profile_export_quick(
        namespace: str,
        job_id: str,
        epoch: str,
        format: Literal["json"] = "json",
    ) -> Response:
        """Quick-export alias for the run-specific summary JSON.

        ``format`` is currently constrained to ``"json"``; the parameter
        exists so future shortcuts (csv/parquet) can be added without a
        new route. See :func:`_profile_export_quick_response`.
        """
        del format  # Reserved for future format shortcuts; only "json" today.
        return await _profile_export_quick_response(base_dir, namespace, job_id, epoch)

    @router.get("/results/{namespace}/{job_id}/runs/{epoch}/{filename:path}")
    async def download_historical_file(
        namespace: str,
        job_id: str,
        epoch: str,
        filename: str,
        *,
        request: Request,
    ) -> StreamingResponse:
        _validate_epoch(epoch)
        job_dir = _resolve_job_dir(base_dir, namespace, job_id, epoch=epoch)
        return _serve_job_file(request, job_dir, filename)

    @router.get("/results/{namespace}/{job_id}/{filename:path}")
    async def download_file(
        namespace: str, job_id: str, filename: str, request: Request
    ) -> StreamingResponse:
        """Reject non-epoch file downloads; callers must pin a run epoch."""
        _require_epoch_for_results(namespace, job_id)


def create_results_files_router(base_dir: Path) -> APIRouter:
    """Create the router for file listing/download endpoints.

    Args:
        base_dir: Base directory containing ``<namespace>/<job_id>/`` result files.
    """
    router = APIRouter(prefix="/api/v1", tags=["results-files"])
    _register_listing_routes(router, base_dir)
    _register_run_routes(router, base_dir)
    return router
