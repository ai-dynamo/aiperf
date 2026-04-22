# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP server for serving stored benchmark results from the operator PVC.

Runs as a sidecar container alongside the kopf operator, sharing the results
PVC volume. Provides two layers:

1. **File serving** — download raw result files with zstd content negotiation
2. **Analytics** — DuckDB-powered query endpoints for leaderboards, history,
   and cross-job comparisons (reads result files directly, no ETL)

Endpoints:
    GET /healthz                                        - health check

    File serving:
    GET /api/v1/results                                 - list all jobs
    GET /api/v1/results/{namespace}/{job_id}             - list files for a job
    GET /api/v1/results/{namespace}/{job_id}/{filename}  - download a file

    Analytics:
    GET /api/v1/analytics/leaderboard                   - rank runs by metric
    GET /api/v1/analytics/history                        - metric over time
    GET /api/v1/analytics/compare                       - compare specific jobs
    GET /api/v1/analytics/summary/{namespace}/{job_id}   - full summary for a job

Run: python -m aiperf.operator.results_server
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.operator.results_db import DEFAULT_COMPARE_METRICS

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024

# Configured via environment variable, matching the operator's AIPERF_RESULTS_DIR
RESULTS_DIR = Path(os.environ.get("AIPERF_RESULTS_DIR", "/data"))
SERVER_PORT = int(os.environ.get("AIPERF_RESULTS_SERVER_PORT", "8081"))


# --- Response models ---


class JobEntry(AIPerfBaseModel):
    """Summary of a stored benchmark job."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    file_count: int = Field(description="Number of stored result files")
    total_size_bytes: int = Field(description="Total size of stored files in bytes")


class JobListResponse(AIPerfBaseModel):
    """Response for listing all jobs with stored results."""

    jobs: list[JobEntry] = Field(
        default_factory=list, description="Available benchmark results"
    )


class FileEntry(AIPerfBaseModel):
    """Metadata for a stored result file."""

    name: str = Field(description="Display filename (without .zst suffix)")
    stored_name: str = Field(description="Actual filename on disk")
    size_bytes: int = Field(description="File size on disk in bytes")
    compressed: bool = Field(description="Whether the file is stored as zstd")


class FileListResponse(AIPerfBaseModel):
    """Response for listing files in a job's results directory."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    files: list[FileEntry] = Field(
        default_factory=list, description="Available result files"
    )


class LeaderboardEntry(AIPerfBaseModel):
    """A single row in a leaderboard ranking."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    value: float | None = Field(description="Metric value")
    unit: str | None = Field(description="Metric unit")
    start_time: str | None = Field(description="Benchmark start time (ISO)")
    end_time: str | None = Field(description="Benchmark end time (ISO)")
    model: str | None = Field(description="Model name")
    endpoint: str | None = Field(description="Endpoint URL")


class LeaderboardResponse(AIPerfBaseModel):
    """Ranked benchmark results for a metric."""

    metric: str = Field(description="Metric name")
    stat: str = Field(description="Statistic used for ranking")
    order: str = Field(description="Sort order (asc or desc)")
    entries: list[LeaderboardEntry] = Field(
        default_factory=list, description="Ranked entries"
    )


class HistoryEntry(AIPerfBaseModel):
    """A single data point in a time-series history."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    value: float | None = Field(description="Metric value")
    unit: str | None = Field(description="Metric unit")
    start_time: str | None = Field(description="Benchmark start time (ISO)")
    model: str | None = Field(description="Model name")
    endpoint: str | None = Field(description="Endpoint URL")


class HistoryResponse(AIPerfBaseModel):
    """Metric values over time."""

    metric: str = Field(description="Metric name")
    stat: str = Field(description="Statistic tracked")
    entries: list[HistoryEntry] = Field(
        default_factory=list, description="Time-ordered entries"
    )


class CompareResponse(AIPerfBaseModel):
    """Side-by-side comparison of specific jobs."""

    job_ids: list[str] = Field(description="Compared job IDs")
    metrics: list[str] = Field(description="Compared metrics")
    entries: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-job metric values"
    )


# --- Helpers ---


def _safe_resolve(base: Path, *parts: str) -> Path | None:
    """Resolve path parts under base, returning None on traversal attempts."""
    try:
        resolved = (base / Path(*parts)).resolve()
        resolved.relative_to(base.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _display_name(path: Path) -> str:
    """Strip .zst suffix for display."""
    if path.suffix == ".zst":
        return path.stem
    return path.name


# --- Streaming generators ---


async def _stream_zstd_raw(file_path: Path) -> AsyncIterator[bytes]:
    """Stream a .zst file directly as raw bytes."""
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(CHUNK_SIZE):
            yield chunk


async def _stream_zstd_to_gzip(file_path: Path) -> AsyncIterator[bytes]:
    """Decompress zstd, recompress as gzip (streaming)."""
    import zlib

    import zstandard

    gzip_obj = zlib.compressobj(level=6, wbits=31)
    dctx = zstandard.ZstdDecompressor()

    with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
        while chunk := await asyncio.to_thread(reader.read, CHUNK_SIZE):
            gzip_chunk = gzip_obj.compress(chunk)
            if gzip_chunk:
                yield gzip_chunk

    final = gzip_obj.flush()
    if final:
        yield final


async def _stream_zstd_decompress(file_path: Path) -> AsyncIterator[bytes]:
    """Decompress zstd on the fly."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()

    with open(file_path, "rb") as f, dctx.stream_reader(f) as reader:
        while chunk := await asyncio.to_thread(reader.read, CHUNK_SIZE):
            yield chunk


# --- FastAPI app ---


def create_app(results_dir: Path | None = None) -> FastAPI:
    """Create the FastAPI application with results and analytics routes.

    Args:
        results_dir: Base directory for stored results. Defaults to RESULTS_DIR.
    """
    from aiperf.operator.results_db import ResultsDB

    base_dir = results_dir or RESULTS_DIR
    db: ResultsDB | None = None

    # Mutable holder for Kubernetes ApiClient - populated during lifespan,
    # read by router. Typed as list to match create_jobs_router signature.
    from kubernetes_asyncio.client import ApiClient

    api_holder: list[ApiClient | None] = [None]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal db
        db = ResultsDB(base_dir)
        logger.info(f"DuckDB analytics engine initialized (results_dir={base_dir})")

        # Initialize kubernetes_asyncio client for live job/cluster endpoints.
        # Load config (in-cluster first, kubeconfig fallback) and build an
        # ApiClient that lives for the process lifetime.
        try:
            from kubernetes_asyncio import config

            from aiperf.common.noisy_loggers import suppress_noisy_http_loggers

            suppress_noisy_http_loggers()
            try:
                config.load_incluster_config()
            except config.ConfigException:
                await config.load_kube_config()
            api_holder[0] = ApiClient()
            logger.info("kubernetes_asyncio client initialized for UI endpoints")
        except Exception as e:
            logger.warning(
                f"Kubernetes client unavailable, live job endpoints disabled: {e}"
            )

        yield

        db.close()
        logger.info("DuckDB analytics engine closed")

        api = api_holder[0]
        if api is not None:
            try:
                await api.close()
            except Exception as e:
                logger.warning(f"Error closing kubernetes_asyncio client: {e}")
            api_holder[0] = None

    app = FastAPI(
        title="AIPerf Operator Results API",
        description="Serves benchmark results and analytics from the operator PVC.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register jobs/cluster router (client populated during lifespan)
    from aiperf.operator.routers.jobs import create_jobs_router

    app.include_router(create_jobs_router(api_holder))

    def _get_db() -> ResultsDB:
        if db is None:
            raise HTTPException(503, "Analytics engine not initialized")
        return db

    # ---------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ---------------------------------------------------------------
    # File serving endpoints
    # ---------------------------------------------------------------

    @app.get("/api/v1/results", response_model=JobListResponse)
    async def list_jobs() -> JobListResponse:
        """List all namespaces and jobs with stored results."""
        if not base_dir.exists():
            return JobListResponse()

        def _scan() -> list[JobEntry]:
            found: list[JobEntry] = []
            for ns_dir in sorted(base_dir.iterdir()):
                if not ns_dir.is_dir():
                    continue
                for job_dir in sorted(ns_dir.iterdir()):
                    if not job_dir.is_dir():
                        continue
                    files = [f for f in job_dir.iterdir() if f.is_file()]
                    if files:
                        found.append(
                            JobEntry(
                                namespace=ns_dir.name,
                                job_id=job_dir.name,
                                file_count=len(files),
                                total_size_bytes=sum(f.stat().st_size for f in files),
                            )
                        )
            return found

        jobs = await asyncio.to_thread(_scan)
        return JobListResponse(jobs=jobs)

    @app.get("/api/v1/results/{namespace}/{job_id}", response_model=FileListResponse)
    async def list_job_files(namespace: str, job_id: str) -> FileListResponse:
        """List files for a specific job."""
        job_dir = _safe_resolve(base_dir, namespace, job_id)
        if job_dir is None or not job_dir.is_dir():
            raise HTTPException(404, f"No results for {namespace}/{job_id}")

        def _list() -> list[FileEntry]:
            return sorted(
                [
                    FileEntry(
                        name=_display_name(f),
                        stored_name=f.name,
                        size_bytes=f.stat().st_size,
                        compressed=f.suffix == ".zst",
                    )
                    for f in job_dir.iterdir()
                    if f.is_file()
                ],
                key=lambda x: x.name,
            )

        files = await asyncio.to_thread(_list)
        return FileListResponse(namespace=namespace, job_id=job_id, files=files)

    @app.get("/api/v1/results/{namespace}/{job_id}/{filename:path}")
    async def download_file(
        namespace: str, job_id: str, filename: str, request: Request
    ) -> StreamingResponse:
        """Download a result file with content negotiation."""
        job_dir = _safe_resolve(base_dir, namespace, job_id)
        if job_dir is None or not job_dir.is_dir():
            raise HTTPException(404, f"No results for {namespace}/{job_id}")

        zst_path = _safe_resolve(job_dir, filename + ".zst")
        raw_path = _safe_resolve(job_dir, filename)

        if zst_path and zst_path.is_file():
            return _serve_zst_file(request, zst_path, filename)

        if raw_path and raw_path.is_file():
            return _serve_raw_file(request, raw_path)

        raise HTTPException(404, f"File not found: {filename}")

    # ---------------------------------------------------------------
    # Analytics endpoints (DuckDB)
    # ---------------------------------------------------------------

    @app.get("/api/v1/analytics/leaderboard", response_model=LeaderboardResponse)
    async def leaderboard(
        metric: str = Query(
            default="request_throughput",
            description="Metric to rank by (e.g. request_throughput, request_latency)",
        ),
        stat: str = Query(
            default="avg",
            description="Statistic (avg, p50, p99, min, max)",
        ),
        order: str = Query(
            default="desc",
            description="Sort order (asc or desc)",
        ),
        limit: int = Query(default=20, ge=1, le=1000, description="Max results"),
    ) -> LeaderboardResponse:
        """Rank all benchmark runs by a metric."""
        rows = await _get_db().leaderboard(
            metric=metric, stat=stat, order=order, limit=limit
        )
        return LeaderboardResponse(
            metric=metric,
            stat=stat,
            order=order,
            entries=[LeaderboardEntry(**r) for r in rows],
        )

    @app.get("/api/v1/analytics/history", response_model=HistoryResponse)
    async def history(
        metric: str = Query(
            default="request_throughput",
            description="Metric to track over time",
        ),
        stat: str = Query(default="avg", description="Statistic"),
        model: str | None = Query(
            default=None, description="Filter by model name (substring)"
        ),
        endpoint: str | None = Query(
            default=None, description="Filter by endpoint URL (substring)"
        ),
        limit: int = Query(default=100, ge=1, le=10000, description="Max results"),
    ) -> HistoryResponse:
        """Get metric values over time, optionally filtered."""
        rows = await _get_db().history(
            metric=metric,
            stat=stat,
            model=model,
            endpoint=endpoint,
            limit=limit,
        )
        return HistoryResponse(
            metric=metric,
            stat=stat,
            entries=[HistoryEntry(**r) for r in rows],
        )

    @app.get("/api/v1/analytics/compare", response_model=CompareResponse)
    async def compare(
        jobs: list[str] = Query(  # noqa: B008
            description="Job IDs to compare (repeat parameter for multiple)"
        ),
        metrics: list[str] | None = Query(  # noqa: B008
            default=None,
            description="Metrics to include (default: key performance metrics)",
        ),
    ) -> CompareResponse:
        """Compare specific jobs side-by-side."""
        rows = await _get_db().compare(job_ids=jobs, metrics=metrics)
        metric_list = metrics or list(DEFAULT_COMPARE_METRICS)
        entries = _pivot_compare_rows(rows, metric_list)
        return CompareResponse(
            job_ids=jobs,
            metrics=metric_list,
            entries=entries,
        )

    @app.get("/api/v1/analytics/summary/{namespace}/{job_id}")
    async def summary(namespace: str, job_id: str) -> dict[str, Any]:
        """Get the full aggregated summary for a single job."""
        result = await _get_db().summary(namespace, job_id)
        if result is None:
            raise HTTPException(404, f"No summary for {namespace}/{job_id}")
        return result

    @app.get("/api/v1/index")
    async def get_index() -> dict[str, Any]:
        """Get the full job index for fast lookups."""
        from aiperf.operator.job_index import get_index as _get_idx

        return await _get_idx()

    @app.get("/api/v1/config/{namespace}/{job_id}")
    async def get_job_config(namespace: str, job_id: str) -> dict[str, Any]:
        """Get the original CR spec/config for a job.

        First checks the index, then falls back to the standalone spec file,
        then falls back to extracting input_config from the summary JSON.
        """
        from aiperf.operator.job_index import get_job_spec

        # Try index first
        spec = await get_job_spec(namespace, job_id)
        if spec is not None:
            return {"source": "index", "spec": spec}

        # Try standalone spec file
        spec_file = base_dir / namespace / job_id / "job_spec.json"
        if spec_file.exists():
            import orjson

            data = orjson.loads(await asyncio.to_thread(spec_file.read_bytes))
            return {"source": "file", "spec": data}

        # Fall back to input_config from profile_export_aiperf.json
        result = await _get_db().summary(namespace, job_id)
        if result and result.get("input_config"):
            return {"source": "summary", "spec": {"benchmark": result["input_config"]}}

        raise HTTPException(404, f"No config found for {namespace}/{job_id}")

    def _pivot_compare_rows(
        rows: list[dict[str, Any]], metric_list: list[str]
    ) -> list[dict[str, Any]]:
        """Pivot raw DuckDB rows (one per job) into the format the UI expects.

        Input:  [{job_id, request_throughput_avg, request_throughput_unit, ...}, ...]
        Output: [{metric, stat, unit, values: {job_id: value}}, ...]
        """
        stats = ["avg", "p50", "p99"]
        entries: list[dict[str, Any]] = []
        for metric in metric_list:
            for stat in stats:
                col = f"{metric}_{stat}"
                unit_col = f"{metric}_unit"
                values: dict[str, float | None] = {}
                unit = None
                has_value = False
                for row in rows:
                    job_id = row.get("job_id", "")
                    namespace = row.get("namespace", "")
                    # Include namespace in key to disambiguate identical
                    # job_ids from different namespaces.
                    key = f"{namespace}/{job_id}" if namespace else job_id
                    val = row.get(col)
                    if val is not None:
                        has_value = True
                    values[key] = val
                    if unit is None and row.get(unit_col):
                        unit = row[unit_col]
                if has_value:
                    entries.append(
                        {
                            "metric": metric,
                            "stat": stat,
                            "unit": unit,
                            "values": values,
                        }
                    )
        return entries

    # Mount UI static files last (catch-all for SPA routing)
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


def _serve_zst_file(
    request: Request, zst_path: Path, display_name: str
) -> StreamingResponse:
    """Serve a .zst file with content negotiation."""
    accept = (request.headers.get("accept-encoding") or "").lower()

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{display_name}"',
        "X-Filename": display_name,
    }

    if "zstd" in accept:
        headers["Content-Encoding"] = "zstd"
        return StreamingResponse(
            _stream_zstd_raw(zst_path),
            media_type="application/octet-stream",
            headers=headers,
        )

    if "gzip" in accept:
        headers["Content-Encoding"] = "gzip"
        return StreamingResponse(
            _stream_zstd_to_gzip(zst_path),
            media_type="application/octet-stream",
            headers=headers,
        )

    return StreamingResponse(
        _stream_zstd_decompress(zst_path),
        media_type="application/octet-stream",
        headers=headers,
    )


def _serve_raw_file(request: Request, file_path: Path) -> StreamingResponse:
    """Serve an uncompressed file, optionally compressing on the fly."""
    from aiperf.common.compression import (
        CompressionEncoding,
        select_encoding,
        stream_file_compressed,
    )

    accept = request.headers.get("accept-encoding")
    encoding = select_encoding(accept, default=CompressionEncoding.IDENTITY)

    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{file_path.name}"',
        "X-Filename": file_path.name,
    }
    if encoding != CompressionEncoding.IDENTITY:
        headers["Content-Encoding"] = encoding

    return StreamingResponse(
        stream_file_compressed(file_path, encoding),
        media_type="application/octet-stream",
        headers=headers,
    )


def main() -> None:
    """Run the results server as a standalone process."""
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=SERVER_PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
