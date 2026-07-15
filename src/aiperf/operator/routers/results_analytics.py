# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytics routes for the operator results API.

DuckDB-backed leaderboard / history / comparison / summary endpoints plus
job-index and per-job config lookups.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException, Query
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.redact import REDACTED_VALUE, redact_headers
from aiperf.kubernetes.client import get_raw_aiperfjob
from aiperf.operator import runs_index
from aiperf.operator.results_db import DEFAULT_COMPARE_METRICS, ResultsDB
from aiperf.operator.results_layout import resolve_run_dir
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers.results_schemas import (
    CompareResponse,
    HistoryEntry,
    HistoryResponse,
    LeaderboardEntry,
    LeaderboardResponse,
)

logger = logging.getLogger("aiperf.operator.ui")


def _compare_response_key(
    job_id: str, namespace: str, qualified_jobs: set[str], bare_jobs: set[str]
) -> str:
    qualified_key = f"{namespace}/{job_id}" if namespace else job_id
    if qualified_key in qualified_jobs:
        return qualified_key
    if job_id in bare_jobs:
        return job_id
    return qualified_key


def _pivot_compare_rows(
    rows: list[dict[str, Any]], metric_list: list[str], requested_jobs: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Pivot raw DuckDB rows (one per job) into the format the UI expects.

    Input:  [{job_id, request_throughput_avg, request_throughput_unit, ...}, ...]
    Output: ([{metric, stat, unit, values: {job_id: value}}, ...],
             {<ns>/<job_id>: {gpu_count, gpu_name, model, endpoint}})

    The ``meta`` map carries per-job context — GPU count and accelerator
    model — so the UI can normalize throughput to per-GPU values and color
    runs by hardware (the InferenceX-style correlation).
    """
    qualified_jobs = {job for job in requested_jobs if "/" in job}
    bare_jobs = {job for job in requested_jobs if "/" not in job}
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
                key = _compare_response_key(
                    job_id, namespace, qualified_jobs, bare_jobs
                )
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

    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id", "")
        namespace = row.get("namespace", "")
        key = _compare_response_key(job_id, namespace, qualified_jobs, bare_jobs)
        meta[key] = {
            "gpu_count": row.get("gpu_count"),
            "gpu_name": row.get("gpu_name"),
            "model": row.get("model"),
            "endpoint": row.get("endpoint"),
        }
    return entries, meta


async def _get_run_spec_from_index(
    base_dir: Path, namespace: str, job_id: str, epoch: str | None = None
) -> dict[str, Any] | None:
    try:
        if epoch is None:
            row = await runs_index.get_latest_run(namespace, job_id)
            if row is None:
                return None
            epoch = row.epoch
        if resolve_run_dir(base_dir, namespace, job_id, epoch) is None:
            return None
        return await runs_index.get_run_spec(namespace, job_id, epoch)
    except (RuntimeError, sqlite3.Error):
        return None


def _redact_exposed_spec(spec: dict[str, Any]) -> dict[str, Any]:
    redacted = deepcopy(spec)
    benchmark = redacted.get("benchmark")
    if not isinstance(benchmark, dict):
        return redacted
    endpoint = benchmark.get("endpoint")
    if not isinstance(endpoint, dict):
        return redacted
    if endpoint.get("api_key") is not None:
        endpoint["api_key"] = REDACTED_VALUE
    headers = endpoint.get("headers")
    if isinstance(headers, dict):
        endpoint["headers"] = redact_headers(headers) or {}
    return redacted


async def _get_live_cr_config(
    api: ApiClient, namespace: str, job_id: str
) -> dict[str, Any] | None:
    try:
        raw = await get_raw_aiperfjob(api, namespace, job_id, suppress_api_errors=False)
    except TypeError as exc:
        if "suppress_api_errors" not in str(exc):
            raise
        raw = await get_raw_aiperfjob(api, namespace, job_id)
    except ApiException as exc:
        status = exc.status or 0
        reason = f" {exc.reason}" if exc.reason else ""
        raise HTTPException(
            503,
            f"Could not read live AIPerfJob config for {namespace}/{job_id}: "
            f"Kubernetes API returned {status}{reason}",
        ) from exc
    if not raw:
        return None
    spec = raw.get("spec")
    if not isinstance(spec, dict) or not spec:
        return None
    return _redact_exposed_spec(spec)


def _register_leaderboard_route(
    router: APIRouter, get_db: Callable[[], ResultsDB]
) -> None:
    """Register the ``/analytics/leaderboard`` endpoint."""

    @router.get("/analytics/leaderboard", response_model=LeaderboardResponse)
    async def leaderboard(
        *,
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
        epoch: str | None = Query(
            default=None,
            description="Restrict to one run epoch. None = latest per (ns, job).",
        ),
    ) -> LeaderboardResponse:
        """Rank all benchmark runs by a metric."""
        rows = await get_db().leaderboard(
            metric=metric, stat=stat, order=order, limit=limit, epoch=epoch
        )
        return LeaderboardResponse(
            metric=metric,
            stat=stat,
            order=order,
            entries=[LeaderboardEntry(**r) for r in rows],
        )


def _register_history_route(router: APIRouter, get_db: Callable[[], ResultsDB]) -> None:
    """Register the ``/analytics/history`` endpoint."""

    @router.get("/analytics/history", response_model=HistoryResponse)
    async def history(
        *,
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
        epoch: str | None = Query(
            default=None,
            description="Restrict to one run epoch. None = latest per (ns, job).",
        ),
    ) -> HistoryResponse:
        """Get metric values over time, optionally filtered."""
        rows = await get_db().history(
            metric=metric,
            stat=stat,
            model=model,
            endpoint=endpoint,
            limit=limit,
            epoch=epoch,
        )
        return HistoryResponse(
            metric=metric,
            stat=stat,
            entries=[HistoryEntry(**r) for r in rows],
        )


def _register_compare_route(router: APIRouter, get_db: Callable[[], ResultsDB]) -> None:
    """Register the ``/analytics/compare`` endpoint."""

    @router.get("/analytics/compare", response_model=CompareResponse)
    async def compare(
        jobs: list[str] = Query(  # noqa: B008
            description="Job IDs to compare (repeat parameter for multiple)"
        ),
        metrics: list[str] | None = Query(  # noqa: B008
            default=None,
            description="Metrics to include (default: key performance metrics)",
        ),
        epoch: str | None = Query(
            default=None,
            description="Restrict every job to one run epoch. None = latest per job.",
        ),
    ) -> CompareResponse:
        """Compare specific jobs side-by-side."""
        rows = await get_db().compare(job_ids=jobs, metrics=metrics, epoch=epoch)
        bare_jobs = {job for job in jobs if "/" not in job}
        ambiguous: dict[str, list[str]] = {}
        for job in bare_jobs:
            matches = sorted(
                {
                    f"{row.get('namespace')}/{row.get('job_id')}"
                    for row in rows
                    if row.get("job_id") == job and row.get("namespace")
                }
            )
            if len(matches) > 1:
                ambiguous[job] = matches
        if ambiguous:
            detail = "; ".join(
                f"{job!r} matches {', '.join(matches)}"
                for job, matches in sorted(ambiguous.items())
            )
            raise HTTPException(
                409,
                f"Ambiguous compare job name; use namespace/job syntax: {detail}",
            )
        metric_list = metrics or list(DEFAULT_COMPARE_METRICS)
        entries, meta = _pivot_compare_rows(rows, metric_list, jobs)
        return CompareResponse(
            job_ids=jobs,
            metrics=metric_list,
            entries=entries,
            meta=meta,
        )


def _register_summary_route(router: APIRouter, get_db: Callable[[], ResultsDB]) -> None:
    """Register the ``/analytics/summary/{namespace}/{job_id}`` endpoint."""

    @router.get("/analytics/summary/{namespace}/{job_id}")
    async def summary(
        namespace: str,
        job_id: str,
        epoch: str | None = Query(
            default=None,
            description="Run epoch to load. None = follow latest.txt.",
        ),
    ) -> dict[str, Any]:
        """Get the full aggregated summary for a single job."""
        validate_results_path_params(namespace, job_id)
        result = await get_db().summary(namespace, job_id, epoch=epoch)
        if result is None:
            raise HTTPException(404, f"No summary for {namespace}/{job_id}")
        return result


async def _config_from_job_spec_file(
    base_dir: Path, namespace: str, job_id: str, epoch: str | None
) -> dict[str, Any] | None:
    """Serve the standalone ``job_spec.json`` fallback as a config response.

    Returns the ``{"source": "file", "spec": ...}`` response body when the
    run directory holds a parseable ``job_spec.json``; None when the file is
    missing or corrupt (logged) so the caller can try the next fallback.
    """
    run = resolve_run_dir(base_dir, namespace, job_id, epoch)
    if run is None:
        return None
    spec_file = run / "job_spec.json"
    if not spec_file.exists():
        return None
    try:
        data = orjson.loads(await asyncio.to_thread(spec_file.read_bytes))
        return {"source": "file", "spec": _redact_exposed_spec(data)}
    except (orjson.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Ignoring corrupt job_spec.json for %s/%s at %s: %s",
            namespace,
            job_id,
            spec_file,
            exc,
        )
        return None


def _register_job_index_route(
    router: APIRouter, get_db: Callable[[], ResultsDB]
) -> None:
    """Register the ``/index`` endpoint."""

    @router.get("/index")
    async def get_index() -> dict[str, Any]:
        """Get the full job index for fast lookups."""
        rows = await get_db().index_entries()
        out: dict[str, Any] = {}
        for row in rows:
            out[f"{row['namespace']}/{row['job_id']}"] = row
        return out


def _register_job_config_route(
    router: APIRouter,
    get_db: Callable[[], ResultsDB],
    base_dir: Path,
    api_holder: list[ApiClient | None],
) -> None:
    """Register the ``/config/{namespace}/{job_id}`` endpoint."""

    @router.get("/config/{namespace}/{job_id}")
    async def get_job_config(
        namespace: str,
        job_id: str,
        epoch: str | None = Query(
            default=None,
            description="Run epoch to load. None = follow latest.txt.",
        ),
    ) -> dict[str, Any]:
        """Get the original CR spec/config for a job.

        Fallback chain (first hit wins):
        1. ``runs_index`` SQLite cache (``get_run_spec``) — populated as jobs land.
        2. Standalone ``<base>/<ns>/<job>/job_spec.json`` file — written by
           the operator after the controller starts.
        3. ``input_config`` from the DuckDB summary — requires a finished run.
        4. Live CR ``spec`` fetched from the apiserver — covers running jobs
           whose artifacts haven't been persisted yet (e.g. dashboard hero
           SLO chips for the currently-running CR).
        """
        validate_results_path_params(namespace, job_id)
        spec = await _get_run_spec_from_index(base_dir, namespace, job_id, epoch)
        if spec is not None:
            return {"source": "index", "spec": _redact_exposed_spec(spec)}

        file_response = await _config_from_job_spec_file(
            base_dir, namespace, job_id, epoch
        )
        if file_response is not None:
            return file_response

        result = await get_db().summary(namespace, job_id, epoch=epoch)
        if result and result.get("input_config"):
            return {
                "source": "summary",
                "spec": _redact_exposed_spec({"benchmark": result["input_config"]}),
            }

        api = api_holder[0] if api_holder else None
        if api is not None:
            spec = await _get_live_cr_config(api, namespace, job_id)
            if spec is not None:
                return {"source": "cr", "spec": spec}

        raise HTTPException(404, f"No config found for {namespace}/{job_id}")


def _register_index_routes(
    router: APIRouter,
    get_db: Callable[[], ResultsDB],
    base_dir: Path,
    api_holder: list[ApiClient | None],
) -> None:
    """Register job-index and per-job config-lookup endpoints."""
    _register_job_index_route(router, get_db)
    _register_job_config_route(router, get_db, base_dir, api_holder)


def create_results_analytics_router(
    get_db: Callable[[], ResultsDB],
    base_dir: Path,
    api_holder: list[ApiClient | None] | None = None,
) -> APIRouter:
    """Create the router for DuckDB analytics + index/config endpoints.

    Args:
        get_db: Callable returning the lifespan-managed ResultsDB instance;
            raises HTTPException(503) if not yet initialized.
        base_dir: Base directory containing ``<namespace>/<job_id>/`` result files,
            used by the config fallback to look up standalone spec files.
        api_holder: Mutable single-element list holding the kubernetes_asyncio
            ApiClient, populated during FastAPI lifespan startup. Used by the
            ``/config/{ns}/{name}`` live-CR fallback so running jobs with no
            on-disk artifacts still return their declared SLOs to the UI. If
            ``None`` or the held client is ``None``, that fallback is skipped.
    """
    router = APIRouter(prefix="/api/v1", tags=["results-analytics"])
    _holder: list[ApiClient | None] = api_holder if api_holder is not None else [None]
    _register_leaderboard_route(router, get_db)
    _register_history_route(router, get_db)
    _register_compare_route(router, get_db)
    _register_summary_route(router, get_db)
    _register_index_routes(router, get_db, base_dir, _holder)
    return router
