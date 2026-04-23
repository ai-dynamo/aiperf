# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytics routes for the operator results API.

DuckDB-backed leaderboard / history / comparison / summary endpoints plus
job-index and per-job config lookups.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes.client import get_raw_aiperfjob
from aiperf.operator.results_db import DEFAULT_COMPARE_METRICS, ResultsDB
from aiperf.operator.routers.results_schemas import (
    CompareResponse,
    HistoryEntry,
    HistoryResponse,
    LeaderboardEntry,
    LeaderboardResponse,
)


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


def _register_leaderboard_route(
    router: APIRouter, get_db: Callable[[], ResultsDB]
) -> None:
    """Register the ``/analytics/leaderboard`` endpoint."""

    @router.get("/analytics/leaderboard", response_model=LeaderboardResponse)
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
        rows = await get_db().leaderboard(
            metric=metric, stat=stat, order=order, limit=limit
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
    ) -> HistoryResponse:
        """Get metric values over time, optionally filtered."""
        rows = await get_db().history(
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
    ) -> CompareResponse:
        """Compare specific jobs side-by-side."""
        rows = await get_db().compare(job_ids=jobs, metrics=metrics)
        metric_list = metrics or list(DEFAULT_COMPARE_METRICS)
        entries = _pivot_compare_rows(rows, metric_list)
        return CompareResponse(
            job_ids=jobs,
            metrics=metric_list,
            entries=entries,
        )


def _register_summary_route(router: APIRouter, get_db: Callable[[], ResultsDB]) -> None:
    """Register the ``/analytics/summary/{namespace}/{job_id}`` endpoint."""

    @router.get("/analytics/summary/{namespace}/{job_id}")
    async def summary(namespace: str, job_id: str) -> dict[str, Any]:
        """Get the full aggregated summary for a single job."""
        result = await get_db().summary(namespace, job_id)
        if result is None:
            raise HTTPException(404, f"No summary for {namespace}/{job_id}")
        return result


def _register_index_routes(
    router: APIRouter,
    get_db: Callable[[], ResultsDB],
    base_dir: Path,
    api_holder: list[ApiClient | None],
) -> None:
    """Register job-index and per-job config-lookup endpoints."""

    @router.get("/index")
    async def get_index() -> dict[str, Any]:
        """Get the full job index for fast lookups."""
        from aiperf.operator.job_index import get_index as _get_idx

        return await _get_idx()

    @router.get("/config/{namespace}/{job_id}")
    async def get_job_config(namespace: str, job_id: str) -> dict[str, Any]:
        """Get the original CR spec/config for a job.

        Fallback chain (first hit wins):
        1. In-memory index (``get_job_spec``) — populated as jobs land.
        2. Standalone ``<base>/<ns>/<job>/job_spec.json`` file — written by
           the operator after the controller starts.
        3. ``input_config`` from the DuckDB summary — requires a finished run.
        4. Live CR ``spec`` fetched from the apiserver — covers running jobs
           whose artifacts haven't been persisted yet (e.g. dashboard hero
           SLO chips for the currently-running CR).
        """
        from aiperf.operator.job_index import get_job_spec

        spec = await get_job_spec(namespace, job_id)
        if spec is not None:
            return {"source": "index", "spec": spec}

        spec_file = base_dir / namespace / job_id / "job_spec.json"
        if spec_file.exists():
            import orjson

            data = orjson.loads(await asyncio.to_thread(spec_file.read_bytes))
            return {"source": "file", "spec": data}

        result = await get_db().summary(namespace, job_id)
        if result and result.get("input_config"):
            return {"source": "summary", "spec": {"benchmark": result["input_config"]}}

        api = api_holder[0] if api_holder else None
        if api is not None:
            raw = await get_raw_aiperfjob(api, namespace, job_id)
            if raw and raw.get("spec"):
                return {"source": "cr", "spec": raw["spec"]}

        raise HTTPException(404, f"No config found for {namespace}/{job_id}")


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
