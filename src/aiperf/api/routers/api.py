# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API router for AIPerf API.

Provides core metrics, status, progress, workers, and config endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from aiperf.api.depends import ServiceDep
from aiperf.api.metrics_utils import format_metrics_json
from aiperf.api.models.responses import ProgressResponse, WorkersResponse
from aiperf.api.prometheus_formatter import format_as_prometheus

api_router = APIRouter()


# Metrics Endpoints
@api_router.get("/metrics", response_class=PlainTextResponse, tags=["Metrics"])
async def prometheus_metrics(svc: ServiceDep) -> PlainTextResponse:
    """Get metrics in Prometheus exposition format."""
    return PlainTextResponse(
        format_as_prometheus(
            metrics=list(svc._metrics),
            info_labels=svc.get_info_labels(),
        )
    )


@api_router.get("/api/metrics", tags=["Metrics"])
async def json_metrics(svc: ServiceDep) -> dict[str, Any]:
    """Get metrics in JSON format."""
    return format_metrics_json(
        metrics=list(svc._metrics),
        info_labels=svc.get_info_labels(),
        benchmark_id=svc.run.benchmark_id,
    )


# API Endpoints
@api_router.get("/api/progress", response_model=ProgressResponse, tags=["API"])
async def get_progress(svc: ServiceDep) -> ProgressResponse:
    """Get benchmark progress with full phase stats."""
    return ProgressResponse(
        phases=svc._progress_tracker._phases,
    )


@api_router.get("/api/workers", response_model=WorkersResponse, tags=["API"])
async def get_workers(svc: ServiceDep) -> WorkersResponse:
    """Get worker status with full stats."""
    return WorkersResponse(
        workers=svc._worker_tracker.workers,
    )


@api_router.get("/api/config", tags=["API"])
async def get_config(svc: ServiceDep) -> dict[str, Any]:
    """Get benchmark configuration."""
    return svc.run.cfg.model_dump(mode="json", exclude_unset=True, exclude_none=True)
