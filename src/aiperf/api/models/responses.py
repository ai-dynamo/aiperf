# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared FastAPI response models.

Consolidated here so both the legacy ``api.py`` router and the per-component
routers (``progress.py``, ``workers.py``, ``results.py``) can agree on a
single schema for OpenAPI and response validation.
"""

from __future__ import annotations

from pydantic import Field

from aiperf.common.enums import CaseInsensitiveStrEnum, CreditPhase
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import AIPerfBaseModel, WorkerStats
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.controller.system_controller import AggregateWorkerStatus


class ProgressResponse(AIPerfBaseModel):
    """Benchmark progress response."""

    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict, description="Per-phase progress stats"
    )
    workers: AggregateWorkerStatus = Field(
        default_factory=AggregateWorkerStatus,
        description="Controller-authored aggregate worker-pod status.",
    )


class WorkersResponse(AIPerfBaseModel):
    """Worker status response."""

    workers: dict[str, WorkerStats] = Field(description="Per-worker stats")


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


class HealthResponse(AIPerfBaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Health status")
    websocket_clients: int = Field(default=0, description="Connected clients")
