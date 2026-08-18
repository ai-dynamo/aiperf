# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared FastAPI response models.

Consolidated here so both the legacy ``api.py`` router and the per-component
routers (``progress.py``, ``workers.py``, ``results.py``) can agree on a
single schema for OpenAPI and response validation.
"""

from __future__ import annotations

from pydantic import Field

from aiperf.common.enums import CaseInsensitiveStrEnum, SystemState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import AIPerfBaseModel, WorkerGroupStats
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.controller.system_controller_models import AggregateWorkerStatus


class ProgressResponse(AIPerfBaseModel):
    """Benchmark progress response."""

    phases: dict[str, CombinedPhaseStats] = Field(
        default_factory=dict, description="Per-phase progress stats"
    )
    workers: AggregateWorkerStatus = Field(
        default_factory=AggregateWorkerStatus,
        description="Controller-authored aggregate worker-pod status.",
    )
    results_exported: bool = Field(
        default=False,
        description=(
            "True only after the SystemController has written all benchmark "
            "artifacts to disk (and, in K8s mode, the readiness marker "
            "``.aiperf_results_ready.json``). The operator gates "
            "``JobProgress.is_complete`` on this field so sub-second "
            "benchmarks cannot let the kopf-timer monitor claim completion "
            "while the exporter is still flushing — without the gate the "
            "operator races the controller and surfaces ``Phase.Failed``."
        ),
    )
    system_state: SystemState = Field(
        default=SystemState.INITIALIZING,
        description=(
            "Controller-side outer-lifecycle state. Distinct from the "
            "AIPerfJob top-level `phase` (which is the operator's view); "
            "this is the controller's view of where it is in the "
            "configure → ready → profiling → processing → stopping flow. "
            "Operator mirrors this to status.subPhase."
        ),
    )


class WorkersResponse(AIPerfBaseModel):
    """Per-worker-group stats payload for /api/workers."""

    worker_groups: dict[str, WorkerGroupStats] = Field(
        description="Per-worker-group aggregated stats keyed by group_id."
    )


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
    websocket_clients: int = Field(default=0, ge=0, description="Connected clients")
