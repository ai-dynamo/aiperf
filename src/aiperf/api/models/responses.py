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
from aiperf.common.models import AIPerfBaseModel, WorkerGroupStats, WorkerStats
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.controller.system_controller_models import AggregateWorkerStatus


class ProgressResponse(AIPerfBaseModel):
    """Benchmark progress response."""

    phases: dict[str, CombinedPhaseStats] = Field(
        default_factory=dict,
        description=(
            "Per-phase progress stats, keyed by the phase's concrete name "
            "(`CombinedPhaseStats.phase_name`, e.g. a DAG/multi-instance "
            "phase's user-assigned identity) falling back to the plain "
            "`CreditPhase` value (e.g. 'profiling') when the phase has no "
            "explicit name. Breaking change: prior releases keyed this dict "
            "by the closed `CreditPhase` enum; consumers indexing by a fixed "
            "enum value should instead iterate the dict or match on each "
            "stats entry's `phase` field."
        ),
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
    """Per-worker-group stats payload for /api/workers.

    Both views are always populated. ``workers`` is the flat per-worker map
    this endpoint has always returned; ``worker_groups`` adds the Kubernetes
    group topology on top. Locally there is one synthetic group, so the two
    carry the same workers.
    """

    workers: dict[str, WorkerStats] = Field(
        default_factory=dict,
        description=(
            "Per-worker stats keyed by worker_id, flattened across all groups. "
            "Stable contract; prefer worker_groups when group topology matters. "
            "Defaulted so responses from an older controller still parse."
        ),
    )
    worker_groups: dict[str, WorkerGroupStats] = Field(
        description="Per-worker-group aggregated stats keyed by group_id."
    )


class BenchmarkStatus(CaseInsensitiveStrEnum):
    """Status of a benchmark run."""

    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class BenchmarkResultsResponse(AIPerfBaseModel):
    """Final benchmark results response."""

    status: BenchmarkStatus = Field(
        description="Benchmark status: running, complete, cancelled, or incomplete "
        "(the run terminated without ever publishing final results)"
    )
    results: ProcessRecordsResult | None = Field(
        default=None, description="Final benchmark results if complete"
    )
