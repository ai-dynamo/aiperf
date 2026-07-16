# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the operator progress client.

Split out of ``progress_client.py`` to keep that module focused on the HTTP
client. Retry tunables (max retries, backoff, multiplier) live on
``OperatorEnvironment.PROGRESS`` in :mod:`aiperf.operator.environment`; only
``RETRYABLE_STATUS_CODES`` (a fixed HTTP-status set, not a tunable) stays here.
"""

from pydantic import Field

from aiperf.common.enums import CreditPhase, SystemState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import AIPerfBaseModel

# HTTP status codes that are retryable (transient failures). Not a tunable —
# these are the canonical RFC-7231/7232 transient codes; do not promote to env.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class ControllerAggregateWorkerStatus(AIPerfBaseModel):
    """Controller-authored aggregate worker status, as seen from the operator.

    Wire-format mirror of
    :class:`aiperf.controller.system_controller.AggregateWorkerStatus`
    returned by the controller's progress API. The two classes have the same
    shape and are intentionally distinct so that either side can add fields
    without forcing a lockstep deploy.
    """

    ready: int = Field(default=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, description="Declared worker count.")
    dispatchable: int = Field(
        default=0,
        description="Workers eligible to receive credits.",
    )
    router_connected: int = Field(
        default=0,
        description="Workers connected to the router.",
    )
    ready_record_processors: int = Field(
        default=0,
        description="Ready record processors.",
    )
    declared_record_processors: int = Field(
        default=0,
        description="Declared record processors.",
    )
    ready_pods: int = Field(default=0, description="Usable worker pods.")
    total_pods: int = Field(default=0, description="Observed worker pods.")
    degraded_pods: int = Field(
        default=0,
        description="Usable but degraded worker pods.",
    )


class JobProgress(AIPerfBaseModel):
    """Aggregated progress across all benchmark phases.

    This model wraps phase-specific progress stats (CombinedPhaseStats) for
    each benchmark phase (warmup, profiling), providing a complete view of
    job execution status.

    Attributes:
        phases: Progress stats for each phase (warmup, profiling).
        workers: Controller-authored aggregate worker status.
        error: Error message if the job failed.
        connection_error: Connection error message if API request failed.
    """

    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict,
        description="Progress stats for each benchmark phase",
    )
    workers: ControllerAggregateWorkerStatus = Field(
        default_factory=ControllerAggregateWorkerStatus,
        description="Controller-authored aggregate worker status.",
    )
    results_exported: bool = Field(
        default=False,
        description=(
            "Whether the controller has written ALL artifacts to disk and (in "
            "K8s mode) the readiness marker. Default False so older "
            "controllers whose progress payload omits this field appear "
            "incomplete to the operator — sub-second benchmarks otherwise let "
            "is_requests_complete && is_records_complete flip True before "
            "ExporterManager.export_data() returns, so the kopf monitor would "
            "claim completion mid-export and fetch a partial artifact tree."
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
    error: str | None = Field(
        default=None,
        description="Error message if job failed",
    )
    connection_error: str | None = Field(
        default=None,
        description="Connection error if progress API was unreachable",
    )

    @property
    def current_phase(self) -> CreditPhase | None:
        """Get the most recently started phase."""
        if not self.phases:
            return None
        return max(
            self.phases.items(),
            key=lambda x: x[1].start_ns or 0,
        )[0]

    @property
    def is_complete(self) -> bool:
        """Check if the job is fully complete and artifacts are fetchable.

        Three conditions must hold:

        1. The profiling phase is present.
        2. ``is_requests_complete`` AND ``is_records_complete`` (existing
           contract — credits issued, records aggregated).
        3. ``results_exported`` is True — the controller has flushed all
           exporter artifacts and (in K8s mode) the readiness marker.

        Without (3) the kopf monitor races the controller's exporter on
        sub-second benchmarks: requests + records can complete in <1s but
        ExporterManager.export_data() takes hundreds of ms more, and the
        operator would otherwise claim completion mid-write and surface
        ``Phase.Failed``.
        """
        if not self.results_exported:
            return False
        profiling = self.phases.get("profiling")
        if profiling is None:
            return False
        if not profiling.is_requests_complete:
            return False
        # Wait for records to finish processing too — the controller won't
        # export results until all records are received.
        return profiling.is_records_complete

    @property
    def profiling_stats(self) -> CombinedPhaseStats | None:
        """Get the profiling phase stats (primary benchmark phase)."""
        return self.phases.get("profiling")

    @property
    def warmup_stats(self) -> CombinedPhaseStats | None:
        """Get the warmup phase stats."""
        return self.phases.get("warmup")
