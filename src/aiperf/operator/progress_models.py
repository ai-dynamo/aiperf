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
from aiperf.common.types import PhaseKind

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

    ready: int = Field(default=0, ge=0, description="Dispatch-ready worker count.")
    total: int = Field(default=0, ge=0, description="Declared worker count.")
    dispatchable: int = Field(
        default=0,
        ge=0,
        description="Workers eligible to receive credits.",
    )
    router_connected: int = Field(
        default=0,
        ge=0,
        description="Workers connected to the router.",
    )
    ready_record_processors: int = Field(
        default=0,
        ge=0,
        description="Ready record processors.",
    )
    declared_record_processors: int = Field(
        default=0,
        ge=0,
        description="Declared record processors.",
    )
    ready_pods: int = Field(default=0, ge=0, description="Usable worker pods.")
    total_pods: int = Field(default=0, ge=0, description="Observed worker pods.")
    degraded_pods: int = Field(
        default=0,
        ge=0,
        description="Usable but degraded worker pods.",
    )


class JobProgress(AIPerfBaseModel):
    """Aggregated progress across all benchmark phases.

    This model wraps phase-specific progress stats (CombinedPhaseStats) for
    each user-named benchmark phase, providing a complete view of job
    execution status.

    Attributes:
        phases: Progress stats keyed by user-provided phase name.
        workers: Controller-authored aggregate worker status.
        error: Error message if the job failed.
        connection_error: Connection error message if API request failed.
    """

    phases: dict[str, CombinedPhaseStats] = Field(
        default_factory=dict,
        description="Progress stats keyed by user-provided benchmark phase name",
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
    def current_phase(self) -> str | None:
        """Get the name of the most recently started phase."""
        if not self.phases:
            return None
        concrete_phases = self._concrete_phases
        return max(
            concrete_phases.items(),
            key=lambda x: x[1].start_ns or 0,
        )[0]

    @property
    def _concrete_phases(self) -> dict[str, CombinedPhaseStats]:
        """Prefer explicit phase identities over legacy aggregate entries."""
        concrete = {
            phase_name: stats
            for phase_name, stats in self.phases.items()
            if stats.phase_name is not None or phase_name != str(stats.phase)
        }
        return concrete or self.phases

    @staticmethod
    def _phase_kind(stats: CombinedPhaseStats) -> PhaseKind:
        """Return semantic kind, including compatibility with older payloads."""
        if stats.phase_kind is not None:
            return stats.phase_kind
        if stats.phase == CreditPhase.WARMUP:
            return "warmup"
        return "profiling"

    @property
    def current_phase_stats(self) -> CombinedPhaseStats | None:
        """Get stats for :attr:`current_phase`."""
        phase = self.current_phase
        return None if phase is None else self.phases.get(phase)

    @property
    def primary_phase(self) -> str | None:
        """Get the latest results-producing phase by semantic kind."""
        profiling_phases = {
            phase_name: stats
            for phase_name, stats in self._concrete_phases.items()
            if self._phase_kind(stats) == "profiling"
        }
        if not profiling_phases:
            return None
        return max(
            profiling_phases.items(),
            key=lambda x: x[1].start_ns or 0,
        )[0]

    @property
    def primary_phase_stats(self) -> CombinedPhaseStats | None:
        """Get stats for :attr:`primary_phase`."""
        phase = self.primary_phase
        return None if phase is None else self.phases.get(phase)

    @property
    def is_benchmark_phase_active(self) -> bool:
        """Return whether the active phase contributes benchmark results."""
        stats = self.current_phase_stats
        return stats is not None and self._phase_kind(stats) == "profiling"

    @property
    def is_complete(self) -> bool:
        """Check if the job is fully complete and artifacts are fetchable.

        Three conditions must hold:

        1. A profiling-kind phase is present, regardless of its name.
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
        primary = self.primary_phase_stats
        if primary is None:
            return False
        if not primary.is_requests_complete:
            return False
        # Wait for records to finish processing too — the controller won't
        # export results until all records are received.
        return primary.is_records_complete

    @property
    def profiling_stats(self) -> CombinedPhaseStats | None:
        """Get stats for the latest profiling-kind phase."""
        return self.primary_phase_stats

    @property
    def warmup_stats(self) -> CombinedPhaseStats | None:
        """Get stats for the latest warmup-kind phase."""
        warmup_phases = {
            phase_name: stats
            for phase_name, stats in self._concrete_phases.items()
            if self._phase_kind(stats) == "warmup"
        }
        if not warmup_phases:
            return None
        return max(
            warmup_phases.values(),
            key=lambda stats: stats.start_ns or 0,
        )
