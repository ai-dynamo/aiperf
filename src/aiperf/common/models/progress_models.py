# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for tracking the progress of the benchmark suite."""

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import ConfigDict, Field

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.credit_models import ProcessingStats
from aiperf.common.models.health_models import ProcessHealth
from aiperf.common.models.worker_models import WorkerTaskStats


class WorkerProcessingStats(AIPerfBaseModel):
    """Model for worker processing stats. Tracks a worker's record processing progress."""

    success_records: int = Field(
        default=0, ge=0, description="The number of records processed successfully."
    )
    error_records: int = Field(
        default=0, ge=0, description="The number of records processed with errors."
    )

    @property
    def total_records(self) -> int:
        """The total number of records processed (success + errors)."""
        return self.success_records + self.error_records


class WorkerStats(AIPerfBaseModel):
    """Stats for a worker."""

    worker_id: str = Field(
        ...,
        description="The ID of the worker",
    )
    task_stats: WorkerTaskStats = Field(
        default_factory=WorkerTaskStats,
        description="The task stats for the worker as reported by the Workers (total, completed, failed)",
    )
    processing_stats: ProcessingStats = Field(
        default_factory=ProcessingStats,
        description="The processing stats for the worker as reported by the RecordsManager (processed, errors)",
    )
    health: ProcessHealth | None = Field(
        default=None,
        description="The health of the worker as reported by the Workers",
    )
    status: WorkerStatus = Field(
        default=WorkerStatus.IDLE,
        description="The status of the worker",
    )
    startup_state: WorkerStartupState | None = Field(
        default=None,
        description="The startup lifecycle state of the worker, or None if it has not reported one",
    )
    startup_state_updated_ns: int | None = Field(
        default=None,
        ge=0,
        description="The last time the worker's startup state changed in nanoseconds",
    )
    last_update_ns: int | None = Field(
        default=None,
        description="The last time the worker was updated in nanoseconds",
    )


@dataclass(slots=True, kw_only=True)
class WorkerGroupStats:
    """Aggregate stats for one worker-group (one WorkerGroupManager).

    Mutable slotted dataclass, shared between msgspec (the ``/api/workers``
    HTTP payload) and Pydantic (``WorkersResponse``); the worker tracker
    rewrites its fields in place as group snapshots arrive, so it is not
    frozen. ``__pydantic_config__`` is required because it participates in
    Pydantic union discrimination.

    ``workers`` is the per-child :class:`WorkerStats` map, used by the local
    web UI when there is exactly one group.

    Example:
        >>> group = WorkerGroupStats(group_id="worker_group_manager_0")
        >>> group.ready_workers = 8
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    group_id: str
    """Stable identifier for this worker group (the WorkerGroupManager service id)."""
    status: WorkerStatus = WorkerStatus.IDLE
    """Coarse health status rolled up across the group's workers."""
    startup_state: WorkerStartupState | None = None
    """Startup lifecycle state, or None before the first report."""
    declared_workers: int = 0
    """Worker count the group intends to run."""
    ready_workers: int = 0
    """Workers that have completed startup and are dispatch-ready."""
    health: ProcessHealth | None = None
    """Latest aggregate process-health sample, if collected."""
    task_stats: WorkerTaskStats = field(default_factory=WorkerTaskStats)
    """Aggregate task counters summed across the group's workers."""
    workers: dict[str, WorkerStats] = field(default_factory=dict)
    """Per-child worker stats, keyed by worker id."""
    last_update_ns: int | None = None
    """Monotonic timestamp of the last update, or None before the first report."""
