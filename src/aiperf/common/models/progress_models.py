# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for tracking the progress of the benchmark suite."""

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import ConfigDict

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models.credit_models import ProcessingStats
from aiperf.common.models.health_models import ProcessHealth, ProcessHealthAggregates
from aiperf.common.models.worker_models import WorkerTaskStats


@dataclass(slots=True, kw_only=True)
class WorkerProcessingStats:
    """Tracks a worker's record-processing progress.

    Mutable slotted dataclass: ``success_records`` / ``error_records`` are
    incremented in place by the records tracker, so not ``frozen``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    success_records: int = 0
    error_records: int = 0

    @property
    def total_records(self) -> int:
        """The total number of records processed (success + errors)."""
        return self.success_records + self.error_records


@dataclass(slots=True, kw_only=True)
class WorkerStats:
    """Stats for a worker.

    Mutable slotted dataclass — shared type usable natively in msgspec
    contexts (the ``/api/workers`` HTTP payload encoded via msgspec) and
    Pydantic contexts (``WorkersResponse``). The worker tracker rewrites
    ``health``, ``task_stats``, ``status``, ``startup_state``, etc. as
    messages arrive; not ``frozen``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    worker_id: str
    task_stats: WorkerTaskStats = field(default_factory=WorkerTaskStats)
    processing_stats: ProcessingStats = field(default_factory=ProcessingStats)
    health: ProcessHealth | None = None
    health_aggregates: ProcessHealthAggregates = field(
        default_factory=ProcessHealthAggregates
    )
    status: WorkerStatus = WorkerStatus.IDLE
    startup_state: WorkerStartupState | None = None
    startup_state_updated_ns: int | None = None
    last_update_ns: int | None = None
