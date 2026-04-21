# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for tracking the progress of the benchmark suite."""

import msgspec

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models.base_models import PydanticStructMixin
from aiperf.common.models.credit_models import ProcessingStats
from aiperf.common.models.health_models import ProcessHealth, ProcessHealthAggregates
from aiperf.common.models.worker_models import WorkerTaskStats


class WorkerProcessingStats(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Tracks a worker's record-processing progress.

    Mutable accumulator: ``success_records`` / ``error_records`` are
    incremented in place by the records tracker, so the struct intentionally
    omits ``frozen``.
    """

    success_records: int = 0
    error_records: int = 0

    @property
    def total_records(self) -> int:
        """The total number of records processed (success + errors)."""
        return self.success_records + self.error_records


class WorkerStats(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
):
    """Stats for a worker.

    Mutable: the worker tracker rewrites ``health``, ``task_stats``,
    ``status``, ``startup_state``, etc. as messages arrive. Not frozen.

    Intentionally does not set ``omit_defaults``: WorkerStats is serialized
    over the /api/workers HTTP endpoint, and downstream consumers (TUI,
    operator) rely on every field (including ``status=idle`` and empty
    defaults) being present on the wire.
    """

    worker_id: str
    task_stats: WorkerTaskStats = msgspec.field(default_factory=WorkerTaskStats)
    processing_stats: ProcessingStats = msgspec.field(default_factory=ProcessingStats)
    health: ProcessHealth | None = None
    health_aggregates: ProcessHealthAggregates = msgspec.field(
        default_factory=ProcessHealthAggregates
    )
    status: WorkerStatus = WorkerStatus.IDLE
    startup_state: WorkerStartupState | None = None
    startup_state_updated_ns: int | None = None
    last_update_ns: int | None = None
