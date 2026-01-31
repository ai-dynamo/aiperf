# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for tracking the progress of the benchmark suite."""

from pydantic import Field

from aiperf.common.enums import WorkerStatus
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
    last_update_ns: int | None = Field(
        default=None,
        description="The last time the worker was updated in nanoseconds",
    )
