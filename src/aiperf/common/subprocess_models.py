# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclass models used by ``SubprocessManager``.

Kept in a separate module so the main manager file stays under the
``file-size`` ergonomic limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.context import ForkProcess, ForkServerProcess, SpawnProcess

from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.types import ServiceTypeT
from aiperf.workers.group_runtime import GroupRuntimeRegistration


@dataclass(slots=True)
class LocalWorkerGroupManagerAdapter:
    """Local runtime adapter that models a WorkerGroupManager boundary."""

    service_id: str
    """Stable identifier for the local worker-group manager boundary."""

    declared_worker_capacity: int
    """Declared local worker capacity exposed by the adapter."""

    declared_record_processor_capacity: int
    """Declared local record-processor capacity exposed by the adapter."""

    @property
    def group_id(self) -> str:
        """Return the stable runtime group identifier."""
        return self.service_id

    @property
    def declared_workers(self) -> int:
        """Return the declared worker capacity using runtime-contract naming."""
        return self.declared_worker_capacity

    @property
    def declared_record_processors(self) -> int:
        """Return the declared record-processor capacity using runtime naming."""
        return self.declared_record_processor_capacity

    def build_registration(self) -> GroupRuntimeRegistration:
        """Build the runtime registration consumed by WorkerGroupManager."""
        return GroupRuntimeRegistration(
            group_id=self.group_id,
            declared_workers=self.declared_workers,
            declared_record_processors=self.declared_record_processors,
        )

    async def publish_summary(self, summary: WorkerStatusSummaryMessage) -> None:
        """Accept summary publication for local mode without external transport."""
        del summary


@dataclass(slots=True)
class SubprocessInfo:
    """Information about a subprocess managed by SubprocessManager."""

    service_type: ServiceTypeT
    """Type of service running in the process"""

    service_id: str
    """ID of the service running in the process"""

    process: Process | SpawnProcess | ForkProcess | ForkServerProcess | None = None
    """The underlying multiprocessing process instance"""

    launch_adapter: LocalWorkerGroupManagerAdapter | None = None
    """Optional runtime adapter associated with this subprocess entry."""

    parent_service_id: str | None = None
    """Optional parent worker-group manager service ID for local children."""

    @property
    def exitcode(self) -> int | None:
        """Exit code of the process, or None if still running or no process."""
        return self.process.exitcode if self.process else None

    @property
    def pid(self) -> int | None:
        """PID of the process, or None if no process."""
        return self.process.pid if self.process else None
