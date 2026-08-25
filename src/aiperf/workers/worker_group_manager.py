# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkerGroupManager service export and shared worker-group state helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import ProcessHealth, ProcessHealthAggregates, WorkerTaskStats
from aiperf.workers.group_dataset_authority import (
    GroupDatasetAuthority,
    GroupDatasetSnapshot,
)
from aiperf.workers.group_lifecycle_transport import GroupLifecycleTransport
from aiperf.workers.group_runtime import GroupRuntimeAdapter, GroupRuntimeRegistration
from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)
from aiperf.workers.worker_pod_manager import WorkerGroupManagerBase


@dataclass(slots=True)
class GroupChildState:
    """Current per-child state tracked by GroupStateManager."""

    child_id: str
    """Stable identifier for the child worker."""

    task_stats: WorkerTaskStats = field(default_factory=WorkerTaskStats)
    """Latest task counters reported by the child."""

    health: ProcessHealth | None = None
    """Latest child process health snapshot."""

    health_aggregates: ProcessHealthAggregates = field(
        default_factory=ProcessHealthAggregates
    )
    """Aggregated child health statistics over time."""

    status: WorkerStatus = WorkerStatus.IDLE
    """Latest aggregate status derived from child health."""

    startup_state: WorkerStartupState | None = None
    """Latest startup state reported by the child."""

    startup_state_updated_ns: int | None = None
    """The last time the child startup state changed in nanoseconds."""

    last_update_ns: int | None = None
    """The last time child health was updated in nanoseconds."""

    last_error_ns: int | None = None
    """The last time the child entered error status."""

    last_high_load_ns: int | None = None
    """The last time the child entered high-load status."""


class WorkerGroupManager(WorkerGroupManagerBase):
    """Kubernetes worker group manager service."""


class GroupStateManager:
    """Tracks group capacity, dataset readiness, and child aggregate state."""

    def __init__(
        self,
        runtime_adapter: GroupRuntimeAdapter,
        dataset_authority: GroupDatasetAuthority | None = None,
        lifecycle_transport: GroupLifecycleTransport | None = None,
    ) -> None:
        self._runtime_adapter = runtime_adapter
        self._dataset_authority = dataset_authority or GroupDatasetAuthority()
        self._lifecycle_transport = lifecycle_transport
        self._registration: GroupRuntimeRegistration | None = None
        self._children: dict[str, GroupChildState] = {}

    @property
    def declared_worker_capacity(self) -> int:
        """Return the runtime-declared worker capacity for the group."""
        return (
            self._registration.declared_workers if self._registration is not None else 0
        )

    @property
    def declared_record_processor_capacity(self) -> int:
        """Return the runtime-declared record-processor capacity for the group."""
        return (
            self._registration.declared_record_processors
            if self._registration is not None
            else 0
        )

    @property
    def dispatchable_children(self) -> int:
        """Return the number of children eligible for dispatch right now."""
        if not self._dataset_authority.is_ready:
            return 0
        return sum(
            1
            for child in self._children.values()
            if child.startup_state == WorkerStartupState.READY
        )

    @property
    def available_capacity(self) -> int:
        """Return remaining group capacity after dispatchable children are counted."""
        return max(0, self.declared_worker_capacity - self.dispatchable_children)

    @property
    def dataset_snapshot(self) -> GroupDatasetSnapshot:
        """Return the current dataset snapshot."""
        return self._dataset_authority.snapshot

    def register_group(self) -> GroupRuntimeRegistration:
        """Register the group using the active runtime adapter."""
        self._registration = self._runtime_adapter.build_registration()
        return self._registration

    def update_dataset_snapshot(
        self, snapshot: GroupDatasetSnapshot
    ) -> GroupDatasetSnapshot:
        """Update the dataset snapshot used for dispatch gating."""
        return self._dataset_authority.update_snapshot(snapshot)

    def update_child_startup_state(
        self,
        child_id: str,
        startup_state: WorkerStartupState,
    ) -> GroupChildState:
        """Record a child startup-state transition."""
        child = self._get_or_create_child(child_id)
        child.startup_state = startup_state
        child.startup_state_updated_ns = time.time_ns()
        return child

    def update_child_health(
        self,
        child_id: str,
        health: ProcessHealth,
        task_stats: WorkerTaskStats,
    ) -> GroupChildState:
        """Record child health and derive the aggregate worker status."""
        child = self._get_or_create_child(child_id)
        child.last_update_ns = time.time_ns()
        child.status = self._derive_status(
            child=child,
            health=health,
            task_stats=task_stats,
        )
        child.health = health
        child.task_stats = task_stats
        self._update_health_aggregates(child=child, health=health)
        return child

    def build_summary(self, service_id: str) -> WorkerStatusSummaryMessage:
        """Build the child health and startup summary for publication."""
        return build_worker_status_summary(
            service_id=service_id,
            worker_infos=self._children,
        )

    async def publish_summary(self, service_id: str) -> WorkerStatusSummaryMessage:
        """Publish the current summary through the runtime adapter."""
        summary = self.build_summary(service_id=service_id)
        await self._runtime_adapter.publish_summary(summary)
        return summary

    async def fanout_command(self, child_ids: list[str], command: str) -> None:
        """Send a lifecycle command to child workers when a transport is available."""
        if self._lifecycle_transport is None:
            return
        await self._lifecycle_transport.fanout_command(child_ids, command)

    def _get_or_create_child(self, child_id: str) -> GroupChildState:
        child = self._children.get(child_id)
        if child is None:
            child = GroupChildState(child_id=child_id)
            self._children[child_id] = child
        return child

    def _derive_status(
        self,
        child: GroupChildState,
        health: ProcessHealth,
        task_stats: WorkerTaskStats,
    ) -> WorkerStatus:
        now_ns = time.time_ns()
        if task_stats.failed > child.task_stats.failed:
            child.last_error_ns = now_ns
            return WorkerStatus.ERROR
        if (
            now_ns - (child.last_error_ns or 0)
        ) / NANOS_PER_SECOND < Environment.WORKER.ERROR_RECOVERY_TIME:
            return WorkerStatus.ERROR
        if health.cpu_usage > Environment.WORKER.HIGH_LOAD_CPU_USAGE:
            child.last_high_load_ns = now_ns
            return WorkerStatus.HIGH_LOAD
        if (
            now_ns - (child.last_high_load_ns or 0)
        ) / NANOS_PER_SECOND < Environment.WORKER.HIGH_LOAD_RECOVERY_TIME:
            return WorkerStatus.HIGH_LOAD
        if task_stats.total == 0 or task_stats.in_progress == 0:
            return WorkerStatus.IDLE
        return WorkerStatus.HEALTHY

    def _update_health_aggregates(
        self,
        child: GroupChildState,
        health: ProcessHealth,
    ) -> None:
        aggregates = child.health_aggregates
        aggregates.memory_usage.update(health.memory_usage)
        aggregates.cpu_usage.update(health.cpu_usage)
        aggregates.num_threads.update(health.num_threads)
        if health.num_ctx_switches:
            aggregates.voluntary_ctx_switches.update(health.num_ctx_switches[0])
            aggregates.involuntary_ctx_switches.update(health.num_ctx_switches[1])
        if health.io_counters:
            aggregates.io_read_bytes.update(health.io_counters[4])
            aggregates.io_write_bytes.update(health.io_counters[5])
        if health.cpu_times:
            aggregates.cpu_time_user.update(health.cpu_times[0])
            aggregates.cpu_time_system.update(health.cpu_times[1])
            aggregates.cpu_time_iowait.update(health.cpu_times[2])


__all__ = [
    "GroupChildState",
    "GroupStateManager",
    "WorkerGroupManager",
    "WorkerStatusInfo",
    "build_worker_status_summary",
    "mark_stale_workers",
    "update_worker_status",
]
