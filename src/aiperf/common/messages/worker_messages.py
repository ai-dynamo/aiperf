# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import Field

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.common.types import MessageTypeT


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: MessageTypeT = MessageType.WORKER_HEALTH

    health: ProcessHealth = Field(..., description="The health of the worker process")

    pod_index: str | None = Field(
        default=None,
        description="Index of the pod owning this worker, used by each "
        "WorkerGroupManager to ignore workers belonging to other pods. This is "
        "a cluster-wide broadcast topic, so without it every group adopts every "
        "worker in the cluster. None outside Kubernetes, where a single group "
        "owns every worker anyway.",
    )

    # Worker specific fields
    task_stats: WorkerTaskStats = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker",
    )

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class WorkerStatusSummaryMessage(BaseServiceMessage):
    """Message for a worker status summary."""

    message_type: MessageTypeT = MessageType.WORKER_STATUS_SUMMARY

    worker_statuses: dict[str, WorkerStatus] = Field(
        ...,
        description="A mapping of worker IDs to their status",
    )
    worker_startup_states: dict[str, WorkerStartupState] = Field(
        default_factory=dict,
        description=(
            "A mapping of worker IDs to their startup lifecycle state. In "
            "Kubernetes mode workers report startup state to their group "
            "manager over DEALER, so this republished aggregate is the only "
            "bus-visible source of per-worker startup state."
        ),
    )


class WorkerPodStateMessage(BaseServiceMessage):
    """Controller-facing aggregate snapshot for a single Kubernetes worker pod.

    Published by a worker pod's manager so the SystemController can tell a pod
    that is merely scheduled from one whose workers have actually connected to
    the credit router and are dispatchable. The declared/ready/degraded counts
    are carried per pod rather than per worker to keep the controller's fan-in
    proportional to pod count at high worker counts.
    """

    message_type: MessageTypeT = MessageType.WORKER_POD_STATE

    # Redeclared with ge=0 so the numeric-bounds invariant in
    # tests/unit/property/test_finite_invariants.py sees a bounded timestamp.
    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the request in nanoseconds",
    )

    pod_index: str = Field(
        ...,
        description="Ordinal index of the pod within its JobSet replicated job",
    )
    declared_workers: int = Field(
        ...,
        ge=0,
        description="Number of worker processes this pod was configured to run",
    )
    declared_record_processors: int = Field(
        ...,
        ge=0,
        description="Number of record processor processes this pod was configured to run",
    )
    pod_state: str = Field(
        ...,
        description="Coarse lifecycle state of the pod as reported by its manager",
    )
    admission_state: str = Field(
        ...,
        description="Whether the controller has admitted this pod into the current benchmark",
    )
    benchmark_generation: str | None = Field(
        default=None,
        description="Benchmark generation the pod is currently serving, or None if unassigned",
    )
    dataset_generation: str | None = Field(
        default=None,
        description="Dataset generation the pod has loaded, or None if no dataset is loaded",
    )
    router_connected_workers: int = Field(
        default=0,
        ge=0,
        description="Workers in this pod with an established connection to the credit router",
    )
    dispatchable_workers: int = Field(
        default=0,
        ge=0,
        description="Workers in this pod eligible to receive credits right now",
    )
    ready_workers: int = Field(
        default=0,
        ge=0,
        description="Workers in this pod that have completed startup",
    )
    ready_record_processors: int = Field(
        default=0,
        ge=0,
        description="Record processors in this pod that have completed startup",
    )
    degraded_workers: int = Field(
        default=0,
        ge=0,
        description="Workers in this pod running in a degraded state",
    )
    degraded_record_processors: int = Field(
        default=0,
        ge=0,
        description="Record processors in this pod running in a degraded state",
    )


class WorkerStartupStateMessage(BaseServiceMessage):
    """Single worker's transition through its startup lifecycle.

    Emitted on every startup-state change so the controller can distinguish
    "still warming up" from "wedged", and surface which stage a slow worker
    is stuck in rather than only a terminal ready/not-ready bit.
    """

    message_type: MessageTypeT = MessageType.WORKER_STARTUP_STATE

    # Declared here rather than via RequiresRequestNSMixin: BaseServiceMessage
    # sits earlier in the MRO, so its Optional/None request_ns would win and the
    # transition timestamp would silently be absent.
    request_ns: int = Field(  # type: ignore[assignment]
        default_factory=time.time_ns,
        ge=0,
        description="Timestamp of the startup-state transition in nanoseconds",
    )
    startup_state: WorkerStartupState = Field(
        ...,
        description="The startup lifecycle state the worker has just entered",
    )
    pod_index: str | None = Field(
        default=None,
        description="Kubernetes pod ordinal this worker runs in, or None outside "
        "Kubernetes. This is a broadcast topic, so a WorkerGroupManager must "
        "filter on it to count only the workers in its OWN pod; without the "
        "filter every pod reports every worker in the cluster and the aggregate "
        "over-counts by the number of pods.",
    )


class WorkerGroupStatsMessage(BaseServiceMessage):
    """Aggregate stats for a single worker-group manager.

    Per-worker maps (statuses, startup states, task stats, health) are carried
    inline rather than as separate messages so the controller can populate a
    full per-child view in one fan-in, which the local web UI renders as a
    drop-down when exactly one group exists.
    """

    message_type: MessageTypeT = MessageType.WORKER_GROUP_STATS

    # Redeclared with ge=0 so the numeric-bounds invariant in
    # tests/unit/property/test_finite_invariants.py sees a bounded timestamp.
    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the request in nanoseconds",
    )

    group_id: str = Field(
        ..., description="ID of the worker group this snapshot describes"
    )
    status: WorkerStatus = Field(
        ..., description="Rolled-up status for the worker group as a whole"
    )
    task_stats: WorkerTaskStats = Field(
        ..., description="Aggregate task stats summed across the group's workers"
    )
    startup_state: WorkerStartupState | None = Field(
        default=None,
        description="Rolled-up startup state, or None if the group has not reported one",
    )
    declared_workers: int = Field(
        default=0,
        ge=0,
        description="Number of workers the group was configured to run",
    )
    ready_workers: int = Field(
        default=0,
        ge=0,
        description="Number of the group's workers that have completed startup",
    )
    health: ProcessHealth | None = Field(
        default=None,
        description="Aggregate process health for the group, or None if unavailable",
    )
    worker_statuses: dict[str, WorkerStatus] = Field(
        default_factory=dict,
        description="Per-worker status, keyed by worker service_id",
    )
    worker_startup_states: dict[str, WorkerStartupState] = Field(
        default_factory=dict,
        description="Per-worker startup state, keyed by worker service_id",
    )
    worker_task_stats: dict[str, WorkerTaskStats] = Field(
        default_factory=dict,
        description="Per-worker task stats, keyed by worker service_id",
    )
    worker_health: dict[str, ProcessHealth] = Field(
        default_factory=dict,
        description="Per-worker process health, keyed by worker service_id",
    )
    last_update_ns: int = Field(
        default_factory=time.time_ns,
        ge=0,
        description="Timestamp of this snapshot in nanoseconds",
    )
