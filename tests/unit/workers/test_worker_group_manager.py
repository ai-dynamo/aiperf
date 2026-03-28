# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the worker-group state core."""

import asyncio
import time
from dataclasses import fields, is_dataclass

import pytest

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import ProcessHealth, ProcessHealthAggregates, WorkerTaskStats
from aiperf.workers.group_dataset_authority import GroupDatasetSnapshot
from aiperf.workers.group_runtime import GroupRuntimeRegistration
from aiperf.workers.worker_group_manager import (
    GroupChildState,
    GroupStateManager,
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)

DEFAULT_MEMORY = 1024 * 1024 * 100


class StubRuntimeAdapter:
    """Runtime adapter stub for GroupStateManager tests."""

    def __init__(self, registration: GroupRuntimeRegistration) -> None:
        self._registration = registration
        self.published_snapshots: list[WorkerStatusSummaryMessage] = []

    def build_registration(self) -> GroupRuntimeRegistration:
        return self._registration

    async def publish_summary(self, summary: WorkerStatusSummaryMessage) -> None:
        self.published_snapshots.append(summary)


def _make_health(cpu_usage: float = 25.0) -> ProcessHealth:
    return ProcessHealth(
        create_time=time.time(),
        uptime=10.0,
        cpu_usage=cpu_usage,
        memory_usage=DEFAULT_MEMORY,
    )


def _make_worker_health_message(
    cpu_usage: float = 25.0,
    *,
    total: int = 4,
    completed: int = 3,
    failed: int = 0,
) -> WorkerHealthMessage:
    return WorkerHealthMessage(
        service_id="worker-0",
        health=_make_health(cpu_usage=cpu_usage),
        task_stats=WorkerTaskStats(
            total=total,
            completed=completed,
            failed=failed,
        ),
    )


def test_group_runtime_registration_is_a_slotted_dataclass() -> None:
    registration = GroupRuntimeRegistration(
        group_id="group-0",
        declared_workers=3,
        declared_record_processors=2,
    )

    assert is_dataclass(registration)
    assert GroupRuntimeRegistration.__slots__ == (
        "group_id",
        "declared_workers",
        "declared_record_processors",
    )
    assert [field.name for field in fields(registration)] == [
        "group_id",
        "declared_workers",
        "declared_record_processors",
    ]
    assert registration.group_id == "group-0"
    assert registration.declared_workers == 3
    assert registration.declared_record_processors == 2


def test_group_dataset_snapshot_is_a_slotted_local_only_dataclass() -> None:
    snapshot = GroupDatasetSnapshot(
        benchmark_generation="bench-1",
        dataset_generation="dataset-1",
        ready=True,
        error_message="boom",
    )

    assert is_dataclass(snapshot)
    assert GroupDatasetSnapshot.__slots__ == (
        "benchmark_generation",
        "dataset_generation",
        "ready",
        "error_message",
    )
    assert [field.name for field in fields(snapshot)] == [
        "benchmark_generation",
        "dataset_generation",
        "ready",
        "error_message",
    ]
    assert not hasattr(GroupDatasetSnapshot, "__struct_fields__")
    assert snapshot.error_message == "boom"


def test_group_child_state_is_a_slotted_dataclass_with_expected_defaults() -> None:
    child = GroupChildState(child_id="worker-0")

    assert is_dataclass(child)
    assert GroupChildState.__slots__ == (
        "child_id",
        "task_stats",
        "health",
        "health_aggregates",
        "status",
        "startup_state",
        "startup_state_updated_ns",
        "last_update_ns",
        "last_error_ns",
        "last_high_load_ns",
    )
    assert child.child_id == "worker-0"
    assert child.task_stats == WorkerTaskStats()
    assert child.health is None
    assert child.health_aggregates == ProcessHealthAggregates()
    assert child.status == WorkerStatus.IDLE
    assert child.startup_state is None
    assert child.startup_state_updated_ns is None
    assert child.last_update_ns is None
    assert child.last_error_ns is None
    assert child.last_high_load_ns is None


def test_shared_worker_status_helpers_preserve_legacy_warning_and_summary_behavior() -> (
    None
):
    warnings: list[str] = []
    worker = WorkerStatusInfo(worker_id="worker-0")

    update_worker_status(
        worker,
        _make_worker_health_message(cpu_usage=97.0),
        warning=warnings.append,
    )

    assert worker.status == WorkerStatus.HIGH_LOAD
    assert warnings == [
        "CPU usage for worker-0 is 97%. AIPerf results may be inaccurate."
    ]
    assert build_worker_status_summary(
        service_id="group-manager",
        worker_infos={"worker-0": worker},
    ) == WorkerStatusSummaryMessage(
        service_id="group-manager",
        worker_statuses={"worker-0": WorkerStatus.HIGH_LOAD},
        worker_startup_states={},
    )


def test_mark_stale_workers_uses_shared_activity_window() -> None:
    worker = WorkerStatusInfo(worker_id="worker-0")
    worker.last_update_ns = 1

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("time.time_ns", lambda: int(1e12))
        mark_stale_workers({"worker-0": worker})

    assert worker.status == WorkerStatus.STALE


def test_register_group_preserves_worker_and_record_processor_capacity() -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=3,
            declared_record_processors=2,
        )
    )

    manager = GroupStateManager(runtime_adapter=runtime)

    registration = manager.register_group()

    assert registration.group_id == "group-0"
    assert registration.declared_workers == 3
    assert registration.declared_record_processors == 2
    assert manager.declared_worker_capacity == 3
    assert manager.declared_record_processor_capacity == 2
    assert manager.available_capacity == 3


def test_dispatchability_is_gated_on_dataset_readiness() -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=2,
            declared_record_processors=1,
        )
    )
    manager = GroupStateManager(runtime_adapter=runtime)
    manager.register_group()

    manager.update_child_startup_state("worker-0", WorkerStartupState.READY)
    manager.update_child_startup_state("worker-1", WorkerStartupState.STARTING)

    assert manager.dispatchable_children == 0

    manager.update_dataset_snapshot(
        GroupDatasetSnapshot(
            benchmark_generation="bench-1",
            dataset_generation="dataset-1",
            ready=True,
        )
    )

    assert manager.dispatchable_children == 1
    assert manager.available_capacity == 1


def test_child_health_aggregation_tracks_summary_statuses() -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=3,
            declared_record_processors=1,
        )
    )
    manager = GroupStateManager(runtime_adapter=runtime)
    manager.register_group()

    manager.update_dataset_snapshot(
        GroupDatasetSnapshot(
            benchmark_generation="bench-1",
            dataset_generation="dataset-1",
            ready=True,
        )
    )
    manager.update_child_startup_state("worker-0", WorkerStartupState.READY)
    manager.update_child_startup_state("worker-1", WorkerStartupState.READY)
    manager.update_child_startup_state("worker-2", WorkerStartupState.STARTING)

    manager.update_child_health(
        "worker-0",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=4, completed=4, failed=0),
    )
    manager.update_child_health(
        "worker-1",
        health=_make_health(cpu_usage=97.0),
        task_stats=WorkerTaskStats(total=3, completed=2, failed=0),
    )
    manager.update_child_health(
        "worker-2",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=2, completed=1, failed=1),
    )

    summary = manager.build_summary(service_id="group-manager")

    assert summary.worker_statuses == {
        "worker-0": WorkerStatus.IDLE,
        "worker-1": WorkerStatus.HIGH_LOAD,
        "worker-2": WorkerStatus.ERROR,
    }
    assert summary.worker_startup_states == {
        "worker-0": WorkerStartupState.READY,
        "worker-1": WorkerStartupState.READY,
        "worker-2": WorkerStartupState.STARTING,
    }
    assert manager.dispatchable_children == 2
    assert manager.available_capacity == 1


@pytest.mark.parametrize(
    ("task_stats", "expected_status"),
    [
        pytest.param(WorkerTaskStats(total=0), WorkerStatus.IDLE, id="no-tasks"),
        pytest.param(
            WorkerTaskStats(total=4, completed=4, failed=0),
            WorkerStatus.IDLE,
            id="completed-without-active-work",
        ),
        pytest.param(
            WorkerTaskStats(total=4, completed=3, failed=0),
            WorkerStatus.HEALTHY,
            id="active-work-in-progress",
        ),
    ],
)  # fmt: skip
def test_update_child_health_preserves_worker_manager_idle_semantics(
    task_stats: WorkerTaskStats,
    expected_status: WorkerStatus,
) -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=1,
            declared_record_processors=1,
        )
    )
    manager = GroupStateManager(runtime_adapter=runtime)

    child = manager.update_child_health(
        "worker-0",
        health=_make_health(cpu_usage=20.0),
        task_stats=task_stats,
    )

    assert child.status == expected_status
    assert manager.build_summary(service_id="group-manager").worker_statuses == {
        "worker-0": expected_status,
    }


def test_update_child_health_preserves_error_and_high_load_recovery_windows() -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=1,
            declared_record_processors=1,
        )
    )
    manager = GroupStateManager(runtime_adapter=runtime)

    errored_child = manager.update_child_health(
        "worker-0",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=1, completed=0, failed=1),
    )
    assert errored_child.status == WorkerStatus.ERROR

    recovered_child = manager.update_child_health(
        "worker-0",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=1, completed=1, failed=1),
    )
    assert recovered_child.status == WorkerStatus.ERROR

    high_load_child = manager.update_child_health(
        "worker-1",
        health=_make_health(cpu_usage=97.0),
        task_stats=WorkerTaskStats(total=2, completed=1, failed=0),
    )
    assert high_load_child.status == WorkerStatus.HIGH_LOAD

    recovered_high_load_child = manager.update_child_health(
        "worker-1",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=2, completed=1, failed=0),
    )
    assert recovered_high_load_child.status == WorkerStatus.HIGH_LOAD


def test_publish_summary_returns_and_publishes_current_snapshot() -> None:
    runtime = StubRuntimeAdapter(
        GroupRuntimeRegistration(
            group_id="group-0",
            declared_workers=2,
            declared_record_processors=1,
        )
    )
    manager = GroupStateManager(runtime_adapter=runtime)
    manager.update_child_startup_state("worker-0", WorkerStartupState.READY)
    manager.update_child_health(
        "worker-0",
        health=_make_health(cpu_usage=20.0),
        task_stats=WorkerTaskStats(total=4, completed=3, failed=0),
    )
    manager.update_child_startup_state("worker-1", WorkerStartupState.STARTING)

    published = asyncio.run(manager.publish_summary(service_id="group-manager"))

    assert published == WorkerStatusSummaryMessage(
        service_id="group-manager",
        worker_statuses={
            "worker-0": WorkerStatus.HEALTHY,
            "worker-1": WorkerStatus.IDLE,
        },
        worker_startup_states={
            "worker-0": WorkerStartupState.READY,
            "worker-1": WorkerStartupState.STARTING,
        },
    )
    assert runtime.published_snapshots == [published]
