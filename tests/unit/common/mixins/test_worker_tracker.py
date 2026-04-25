# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone WorkerGroupTracker."""

from __future__ import annotations

import pytest

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.mixins.worker_tracker_mixin import WorkerGroupTracker
from aiperf.common.models import ProcessHealth, WorkerTaskStats


@pytest.fixture
def tracker() -> WorkerGroupTracker:
    return WorkerGroupTracker()


def _health(cpu: float = 10.0, mem: int = 1024) -> ProcessHealth:
    return ProcessHealth(
        pid=1, create_time=0.0, uptime=1.0, cpu_usage=cpu, memory_usage=mem
    )


def _group_msg(**overrides) -> WorkerGroupStatsMessage:
    base = dict(
        service_id="wgm-0",
        group_id="wgm-0",
        status=WorkerStatus.HEALTHY,
        declared_workers=2,
        ready_workers=2,
        health=_health(),
        task_stats=WorkerTaskStats(total=4),
        worker_statuses={
            "w-0": WorkerStatus.HEALTHY,
            "w-1": WorkerStatus.HEALTHY,
        },
        worker_startup_states={"w-0": WorkerStartupState.READY},
        worker_task_stats={
            "w-0": WorkerTaskStats(total=2),
            "w-1": WorkerTaskStats(total=2),
        },
        worker_health={"w-0": _health(), "w-1": _health()},
    )
    base.update(overrides)
    return WorkerGroupStatsMessage(**base)


class TestUpdateFromGroupMessage:
    def test_creates_group_entry(self, tracker: WorkerGroupTracker) -> None:
        group = tracker.update_from_group_message(_group_msg())
        assert group.group_id == "wgm-0"
        assert group.status == WorkerStatus.HEALTHY
        assert set(group.workers.keys()) == {"w-0", "w-1"}

    def test_replaces_group_entry_on_subsequent_update(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_group_message(_group_msg())
        tracker.update_from_group_message(
            _group_msg(
                status=WorkerStatus.HIGH_LOAD,
                worker_statuses={"w-0": WorkerStatus.HIGH_LOAD},
                worker_task_stats={"w-0": WorkerTaskStats(total=2)},
                worker_health={"w-0": _health()},
            )
        )
        group = tracker.get_group("wgm-0")
        assert group is not None
        assert group.status == WorkerStatus.HIGH_LOAD
        assert set(group.workers.keys()) == {"w-0"}

    def test_per_child_stats_populated(self, tracker: WorkerGroupTracker) -> None:
        group = tracker.update_from_group_message(_group_msg())
        assert group.workers["w-0"].task_stats.total == 2
        assert group.workers["w-0"].health is not None
        assert group.workers["w-0"].startup_state == WorkerStartupState.READY


class TestFakeInProcessFallback:
    def test_worker_health_creates_local_group(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_worker_health(
            "w-0", _health(cpu=20.0), WorkerTaskStats(total=3)
        )
        group = tracker.get_group("local")
        assert group is not None
        assert group.workers["w-0"].task_stats.total == 3

    def test_multiple_workers_aggregated_in_local_group(
        self, tracker: WorkerGroupTracker
    ) -> None:
        tracker.update_from_worker_health(
            "w-0", _health(cpu=10.0, mem=1000), WorkerTaskStats(total=2)
        )
        tracker.update_from_worker_health(
            "w-1", _health(cpu=30.0, mem=2000), WorkerTaskStats(total=4)
        )
        group = tracker.get_group("local")
        assert group is not None
        assert group.task_stats.total == 6
        assert group.health is not None
        assert group.health.memory_usage == 3000
        assert group.health.cpu_usage == 20.0


class TestWorkerGroupsProperty:
    def test_empty_initially(self, tracker: WorkerGroupTracker) -> None:
        assert tracker.worker_groups == {}

    def test_keyed_by_group_id(self, tracker: WorkerGroupTracker) -> None:
        tracker.update_from_group_message(
            _group_msg(service_id="wgm-0", group_id="wgm-0")
        )
        tracker.update_from_group_message(
            _group_msg(service_id="wgm-1", group_id="wgm-1")
        )
        assert set(tracker.worker_groups.keys()) == {"wgm-0", "wgm-1"}
