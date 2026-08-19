# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone WorkerTracker class."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.mixins.worker_tracker_mixin import (
    LOCAL_GROUP_ID,
    WorkerTracker,
    worst_status,
)
from aiperf.common.models import ProcessHealth, WorkerStats, WorkerTaskStats


@pytest.fixture
def tracker() -> WorkerTracker:
    """Create a fresh WorkerTracker."""
    return WorkerTracker()


@pytest.fixture
def sample_health() -> ProcessHealth:
    """Create sample ProcessHealth data."""
    return ProcessHealth(
        pid=1234,
        create_time=1000.0,
        uptime=60.0,
        cpu_usage=25.0,
        memory_usage=1024 * 1024,
    )


@pytest.fixture
def sample_task_stats() -> WorkerTaskStats:
    """Create sample WorkerTaskStats data."""
    return WorkerTaskStats(total=10, failed=1)


class TestWorkerTrackerUpdateStats:
    """Test WorkerTracker.update_worker_stats."""

    def test_creates_worker_on_first_update(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """Test that a new worker is created on first stats update."""
        result = tracker.update_worker_stats(
            "worker-1", sample_health, sample_task_stats
        )
        assert result.worker_id == "worker-1"
        assert result.health == sample_health
        assert result.task_stats == sample_task_stats

    def test_updates_existing_worker(
        self, tracker: WorkerTracker, sample_health: ProcessHealth
    ) -> None:
        """Test that subsequent calls update the existing worker."""
        initial_stats = WorkerTaskStats(total=5, failed=0)
        tracker.update_worker_stats("worker-1", sample_health, initial_stats)

        updated_stats = WorkerTaskStats(total=20, failed=2)
        result = tracker.update_worker_stats("worker-1", sample_health, updated_stats)
        assert result.task_stats.total == 20
        assert result.task_stats.failed == 2

    def test_returns_same_worker_stats_object(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """Test that update returns the stored WorkerStats (same reference)."""
        result = tracker.update_worker_stats(
            "worker-1", sample_health, sample_task_stats
        )
        assert tracker.get_worker_stats("worker-1") is result


class TestWorkerTrackerUpdateStatuses:
    """Test WorkerTracker.update_worker_statuses."""

    def test_creates_workers_from_status_summary(self, tracker: WorkerTracker) -> None:
        """Test that workers are created if they don't exist during status update."""
        tracker.update_worker_statuses(
            {"w-1": WorkerStatus.HEALTHY, "w-2": WorkerStatus.IDLE}
        )
        assert tracker.get_worker_stats("w-1").status == WorkerStatus.HEALTHY
        assert tracker.get_worker_stats("w-2").status == WorkerStatus.IDLE

    def test_overwrites_existing_status(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """Test that status update overwrites existing worker status."""
        tracker.update_worker_stats("w-1", sample_health, sample_task_stats)
        assert tracker.get_worker_stats("w-1").status == WorkerStatus.IDLE

        tracker.update_worker_statuses({"w-1": WorkerStatus.HEALTHY})
        assert tracker.get_worker_stats("w-1").status == WorkerStatus.HEALTHY

    def test_empty_statuses_dict(self, tracker: WorkerTracker) -> None:
        """Test that empty status dict is a no-op."""
        tracker.update_worker_statuses({})
        assert tracker.workers == {}


class TestWorkerTrackerGetWorkerStats:
    """Test WorkerTracker.get_worker_stats."""

    def test_returns_none_for_unknown_worker(self, tracker: WorkerTracker) -> None:
        """Test that getting stats for unknown worker returns None."""
        assert tracker.get_worker_stats("nonexistent") is None

    def test_returns_stats_for_known_worker(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """Test that getting stats for known worker returns WorkerStats."""
        tracker.update_worker_stats("worker-1", sample_health, sample_task_stats)
        stats = tracker.get_worker_stats("worker-1")
        assert stats is not None
        assert stats.worker_id == "worker-1"


class TestWorkerTrackerWorkersProperty:
    """Test WorkerTracker.workers property."""

    def test_empty_initially(self, tracker: WorkerTracker) -> None:
        """Test that workers dict is empty initially."""
        assert tracker.workers == {}

    def test_tracks_multiple_workers(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """Test tracking multiple workers simultaneously."""
        tracker.update_worker_stats("w-1", sample_health, sample_task_stats)
        tracker.update_worker_stats("w-2", sample_health, sample_task_stats)
        tracker.update_worker_stats("w-3", sample_health, sample_task_stats)
        assert len(tracker.workers) == 3
        assert set(tracker.workers.keys()) == {"w-1", "w-2", "w-3"}

    @pytest.mark.parametrize(
        "status",
        [
            param(WorkerStatus.IDLE, id="idle"),
            param(WorkerStatus.HEALTHY, id="healthy"),
            param(WorkerStatus.HIGH_LOAD, id="high-load"),
            param(WorkerStatus.ERROR, id="error"),
            param(WorkerStatus.STALE, id="stale"),
        ],
    )  # fmt: skip
    def test_preserves_status_values(
        self, tracker: WorkerTracker, status: WorkerStatus
    ) -> None:
        """Test that all WorkerStatus values are tracked correctly."""
        tracker.update_worker_statuses({"w-1": status})
        assert tracker.workers["w-1"].status == status


def _group_message(
    group_id: str,
    worker_statuses: dict[str, WorkerStatus],
    **kwargs,
) -> WorkerGroupStatsMessage:
    """Build a WorkerGroupStatsMessage for ``group_id`` with the given children."""
    return WorkerGroupStatsMessage(
        service_id=group_id,
        group_id=group_id,
        status=kwargs.pop("status", WorkerStatus.HEALTHY),
        task_stats=kwargs.pop("task_stats", WorkerTaskStats()),
        worker_statuses=worker_statuses,
        **kwargs,
    )


class TestWorkerTrackerGroups:
    """Group-keyed storage: update_from_group_message, get_group, flat view."""

    def test_update_from_group_message_populates_group(
        self, tracker: WorkerTracker, sample_health: ProcessHealth
    ) -> None:
        group = tracker.update_from_group_message(
            _group_message(
                "wgm-0",
                {"w-0": WorkerStatus.HEALTHY, "w-1": WorkerStatus.IDLE},
                startup_state=WorkerStartupState.READY,
                declared_workers=2,
                ready_workers=1,
                health=sample_health,
                task_stats=WorkerTaskStats(total=7, completed=5, failed=2),
                worker_startup_states={"w-0": WorkerStartupState.READY},
                worker_task_stats={"w-0": WorkerTaskStats(total=4)},
                worker_health={"w-0": sample_health},
            )
        )
        assert group is tracker.get_group("wgm-0")
        assert group.startup_state == WorkerStartupState.READY
        assert (group.declared_workers, group.ready_workers) == (2, 1)
        assert group.task_stats.total == 7
        assert group.health == sample_health
        assert set(group.workers) == {"w-0", "w-1"}
        assert group.workers["w-0"].startup_state == WorkerStartupState.READY
        assert group.workers["w-0"].task_stats.total == 4
        assert group.workers["w-0"].health == sample_health
        # Children absent from the per-worker maps still land, with defaults.
        assert group.workers["w-1"].startup_state is None
        assert group.workers["w-1"].task_stats.total == 0

    def test_update_from_group_message_replaces_prior_snapshot(
        self, tracker: WorkerTracker
    ) -> None:
        tracker.update_from_group_message(
            _group_message("wgm-0", {"w-0": WorkerStatus.HEALTHY})
        )
        tracker.update_from_group_message(
            _group_message("wgm-0", {"w-1": WorkerStatus.HEALTHY})
        )
        assert set(tracker.get_group("wgm-0").workers) == {"w-1"}
        assert list(tracker.worker_groups) == ["wgm-0"]

    def test_get_group_missing_returns_none(self, tracker: WorkerTracker) -> None:
        assert tracker.get_group("nope") is None

    def test_flat_view_spans_every_group(self, tracker: WorkerTracker) -> None:
        """A worker in the second group must be visible from the flat view."""
        tracker.update_from_group_message(
            _group_message(
                "wgm-0", {"w-0": WorkerStatus.HEALTHY, "w-1": WorkerStatus.IDLE}
            )
        )
        tracker.update_from_group_message(
            _group_message(
                "wgm-1", {"w-2": WorkerStatus.ERROR, "w-3": WorkerStatus.HIGH_LOAD}
            )
        )
        assert set(tracker.workers) == {"w-0", "w-1", "w-2", "w-3"}
        assert tracker.workers["w-2"].status == WorkerStatus.ERROR
        assert tracker.workers["w-3"].status == WorkerStatus.HIGH_LOAD
        assert tracker.get_worker_stats("w-3") is tracker.workers["w-3"]

    def test_flat_view_is_a_snapshot_not_live_storage(
        self, tracker: WorkerTracker
    ) -> None:
        """Mutating the returned mapping must not alter tracker state."""
        tracker.update_from_group_message(
            _group_message("wgm-0", {"w-0": WorkerStatus.HEALTHY})
        )
        snapshot = tracker.workers
        snapshot["injected"] = WorkerStats(worker_id="injected")
        del snapshot["w-0"]
        assert set(tracker.workers) == {"w-0"}
        assert tracker.get_worker_stats("injected") is None

    def test_duplicate_worker_id_first_group_wins_for_both_accessors(
        self, tracker: WorkerTracker
    ) -> None:
        """Both accessors must agree on the winner: first group in insertion order."""
        tracker.update_from_group_message(
            _group_message("wgm-0", {"dup": WorkerStatus.HEALTHY})
        )
        tracker.update_from_group_message(
            _group_message("wgm-1", {"dup": WorkerStatus.ERROR})
        )
        first = tracker.get_group("wgm-0").workers["dup"]
        assert tracker.workers["dup"] is first
        assert tracker.get_worker_stats("dup") is first
        assert tracker.workers["dup"].status == WorkerStatus.HEALTHY

    def test_local_and_group_entries_coexist(
        self,
        tracker: WorkerTracker,
        sample_health: ProcessHealth,
        sample_task_stats: WorkerTaskStats,
    ) -> None:
        """In-process health folds into "local" without displacing real groups."""
        tracker.update_from_group_message(
            _group_message("wgm-0", {"w-0": WorkerStatus.HEALTHY})
        )
        tracker.update_worker_stats("w-local", sample_health, sample_task_stats)
        assert set(tracker.worker_groups) == {"wgm-0", LOCAL_GROUP_ID}
        assert set(tracker.workers) == {"w-0", "w-local"}
        assert tracker.get_group(LOCAL_GROUP_ID).declared_workers == 1


class TestWorstStatus:
    """Rollup precedence used for the synthetic local group."""

    def test_empty_is_idle(self) -> None:
        assert worst_status([]) == WorkerStatus.IDLE

    @pytest.mark.parametrize(
        "statuses,expected",
        [
            param([WorkerStatus.IDLE, WorkerStatus.HEALTHY], WorkerStatus.HEALTHY, id="healthy-beats-idle"),
            param([WorkerStatus.HEALTHY, WorkerStatus.HIGH_LOAD], WorkerStatus.HIGH_LOAD, id="high-load-beats-healthy"),
            param([WorkerStatus.HIGH_LOAD, WorkerStatus.STALE], WorkerStatus.STALE, id="stale-beats-high-load"),
            param([WorkerStatus.STALE, WorkerStatus.ERROR], WorkerStatus.ERROR, id="error-beats-stale"),
            param([WorkerStatus.ERROR, WorkerStatus.IDLE], WorkerStatus.ERROR, id="order-independent"),
        ],
    )  # fmt: skip
    def test_precedence(
        self, statuses: list[WorkerStatus], expected: WorkerStatus
    ) -> None:
        assert worst_status(statuses) == expected

    def test_local_group_status_rolls_up_worst_child(
        self, tracker: WorkerTracker
    ) -> None:
        tracker.update_worker_statuses(
            {"w-0": WorkerStatus.HEALTHY, "w-1": WorkerStatus.ERROR}
        )
        assert tracker.get_group(LOCAL_GROUP_ID).status == WorkerStatus.ERROR

    def test_local_group_task_stats_sum_across_children(
        self, tracker: WorkerTracker, sample_health: ProcessHealth
    ) -> None:
        tracker.update_worker_stats(
            "w-0", sample_health, WorkerTaskStats(total=3, completed=2, failed=1)
        )
        tracker.update_worker_stats(
            "w-1", sample_health, WorkerTaskStats(total=5, completed=4, failed=1)
        )
        group = tracker.get_group(LOCAL_GROUP_ID)
        assert (group.task_stats.total, group.task_stats.completed) == (8, 6)
        assert group.task_stats.failed == 2
        assert group.declared_workers == 2
