# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone WorkerTracker class."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import WorkerStatus
from aiperf.common.mixins.worker_tracker_mixin import WorkerTracker
from aiperf.common.models import ProcessHealth, WorkerTaskStats


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
