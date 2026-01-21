# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from aiperf.common.models.worker_models import WorkerTaskStats


class TestWorkerTaskStatsInitialization:
    """Test WorkerTaskStats initialization and defaults."""

    def test_default_initialization(self):
        """Test that WorkerTaskStats initializes with default values."""
        stats = WorkerTaskStats()
        assert stats.total == 0
        assert stats.failed == 0
        assert stats.completed == 0

    def test_custom_initialization(self):
        """Test WorkerTaskStats initialization with custom values."""
        stats = WorkerTaskStats(total=10, failed=2, completed=7)
        assert stats.total == 10
        assert stats.failed == 2
        assert stats.completed == 7

    def test_validation_requires_int(self):
        """Test that fields require integer values."""
        with pytest.raises(ValidationError):
            WorkerTaskStats(total="not an int")

        with pytest.raises(ValidationError):
            WorkerTaskStats(failed=3.5)

        with pytest.raises(ValidationError):
            WorkerTaskStats(completed=None)


class TestWorkerTaskStatsTaskFinished:
    """Test the task_finished method."""

    def test_task_finished_with_failure(self):
        """Test that task_finished increments failed count when valid=False."""
        stats = WorkerTaskStats(total=10, failed=1, completed=5)
        stats.task_finished(valid=False)
        assert stats.failed == 2
        assert stats.completed == 5  # Should not change
        assert stats.total == 10  # Should not change

    def test_task_finished_with_success(self):
        """Test that task_finished increments completed count when valid=True."""
        stats = WorkerTaskStats(total=10, failed=2, completed=5)
        stats.task_finished(valid=True)
        assert stats.completed == 6
        assert stats.failed == 2  # Should not change
        assert stats.total == 10  # Should not change

    def test_task_finished_multiple_failures(self):
        """Test multiple failed tasks."""
        stats = WorkerTaskStats(total=10)
        stats.task_finished(valid=False)
        stats.task_finished(valid=False)
        stats.task_finished(valid=False)
        assert stats.failed == 3
        assert stats.completed == 0

    def test_task_finished_multiple_successes(self):
        """Test multiple successful tasks."""
        stats = WorkerTaskStats(total=10)
        stats.task_finished(valid=True)
        stats.task_finished(valid=True)
        stats.task_finished(valid=True)
        assert stats.completed == 3
        assert stats.failed == 0

    def test_task_finished_mixed(self):
        """Test a mix of successful and failed tasks."""
        stats = WorkerTaskStats(total=10)
        stats.task_finished(valid=True)
        stats.task_finished(valid=False)
        stats.task_finished(valid=True)
        stats.task_finished(valid=False)
        stats.task_finished(valid=True)
        assert stats.completed == 3
        assert stats.failed == 2


class TestWorkerTaskStatsInProgress:
    """Test the in_progress property."""

    def test_in_progress_with_no_completed_tasks(self):
        """Test in_progress when no tasks have completed."""
        stats = WorkerTaskStats(total=10, failed=0, completed=0)
        assert stats.in_progress == 10

    def test_in_progress_with_some_completed(self):
        """Test in_progress with some completed tasks."""
        stats = WorkerTaskStats(total=10, failed=2, completed=5)
        assert stats.in_progress == 3

    def test_in_progress_with_all_completed(self):
        """Test in_progress when all tasks are done."""
        stats = WorkerTaskStats(total=10, failed=3, completed=7)
        assert stats.in_progress == 0

    def test_in_progress_after_task_finished(self):
        """Test that in_progress updates correctly after task_finished calls."""
        stats = WorkerTaskStats(total=10, failed=0, completed=0)
        assert stats.in_progress == 10

        stats.task_finished(valid=True)
        assert stats.in_progress == 9

        stats.task_finished(valid=False)
        assert stats.in_progress == 8

        stats.task_finished(valid=True)
        assert stats.in_progress == 7

    def test_in_progress_calculation(self):
        """Test various scenarios for in_progress calculation."""
        # All in progress
        stats = WorkerTaskStats(total=100, failed=0, completed=0)
        assert stats.in_progress == 100

        # Half completed successfully
        stats = WorkerTaskStats(total=100, failed=0, completed=50)
        assert stats.in_progress == 50

        # All failed
        stats = WorkerTaskStats(total=100, failed=100, completed=0)
        assert stats.in_progress == 0

        # Mix
        stats = WorkerTaskStats(total=100, failed=25, completed=60)
        assert stats.in_progress == 15
