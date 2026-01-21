# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for error tracking functionality."""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ErrorDetails
from aiperf.records.error_tracker import ErrorTracker, PhaseErrorTracker


@pytest.fixture
def sample_error():
    """Create a sample error for testing."""
    return ErrorDetails(message="Connection timeout", type="TimeoutError")


@pytest.fixture
def another_error():
    """Create another sample error for testing."""
    return ErrorDetails(message="Invalid response", type="ValueError")


@pytest.fixture
def third_error():
    """Create a third sample error for testing."""
    return ErrorDetails(message="Rate limit exceeded", type="RateLimitError")


class TestPhaseErrorTracker:
    """Tests for PhaseErrorTracker class."""

    def test_initialization(self):
        """PhaseErrorTracker initializes with a phase."""
        phase = CreditPhase.WARMUP
        tracker = PhaseErrorTracker(phase)

        assert tracker.phase == phase

    def test_initial_error_summary_empty(self):
        """Error summary is empty when no errors have been recorded."""
        tracker = PhaseErrorTracker(CreditPhase.WARMUP)

        summary = tracker.get_error_summary()

        assert summary == []

    def test_increment_error_count_single_error(self, sample_error):
        """Error count increments correctly for a single error."""
        tracker = PhaseErrorTracker(CreditPhase.WARMUP)

        tracker.increment_error_count(sample_error)

        summary = tracker.get_error_summary()
        assert len(summary) == 1
        assert summary[0].error_details == sample_error
        assert summary[0].count == 1

    def test_increment_error_count_multiple_times_same_error(self, sample_error):
        """Error count increments correctly when same error occurs multiple times."""
        tracker = PhaseErrorTracker(CreditPhase.WARMUP)

        # Increment same error multiple times
        for _ in range(5):
            tracker.increment_error_count(sample_error)

        summary = tracker.get_error_summary()
        assert len(summary) == 1
        assert summary[0].error_details == sample_error
        assert summary[0].count == 5

    def test_increment_error_count_different_errors(
        self, sample_error, another_error, third_error
    ):
        """Multiple different errors are tracked separately."""
        tracker = PhaseErrorTracker(CreditPhase.WARMUP)

        tracker.increment_error_count(sample_error)
        tracker.increment_error_count(another_error)
        tracker.increment_error_count(sample_error)  # Increment first again
        tracker.increment_error_count(third_error)
        tracker.increment_error_count(another_error)  # Increment second again

        summary = tracker.get_error_summary()
        assert len(summary) == 3

        # Find each error in summary
        error_counts = {item.error_details: item.count for item in summary}
        assert error_counts[sample_error] == 2
        assert error_counts[another_error] == 2
        assert error_counts[third_error] == 1

    def test_phase_property_returns_correct_phase(self):
        """Phase property returns the phase set during initialization."""
        phase = CreditPhase.PROFILING
        tracker = PhaseErrorTracker(phase)

        assert tracker.phase == phase

    @pytest.mark.parametrize(
        "phase",  # fmt: skip
        [
            CreditPhase.WARMUP,
            CreditPhase.PROFILING,
        ],
    )
    def test_works_with_different_phases(self, phase, sample_error):
        """PhaseErrorTracker works correctly with different credit phases."""
        tracker = PhaseErrorTracker(phase)

        tracker.increment_error_count(sample_error)

        assert tracker.phase == phase
        summary = tracker.get_error_summary()
        assert len(summary) == 1
        assert summary[0].count == 1

    def test_error_summary_reflects_current_state(self, sample_error, another_error):
        """Error summary always reflects the current state of error counts."""
        tracker = PhaseErrorTracker(CreditPhase.WARMUP)

        # First increment
        tracker.increment_error_count(sample_error)
        summary1 = tracker.get_error_summary()
        assert len(summary1) == 1

        # Add different error
        tracker.increment_error_count(another_error)
        summary2 = tracker.get_error_summary()
        assert len(summary2) == 2

        # Increment first error again
        tracker.increment_error_count(sample_error)
        summary3 = tracker.get_error_summary()
        assert len(summary3) == 2
        error_counts = {item.error_details: item.count for item in summary3}
        assert error_counts[sample_error] == 2
        assert error_counts[another_error] == 1

    def test_concurrent_increments_same_error(self, sample_error):
        """Multiple increments of the same error are accumulated correctly."""
        tracker = PhaseErrorTracker(CreditPhase.PROFILING)

        # Simulate rapid concurrent increments
        for _ in range(100):
            tracker.increment_error_count(sample_error)

        summary = tracker.get_error_summary()
        assert len(summary) == 1
        assert summary[0].count == 100


class TestErrorTracker:
    """Tests for ErrorTracker class."""

    def test_initialization(self):
        """ErrorTracker initializes without errors."""
        tracker = ErrorTracker()

        # Should not have any phase trackers initially
        summary_warmup = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        summary_run = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)

        assert summary_warmup == []
        assert summary_run == []

    def test_increment_error_creates_phase_tracker_lazily(self, sample_error):
        """Phase tracker is created lazily when first error is recorded."""
        tracker = ErrorTracker()

        tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)

        summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        assert len(summary) == 1
        assert summary[0].count == 1

    def test_errors_tracked_separately_by_phase(self, sample_error, another_error):
        """Errors in different phases are tracked independently."""
        tracker = ErrorTracker()

        # Add errors to different phases
        tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, another_error)

        warmup_summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        run_summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)

        assert len(warmup_summary) == 1
        assert warmup_summary[0].error_details == sample_error
        assert warmup_summary[0].count == 2

        assert len(run_summary) == 1
        assert run_summary[0].error_details == another_error
        assert run_summary[0].count == 1

    def test_same_error_tracked_separately_across_phases(self, sample_error):
        """Same error type in different phases is tracked separately."""
        tracker = ErrorTracker()

        tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)

        warmup_summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        run_summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)

        assert len(warmup_summary) == 1
        assert warmup_summary[0].count == 2

        assert len(run_summary) == 1
        assert run_summary[0].count == 3

    def test_multiple_errors_in_same_phase(
        self, sample_error, another_error, third_error
    ):
        """Multiple different errors in same phase are tracked correctly."""
        tracker = ErrorTracker()

        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, another_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, third_error)
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)

        summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
        assert len(summary) == 3

        error_counts = {item.error_details: item.count for item in summary}
        assert error_counts[sample_error] == 2
        assert error_counts[another_error] == 1
        assert error_counts[third_error] == 1

    def test_get_summary_for_untracked_phase_returns_empty(self):
        """Getting summary for phase with no errors returns empty list."""
        tracker = ErrorTracker()

        summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)

        assert summary == []

    def test_realistic_error_tracking_scenario(self, sample_error, another_error):
        """Realistic scenario: tracking errors across multiple phases."""
        tracker = ErrorTracker()

        # Warmup phase: some timeout errors
        for _ in range(3):
            tracker.increment_error_count_for_phase(CreditPhase.WARMUP, sample_error)

        # Run phase: mix of errors
        for _ in range(10):
            tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)
        for _ in range(5):
            tracker.increment_error_count_for_phase(
                CreditPhase.PROFILING, another_error
            )

        # Verify warmup phase errors
        warmup_summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        assert len(warmup_summary) == 1
        assert warmup_summary[0].error_details == sample_error
        assert warmup_summary[0].count == 3

        # Verify run phase errors
        run_summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
        assert len(run_summary) == 2
        error_counts = {item.error_details: item.count for item in run_summary}
        assert error_counts[sample_error] == 10
        assert error_counts[another_error] == 5

    @pytest.mark.parametrize(
        "phase",  # fmt: skip
        [
            CreditPhase.WARMUP,
            CreditPhase.PROFILING,
        ],
    )
    def test_works_with_all_credit_phases(self, phase, sample_error):
        """ErrorTracker works correctly with all credit phases."""
        tracker = ErrorTracker()

        tracker.increment_error_count_for_phase(phase, sample_error)

        summary = tracker.get_error_summary_for_phase(phase)
        assert len(summary) == 1
        assert summary[0].count == 1

    def test_high_volume_error_tracking(self, sample_error, another_error):
        """ErrorTracker handles high volume of errors correctly."""
        tracker = ErrorTracker()

        # Simulate many errors
        for _ in range(1000):
            tracker.increment_error_count_for_phase(CreditPhase.PROFILING, sample_error)
        for _ in range(500):
            tracker.increment_error_count_for_phase(
                CreditPhase.PROFILING, another_error
            )

        summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
        assert len(summary) == 2

        error_counts = {item.error_details: item.count for item in summary}
        assert error_counts[sample_error] == 1000
        assert error_counts[another_error] == 500


class TestErrorDetailsEquality:
    """Tests for ErrorDetails equality behavior in error tracking."""

    def test_identical_errors_counted_together(self):
        """Identical ErrorDetails instances are counted as the same error."""
        tracker = PhaseErrorTracker(CreditPhase.PROFILING)

        error1 = ErrorDetails(message="Connection timeout", type="TimeoutError")
        error2 = ErrorDetails(message="Connection timeout", type="TimeoutError")

        tracker.increment_error_count(error1)
        tracker.increment_error_count(error2)

        summary = tracker.get_error_summary()
        # Should be counted as one error type with count of 2
        assert len(summary) == 1
        assert summary[0].count == 2

    def test_different_messages_counted_separately(self):
        """ErrorDetails with different messages are counted separately."""
        tracker = PhaseErrorTracker(CreditPhase.PROFILING)

        error1 = ErrorDetails(message="Connection timeout", type="TimeoutError")
        error2 = ErrorDetails(message="Read timeout", type="TimeoutError")

        tracker.increment_error_count(error1)
        tracker.increment_error_count(error2)

        summary = tracker.get_error_summary()
        assert len(summary) == 2

    def test_different_types_counted_separately(self):
        """ErrorDetails with different types are counted separately."""
        tracker = PhaseErrorTracker(CreditPhase.PROFILING)

        error1 = ErrorDetails(message="Error occurred", type="TimeoutError")
        error2 = ErrorDetails(message="Error occurred", type="ValueError")

        tracker.increment_error_count(error1)
        tracker.increment_error_count(error2)

        summary = tracker.get_error_summary()
        assert len(summary) == 2


class TestErrorTrackerIntegration:
    """Integration tests for error tracking across multiple components."""

    def test_complete_benchmark_error_tracking_workflow(self):
        """Realistic workflow: tracking errors throughout a benchmark run."""
        tracker = ErrorTracker()

        # Phase 1: Warmup - a few connection errors
        warmup_error = ErrorDetails(
            message="Connection refused", type="ConnectionError"
        )
        for _ in range(2):
            tracker.increment_error_count_for_phase(CreditPhase.WARMUP, warmup_error)

        # Phase 2: Run - various errors
        timeout_error = ErrorDetails(message="Request timeout", type="TimeoutError")
        rate_limit_error = ErrorDetails(message="Rate limit", type="RateLimitError")
        validation_error = ErrorDetails(message="Invalid input", type="ValidationError")

        for _ in range(15):
            tracker.increment_error_count_for_phase(
                CreditPhase.PROFILING, timeout_error
            )
        for _ in range(8):
            tracker.increment_error_count_for_phase(
                CreditPhase.PROFILING, rate_limit_error
            )
        for _ in range(3):
            tracker.increment_error_count_for_phase(
                CreditPhase.PROFILING, validation_error
            )

        # Verify warmup errors
        warmup_summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        assert len(warmup_summary) == 1
        assert warmup_summary[0].count == 2

        # Verify run errors
        run_summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)
        assert len(run_summary) == 3

        error_counts = {item.error_details: item.count for item in run_summary}
        assert error_counts[timeout_error] == 15
        assert error_counts[rate_limit_error] == 8
        assert error_counts[validation_error] == 3

        # Total errors in run phase
        total_run_errors = sum(item.count for item in run_summary)
        assert total_run_errors == 26

    def test_error_tracking_with_no_warmup_phase(self):
        """Error tracking works when benchmark has no warmup phase."""
        tracker = ErrorTracker()

        # Only run phase errors
        error = ErrorDetails(message="Server error", type="ServerError")
        tracker.increment_error_count_for_phase(CreditPhase.PROFILING, error)

        warmup_summary = tracker.get_error_summary_for_phase(CreditPhase.WARMUP)
        run_summary = tracker.get_error_summary_for_phase(CreditPhase.PROFILING)

        assert len(warmup_summary) == 0
        assert len(run_summary) == 1
        assert run_summary[0].count == 1
