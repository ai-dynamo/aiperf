# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ProgressTracker."""

import time
from unittest.mock import patch

from aiperf.progress.progress_models import (
    BenchmarkSuiteType,
    ProfileProgress,
    ProfileSuiteProgress,
    SweepSuiteProgress,
)
from aiperf.progress.progress_tracker import ProgressTracker


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_progress_tracker_creation(self):
        """Test creating a ProgressTracker instance."""
        tracker = ProgressTracker()

        assert tracker.suite is None
        assert tracker.current_profile is None
        assert tracker.current_sweep is None
        assert tracker.logger is not None

    def test_configure_single_profile(self):
        """Test configuring for single profile."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        assert tracker.suite is not None
        assert isinstance(tracker.suite, ProfileSuiteProgress)
        assert len(tracker.suite.profiles) == 1
        assert tracker.suite.total_profiles == 1
        assert tracker.suite.profiles[0].profile_id == "0"

    def test_configure_multi_profile(self):
        """Test configuring for multi profile."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.MULTI_PROFILE)

        assert tracker.suite is not None
        assert isinstance(tracker.suite, ProfileSuiteProgress)
        assert len(tracker.suite.profiles) == 1
        assert tracker.suite.total_profiles == 1

    def test_configure_single_sweep(self):
        """Test configuring for single sweep."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)

        assert tracker.suite is not None
        assert isinstance(tracker.suite, SweepSuiteProgress)
        assert len(tracker.suite.sweeps) == 1
        assert tracker.suite.total_sweeps == 1
        assert tracker.suite.sweeps[0].sweep_id == "0"
        assert len(tracker.suite.sweeps[0].profiles) == 1

    def test_configure_multi_sweep(self):
        """Test configuring for multi sweep."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.MULTI_SWEEP)

        assert tracker.suite is not None
        assert isinstance(tracker.suite, SweepSuiteProgress)
        assert len(tracker.suite.sweeps) == 1
        assert tracker.suite.total_sweeps == 1

    def test_current_profile_property_profile_suite(self):
        """Test current_profile property with profile suite."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # Initially no current profile
        assert tracker.current_profile is None

        # Set current profile index
        tracker.suite.current_profile_idx = 0
        assert tracker.current_profile is not None
        assert tracker.current_profile.profile_id == "0"

    def test_current_profile_property_sweep_suite(self):
        """Test current_profile property with sweep suite."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)

        # Initially no current profile
        assert tracker.current_profile is None

        # Set current sweep and profile
        tracker.suite.current_sweep_idx = 0
        tracker.suite.current_sweep.current_profile_idx = 0
        assert tracker.current_profile is not None
        assert tracker.current_profile.profile_id == "0"

    def test_current_sweep_property(self):
        """Test current_sweep property."""
        tracker = ProgressTracker()

        # Profile suite should return None
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        assert tracker.current_sweep is None

        # Sweep suite should return sweep
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)
        assert tracker.current_sweep is None  # No current sweep index set

        tracker.suite.current_sweep_idx = 0
        assert tracker.current_sweep is not None
        assert tracker.current_sweep.sweep_id == "0"

    def test_update_profile_progress_no_suite(self, profile_progress_message):
        """Test update_profile_progress with no suite configured."""
        tracker = ProgressTracker()

        # Should not raise error
        tracker.update_profile_progress(profile_progress_message)

        assert tracker.suite is None

    def test_update_profile_progress_no_current_profile(self, profile_progress_message):
        """Test update_profile_progress with no current profile."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # Should not raise error
        tracker.update_profile_progress(profile_progress_message)

        # Profile should not be updated
        assert tracker.current_profile is None

    def test_update_profile_progress_first_time(self, profile_progress_message):
        """Test update_profile_progress for the first time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = 1000000000  # 1 second in nanoseconds

            tracker.update_profile_progress(profile_progress_message)

            profile = tracker.current_profile
            assert profile.start_time_ns == profile_progress_message.start_ns
            assert profile.total_expected_requests == profile_progress_message.total
            assert profile.requests_completed == profile_progress_message.completed

    def test_update_profile_progress_subsequent_times(self, profile_progress_message):
        """Test update_profile_progress for subsequent times."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # First update
        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = (
                profile_progress_message.start_ns + 1000000000
            )  # 1 second after start
            tracker.update_profile_progress(profile_progress_message)

            # Second update with more progress
            profile_progress_message.completed = 75
            mock_time.return_value = (
                profile_progress_message.start_ns + 2000000000
            )  # 2 seconds after start
            tracker.update_profile_progress(profile_progress_message)

            profile = tracker.current_profile
            assert profile.requests_completed == 75
            assert profile.requests_per_second is not None
            assert profile.requests_per_second > 0
            assert profile.elapsed_time > 0
            assert profile.eta is not None

    def test_update_profile_progress_completed(self, profile_progress_message):
        """Test update_profile_progress when profile is completed."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set completed equal to total
        profile_progress_message.completed = profile_progress_message.total

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = 1000000000
            tracker.update_profile_progress(profile_progress_message)

            profile = tracker.current_profile
            assert profile.requests_completed == profile_progress_message.total
            # Should not calculate rates when completed
            assert (
                profile.requests_per_second is None or profile.requests_per_second == 0
            )

    def test_update_processing_stats_no_suite(self, processing_stats_message):
        """Test update_processing_stats with no suite configured."""
        tracker = ProgressTracker()

        # Should not raise error
        tracker.update_processing_stats(processing_stats_message)

        assert tracker.suite is None

    def test_update_processing_stats_no_current_profile(self, processing_stats_message):
        """Test update_processing_stats with no current profile."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # Should not raise error
        tracker.update_processing_stats(processing_stats_message)

        assert tracker.current_profile is None

    def test_update_processing_stats_normal(self, processing_stats_message):
        """Test update_processing_stats with normal operation."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with start time
        profile = tracker.current_profile
        start_time = 1000000000  # Use fixed start time
        profile.start_time_ns = start_time
        profile.total_expected_requests = 100

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = start_time + 1000000000  # 1 second later

            tracker.update_processing_stats(processing_stats_message)

            assert profile.request_errors == processing_stats_message.error_count
            assert (
                profile.successful_requests
                == processing_stats_message.completed
                - processing_stats_message.error_count
            )
            assert profile.requests_processed == processing_stats_message.completed
            assert profile.worker_completed == processing_stats_message.worker_completed
            assert profile.worker_errors == processing_stats_message.worker_errors

    def test_update_processing_stats_completion(self, processing_stats_message):
        """Test update_processing_stats when profile is completed."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        start_time = 1000000000  # Use fixed start time
        profile.start_time_ns = start_time
        profile.total_expected_requests = 45  # Same as completed in message

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = start_time + 1000000000  # 1 second later

            tracker.update_processing_stats(processing_stats_message)

            assert profile.is_complete is True
            assert profile.end_time_ns is not None

    def test_update_processing_stats_with_processing_rate(
        self, processing_stats_message
    ):
        """Test update_processing_stats calculates processing rate."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        start_time = 1000000000  # Use fixed start time
        profile.start_time_ns = start_time
        profile.total_expected_requests = 100

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = start_time + 1000000000  # 1 second later

            tracker.update_processing_stats(processing_stats_message)

            assert profile.processed_per_second is not None
            assert profile.processed_per_second > 0
            assert profile.processing_eta is not None

    def test_update_profile_results_no_suite(self, profile_results_message):
        """Test update_profile_results with no suite configured."""
        tracker = ProgressTracker()

        # Should not raise error
        tracker.update_profile_results(profile_results_message)

        assert tracker.suite is None

    def test_update_profile_results_no_current_profile(self, profile_results_message):
        """Test update_profile_results with no current profile."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # Should not raise error
        tracker.update_profile_results(profile_results_message)

        assert tracker.current_profile is None

    def test_update_profile_results_normal(self, profile_results_message):
        """Test update_profile_results with normal operation."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with start time
        profile = tracker.current_profile
        profile.start_time_ns = profile_results_message.start_ns

        tracker.update_profile_results(profile_results_message)

        assert profile.end_time_ns == profile_results_message.end_ns
        assert profile.was_cancelled == profile_results_message.was_cancelled
        assert profile.elapsed_time > 0
        assert profile.eta is None
        assert profile.requests_per_second == 0
        assert profile.records == profile_results_message.records
        assert profile.errors_by_type == profile_results_message.errors_by_type

    def test_update_profile_results_no_start_time(self, profile_results_message):
        """Test update_profile_results when profile has no start time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Profile has no start time
        profile = tracker.current_profile
        profile.start_time_ns = None

        tracker.update_profile_results(profile_results_message)

        assert profile.end_time_ns == profile_results_message.end_ns
        assert profile.elapsed_time == 0  # Should remain 0 since no start time

    def test_update_sweep_progress_no_suite(self, sweep_progress_message):
        """Test update_sweep_progress with no suite configured."""
        tracker = ProgressTracker()

        # Should not raise error
        tracker.update_sweep_progress(sweep_progress_message)

        assert tracker.suite is None

    def test_update_sweep_progress_no_current_sweep(self, sweep_progress_message):
        """Test update_sweep_progress with no current sweep."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)  # Profile suite, not sweep

        # Should not raise error
        tracker.update_sweep_progress(sweep_progress_message)

        assert tracker.current_sweep is None

    def test_update_sweep_progress_first_time(self, sweep_progress_message):
        """Test update_sweep_progress for the first time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)
        tracker.suite.current_sweep_idx = 0

        tracker.update_sweep_progress(sweep_progress_message)

        current_sweep = tracker.current_sweep
        assert current_sweep.start_time_ns == sweep_progress_message.sweep_start_ns

    def test_update_sweep_progress_with_end_time(self, sweep_progress_message):
        """Test update_sweep_progress with end time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)
        tracker.suite.current_sweep_idx = 0

        # Set end time
        sweep_progress_message.end_ns = time.time_ns()

        tracker.update_sweep_progress(sweep_progress_message)

        current_sweep = tracker.current_sweep
        assert current_sweep.end_time_ns == sweep_progress_message.end_ns

    def test_update_sweep_progress_subsequent_times(self, sweep_progress_message):
        """Test update_sweep_progress for subsequent times."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_SWEEP)
        tracker.suite.current_sweep_idx = 0

        # First update
        tracker.update_sweep_progress(sweep_progress_message)

        # Second update should not change start time
        original_start = tracker.current_sweep.start_time_ns
        tracker.update_sweep_progress(sweep_progress_message)

        assert tracker.current_sweep.start_time_ns == original_start

    def test_eta_calculation_edge_cases(self, profile_progress_message):
        """Test ETA calculation edge cases."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Test when requests_per_second is 0
        profile_progress_message.completed = 0

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = (
                profile_progress_message.start_ns + 1000000000
            )  # 1 second later

            tracker.update_profile_progress(profile_progress_message)

            profile = tracker.current_profile
            assert profile.eta is None  # Should be None when no progress

    def test_processing_eta_calculation_edge_cases(self, processing_stats_message):
        """Test processing ETA calculation edge cases."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        start_time = 1000000000  # Use fixed start time
        profile.start_time_ns = start_time
        profile.total_expected_requests = 100

        # Test when processed_per_second is 0
        processing_stats_message.completed = 0

        with patch("aiperf.progress.progress_tracker.time.time_ns") as mock_time:
            mock_time.return_value = start_time + 1000000000  # 1 second later

            tracker.update_processing_stats(processing_stats_message)

            assert profile.processing_eta is None  # Should be None when no progress

    def test_thread_safety_considerations(self):
        """Test that tracker operations are generally thread-safe."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # This test mainly ensures no obvious race conditions
        # More comprehensive thread safety testing would require more complex setup
        assert tracker.suite is not None
        assert tracker.current_profile is None
        assert tracker.current_sweep is None

    def test_memory_management(self):
        """Test that tracker doesn't leak memory with large datasets."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)

        # Create large profile data
        profile = ProfileProgress(
            profile_id="test",
            total_expected_requests=10000,
            worker_completed={f"worker-{i}": 100 for i in range(100)},
            worker_errors={f"worker-{i}": 5 for i in range(100)},
        )

        tracker.suite.profiles = [profile]
        tracker.suite.current_profile_idx = 0

        # Should handle large datasets without issues
        assert len(tracker.current_profile.worker_completed) == 100
        assert len(tracker.current_profile.worker_errors) == 100
