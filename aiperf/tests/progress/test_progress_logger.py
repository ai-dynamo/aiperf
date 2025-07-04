# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleProgressLogger."""

from unittest.mock import Mock, patch

import pytest

from aiperf.progress.progress_logger import SimpleProgressLogger
from aiperf.progress.progress_models import (
    BenchmarkSuiteType,
)
from aiperf.progress.progress_tracker import ProgressTracker


class TestSimpleProgressLogger:
    """Test SimpleProgressLogger functionality."""

    def test_simple_progress_logger_creation(self):
        """Test creating a SimpleProgressLogger instance."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        assert logger.progress_tracker is tracker
        assert logger.tqdm_requests is None
        assert logger.tqdm_records is None
        assert logger.logger is not None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_no_current_profile(self, mock_tqdm):
        """Test update_progress when no current profile exists."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        await logger.update_progress()

        # Should not create tqdm if no current profile
        mock_tqdm.assert_not_called()
        assert logger.tqdm_requests is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_first_time(self, mock_tqdm):
        """Test update_progress for the first time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 25

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_progress()

        # Should create tqdm with correct parameters
        mock_tqdm.assert_called_once_with(
            total=100,
            desc="Requests Completed",
            colour="green",
        )

        # Should update tqdm
        assert mock_tqdm_instance.n == 25
        mock_tqdm_instance.refresh.assert_called_once()

        assert logger.tqdm_requests is mock_tqdm_instance

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_subsequent_times(self, mock_tqdm):
        """Test update_progress for subsequent times."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 25

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        # First update
        await logger.update_progress()

        # Second update with more progress
        profile.requests_completed = 50
        await logger.update_progress()

        # Should only create tqdm once
        mock_tqdm.assert_called_once()

        # Should update tqdm with new progress
        assert mock_tqdm_instance.n == 50
        assert mock_tqdm_instance.refresh.call_count == 2

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_completion(self, mock_tqdm):
        """Test update_progress when profile is completed."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 100  # Completed

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_progress()

        # Should create and immediately close tqdm
        mock_tqdm.assert_called_once()
        mock_tqdm_instance.close.assert_called_once()
        assert logger.tqdm_requests is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_zero_total(self, mock_tqdm):
        """Test update_progress when total_expected_requests is 0."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with zero total
        profile = tracker.current_profile
        profile.total_expected_requests = 0
        profile.requests_completed = 0

        logger = SimpleProgressLogger(tracker)

        await logger.update_progress()

        # Should not create tqdm with zero total
        mock_tqdm.assert_not_called()
        assert logger.tqdm_requests is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_progress_none_total(self, mock_tqdm):
        """Test update_progress when total_expected_requests is None."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with None total
        profile = tracker.current_profile
        profile.total_expected_requests = None
        profile.requests_completed = 25

        logger = SimpleProgressLogger(tracker)

        await logger.update_progress()

        # Should not create tqdm with None total
        mock_tqdm.assert_not_called()
        assert logger.tqdm_requests is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_no_current_profile(self, mock_tqdm):
        """Test update_stats when no current profile exists."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        await logger.update_stats()

        # Should not create tqdm if no current profile
        mock_tqdm.assert_not_called()
        assert logger.tqdm_records is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_first_time(self, mock_tqdm):
        """Test update_stats for the first time."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_processed = 30

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_stats()

        # Should create tqdm with correct parameters
        mock_tqdm.assert_called_once_with(
            total=100,
            desc=" Records Processed",
            colour="blue",
        )

        # Should update tqdm
        assert mock_tqdm_instance.n == 30
        mock_tqdm_instance.refresh.assert_called_once()

        assert logger.tqdm_records is mock_tqdm_instance

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_subsequent_times(self, mock_tqdm):
        """Test update_stats for subsequent times."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_processed = 30

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        # First update
        await logger.update_stats()

        # Second update with more progress
        profile.requests_processed = 60
        await logger.update_stats()

        # Should only create tqdm once
        mock_tqdm.assert_called_once()

        # Should update tqdm with new progress
        assert mock_tqdm_instance.n == 60
        assert mock_tqdm_instance.refresh.call_count == 2

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_completion(self, mock_tqdm):
        """Test update_stats when profile is completed."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_processed = 100  # Completed

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_stats()

        # Should create and immediately close tqdm
        mock_tqdm.assert_called_once()
        mock_tqdm_instance.close.assert_called_once()
        assert logger.tqdm_records is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_zero_total(self, mock_tqdm):
        """Test update_stats when total_expected_requests is 0."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with zero total
        profile = tracker.current_profile
        profile.total_expected_requests = 0
        profile.requests_processed = 0

        logger = SimpleProgressLogger(tracker)

        await logger.update_stats()

        # Should not create tqdm with zero total
        mock_tqdm.assert_not_called()
        assert logger.tqdm_records is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_update_stats_none_total(self, mock_tqdm):
        """Test update_stats when total_expected_requests is None."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with None total
        profile = tracker.current_profile
        profile.total_expected_requests = None
        profile.requests_processed = 25

        logger = SimpleProgressLogger(tracker)

        await logger.update_stats()

        # Should not create tqdm with None total
        mock_tqdm.assert_not_called()
        assert logger.tqdm_records is None

    async def test_update_results_no_tqdm(self):
        """Test update_results when no tqdm instances exist."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        # Should not raise error
        await logger.update_results()

        assert logger.tqdm_requests is None
        assert logger.tqdm_records is None

    async def test_update_results_with_tqdm(self):
        """Test update_results with existing tqdm instances."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        # Create mock tqdm instances
        mock_requests = Mock()
        mock_records = Mock()

        logger.tqdm_requests = mock_requests
        logger.tqdm_records = mock_records

        await logger.update_results()

        # Should close both tqdm instances
        mock_requests.close.assert_called_once()
        mock_records.close.assert_called_once()

        # Should reset references
        assert logger.tqdm_requests is None
        assert logger.tqdm_records is None

    async def test_update_results_partial_tqdm(self):
        """Test update_results with only one tqdm instance."""
        tracker = ProgressTracker()
        logger = SimpleProgressLogger(tracker)

        # Create only one mock tqdm instance
        mock_requests = Mock()
        logger.tqdm_requests = mock_requests
        logger.tqdm_records = None

        await logger.update_results()

        # Should close the existing tqdm instance
        mock_requests.close.assert_called_once()

        # Should reset reference
        assert logger.tqdm_requests is None
        assert logger.tqdm_records is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_tqdm_error_handling(self, mock_tqdm):
        """Test error handling when tqdm fails."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 25

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm to raise an exception
        mock_tqdm.side_effect = Exception("tqdm error")

        # Should handle error gracefully
        with pytest.raises(Exception, match="tqdm error"):
            await logger.update_progress()

        assert logger.tqdm_requests is None

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_concurrent_updates(self, mock_tqdm):
        """Test concurrent update calls."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 25
        profile.requests_processed = 20

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instances
        mock_requests = Mock()
        mock_records = Mock()
        mock_tqdm.side_effect = [mock_requests, mock_records]

        # Concurrent updates
        await logger.update_progress()
        await logger.update_stats()

        # Should create separate tqdm instances
        assert mock_tqdm.call_count == 2
        assert logger.tqdm_requests is mock_requests
        assert logger.tqdm_records is mock_records

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_progress_over_100_percent(self, mock_tqdm):
        """Test handling progress over 100%."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with progress over 100%
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 120  # Over 100%

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_progress()

        # Should handle over 100% gracefully
        mock_tqdm.assert_called_once()
        assert mock_tqdm_instance.n == 120
        mock_tqdm_instance.refresh.assert_called_once()

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_negative_progress(self, mock_tqdm):
        """Test handling negative progress values."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with negative progress
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = -5  # Negative progress

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_progress()

        # Should handle negative progress gracefully
        mock_tqdm.assert_called_once()
        assert mock_tqdm_instance.n == -5
        mock_tqdm_instance.refresh.assert_called_once()

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_logger_debug_calls(self, mock_tqdm):
        """Test that debug logging is called appropriately."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 25
        profile.requests_processed = 20

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instances
        mock_requests = Mock()
        mock_records = Mock()
        mock_tqdm.side_effect = [mock_requests, mock_records]

        with patch.object(logger.logger, "debug") as mock_debug:
            await logger.update_progress()
            await logger.update_stats()

            # Should call debug logging
            assert mock_debug.call_count >= 2

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_memory_cleanup(self, mock_tqdm):
        """Test that memory is properly cleaned up."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile
        profile = tracker.current_profile
        profile.total_expected_requests = 100
        profile.requests_completed = 100

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        # Create and complete progress
        await logger.update_progress()

        # Should clean up references
        assert logger.tqdm_requests is None
        mock_tqdm_instance.close.assert_called_once()

    @patch("aiperf.progress.progress_logger.tqdm")
    async def test_large_numbers(self, mock_tqdm):
        """Test handling of large progress numbers."""
        tracker = ProgressTracker()
        tracker.configure(BenchmarkSuiteType.SINGLE_PROFILE)
        tracker.suite.current_profile_idx = 0

        # Set up profile with large numbers
        profile = tracker.current_profile
        profile.total_expected_requests = 1000000
        profile.requests_completed = 500000

        logger = SimpleProgressLogger(tracker)

        # Mock tqdm instance
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value = mock_tqdm_instance

        await logger.update_progress()

        # Should handle large numbers
        mock_tqdm.assert_called_once_with(
            total=1000000,
            desc="Requests Completed",
            colour="green",
        )
        assert mock_tqdm_instance.n == 500000
