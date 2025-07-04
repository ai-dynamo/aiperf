# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for progress module tests."""

import time
from unittest.mock import Mock

import pytest

from aiperf.progress.progress_logger import SimpleProgressLogger
from aiperf.progress.progress_models import (
    ProcessingStatsMessage,
    ProfileProgress,
    ProfileProgressMessage,
    ProfileResultsMessage,
    SweepProgress,
    SweepProgressMessage,
)
from aiperf.progress.progress_tracker import ProgressTracker


@pytest.fixture
def profile_progress():
    """Create a basic ProfileProgress instance."""
    return ProfileProgress(
        profile_id="test-profile",
        total_expected_requests=100,
        start_time_ns=time.time_ns(),
    )


@pytest.fixture
def sweep_progress():
    """Create a basic SweepProgress instance."""
    return SweepProgress(
        sweep_id="test-sweep",
        profiles=[
            ProfileProgress(profile_id="profile-1", total_expected_requests=50),
            ProfileProgress(profile_id="profile-2", total_expected_requests=75),
        ],
    )


@pytest.fixture
def progress_tracker():
    """Create a fresh ProgressTracker instance."""
    return ProgressTracker()


@pytest.fixture
def simple_progress_logger(progress_tracker):
    """Create a SimpleProgressLogger instance."""
    return SimpleProgressLogger(progress_tracker)


@pytest.fixture
def profile_progress_message():
    """Create a ProfileProgressMessage instance."""
    return ProfileProgressMessage(
        service_id="test-service",
        start_ns=time.time_ns(),
        total=100,
        completed=50,
    )


@pytest.fixture
def processing_stats_message():
    """Create a ProcessingStatsMessage instance."""
    return ProcessingStatsMessage(
        service_id="test-service",
        error_count=5,
        completed=45,
        worker_completed={"worker-1": 25, "worker-2": 20},
        worker_errors={"worker-1": 3, "worker-2": 2},
    )


@pytest.fixture
def profile_results_message():
    """Create a ProfileResultsMessage instance."""
    return ProfileResultsMessage(
        service_id="test-service",
        records=[],
        total=100,
        completed=100,
        start_ns=time.time_ns() - 1000000000,  # 1 second ago
        end_ns=time.time_ns(),
        was_cancelled=False,
        errors_by_type=[],
    )


@pytest.fixture
def sweep_progress_message():
    """Create a SweepProgressMessage instance."""
    return SweepProgressMessage(
        service_id="test-service",
        sweep_id="test-sweep",
        sweep_start_ns=time.time_ns(),
    )


@pytest.fixture
def mock_tqdm():
    """Create a mock tqdm instance."""
    mock = Mock()
    mock.n = 0
    mock.total = 100
    mock.refresh = Mock()
    mock.close = Mock()
    return mock
