# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for progress models."""

import time

from aiperf.progress.progress_models import (
    BenchmarkSuiteCompletionTrigger,
    BenchmarkSuiteType,
    ProcessingStatsMessage,
    ProfileCompletionTrigger,
    ProfileProgress,
    ProfileProgressMessage,
    ProfileResultsMessage,
    ProfileSuiteProgress,
    SweepCompletionTrigger,
    SweepProgress,
    SweepProgressMessage,
    SweepSuiteProgress,
)


class TestProfileProgress:
    """Test ProfileProgress model."""

    def test_profile_progress_creation(self):
        """Test creating a ProfileProgress instance."""
        profile = ProfileProgress(
            profile_id="test-profile",
            total_expected_requests=100,
        )

        assert profile.profile_id == "test-profile"
        assert profile.total_expected_requests == 100
        assert profile.requests_completed == 0
        assert profile.request_errors == 0
        assert profile.successful_requests == 0
        assert profile.requests_processed == 0
        assert profile.requests_per_second is None
        assert profile.processed_per_second is None
        assert profile.was_cancelled is False
        assert profile.elapsed_time == 0
        assert profile.eta is None
        assert profile.processing_eta is None
        assert profile.records == []
        assert profile.errors_by_type == []
        assert profile.is_complete is False
        assert (
            profile.profile_completion_trigger == ProfileCompletionTrigger.REQUEST_COUNT
        )

    def test_profile_progress_with_timing(self):
        """Test ProfileProgress with timing information."""
        start_time = time.time_ns()
        profile = ProfileProgress(
            profile_id="test-profile",
            start_time_ns=start_time,
            total_expected_requests=100,
        )

        assert profile.start_time_ns == start_time
        assert profile.end_time_ns is None

    def test_profile_progress_defaults(self):
        """Test ProfileProgress with default values."""
        profile = ProfileProgress(profile_id="test-profile")

        assert profile.total_expected_requests is None
        assert profile.worker_completed == {}
        assert profile.worker_errors == {}


class TestSweepProgress:
    """Test SweepProgress model."""

    def test_sweep_progress_creation(self):
        """Test creating a SweepProgress instance."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        sweep = SweepProgress(
            sweep_id="test-sweep",
            profiles=profiles,
        )

        assert sweep.sweep_id == "test-sweep"
        assert len(sweep.profiles) == 2
        assert sweep.current_profile_idx is None
        assert sweep.completed_profiles == 0
        assert sweep.start_time_ns is None
        assert sweep.end_time_ns is None
        assert sweep.was_cancelled is False
        assert (
            sweep.sweep_completion_trigger == SweepCompletionTrigger.COMPLETED_PROFILES
        )

    def test_sweep_progress_current_profile(self):
        """Test current_profile property."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        sweep = SweepProgress(
            sweep_id="test-sweep",
            profiles=profiles,
        )

        # Initially no current profile
        assert sweep.current_profile is None

        # Set current profile index
        sweep.current_profile_idx = 0
        assert sweep.current_profile is not None
        assert sweep.current_profile.profile_id == "profile-1"

        sweep.current_profile_idx = 1
        assert sweep.current_profile.profile_id == "profile-2"

    def test_sweep_progress_next_profile(self):
        """Test next_profile method."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        sweep = SweepProgress(
            sweep_id="test-sweep",
            profiles=profiles,
        )

        # First call should return first profile
        profile1 = sweep.next_profile()
        assert profile1 is not None
        assert profile1.profile_id == "profile-1"
        assert sweep.current_profile_idx == 0

        # Second call should return second profile
        profile2 = sweep.next_profile()
        assert profile2 is not None
        assert profile2.profile_id == "profile-2"
        assert sweep.current_profile_idx == 1

        # Third call should return None
        profile3 = sweep.next_profile()
        assert profile3 is None

    def test_sweep_progress_next_profile_empty_profiles(self):
        """Test next_profile with empty profiles list."""
        sweep = SweepProgress(
            sweep_id="test-sweep",
            profiles=[],
        )

        profile = sweep.next_profile()
        assert profile is None


class TestProfileSuiteProgress:
    """Test ProfileSuiteProgress model."""

    def test_profile_suite_progress_creation(self):
        """Test creating a ProfileSuiteProgress instance."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        suite = ProfileSuiteProgress(
            profiles=profiles,
            total_profiles=2,
        )

        assert len(suite.profiles) == 2
        assert suite.total_profiles == 2
        assert suite.completed_profiles == 0
        assert suite.current_profile_idx is None
        assert suite.suite_type == BenchmarkSuiteType.SINGLE_PROFILE
        assert (
            suite.suite_completion_trigger
            == BenchmarkSuiteCompletionTrigger.COMPLETED_PROFILES
        )

    def test_profile_suite_progress_current_profile(self):
        """Test current_profile property."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        suite = ProfileSuiteProgress(
            profiles=profiles,
            total_profiles=2,
        )

        # Initially no current profile
        assert suite.current_profile is None

        # Set current profile index
        suite.current_profile_idx = 0
        assert suite.current_profile is not None
        assert suite.current_profile.profile_id == "profile-1"

    def test_profile_suite_progress_next_profile(self):
        """Test next_profile method."""
        profiles = [
            ProfileProgress(profile_id="profile-1"),
            ProfileProgress(profile_id="profile-2"),
        ]

        suite = ProfileSuiteProgress(
            profiles=profiles,
            total_profiles=2,
        )

        # First call should return first profile
        profile1 = suite.next_profile()
        assert profile1 is not None
        assert profile1.profile_id == "profile-1"
        assert suite.current_profile_idx == 0

        # Second call should return second profile
        profile2 = suite.next_profile()
        assert profile2 is not None
        assert profile2.profile_id == "profile-2"
        assert suite.current_profile_idx == 1

        # Third call should return None
        profile3 = suite.next_profile()
        assert profile3 is None

    def test_profile_suite_progress_current_sweep(self):
        """Test current_sweep property returns None for ProfileSuiteProgress."""
        suite = ProfileSuiteProgress(profiles=[], total_profiles=0)
        assert suite.current_sweep is None


class TestSweepSuiteProgress:
    """Test SweepSuiteProgress model."""

    def test_sweep_suite_progress_creation(self):
        """Test creating a SweepSuiteProgress instance."""
        sweeps = [
            SweepProgress(
                sweep_id="sweep-1",
                profiles=[ProfileProgress(profile_id="profile-1")],
            ),
        ]

        suite = SweepSuiteProgress(
            sweeps=sweeps,
            total_sweeps=1,
        )

        assert len(suite.sweeps) == 1
        assert suite.total_sweeps == 1
        assert suite.completed_sweeps == 0
        assert suite.current_sweep_idx is None

    def test_sweep_suite_progress_current_sweep(self):
        """Test current_sweep property."""
        sweeps = [
            SweepProgress(
                sweep_id="sweep-1",
                profiles=[ProfileProgress(profile_id="profile-1")],
            ),
        ]

        suite = SweepSuiteProgress(
            sweeps=sweeps,
            total_sweeps=1,
        )

        # Initially no current sweep
        assert suite.current_sweep is None

        # Set current sweep index
        suite.current_sweep_idx = 0
        assert suite.current_sweep is not None
        assert suite.current_sweep.sweep_id == "sweep-1"

    def test_sweep_suite_progress_current_profile(self):
        """Test current_profile property."""
        sweeps = [
            SweepProgress(
                sweep_id="sweep-1",
                profiles=[ProfileProgress(profile_id="profile-1")],
            ),
        ]

        suite = SweepSuiteProgress(
            sweeps=sweeps,
            total_sweeps=1,
        )

        # Initially no current profile
        assert suite.current_profile is None

        # Set current sweep and profile
        suite.current_sweep_idx = 0
        suite.current_sweep.current_profile_idx = 0
        assert suite.current_profile is not None
        assert suite.current_profile.profile_id == "profile-1"

    def test_sweep_suite_progress_next_sweep(self):
        """Test next_sweep method."""
        sweeps = [
            SweepProgress(
                sweep_id="sweep-1",
                profiles=[ProfileProgress(profile_id="profile-1")],
            ),
            SweepProgress(
                sweep_id="sweep-2",
                profiles=[ProfileProgress(profile_id="profile-2")],
            ),
        ]

        suite = SweepSuiteProgress(
            sweeps=sweeps,
            total_sweeps=2,
        )

        # First call should return first sweep
        sweep1 = suite.next_sweep()
        assert sweep1 is not None
        assert sweep1.sweep_id == "sweep-1"
        assert suite.current_sweep_idx == 0

        # Second call should return second sweep
        sweep2 = suite.next_sweep()
        assert sweep2 is not None
        assert sweep2.sweep_id == "sweep-2"
        assert suite.current_sweep_idx == 1

        # Third call should return None
        sweep3 = suite.next_sweep()
        assert sweep3 is None

    def test_sweep_suite_progress_next_profile(self):
        """Test next_profile method for sweep suite."""
        sweeps = [
            SweepProgress(
                sweep_id="sweep-1",
                profiles=[
                    ProfileProgress(profile_id="profile-1"),
                    ProfileProgress(profile_id="profile-2"),
                ],
            ),
            SweepProgress(
                sweep_id="sweep-2",
                profiles=[
                    ProfileProgress(profile_id="profile-3"),
                ],
            ),
        ]

        suite = SweepSuiteProgress(
            sweeps=sweeps,
            total_sweeps=2,
        )

        # First call should start first sweep and return first profile
        profile1 = suite.next_profile()
        assert profile1 is not None
        assert profile1.profile_id == "profile-1"
        assert suite.current_sweep_idx == 0

        # Second call should return second profile from first sweep
        profile2 = suite.next_profile()
        assert profile2 is not None
        assert profile2.profile_id == "profile-2"

        # Third call should move to second sweep and return its first profile
        profile3 = suite.next_profile()
        assert profile3 is not None
        assert profile3.profile_id == "profile-3"
        assert suite.current_sweep_idx == 1

        # Fourth call should return None (no more profiles)
        profile4 = suite.next_profile()
        assert profile4 is None


class TestMessageModels:
    """Test message models."""

    def test_profile_progress_message(self):
        """Test ProfileProgressMessage creation."""
        message = ProfileProgressMessage(
            service_id="test-service",
            start_ns=time.time_ns(),
            total=100,
            completed=50,
        )

        assert message.service_id == "test-service"
        assert message.total == 100
        assert message.completed == 50
        assert message.profile_id is None
        assert message.end_ns is None

    def test_profile_progress_message_with_optional_fields(self):
        """Test ProfileProgressMessage with optional fields."""
        start_time = time.time_ns()
        end_time = start_time + 1000000000  # 1 second later

        message = ProfileProgressMessage(
            service_id="test-service",
            profile_id="test-profile",
            start_ns=start_time,
            end_ns=end_time,
            total=100,
            completed=100,
        )

        assert message.profile_id == "test-profile"
        assert message.end_ns == end_time

    def test_processing_stats_message(self):
        """Test ProcessingStatsMessage creation."""
        message = ProcessingStatsMessage(
            service_id="test-service",
            error_count=5,
            completed=95,
            worker_completed={"worker-1": 50, "worker-2": 45},
            worker_errors={"worker-1": 3, "worker-2": 2},
        )

        assert message.service_id == "test-service"
        assert message.error_count == 5
        assert message.completed == 95
        assert message.worker_completed == {"worker-1": 50, "worker-2": 45}
        assert message.worker_errors == {"worker-1": 3, "worker-2": 2}

    def test_processing_stats_message_defaults(self):
        """Test ProcessingStatsMessage with default values."""
        message = ProcessingStatsMessage(service_id="test-service")

        assert message.error_count == 0
        assert message.completed == 0
        assert message.worker_completed == {}
        assert message.worker_errors == {}

    def test_profile_results_message(self):
        """Test ProfileResultsMessage creation."""
        start_time = time.time_ns()
        end_time = start_time + 1000000000  # 1 second later

        message = ProfileResultsMessage(
            service_id="test-service",
            records=[],
            total=100,
            completed=100,
            start_ns=start_time,
            end_ns=end_time,
        )

        assert message.service_id == "test-service"
        assert message.records == []
        assert message.total == 100
        assert message.completed == 100
        assert message.start_ns == start_time
        assert message.end_ns == end_time
        assert message.was_cancelled is False
        assert message.errors_by_type == []

    def test_sweep_progress_message(self):
        """Test SweepProgressMessage creation."""
        message = SweepProgressMessage(
            service_id="test-service",
            sweep_id="test-sweep",
            sweep_start_ns=time.time_ns(),
        )

        assert message.service_id == "test-service"
        assert message.sweep_id == "test-sweep"
        assert message.end_ns is None

    def test_sweep_progress_message_with_end_time(self):
        """Test SweepProgressMessage with end time."""
        start_time = time.time_ns()
        end_time = start_time + 1000000000  # 1 second later

        message = SweepProgressMessage(
            service_id="test-service",
            sweep_id="test-sweep",
            sweep_start_ns=start_time,
            end_ns=end_time,
        )

        assert message.end_ns == end_time


class TestEnumModels:
    """Test enum models."""

    def test_profile_completion_trigger_enum(self):
        """Test ProfileCompletionTrigger enum values."""
        assert ProfileCompletionTrigger.REQUEST_COUNT == "request_count"
        assert ProfileCompletionTrigger.TIME_BASED == "time_based"
        assert ProfileCompletionTrigger.STABILIZATION_BASED == "stabilization_based"
        assert ProfileCompletionTrigger.GOODPUT_THRESHOLD == "goodput_threshold"
        assert ProfileCompletionTrigger.CUSTOM == "custom"

    def test_sweep_completion_trigger_enum(self):
        """Test SweepCompletionTrigger enum values."""
        assert SweepCompletionTrigger.COMPLETED_PROFILES == "completed_profiles"
        assert SweepCompletionTrigger.STABILIZATION_BASED == "stabilization_based"
        assert SweepCompletionTrigger.GOODPUT_THRESHOLD == "goodput_threshold"
        assert SweepCompletionTrigger.CUSTOM == "custom"

    def test_benchmark_suite_type_enum(self):
        """Test BenchmarkSuiteType enum values."""
        assert BenchmarkSuiteType.SINGLE_PROFILE == "single_profile"
        assert BenchmarkSuiteType.MULTI_PROFILE == "multi_profile"
        assert BenchmarkSuiteType.SINGLE_SWEEP == "single_sweep"
        assert BenchmarkSuiteType.MULTI_SWEEP == "multi_sweep"
        assert BenchmarkSuiteType.CUSTOM == "custom"

    def test_benchmark_suite_completion_trigger_enum(self):
        """Test BenchmarkSuiteCompletionTrigger enum values."""
        assert BenchmarkSuiteCompletionTrigger.UNKNOWN == "unknown"
        assert BenchmarkSuiteCompletionTrigger.COMPLETED_SWEEPS == "completed_sweeps"
        assert (
            BenchmarkSuiteCompletionTrigger.COMPLETED_PROFILES == "completed_profiles"
        )
        assert (
            BenchmarkSuiteCompletionTrigger.STABILIZATION_BASED == "stabilization_based"
        )
        assert BenchmarkSuiteCompletionTrigger.CUSTOM == "custom"
