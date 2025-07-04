# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.progress.progress_models import (
    BenchmarkSuiteType,
    ProcessingStatsMessage,
    ProfileProgress,
    ProfileProgressMessage,
    ProfileResultsMessage,
    ProfileSuiteProgress,
    SweepProgress,
    SweepProgressMessage,
    SweepSuiteProgress,
)


class ProgressTracker:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.suite: ProfileSuiteProgress | SweepSuiteProgress | None = None

    @property
    def current_sweep(self) -> SweepProgress | None:
        if self.suite is None or not isinstance(self.suite, SweepSuiteProgress):
            return None
        return self.suite.current_sweep

    @property
    def current_profile(self) -> ProfileProgress | None:
        if self.suite is None:
            return None
        return self.suite.current_profile

    def configure(self, suite_type: BenchmarkSuiteType) -> None:
        if suite_type in [
            BenchmarkSuiteType.SINGLE_PROFILE,
            BenchmarkSuiteType.MULTI_PROFILE,
        ]:
            self.suite = ProfileSuiteProgress(
                suite_type=suite_type,
                profiles=[
                    ProfileProgress(
                        profile_id="0",
                        total_expected_requests=0,
                    )
                ],
                total_profiles=1,
            )

        elif suite_type in [
            BenchmarkSuiteType.SINGLE_SWEEP,
            BenchmarkSuiteType.MULTI_SWEEP,
        ]:
            self.suite = SweepSuiteProgress(
                suite_type=suite_type,
                sweeps=[
                    SweepProgress(
                        sweep_id="0",
                        profiles=[
                            ProfileProgress(
                                profile_id="0",
                                total_expected_requests=0,
                            )
                        ],
                    )
                ],
                total_sweeps=1,
            )

    def update_profile_progress(self, message: ProfileProgressMessage) -> None:
        current_time_ns = time.time_ns()
        if self.suite is None or self.suite.current_profile is None:
            return

        profile = self.suite.current_profile

        if profile.start_time_ns is None:
            profile.start_time_ns = message.start_ns
            profile.total_expected_requests = message.total
            self.logger.info(
                f"Starting performance test: {message.total:,} total requests"
            )

        profile.total_expected_requests = message.total
        profile.requests_completed = message.completed

        if message.completed < profile.total_expected_requests:
            profile.requests_per_second = (
                (
                    message.completed
                    / (current_time_ns - profile.start_time_ns)
                    * NANOS_PER_SECOND
                )
                if current_time_ns > profile.start_time_ns
                else None
            )

            profile.elapsed_time = (
                current_time_ns - profile.start_time_ns
            ) / NANOS_PER_SECOND

            profile.eta = (
                (
                    (profile.total_expected_requests - profile.requests_completed)
                    / profile.requests_per_second
                    if profile.requests_per_second and profile.requests_per_second > 0
                    else None
                )
                if profile.requests_per_second is not None
                else None
            )

    def update_processing_stats(self, message: ProcessingStatsMessage) -> None:
        if self.suite is None or self.suite.current_profile is None:
            return

        current_time_ns = time.time_ns()
        profile = self.suite.current_profile
        profile.request_errors = message.error_count
        profile.successful_requests = message.completed - profile.request_errors
        profile.requests_processed = message.completed

        profile.worker_completed = message.worker_completed
        profile.worker_errors = message.worker_errors
        if (
            profile.total_expected_requests is not None
            and profile.requests_processed >= profile.total_expected_requests
        ):
            self.logger.info("Profile completed")
            profile.end_time_ns = current_time_ns
            profile.is_complete = True

        if not profile.is_complete:
            profile.processed_per_second = (
                (
                    message.completed
                    / (current_time_ns - profile.start_time_ns)
                    * NANOS_PER_SECOND
                )
                if profile.start_time_ns is not None
                and current_time_ns > profile.start_time_ns
                else None
            )

            profile.processing_eta = (
                (
                    (profile.total_expected_requests - profile.requests_processed)
                    / profile.processed_per_second
                    if profile.processed_per_second
                    and profile.processed_per_second > 0
                    and profile.total_expected_requests
                    else None
                )
                if profile.processed_per_second is not None
                else None
            )

    def update_profile_results(self, message: ProfileResultsMessage) -> None:
        if self.suite is None or self.suite.current_profile is None:
            return

        profile = self.suite.current_profile
        profile.end_time_ns = message.end_ns
        profile.was_cancelled = message.was_cancelled
        if profile.start_time_ns is not None and profile.end_time_ns is not None:
            profile.elapsed_time = (
                profile.end_time_ns - profile.start_time_ns
            ) / NANOS_PER_SECOND
        profile.eta = None
        profile.requests_per_second = 0
        profile.records = message.records
        profile.errors_by_type = message.errors_by_type

    def update_sweep_progress(self, message: SweepProgressMessage) -> None:
        """Update sweep progress with the provided message."""
        if self.suite is None or self.suite.current_sweep is None:
            return

        current_sweep = self.suite.current_sweep

        if current_sweep.start_time_ns is None:
            current_sweep.start_time_ns = message.sweep_start_ns
            self.logger.info(f"Starting sweep: {current_sweep.sweep_id}")

        if message.end_ns is not None:
            current_sweep.end_time_ns = message.end_ns
            self.logger.info(f"Completed sweep: {current_sweep.sweep_id}")
