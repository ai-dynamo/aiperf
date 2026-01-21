# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ErrorDetails, ErrorDetailsCount


class PhaseErrorTracker:
    """Phase Error Tracker. This is used to track the errors encountered during a credit phase.

    Operations are atomic only when used in a single thread asyncio context.
    """

    def __init__(self, phase: CreditPhase) -> None:
        self._phase: CreditPhase = phase
        self._error_counts: dict[ErrorDetails, int] = defaultdict(int)

    @property
    def phase(self) -> CreditPhase:
        """Get the phase."""
        return self._phase

    def get_error_summary(self) -> list[ErrorDetailsCount]:
        """Get the error summary."""
        return [
            ErrorDetailsCount(error_details=error_details, count=count)
            for error_details, count in self._error_counts.items()
        ]

    def increment_error_count(self, error: ErrorDetails) -> None:
        """Increment the count for a specific error."""
        self._error_counts[error] += 1


class ErrorTracker:
    """Error Tracker. This is used to track the errors encountered during the benchmark.

    Operations are atomic only when used in a single thread asyncio context.
    """

    def __init__(self) -> None:
        self._phase_error_trackers: dict[CreditPhase, PhaseErrorTracker] = {}

    def _get_phase_error_tracker(self, phase: CreditPhase) -> PhaseErrorTracker:
        """Get the phase error tracker."""
        if phase not in self._phase_error_trackers:
            self._phase_error_trackers.setdefault(phase, PhaseErrorTracker(phase))
        return self._phase_error_trackers[phase]

    def increment_error_count_for_phase(
        self, phase: CreditPhase, error: ErrorDetails
    ) -> None:
        """Increment the error count for an error in a phase."""
        phase_error_tracker = self._get_phase_error_tracker(phase)
        phase_error_tracker.increment_error_count(error)

    def get_error_summary_for_phase(
        self, phase: CreditPhase
    ) -> list[ErrorDetailsCount]:
        """Get the error summary for a phase."""
        phase_error_tracker = self._get_phase_error_tracker(phase)
        return phase_error_tracker.get_error_summary()
