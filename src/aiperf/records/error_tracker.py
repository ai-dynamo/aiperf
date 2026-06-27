# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ErrorDetails, ErrorDetailsCount

# Cap distinct error keys per phase. ``ErrorDetails.__hash__`` includes the full
# message, so backends echoing a per-request-unique body (request id, timestamp,
# prompt) under a sustained error storm would otherwise grow ``_error_counts``
# without bound. Past the cap, further new (code, type) message variants fold
# into a single "(other)" bucket keyed by (code, type) so memory and the
# materialized error_summary stay bounded while identical-message dedup is
# unaffected for everything seen before the cap.
MAX_DISTINCT_ERROR_KEYS = 4096
_OTHER_MESSAGE = "(other errors with this code/type)"


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
        """Increment the count for a specific error.

        Identical errors dedup by ``(code, type, message)``. Once the number of
        distinct keys reaches ``MAX_DISTINCT_ERROR_KEYS``, an otherwise-new error
        is folded into a per-(code, type) "(other)" bucket rather than adding an
        unbounded number of keys for per-request-unique messages.
        """
        if error in self._error_counts or len(self._error_counts) < (
            MAX_DISTINCT_ERROR_KEYS
        ):
            self._error_counts[error] += 1
            return
        other = ErrorDetails(message=_OTHER_MESSAGE, code=error.code, type=error.type)
        self._error_counts[other] += 1


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
