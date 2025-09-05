# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum

from aiperf.common.models import ProcessingStats
from aiperf.common.models.base_models import AIPerfBaseModel


class CompletionReason(Enum):
    """Reasons why a phase completed."""

    ALL_REQUESTS_PROCESSED = "all_requests_processed"
    DURATION_TIMEOUT = "duration_timeout"


class PhaseCompletionContext(AIPerfBaseModel):
    """Context object containing all state needed for completion checking."""

    processing_stats: ProcessingStats
    final_request_count: int | None = None
    timeout_triggered: bool = False
    expected_duration_sec: float | None = None


class PhaseCompletionCondition(ABC):
    """Abstract base class for phase completion conditions."""

    @abstractmethod
    def is_satisfied(self, context: PhaseCompletionContext) -> bool:
        """Check if this completion condition is satisfied."""
        pass

    @property
    @abstractmethod
    def reason(self) -> CompletionReason:
        """The completion reason this condition represents."""
        pass


class AllRequestsProcessedCondition(PhaseCompletionCondition):
    """Completion condition for when all expected requests have been processed."""

    def is_satisfied(self, context: PhaseCompletionContext) -> bool:
        # Only trigger for request-count-based benchmarks, not duration-based ones
        is_request_count_based = context.expected_duration_sec is None
        return (
            is_request_count_based
            and context.final_request_count is not None
            and context.processing_stats.total_records >= context.final_request_count
        )

    @property
    def reason(self) -> CompletionReason:
        return CompletionReason.ALL_REQUESTS_PROCESSED


class DurationTimeoutCondition(PhaseCompletionCondition):
    """Completion condition for when the benchmark duration has elapsed."""

    def is_satisfied(self, context: PhaseCompletionContext) -> bool:
        return context.timeout_triggered and context.final_request_count is not None

    @property
    def reason(self) -> CompletionReason:
        return CompletionReason.DURATION_TIMEOUT


class PhaseCompletionChecker:
    """Orchestrates checking multiple completion conditions."""

    def __init__(self):
        self.conditions: list[PhaseCompletionCondition] = [
            AllRequestsProcessedCondition(),
            DurationTimeoutCondition(),
        ]

    def is_complete(
        self,
        processing_stats: ProcessingStats,
        final_request_count: int | None = None,
        timeout_triggered: bool = False,
        expected_duration_sec: float | None = None,
    ) -> tuple[bool, CompletionReason | None]:
        """Check if the phase is complete based on registered conditions.

        Args:
            processing_stats: Current processing statistics
            final_request_count: Expected number of requests to process (None for duration-based)
            timeout_triggered: Whether a benchmark duration timeout has occurred
            expected_duration_sec: Duration for duration-based benchmarks (None for request-count-based)

        Returns:
            Tuple of (is_complete: bool, reason: CompletionReason | None)
            If is_complete is False, reason will be None.
        """
        context = PhaseCompletionContext(
            processing_stats=processing_stats,
            final_request_count=final_request_count,
            timeout_triggered=timeout_triggered,
            expected_duration_sec=expected_duration_sec,
        )

        for condition in self.conditions:
            if condition.is_satisfied(context):
                return True, condition.reason

        return False, None

    def add_condition(self, condition: PhaseCompletionCondition) -> None:
        """Add a custom completion condition."""
        self.conditions.append(condition)
