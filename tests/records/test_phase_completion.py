# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.models import ProcessingStats
from aiperf.records.phase_completion import (
    AllRequestsProcessedCondition,
    CompletionReason,
    DurationTimeoutCondition,
    PhaseCompletionChecker,
    PhaseCompletionContext,
)


class TestPhaseCompletionContext:
    """Test the PhaseCompletionContext model."""

    def test_context_creation(self):
        """Test creating a completion context."""
        stats = ProcessingStats(processed=5, errors=2)
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=True,
            expected_duration_sec=5.0,
        )

        assert context.processing_stats == stats
        assert context.final_request_count == 10
        assert context.timeout_triggered is True
        assert context.expected_duration_sec == 5.0

    def test_context_defaults(self):
        """Test default values in completion context."""
        stats = ProcessingStats()
        context = PhaseCompletionContext(processing_stats=stats)

        assert context.final_request_count is None
        assert context.timeout_triggered is False
        assert context.expected_duration_sec is None


class TestAllRequestsProcessedCondition:
    """Test the AllRequestsProcessedCondition."""

    def test_satisfied_when_all_requests_processed_request_count_mode(self):
        """Test condition is satisfied for request-count-based benchmarks."""
        stats = ProcessingStats(processed=10, errors=0)  # total_records = 10
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            expected_duration_sec=None,  # Request-count-based
        )

        condition = AllRequestsProcessedCondition()
        assert condition.is_satisfied(context) is True
        assert condition.reason == CompletionReason.ALL_REQUESTS_PROCESSED

    def test_not_satisfied_for_duration_based_benchmarks(self):
        """Test condition is NOT satisfied for duration-based benchmarks."""
        stats = ProcessingStats(processed=10, errors=0)  # total_records = 10
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,  # All requests processed
            expected_duration_sec=5.0,  # But it's duration-based
        )

        condition = AllRequestsProcessedCondition()
        assert condition.is_satisfied(context) is False

    def test_satisfied_when_more_requests_processed(self):
        """Test condition is satisfied when more requests processed than expected."""
        stats = ProcessingStats(processed=12, errors=0)  # total_records = 12
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            expected_duration_sec=None,  # Request-count-based
        )

        condition = AllRequestsProcessedCondition()
        assert condition.is_satisfied(context) is True

    def test_not_satisfied_when_requests_pending(self):
        """Test condition is not satisfied when requests are still pending."""
        stats = ProcessingStats(processed=8, errors=0)  # total_records = 8
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            expected_duration_sec=None,  # Request-count-based
        )

        condition = AllRequestsProcessedCondition()
        assert condition.is_satisfied(context) is False

    def test_not_satisfied_when_no_final_count(self):
        """Test condition is not satisfied when final_request_count is None."""
        stats = ProcessingStats(processed=10, errors=0)
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=None,
            expected_duration_sec=None,  # Request-count-based
        )

        condition = AllRequestsProcessedCondition()
        assert condition.is_satisfied(context) is False


class TestDurationTimeoutCondition:
    """Test the DurationTimeoutCondition."""

    def test_satisfied_when_timeout_and_count_set(self):
        """Test condition is satisfied when timeout occurred and final count is set."""
        stats = ProcessingStats(processed=5, errors=0)
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=True,
        )

        condition = DurationTimeoutCondition()
        assert condition.is_satisfied(context) is True
        assert condition.reason == CompletionReason.DURATION_TIMEOUT

    def test_not_satisfied_when_no_timeout(self):
        """Test condition is not satisfied when timeout has not occurred."""
        stats = ProcessingStats(processed=5, errors=0)
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=False,
        )

        condition = DurationTimeoutCondition()
        assert condition.is_satisfied(context) is False

    def test_not_satisfied_when_timeout_but_no_final_count(self):
        """Test condition is not satisfied when timeout occurred but final count not set."""
        stats = ProcessingStats(processed=5, errors=0)
        context = PhaseCompletionContext(
            processing_stats=stats,
            final_request_count=None,
            timeout_triggered=True,
        )

        condition = DurationTimeoutCondition()
        assert condition.is_satisfied(context) is False


class TestPhaseCompletionChecker:
    """Test the PhaseCompletionChecker orchestrator."""

    def test_completion_by_request_count(self):
        """Test completion when all requests are processed in request-count mode."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=10, errors=2)  # total_records = 12

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=False,
            expected_duration_sec=None,  # Request-count-based
        )

        assert is_complete is True
        assert reason == CompletionReason.ALL_REQUESTS_PROCESSED

    def test_completion_by_duration_timeout(self):
        """Test completion when duration timeout occurs."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=5, errors=0)

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=20,  # More than processed
            timeout_triggered=True,
            expected_duration_sec=5.0,  # Duration-based
        )

        assert is_complete is True
        assert reason == CompletionReason.DURATION_TIMEOUT

    def test_no_completion_when_pending_request_count_mode(self):
        """Test no completion when requests are still pending in request-count mode."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=5, errors=0)

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=False,
            expected_duration_sec=None,  # Request-count-based
        )

        assert is_complete is False
        assert reason is None

    def test_no_completion_when_duration_not_elapsed(self):
        """Test no completion for duration-based benchmark before timeout."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=5, errors=0)

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=5,  # All sent requests processed
            timeout_triggered=False,  # But duration hasn't elapsed
            expected_duration_sec=5.0,  # Duration-based
        )

        assert is_complete is False
        assert reason is None

    def test_no_completion_when_timeout_but_no_final_count(self):
        """Test no completion when timeout but final_request_count not set."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=5, errors=0)

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=None,
            timeout_triggered=True,
            expected_duration_sec=5.0,
        )

        assert is_complete is False
        assert reason is None

    def test_request_count_takes_precedence_when_both_conditions_true(self):
        """Test that request count completion takes precedence over timeout."""
        checker = PhaseCompletionChecker()
        stats = ProcessingStats(processed=10, errors=0)  # total_records = 10

        # This should NOT happen in practice (request-count benchmark with timeout),
        # but test the precedence logic
        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=True,
            expected_duration_sec=None,  # Request-count-based
        )

        assert is_complete is True
        assert reason == CompletionReason.ALL_REQUESTS_PROCESSED

    def test_add_custom_condition(self):
        """Test adding a custom completion condition."""
        checker = PhaseCompletionChecker()

        class AlwaysTrueCondition:
            def is_satisfied(self, context):
                return True

            @property
            def reason(self):
                return "custom_reason"

        checker.add_condition(AlwaysTrueCondition())
        stats = ProcessingStats(processed=1, errors=0)

        is_complete, reason = checker.is_complete(
            processing_stats=stats,
            final_request_count=10,
            timeout_triggered=False,
        )

        # The custom condition should trigger first (it was added last, but first satisfied wins)
        # Actually, the AllRequestsProcessedCondition will be checked first and will return False
        # So this will test that custom conditions work
        assert is_complete is True
        assert str(reason) == "custom_reason"
