# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prefill concurrency control.

Prefill concurrency limits how many requests can be in the prefill (prompt-processing)
stage simultaneously. This is separate from the overall concurrency limit.

Validation approach:
- Use mock server with controlled TTFT (time to first token) to create measurable prefill phases
- Analyze exported JSONL records to verify prefill overlap doesn't exceed the limit
- Compare throughput/timing between different prefill_concurrency values
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI, AIPerfResults


def _calculate_max_overlap_from_intervals(
    intervals: list[tuple[int, int]],
) -> int:
    """Calculate maximum overlap using sweep line algorithm.

    Args:
        intervals: List of (start_ns, end_ns) tuples

    Returns:
        Maximum number of overlapping intervals
    """
    if not intervals:
        return 0

    events = []
    for start, end in intervals:
        events.append((start, 1))  # +1 at start
        events.append((end, -1))  # -1 at end

    events.sort()

    max_overlap = 0
    current_overlap = 0
    for _, delta in events:
        current_overlap += delta
        max_overlap = max(max_overlap, current_overlap)

    return max_overlap


def calculate_max_prefill_overlap(results: AIPerfResults) -> int:
    """Calculate the maximum number of overlapping prefill phases.

    This analyzes the timing data to find the peak number of requests
    that were in prefill phase simultaneously.

    Args:
        results: AIPerfResults containing jsonl records with timing data

    Returns:
        Maximum number of overlapping prefill phases observed
    """
    if not results.jsonl:
        return 0

    # Build list of prefill intervals: (start_ns, end_ns)
    prefill_intervals = []
    for record in results.jsonl:
        # Skip cancelled or errored requests
        if record.error is not None:
            continue

        # Get TTFT metric in nanoseconds
        ttft_metric = record.metrics.get("time_to_first_token")
        if ttft_metric is None:
            continue

        # ttft is in milliseconds, convert to nanoseconds
        ttft_ns = int(ttft_metric.value * 1_000_000)

        # Calculate prefill interval using wall clock timestamps
        prefill_start = record.metadata.request_start_ns
        prefill_end = prefill_start + ttft_ns

        prefill_intervals.append((prefill_start, prefill_end))

    return _calculate_max_overlap_from_intervals(prefill_intervals)


def calculate_max_request_overlap(results: AIPerfResults) -> int:
    """Calculate the maximum number of overlapping requests (overall concurrency).

    This analyzes the timing data to find the peak number of requests
    that were in-flight simultaneously (from start to end).

    Args:
        results: AIPerfResults containing jsonl records with timing data

    Returns:
        Maximum number of overlapping requests observed
    """
    if not results.jsonl:
        return 0

    # Build list of request intervals: (start_ns, end_ns)
    request_intervals = []
    for record in results.jsonl:
        # Skip cancelled or errored requests for clean measurement
        if record.error is not None:
            continue

        request_start = record.metadata.request_start_ns
        request_end = record.metadata.request_end_ns

        request_intervals.append((request_start, request_end))

    return _calculate_max_overlap_from_intervals(request_intervals)


def count_requests_started_before_previous_finished(results: AIPerfResults) -> int:
    """Count how many requests started before at least one other request finished.

    This validates that requests are actually overlapping in time, proving
    that prefill_concurrency only limits prefill phase, not entire requests.

    Args:
        results: AIPerfResults containing jsonl records with timing data

    Returns:
        Number of requests that started while another request was still in progress
    """
    if not results.jsonl:
        return 0

    # Collect valid request intervals
    intervals = []
    for record in results.jsonl:
        if record.error is not None:
            continue
        intervals.append(
            (record.metadata.request_start_ns, record.metadata.request_end_ns)
        )

    if len(intervals) < 2:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Count requests that started before any previous request finished
    # Track the latest end time - if a new request starts before this,
    # it overlapped with at least one previous request
    overlapping_count = 0
    latest_end_so_far = intervals[0][1]

    for i in range(1, len(intervals)):
        start, end = intervals[i]
        # If this request started before the latest previous request finished
        if start < latest_end_so_far:
            overlapping_count += 1
        # Update latest end time
        latest_end_so_far = max(latest_end_so_far, end)

    return overlapping_count


@pytest.mark.component_integration
class TestPrefillConcurrencyBasic:
    """Basic tests for prefill concurrency configuration."""

    @pytest.mark.slow
    def test_prefill_concurrency_limits_overlap(self, cli: AIPerfCLI):
        """Verify prefill concurrency limits simultaneous prefill phases.

        With a controlled TTFT of 100ms and prefill_concurrency=2,
        no more than 2 requests should be in prefill phase simultaneously.
        But requests SHOULD overlap overall (new ones start before previous finish).
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count 20 \
                --concurrency 10 \
                --prefill-concurrency 2 \
                --osl 10 \
                --workers-max 5 \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.request_count == 20
        assert result.has_streaming_metrics

        # Verify prefill overlap respects the limit with duration checking
        max_prefill_overlap = calculate_max_prefill_overlap(result)

        # Check overlap durations to distinguish timing artifacts from real violations
        MAX_OVERLAP_TIME_MS = 2.0
        prefill_intervals = []
        for record in result.jsonl:
            if record.error is not None:
                continue
            ttft_metric = record.metrics.get("time_to_first_token")
            if ttft_metric:
                ttft_ns = int(ttft_metric.value * 1_000_000)
                start = record.metadata.request_start_ns
                end = start + ttft_ns
                prefill_intervals.append((start, end))

        # Count sustained overlaps (duration > threshold)
        max_sustained_overlap = 0
        if prefill_intervals:
            prefill_intervals.sort()
            for i in range(len(prefill_intervals)):
                concurrent = 1
                for j in range(i + 1, len(prefill_intervals)):
                    if prefill_intervals[j][0] < prefill_intervals[i][1]:
                        overlap_ns = (
                            min(prefill_intervals[i][1], prefill_intervals[j][1])
                            - prefill_intervals[j][0]
                        )
                        if overlap_ns > MAX_OVERLAP_TIME_MS * 1_000_000:
                            concurrent += 1
                max_sustained_overlap = max(max_sustained_overlap, concurrent)

        # Allow brief overlaps from xdist jitter, but sustained overlaps must respect limit
        assert max_sustained_overlap <= 2, (
            f"Sustained prefill overlaps (>{MAX_OVERLAP_TIME_MS}ms) exceeded limit: "
            f"{max_sustained_overlap} > 2. Total overlap count: {max_prefill_overlap}"
        )

        # Verify requests DO overlap overall (new requests start before previous finish)
        # This confirms prefill_concurrency only limits prefill, not entire requests
        max_request_overlap = calculate_max_request_overlap(result)
        assert max_request_overlap > max_prefill_overlap, (
            f"Requests should overlap more than prefill phases: "
            f"request_overlap={max_request_overlap}, prefill_overlap={max_prefill_overlap}"
        )

        # Verify multiple requests started while others were still in progress
        overlapping_starts = count_requests_started_before_previous_finished(result)
        assert overlapping_starts >= 5, (
            f"Expected at least 5 requests to start while others in progress, got {overlapping_starts}"
        )

    @pytest.mark.slow
    def test_prefill_concurrency_one_serializes_prefill(self, cli: AIPerfCLI):
        """Verify prefill_concurrency=1 serializes prefill phases.

        With prefill_concurrency=1, requests should go through prefill one at a time,
        even with high overall concurrency. But requests SHOULD still overlap overall
        (decode phases can run concurrently while new requests wait for prefill slot).
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --request-count 15 \
                --concurrency 10 \
                --prefill-concurrency 1 \
                --osl 10 \
                --workers-max 5 \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        assert result.request_count == 15
        assert result.has_streaming_metrics

        # With prefill_concurrency=1, max prefill overlap should be 1 (or 2 with timing tolerance)
        max_prefill_overlap = calculate_max_prefill_overlap(result)
        assert max_prefill_overlap <= 2, (
            f"Expected max prefill overlap <= 2 (limit=1 + tolerance), got {max_prefill_overlap}"
        )

        # Verify requests DO overlap overall (decode phases run concurrently)
        # This is the key test: prefill is serialized but overall requests overlap
        max_request_overlap = calculate_max_request_overlap(result)
        assert max_request_overlap > max_prefill_overlap, (
            f"Requests should overlap more than prefill phases: "
            f"request_overlap={max_request_overlap}, prefill_overlap={max_prefill_overlap}"
        )

        # Verify multiple requests started while others were still in progress
        overlapping_starts = count_requests_started_before_previous_finished(result)
        assert overlapping_starts >= 3, (
            f"Expected at least 3 requests to start while others in progress, got {overlapping_starts}"
        )


@pytest.mark.component_integration
class TestPrefillConcurrencyMultiTurn:
    """Tests for prefill concurrency with multi-turn conversations."""

    def test_prefill_concurrency_with_conversations(self, cli: AIPerfCLI):
        """Verify prefill concurrency works correctly with multi-turn conversations.

        Requests from different conversations should overlap, but prefill phases
        should still respect the prefill_concurrency limit.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --endpoint-type chat \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --concurrency 5 \
                --prefill-concurrency 2 \
                --osl 50 \
                --workers-max 1 \
                --ui {defaults.ui}
            """,
            timeout=60.0,
        )

        # Should complete all 30 requests (10 conversations x 3 turns)
        assert result.request_count == 30
        assert result.has_streaming_metrics

        # Verify turn indices
        turn_indices = set(record.metadata.turn_index for record in result.jsonl)
        assert len(turn_indices) == 3  # Turns 0, 1, 2

        # Verify prefill overlap respects limit (with timing tolerance for test jitter)
        max_prefill_overlap = calculate_max_prefill_overlap(result)
        assert max_prefill_overlap <= 4, (
            f"Expected max prefill overlap <= 4 (limit=2 + tolerance), got {max_prefill_overlap}"
        )

        # Verify requests from different conversations overlap
        max_request_overlap = calculate_max_request_overlap(result)
        assert max_request_overlap >= 2, (
            f"Expected at least 2 concurrent requests, got {max_request_overlap}"
        )

        # Verify requests started while others were still in progress
        overlapping_starts = count_requests_started_before_previous_finished(result)
        assert overlapping_starts >= 3, (
            f"Expected at least 3 requests to start while others in progress, got {overlapping_starts}"
        )


@pytest.mark.component_integration
class TestPrefillConcurrencyDeadlockPrevention:
    """Tests for deadlock prevention when requests fail before FirstToken."""

    def test_prefill_slots_released_on_error(self, cli: AIPerfCLI):
        """Verify benchmark completes when requests error before sending FirstToken.

        This tests the deadlock prevention mechanism: when a request errors before
        TTFT (first_token_sent=False), the orchestrator must release the prefill
        slot via CreditReturn instead of waiting for a FirstToken that never comes.

        With prefill_concurrency=1 and 50% error rate, if slots weren't released
        on error, the benchmark would deadlock after the first error.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                    --endpoint-type chat \
                    --streaming \
                    --request-count 20 \
                    --concurrency 5 \
                    --prefill-concurrency 1 \
                    --osl 5 \
                    --workers-max 3 \
                    --request-cancellation-rate 50 \
                    --request-cancellation-delay 0 \
                    --ui {defaults.ui}
                """,
            timeout=60.0,
        )

        # Key assertion: benchmark completed all 20 requests (no deadlock)
        # Use len(jsonl) because request_count only counts successful requests
        total_requests = len(result.jsonl)
        assert total_requests == 20, (
            f"Expected 20 requests to complete, got {total_requests}. "
            "Benchmark may have deadlocked due to unreleased prefill slots."
        )

        # Count errors - should have ~10 with 50% error rate
        error_count = sum(1 for r in result.jsonl if r.error is not None)
        assert error_count > 0, "Expected some errors with 50% error rate"
        assert error_count < 20, "Expected some successful requests"

        # Verify successful requests have valid streaming metrics
        successful = [r for r in result.jsonl if r.error is None]
        assert successful
        for record in successful:
            assert "time_to_first_token" in record.metrics
