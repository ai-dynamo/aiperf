# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation with multi-turn sessions.

This tests the behavior where cancellation properly works
with multi-turn sessions and sticky routing.
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI

# Test configuration constants
TEST_QPS = 50.0  # High QPS for fast test execution (not testing rate timing)
TEST_CANCELLATION_RATE = 25  # 25% cancellation rate for tests
TEST_CANCELLATION_DELAY = 0  # Immediate cancellation
TEST_OSL = 5  # Small output sequence length for fast generation


@pytest.mark.component_integration
class TestRequestCancellationMultiTurn:
    """Tests for request cancellation with multi-turn sessions."""

    @pytest.mark.slow
    def test_cancellation_with_multi_turn_request_rate(self, cli: AIPerfCLI):
        """Test request cancellation with multi-turn sessions and request rate.

        Cancellation should work properly with multi-turn sessions,
        allowing the session to continue to subsequent turns even if
        one turn is cancelled.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --num-sessions 10 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --request-cancellation-rate {TEST_CANCELLATION_RATE} \
                --request-cancellation-delay {TEST_CANCELLATION_DELAY} \
                --osl {TEST_OSL} \
                --random-seed 42 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # 10 sessions × 4 turns = 40 total requests (including cancelled)
        total_requests = len(result.jsonl)
        assert total_requests == 40, f"Expected 40 total requests, got {total_requests}"

        # Count cancelled requests
        cancelled_requests = 0
        for record in result.jsonl:
            if record.metadata.was_cancelled:
                cancelled_requests += 1
                assert record.error is not None
                assert record.error.code == 499
                assert record.error.type == "RequestCancellationError"

        # Should have some cancellations with 25% rate (~10 out of 40)
        assert cancelled_requests > 5, (
            f"Expected >5 cancellations with 25% rate, got {cancelled_requests}"
        )

        # Valid (completed) requests should be roughly 75% of total
        valid_requests = result.request_count
        assert valid_requests == total_requests - cancelled_requests

        # Verify error summary captures cancellations
        assert result.json.error_summary is not None
        assert result.json.error_summary
        cancellation_error = next(
            (
                e
                for e in result.json.error_summary
                if e.error_details.type == "RequestCancellationError"
            ),
            None,
        )
        assert cancellation_error is not None
        assert cancellation_error.count == cancelled_requests

        # Verify sessions still completed their turns despite cancellations
        session_turns = {}
        for record in result.jsonl:
            session_id = record.metadata.x_correlation_id
            turn = record.metadata.turn_index
            if session_id not in session_turns:
                session_turns[session_id] = set()
            session_turns[session_id].add(turn)

        # All sessions should have turn 0
        assert len(session_turns) == 10

    def test_cancellation_with_multi_turn_concurrency(self, cli: AIPerfCLI):
        """Test request cancellation with multi-turn sessions and concurrency."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --concurrency 4 \
                --num-sessions 8 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --request-cancellation-rate 30 \
                --request-cancellation-delay 0 \
                --osl 5 \
                --random-seed 42 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # 8 sessions × 3 turns = 24 total requests
        assert len(result.jsonl) == 24

        cancelled_requests = sum(
            1 for record in result.jsonl if record.metadata.was_cancelled
        )
        assert cancelled_requests > 1
        assert result.request_count == len(result.jsonl) - cancelled_requests

    @pytest.mark.slow
    def test_cancellation_mid_session(self, cli: AIPerfCLI):
        """Test cancellation in the middle of a multi-turn session.

        When a turn is cancelled, subsequent turns should still proceed.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --num-sessions 6 \
                --session-turns-mean 5 \
                --session-turns-stddev 0 \
                --request-cancellation-rate 20 \
                --request-cancellation-delay {TEST_CANCELLATION_DELAY} \
                --osl {TEST_OSL} \
                --random-seed 123 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # 6 sessions × 5 turns = 30 total requests
        assert len(result.jsonl) == 30

        # Track which turns were cancelled per session
        session_data = {}
        for record in result.jsonl:
            session_id = record.metadata.x_correlation_id
            turn = record.metadata.turn_index
            cancelled = record.metadata.was_cancelled

            if session_id not in session_data:
                session_data[session_id] = []
            session_data[session_id].append({"turn": turn, "cancelled": cancelled})

        # Verify that if a mid-session turn was cancelled,
        # subsequent turns still executed
        mid_session_cancelled = False
        for _, turns in session_data.items():
            turns.sort(key=lambda x: x["turn"])
            # If any turn except the last was cancelled, verify next turn still executed
            for i in range(len(turns) - 1):
                if turns[i]["cancelled"]:
                    # Next turn should still exist
                    assert turns[i + 1]["turn"] == turns[i]["turn"] + 1
                    mid_session_cancelled = True

        assert mid_session_cancelled

    @pytest.mark.slow
    def test_cancellation_sticky_routing_integrity(self, cli: AIPerfCLI):
        """Test that cancellation doesn't break sticky routing.

        Even when requests are cancelled, the session should maintain
        sticky routing to the same worker.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --num-sessions 5 \
                --session-turns-mean 5 \
                --session-turns-stddev 0 \
                --request-cancellation-rate 30 \
                --request-cancellation-delay {TEST_CANCELLATION_DELAY} \
                --osl {TEST_OSL} \
                --random-seed 42 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # 5 sessions × 5 turns = 25 total requests
        assert len(result.jsonl) == 25

        cancelled_requests = sum(
            1 for record in result.jsonl if record.metadata.was_cancelled
        )
        assert result.request_count == len(result.jsonl) - cancelled_requests

        # Verify all turns for each session exist (cancelled or not)
        session_turns = {}
        session_workers = {}
        for record in result.jsonl:
            session_id = record.metadata.x_correlation_id
            turn = record.metadata.turn_index
            worker_id = record.metadata.worker_id
            if session_id not in session_turns:
                session_turns[session_id] = set()
                session_workers[session_id] = set()
            session_turns[session_id].add(turn)
            session_workers[session_id].add(worker_id)

        # Each session should have all 5 turns (0-4)
        assert len(session_turns) == 5
        for session_id, turns in session_turns.items():
            assert turns == {0, 1, 2, 3, 4}, f"Session {session_id} missing turns"

        # Validate sticky routing - each session should use only one worker
        for session_id, workers in session_workers.items():
            assert len(workers) == 1, (
                f"Session {session_id} violated sticky routing: used {len(workers)} workers"
            )
