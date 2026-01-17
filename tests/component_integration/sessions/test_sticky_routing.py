# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sticky routing with multi-turn sessions."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.conftest import (
    assert_jsonl_turns_sequential,
    assert_sticky_routing,
)
from tests.harness.utils import AIPerfCLI

# Test configuration constants
TEST_QPS = 50.0  # High QPS for fast test execution (not testing rate timing)


@pytest.mark.component_integration
class TestStickyRouting:
    """Tests for sticky routing behavior with multi-turn sessions."""

    @pytest.mark.slow
    def test_sticky_routing_multi_turn_request_rate(self, cli: AIPerfCLI):
        """Test sticky routing maintains session affinity with request rate.

        Each session should maintain sticky routing via x-correlation-id.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --num-sessions 5 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # 5 sessions × 4 turns = 20 requests
        assert result.request_count == 20

        # Verify sticky routing and session integrity
        sessions = assert_sticky_routing(result.jsonl)
        assert len(sessions) == 5
        for session_id, records in sessions.items():
            assert len(records) == 4, f"Session {session_id} missing turns"
        assert_jsonl_turns_sequential(result.jsonl)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sticky_routing_high_concurrency_multi_turn(self, cli: AIPerfCLI):
        """Test sticky routing with high worker count and many sessions.

        Validates that sticky routing works correctly when sessions
        are distributed across many workers.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --num-sessions 15 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --workers-max 10 \
                --ui {defaults.ui}
            """
        )
        # 15 sessions × 3 turns = 45 requests
        assert result.request_count == 45

        # Verify sticky routing and all sessions completed all turns
        sessions = assert_sticky_routing(result.jsonl)
        assert len(sessions) == 15
        for session_id, records in sessions.items():
            assert len(records) == 3, f"Session {session_id} incomplete"
        assert_jsonl_turns_sequential(result.jsonl)

    @pytest.mark.slow
    def test_sticky_routing_with_concurrency_limit(self, cli: AIPerfCLI):
        """Test sticky routing with concurrency limit and request rate.

        Validates that concurrency limits don't break sticky routing.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --concurrency 3 \
                --num-sessions 9 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # 9 sessions × 4 turns = 36 requests
        assert result.request_count == 36

        # Verify sticky routing
        assert_sticky_routing(result.jsonl)
        assert_jsonl_turns_sequential(result.jsonl)

    @pytest.mark.slow
    def test_sticky_routing_with_turn_delays(self, cli: AIPerfCLI):
        """Test sticky routing with inter-turn delays.

        Validates that delays between turns don't affect sticky routing.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --random-seed 42 \
                --num-sessions 6 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --session-turn-delay-stddev 20 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # 6 sessions × 3 turns = 18 requests
        assert result.request_count == 18

        # Verify sticky routing and session integrity
        sessions = assert_sticky_routing(result.jsonl)
        for session_id, records in sessions.items():
            assert len(records) == 3, f"Session {session_id} has incorrect turn count"
        assert_jsonl_turns_sequential(result.jsonl)

    @pytest.mark.slow
    def test_sticky_routing_variable_turn_counts(self, cli: AIPerfCLI):
        """Test sticky routing with variable turn counts per session."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --request-rate {TEST_QPS} \
                --random-seed 42 \
                --num-sessions 8 \
                --session-turns-mean 3 \
                --session-turns-stddev 1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        # With stddev=1, turn counts will vary (2-4 typically)
        # Total requests will vary but should be around 24 (8 * 3)
        assert result.request_count >= 16
        assert result.request_count <= 32

        # Verify sticky routing and turn indices are sequential
        assert_sticky_routing(result.jsonl)
        assert_jsonl_turns_sequential(result.jsonl)
