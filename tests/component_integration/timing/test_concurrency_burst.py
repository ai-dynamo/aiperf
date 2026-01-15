# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for concurrency burst timing mode.

Concurrency burst mode issues credits with zero delay - throughput is
controlled entirely by the concurrency semaphore.

Key characteristics:
- No rate limiting - credits issued as fast as concurrency allows
- Effective rate = concurrency / avg_response_time
- Requires concurrency to be set (no request-rate)
- Maximum throughput mode

Tests cover:
- Basic burst mode completion
- Credit flow verification (balance, per-session, sequential turns)
- Concurrency limit enforcement (total and prefill)
- Multi-turn conversations
- Edge cases (low concurrency, high sessions)
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    TimingTestConfig,
    assert_concurrency_limit_hit,
    assert_concurrency_limit_respected,
    assert_request_count,
    build_burst_command,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestConcurrencyBurstBasic:
    """Basic functionality tests for concurrency burst timing."""

    def test_burst_mode_completes(self, cli: AIPerfCLI):
        """Test burst mode completes successfully."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=0,  # No rate for burst mode
            concurrency=6,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions
        assert result.has_streaming_metrics

    def test_burst_mode_multi_turn(self, cli: AIPerfCLI):
        """Test burst mode with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=12,
            qps=0,
            turns_per_session=4,
            concurrency=6,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestConcurrencyBurstCreditFlow:
    """Credit flow verification for concurrency burst timing."""

    def test_credits_balanced(self, cli: AIPerfCLI):
        """Verify all credits sent are returned."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=0,
            concurrency=8,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced: {analyzer.total_credits} sent, "
            f"{analyzer.total_returns} returned"
        )

    def test_credits_per_session_with_sequential_turns(self, cli: AIPerfCLI):
        """Verify each session gets expected credits with sequential turn indices."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,
            turns_per_session=4,
            concurrency=5,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == config.num_sessions
        assert analyzer.session_credits_match(config.turns_per_session)
        assert analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestConcurrencyBurstLimits(BaseConcurrencyTests):
    """Tests for concurrency limit enforcement in burst mode.

    Inherits common concurrency tests from BaseConcurrencyTests, with customization
    for burst mode (qps=0) behavior.
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build burst mode command."""
        return build_burst_command(config)

    @pytest.mark.parametrize("concurrency", [4, 10])  # fmt: skip
    def test_with_concurrency_limit(self, cli: AIPerfCLI, concurrency: int):
        """Test burst mode respects and reaches concurrency limit.

        Override base class to use concurrency-only parameters (no QPS).
        Burst mode (qps=0) issues credits as fast as possible.
        """
        num_sessions = max(30, concurrency * 3)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=concurrency,
        )

        assert config.will_hit_concurrency_limit(), (
            f"Test config won't hit concurrency limit: "
            f"num_sessions={num_sessions}, concurrency={concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, concurrency)
        assert_concurrency_limit_hit(result, concurrency)

    def test_with_prefill_concurrency(self, cli: AIPerfCLI):
        """Test burst mode with prefill concurrency limit."""
        prefill_concurrency = 3
        num_sessions = max(25, prefill_concurrency * 5)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=10,
            prefill_concurrency=prefill_concurrency,
        )

        assert config.will_hit_prefill_limit(), (
            f"Test config won't hit prefill limit: "
            f"num_sessions={num_sessions}, prefill_concurrency={prefill_concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, prefill_concurrency, prefill=True)
        assert_concurrency_limit_hit(result, prefill_concurrency, prefill=True)

    def test_multi_turn_with_concurrency(self, cli: AIPerfCLI):
        """Test multi-turn burst with concurrency limit enforcement."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,  # Burst mode
            turns_per_session=4,
            concurrency=4,
        )

        assert config.will_hit_concurrency_limit()

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.expected_requests)
        assert_concurrency_limit_hit(result, config.concurrency)

    def test_low_concurrency_high_sessions(self, cli: AIPerfCLI):
        """Test low concurrency with many sessions (queuing behavior).

        Verifies that burst mode correctly queues requests when
        concurrency is much lower than total sessions.
        """
        config = TimingTestConfig(
            num_sessions=40,
            qps=0,
            concurrency=2,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_hit(result, config.concurrency)
