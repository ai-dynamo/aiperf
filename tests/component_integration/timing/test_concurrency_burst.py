# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for concurrency burst timing mode.

Concurrency burst mode issues credits with zero delay - throughput is
controlled entirely by the concurrency semaphore.

Key characteristics:
- No rate limiting - credits issued as fast as concurrency allows
- Effective rate = concurrency / avg_response_time
- Requires concurrency to be set (no request-rate)
- Maximum throughput mode

Tests cover:
- Basic functionality at various concurrency levels
- Credit flow verification
- Timing behavior (minimal gaps expected)
- Multi-turn conversations
- Concurrency limit enforcement
- Edge cases
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
    defaults,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
)
from tests.harness.utils import AIPerfCLI

# Fast OSL for concurrency burst tests (10 for slightly longer decode phase)
TEST_OSL_BURST = 10


def build_burst_command(
    config: TimingTestConfig,
    *,
    extra_args: str = "",
    osl: int | None = None,
) -> str:
    """Build a CLI command for concurrency burst tests.

    Burst mode requires concurrency but no request-rate.
    """
    osl_value = osl if osl is not None else TEST_OSL_BURST
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --concurrency {config.concurrency} \
            --osl {osl_value} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


@pytest.mark.component_integration
class TestConcurrencyBurstBasic:
    """Basic functionality tests for concurrency burst timing."""

    @pytest.mark.parametrize(  # fmt: skip
        "num_sessions,concurrency",
        [
            (15, 3),
            (25, 5),
            (40, 10),
            (60, 15),
        ],
    )
    def test_burst_mode_completes(
        self, cli: AIPerfCLI, num_sessions: int, concurrency: int
    ):
        """Test burst mode completes at various concurrency levels."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # No rate for burst mode
            concurrency=concurrency,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
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

    def test_credits_per_session(self, cli: AIPerfCLI):
        """Verify each session gets expected credits."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,
            turns_per_session=3,
            concurrency=5,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == config.num_sessions
        assert analyzer.session_credits_match(config.turns_per_session)

    @pytest.mark.slow
    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Verify turn indices are sequential per session."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            turns_per_session=5,
            concurrency=4,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestConcurrencyBurstLimits(BaseConcurrencyTests):
    """Tests for concurrency limit enforcement in burst mode.

    Inherits common concurrency tests from BaseConcurrencyTests, with customization
    for burst mode (qps=0) behavior. Uses parametrized test for single values instead
    of (concurrency, qps) pairs since burst mode has qps=0.

    Tests: test_with_concurrency_limit (customized), test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build burst mode command."""
        return build_burst_command(config)

    @pytest.mark.parametrize(  # fmt: skip
        "concurrency",
        [2, 4, 8, 12],
    )
    def test_with_concurrency_limit(self, cli: AIPerfCLI, concurrency: int):
        """Test burst mode respects and reaches concurrency limit.

        Override base class to use concurrency-only parameters (no QPS).
        Burst mode (qps=0) issues credits as fast as possible.
        """
        # Ensure we have enough sessions to hit the limit
        num_sessions = max(30, concurrency * 3)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=concurrency,
        )

        # Validate test parameters will actually hit the limit
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
        """Test burst mode with prefill concurrency limit.

        Override base class to ensure enough sessions for burst mode.
        """
        prefill_concurrency = 3
        # Ensure we have enough sessions to hit the prefill limit
        num_sessions = max(25, prefill_concurrency * 5)
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,  # Burst mode
            concurrency=10,
            prefill_concurrency=prefill_concurrency,
        )

        # Validate test parameters will hit the prefill limit
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
        """Test multi-turn burst with concurrency.

        Override base class to use burst mode (qps=0).
        """
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
        """Test low concurrency with many sessions (queuing behavior)."""
        config = TimingTestConfig(
            num_sessions=40,
            qps=0,
            concurrency=2,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)


@pytest.mark.component_integration
class TestConcurrencyBurstEdgeCases:
    """Edge case tests for concurrency burst timing."""

    def test_many_turns_burst(self, cli: AIPerfCLI):
        """Test many turns per session in burst mode."""
        config = TimingTestConfig(
            num_sessions=5,
            qps=0,
            turns_per_session=8,
            concurrency=3,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
