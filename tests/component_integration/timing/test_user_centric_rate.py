# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for user-centric rate timing mode.

User-centric rate mode implements LMBenchmark-style per-user rate limiting:
- N users (sessions), each with gap = num_users / qps between their turns
- Each user blocks on their previous turn (no interleaving within a user)
- Round-robin conversation assignment to users
- First turns staggered by gap / num_users (1/qps stagger)

Key characteristics:
- Per-user rate control vs global rate control
- Sequential turns within each user session
- Staggered first-turn timing
- Global QPS approximately maintained

Tests cover:
- Basic functionality at various QPS/session combinations
- Credit flow verification
- Stagger timing accuracy
- Per-user gap timing
- Sequential ordering within sessions
- Session interleaving globally
- Multi-turn conversation handling
- Race conditions and edge cases
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_timing_command,
    defaults,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    StatisticalAnalyzer,
    TimingAnalyzer,
    verify_no_interleaving_within_session,
)
from tests.harness.utils import AIPerfCLI


def build_user_centric_command(
    num_users: int,
    qps: float,
    *,
    num_sessions: int | None = None,
    request_count: int | None = None,
    turns_per_session: int = 2,
    osl: int = 50,
    benchmark_duration: float | None = None,
    benchmark_grace_period: float = 0.5,
    skip_multi_turn: bool = False,
) -> str:
    """Build CLI command for user-centric rate tests.

    Unified helper that supports all stop conditions:
    - num_sessions: Stop after completing N sessions
    - request_count: Stop after sending N requests
    - benchmark_duration: Stop after N seconds

    Args:
        num_users: Number of concurrent user slots
        qps: User-centric QPS (--user-centric-rate)
        num_sessions: Session count stop condition (optional)
        request_count: Request count stop condition (optional)
        turns_per_session: Turns per session (min 2 for user-centric)
        osl: Output sequence length
        benchmark_duration: Duration stop condition (optional)
        benchmark_grace_period: Grace period after duration
        skip_multi_turn: Skip multi-turn args (for validation tests)
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --osl {osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui} \
            --num-users {num_users} \
            --user-centric-rate {qps}
    """

    # User-centric mode requires multi-turn (unless testing validation)
    if not skip_multi_turn:
        turns = max(turns_per_session, 2)
        cmd += f" --session-turns-mean {turns} --session-turns-stddev 0"

    if num_sessions is not None:
        cmd += f" --num-sessions {num_sessions}"

    if request_count is not None:
        cmd += f" --request-count {request_count}"

    if benchmark_duration is not None:
        cmd += f" --benchmark-duration {benchmark_duration}"
        cmd += f" --benchmark-grace-period {benchmark_grace_period}"

    return cmd


@pytest.mark.component_integration
class TestUserCentricRateBasic:
    """Basic functionality tests for user-centric rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (10, 50.0),
            (20, 100.0),
        ],
    )
    def test_user_centric_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Test user-centric rate mode completes at various configurations.

        Uses --num-sessions as stop condition to verify exact session count.
        """
        cmd = build_user_centric_command(
            num_users=num_sessions,
            qps=qps,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.num_sessions == num_sessions
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestUserCentricRateCreditFlow:
    """Credit flow verification for user-centric rate timing."""

    def test_credits_balanced(self, cli: AIPerfCLI):
        """Verify all credits sent are returned."""
        config = TimingTestConfig(num_sessions=20, qps=100.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced: {analyzer.total_credits} sent, "
            f"{analyzer.total_returns} returned"
        )

    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Verify turn indices are sequential within each session.

        Turn indices must be 0, 1, 2, ... for each session's credits.
        """
        num_users = 10
        num_sessions = 15
        turns_per_session = 4

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            num_sessions=num_sessions,
            turns_per_session=turns_per_session,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestUserCentricRateStaggerTiming:
    """Tests for staggered first-turn timing."""

    def test_first_turns_staggered(self, cli: AIPerfCLI):
        """Verify first turns are staggered by 1/qps."""
        num_sessions = 10
        qps = 100.0
        config = TimingTestConfig(
            num_sessions=num_sessions, qps=qps, turns_per_session=2
        )
        expected_stagger = 1.0 / qps

        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --osl {config.osl} \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-users {config.num_sessions} \
                --num-sessions {config.num_sessions} \
                --user-centric-rate {qps} \
                --session-turns-mean {config.turns_per_session} --session-turns-stddev 0
        """
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        first_turn_times = timing.get_first_turn_issue_times_ns()

        assert len(first_turn_times) == num_sessions
        passed, reason = StatisticalAnalyzer.verify_stagger(
            first_turn_times,
            expected_stagger_sec=expected_stagger,
            tolerance_pct=50.0,
        )
        assert passed, f"First turns not properly staggered: {reason}"


@pytest.mark.component_integration
class TestUserCentricRatePerUserGap:
    """Tests for per-user gap timing."""

    def test_per_user_gap_respected(self, cli: AIPerfCLI):
        """Verify gap = num_sessions / qps between each user's turns."""
        num_sessions = 10
        qps = 100.0
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=qps,
            turns_per_session=5,
        )
        expected_gap = num_sessions / qps

        cmd = build_timing_command(config, user_centric_rate=qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        times_by_session = timing.get_issue_times_by_session()

        passed, reason = StatisticalAnalyzer.verify_per_user_gaps(
            times_by_session,
            expected_gap_sec=expected_gap,
            tolerance_pct=50.0,
        )
        assert passed, f"Per-user gap not respected: {reason}"


@pytest.mark.component_integration
class TestUserCentricRateSequentialOrdering:
    """Tests for sequential ordering within each user."""

    def test_no_interleaving_within_user(self, cli: AIPerfCLI):
        """Verify users block on their previous turn (no interleaving)."""
        config = TimingTestConfig(
            num_sessions=12,
            qps=75.0,
            turns_per_session=5,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        passed, reason = verify_no_interleaving_within_session(credit_analyzer)
        assert passed, f"Interleaving detected: {reason}"


@pytest.mark.component_integration
class TestUserCentricRateSessionCountStop:
    """Tests verifying session count can restrict user-centric mode."""

    def test_session_count_only_stop_condition(self, cli: AIPerfCLI):
        """Test user-centric mode with --num-sessions as the only stop condition."""
        num_users = 10
        num_sessions = 20
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )

    def test_partial_completion_multi_turn(self, cli: AIPerfCLI):
        """Test 1.5x users with multi-turn conversations."""
        num_users = 10
        num_sessions = 15
        turns_per_session = 3
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            num_sessions=num_sessions,
            qps=qps,
            turns_per_session=turns_per_session,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions
        assert analyzer.turn_indices_sequential()
        assert analyzer.credits_balanced()


@pytest.mark.component_integration
class TestUserCentricRateRequestCountStop:
    """Tests verifying request count can restrict user-centric mode."""

    def test_request_count_only_stop_condition(self, cli: AIPerfCLI):
        """Test user-centric mode with --request-count as the only stop condition."""
        num_users = 10
        request_count = 25
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == request_count

    def test_request_count_multi_turn_partial(self, cli: AIPerfCLI):
        """Test request count that stops mid-session."""
        num_users = 10
        turns_per_session = 3
        request_count = 25  # Less than 10 sessions × 3 turns = 30
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
            turns_per_session=turns_per_session,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        assert result.request_count == request_count

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()


@pytest.mark.component_integration
class TestUserCentricRateValidationErrors:
    """Tests verifying CLI validation errors for user-centric mode constraints.

    These tests verify that the CLI properly rejects invalid configurations:
    - --num-sessions < --num-users (each user needs at least one session)
    - --request-count < --num-users (each user needs at least one request)
    """

    def test_num_sessions_less_than_num_users_fails(self, cli: AIPerfCLI):
        """Verify CLI fails when --num-sessions < --num-users.

        Each user needs at least one session to process.
        """
        num_users = 20
        num_sessions = 10  # Invalid: less than num_users

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        assert result.exit_code == 1, "Expected CLI to fail with exit code 1"
        assert (
            "num-sessions" in result.stderr.lower()
            or "num_sessions" in result.stderr.lower()
        ), f"Expected error message about num-sessions, got: {result.stderr}"
        assert (
            "num-users" in result.stderr.lower() or "num_users" in result.stderr.lower()
        ), f"Expected error message about num-users, got: {result.stderr}"

    def test_request_count_less_than_num_users_fails(self, cli: AIPerfCLI):
        """Verify CLI fails when --request-count < --num-users.

        Each user needs at least one request to process.
        """
        num_users = 20
        request_count = 15  # Invalid: less than num_users

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            request_count=request_count,
        )
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        assert result.exit_code == 1, "Expected CLI to fail with exit code 1"
        assert (
            "request-count" in result.stderr.lower()
            or "request_count" in result.stderr.lower()
        ), f"Expected error message about request-count, got: {result.stderr}"
        assert (
            "num-users" in result.stderr.lower() or "num_users" in result.stderr.lower()
        ), f"Expected error message about num-users, got: {result.stderr}"

    def test_num_sessions_equals_num_users_succeeds(self, cli: AIPerfCLI):
        """Verify CLI succeeds when --num-sessions == --num-users (boundary case)."""
        num_users = 10
        num_sessions = 10  # Valid: exactly one session per user

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        # Verify session count (request count varies due to virtual history)
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )

    def test_request_count_equals_num_users_succeeds(self, cli: AIPerfCLI):
        """Verify CLI succeeds when --request-count == --num-users (boundary case).

        Note: With 2-turn sessions and virtual history, the exact request count
        is the stop condition. Some users may not complete their sessions.
        """
        num_users = 10
        request_count = 10  # Valid: exactly one request per user

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            request_count=request_count,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        # Request count should match the stop condition
        assert result.request_count == request_count

    def test_single_turn_user_centric_fails(self, cli: AIPerfCLI):
        """Verify user-centric mode rejects single-turn conversations.

        User-centric rate limiting only makes sense for multi-turn (>=2) conversations.
        For single-turn workloads, it degenerates to request-rate mode with extra overhead,
        so we reject it at config validation time to guide users to the right mode.
        """
        cmd = build_user_centric_command(
            num_users=10,
            qps=100.0,
            num_sessions=10,
            skip_multi_turn=True,  # Test single-turn rejection
        )
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        assert result.exit_code == 1, (
            "Expected CLI to fail for single-turn user-centric mode"
        )
        assert (
            "multi-turn" in result.stderr.lower()
            or "session-turns" in result.stderr.lower()
            or "--request-rate" in result.stderr
        ), f"Expected error message about multi-turn requirement, got: {result.stderr}"
