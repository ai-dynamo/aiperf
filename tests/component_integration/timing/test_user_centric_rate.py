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
    verify_sessions_can_interleave,
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
            (30, 150.0),
            (50, 200.0),
        ],
    )
    def test_user_centric_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Test user-centric rate mode completes at various configurations."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_timing_command(config, user_centric_rate=qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests (initial user turns)
        assert result.request_count >= num_sessions
        assert result.has_streaming_metrics

    def test_user_centric_rate_multi_turn(self, cli: AIPerfCLI):
        """Test user-centric rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=5,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least the initial user count
        assert result.request_count >= config.num_sessions
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

    def test_credits_per_session_sequential(self, cli: AIPerfCLI):
        """Verify each session's credits have sequential turn indices.

        Note: User-centric mode is duration-based (runs for --benchmark-duration),
        not session-count-based. With user replacement enabled, total sessions
        can exceed num_users as users complete and are replaced.

        Key verification: turn indices must be sequential within each session.
        """
        config = TimingTestConfig(
            num_sessions=12,
            qps=60.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Duration-based with user replacement: session count can exceed num_users
        # The key invariant is that turn indices are sequential within each session
        assert analyzer.turn_indices_sequential()

    @pytest.mark.slow
    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Verify turn indices are sequential per session."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=50.0,
            turns_per_session=6,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.turn_indices_sequential()


@pytest.mark.component_integration
class TestUserCentricRateStaggerTiming:
    """Tests for staggered first-turn timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (10, 100.0),  # 10ms stagger
            (15, 150.0),  # ~6.7ms stagger
            (20, 200.0),  # 5ms stagger
        ],
    )
    def test_first_turns_staggered(self, cli: AIPerfCLI, num_sessions: int, qps: float):
        """Verify first turns are staggered by 1/qps.

        Uses session-count stopping (not duration) to prevent user replacement,
        which would cause variable gaps as new users start when old ones finish.
        We want to verify the initial stagger is consistent.
        """
        config = TimingTestConfig(
            num_sessions=num_sessions, qps=qps, turns_per_session=2
        )
        expected_stagger = 1.0 / qps

        # Use --num-sessions to prevent user replacement (which skews stagger measurement)
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

        assert len(first_turn_times) == num_sessions, (
            f"Expected {num_sessions} first turns with session-count stopping, "
            f"got {len(first_turn_times)}"
        )
        passed, reason = StatisticalAnalyzer.verify_stagger(
            first_turn_times,
            expected_stagger_sec=expected_stagger,
            tolerance_pct=50.0,
        )
        assert passed, f"First turns not properly staggered: {reason}"

    def test_stagger_consistency(self, cli: AIPerfCLI):
        """Test that stagger is consistent across first turns.

        Uses session-count stopping (not duration) to prevent user replacement,
        which would cause variable gaps as new users start when old ones finish.
        We want to verify the initial stagger is consistent.
        """
        config = TimingTestConfig(num_sessions=15, qps=100.0, turns_per_session=2)
        # Use custom command to avoid duration-based stopping (which enables user replacement)
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --osl {config.osl} \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-users {config.num_sessions} \
                --num-sessions {config.num_sessions} \
                --user-centric-rate {config.qps} \
                --session-turns-mean {config.turns_per_session} --session-turns-stddev 0
        """
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        first_turn_times = timing.get_first_turn_issue_times_ns()

        # With session-count stopping, we should get exactly num_sessions first turns
        assert len(first_turn_times) == config.num_sessions, (
            f"Expected {config.num_sessions} first turns with session-count stopping, "
            f"got {len(first_turn_times)}"
        )

        gaps = timing.calculate_gaps_sec(first_turn_times)
        assert len(gaps) >= 5, (
            f"Insufficient data for stagger consistency analysis: got {len(gaps)} gaps, "
            f"need >= 5. 15 sessions should produce enough data."
        )
        cv = timing.calculate_cv(gaps)
        # Stagger should be relatively consistent (CV < 0.6)
        assert cv < 0.6, f"Stagger CV {cv:.4f} too variable"


@pytest.mark.component_integration
class TestUserCentricRatePerUserGap:
    """Tests for per-user gap timing (gap = num_sessions / qps)."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (8, 50.0),  # gap = 0.16s
            (10, 100.0),  # gap = 0.1s
            (12, 60.0),  # gap = 0.2s
        ],
    )
    def test_per_user_gap_respected(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Verify gap = num_sessions / qps between each user's turns."""
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

    def test_sessions_can_interleave_globally(self, cli: AIPerfCLI):
        """Verify different sessions CAN interleave (not blocked by others)."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=100.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        passed, reason = verify_sessions_can_interleave(credit_analyzer)
        assert passed, f"Sessions not interleaving: {reason}"


@pytest.mark.component_integration
class TestUserCentricRateMultiTurn:
    """Tests for multi-turn conversation handling."""

    @pytest.mark.parametrize(
        "num_sessions,turns_per_session",
        [
            (8, 4),
            (10, 6),
            (12, 8),
            (15, 5),
        ],
    )
    def test_multi_turn_runs(
        self, cli: AIPerfCLI, num_sessions: int, turns_per_session: int
    ):
        """Verify multi-turn conversations run with sequential turn indices."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=100.0,
            turns_per_session=turns_per_session,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Duration-based: at least num_sessions requests
        assert result.request_count >= config.num_sessions

        # Verify multiple turn indices present (multi-turn working)
        turn_indices = set(r.metadata.turn_index for r in result.jsonl)
        assert len(turn_indices) > 1, "Expected multiple turn indices for multi-turn"

    def test_multi_turn_credit_returns_balanced(self, cli: AIPerfCLI):
        """Verify all credits sent are returned."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=75.0,
            turns_per_session=5,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # All credits should be returned
        assert analyzer.credits_balanced()


@pytest.mark.component_integration
class TestUserCentricRateEdgeCases:
    """Edge case tests for user-centric rate timing."""

    def test_single_user(self, cli: AIPerfCLI):
        """Test with single user (edge case: num_users = 1)."""
        config = TimingTestConfig(num_sessions=1, qps=100.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 1

    def test_single_user_multi_turn(self, cli: AIPerfCLI):
        """Test single user with multiple turns."""
        config = TimingTestConfig(
            num_sessions=1,
            qps=50.0,
            turns_per_session=5,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count >= 1

    def test_very_high_qps(self, cli: AIPerfCLI):
        """Test very high QPS (stress test)."""
        config = TimingTestConfig(num_sessions=50, qps=300.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions

    def test_low_qps(self, cli: AIPerfCLI):
        """Test low QPS with longer gaps."""
        config = TimingTestConfig(
            num_sessions=5,
            qps=5.0,
            timeout=30.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= 1

    def test_many_users_few_turns(self, cli: AIPerfCLI):
        """Test many users with single turn each."""
        config = TimingTestConfig(num_sessions=60, qps=200.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions

    def test_few_users_many_turns(self, cli: AIPerfCLI):
        """Test few users with many turns."""
        config = TimingTestConfig(
            num_sessions=5,
            qps=50.0,
            turns_per_session=10,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestUserCentricRateStress:
    """Stress tests for user-centric rate timing."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Test high volume user-centric workload."""
        config = TimingTestConfig(
            num_sessions=100,
            qps=300.0,
            timeout=90.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn user-centric workload."""
        config = TimingTestConfig(
            num_sessions=25,
            qps=100.0,
            turns_per_session=5,
            timeout=90.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_high_user_count(self, cli: AIPerfCLI):
        """Test with very high user count."""
        config = TimingTestConfig(
            num_sessions=150,
            qps=400.0,
            timeout=120.0,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count >= config.num_sessions


@pytest.mark.component_integration
class TestUserCentricRateVsConstantRate:
    """Comparison tests between user-centric and constant rate modes."""

    def test_global_rate_approximately_maintained(self, cli: AIPerfCLI):
        """Verify user-centric approximately maintains global QPS."""
        config = TimingTestConfig(num_sessions=30, qps=100.0)
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()

        assert len(issue_times) >= 10, (
            f"Insufficient data for rate analysis: got {len(issue_times)} issue times, "
            f"need >= 10. 30 sessions should produce enough data."
        )
        total_duration = (issue_times[-1] - issue_times[0]) / 1e9
        actual_rate = (
            (len(issue_times) - 1) / total_duration if total_duration > 0 else 0
        )

        # User-centric should approximately maintain global rate
        # (may be slightly lower due to blocking behavior)
        expected_rate = config.qps
        tolerance = expected_rate * 0.5  # 50% tolerance

        assert actual_rate > expected_rate - tolerance, (
            f"Actual rate {actual_rate:.1f} QPS much lower than "
            f"expected {expected_rate:.1f} QPS"
        )


@pytest.mark.component_integration
class TestUserCentricRateSessionCountStop:
    """Tests verifying session count can restrict user-centric mode.

    These tests verify that --num-sessions works as a stop condition for
    user-centric mode, separate from duration-based stopping:
    - num_users: concurrent user slots
    - num_sessions: total sessions to complete (stop condition)

    This allows configurations like:
    - 10 concurrent users, complete 15 total sessions (1.5x users)
    - Long duration but sessions should still restrict execution
    """

    def test_session_count_only_stop_condition(self, cli: AIPerfCLI):
        """Test user-centric mode with --num-sessions as the only stop condition.

        No --benchmark-duration is set, so session count should be the
        sole stop condition.

        Note: Request count varies due to virtual history (ramp-up users have fewer
        real turns), but session count should be exact.
        """
        num_users = 10
        num_sessions = 20
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        # Session count should be exact (request count varies due to virtual history)
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )

    def test_session_count_with_long_duration(self, cli: AIPerfCLI):
        """Test session count restricts even with very long duration.

        Duration is long (60s) but num_sessions should stop execution first.
        """
        num_users = 15
        num_sessions = 30
        qps = 150.0

        cmd = build_user_centric_command(
            num_users=num_users,
            qps=qps,
            num_sessions=num_sessions,
            benchmark_duration=60.0,  # Long duration that won't be reached
        )
        result = cli.run_sync(cmd, timeout=45.0)

        # Session count should be exact (request count varies due to virtual history)
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )

    def test_partial_completion_1_5x_users(self, cli: AIPerfCLI):
        """Test 1.5x users scenario for partial completion.

        With 10 users and 15 sessions, exactly 15 sessions should complete.
        This tests that user replacement works correctly when num_sessions > num_users.
        """
        num_users = 10
        num_sessions = 15  # 1.5x users
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            num_sessions=num_sessions,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    def test_partial_completion_2x_users(self, cli: AIPerfCLI):
        """Test 2x users scenario (double replacement cycle).

        With 10 users and 20 sessions, all users complete twice.
        """
        num_users = 10
        num_sessions = 20  # 2x users
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            num_sessions=num_sessions,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()

    def test_partial_completion_multi_turn(self, cli: AIPerfCLI):
        """Test 1.5x users with multi-turn conversations.

        10 users, 15 sessions, 3 turns each. Session count should be exact,
        but request count varies due to virtual history (ramp-up users have
        fewer real turns).
        """
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

        # Session count should be exact (request count varies due to virtual history)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.turn_indices_sequential()
        assert analyzer.credits_balanced()

    def test_users_equal_sessions(self, cli: AIPerfCLI):
        """Test when num_users equals num_sessions (no replacement needed).

        With 10 users and 10 sessions, each user runs one session.
        """
        num_users = 10
        num_sessions = 10
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            num_sessions=num_sessions,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )

    def test_fewer_users_than_sessions(self, cli: AIPerfCLI):
        """Test with many more sessions than users (multiple replacement cycles).

        5 users completing 25 sessions = 5 full cycles.
        """
        num_users = 5
        num_sessions = 25  # 5x users
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            num_sessions=num_sessions,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=90.0)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.num_sessions == num_sessions, (
            f"Expected {num_sessions} sessions, got {analyzer.num_sessions}"
        )
        assert analyzer.credits_balanced()


@pytest.mark.component_integration
class TestUserCentricRateRequestCountStop:
    """Tests verifying request count can restrict user-centric mode.

    These tests verify that --request-count works as a stop condition for
    user-centric mode:
    - num_users: concurrent user slots
    - request_count: total requests to complete (stop condition)

    Unlike --num-sessions which counts complete conversations, --request-count
    counts individual requests (turns). With multi-turn conversations, this
    can stop execution mid-session.
    """

    def test_request_count_only_stop_condition(self, cli: AIPerfCLI):
        """Test user-centric mode with --request-count as the only stop condition.

        No --benchmark-duration is set, so request count should be the
        sole stop condition.
        """
        num_users = 10
        request_count = 25
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == request_count, (
            f"Expected exactly {request_count} requests, got {result.request_count}"
        )

    def test_request_count_with_long_duration(self, cli: AIPerfCLI):
        """Test request count restricts even with very long duration.

        Duration is long (60s) but request_count should stop execution first.
        """
        num_users = 15
        request_count = 40
        qps = 150.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
            benchmark_duration=60.0,  # Long duration that won't be reached
        )
        result = cli.run_sync(cmd, timeout=45.0)

        assert result.request_count == request_count, (
            f"Expected exactly {request_count} requests, got {result.request_count}"
        )

    def test_request_count_equals_users(self, cli: AIPerfCLI):
        """Test request count equals num_users (minimum valid configuration).

        With 20 users and 20 requests, each user gets exactly one turn.
        Note: request_count cannot be less than num_users (validation rule).
        """
        num_users = 20
        request_count = 20  # Minimum valid: one request per user
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == request_count, (
            f"Expected exactly {request_count} requests, got {result.request_count}"
        )

    def test_request_count_multi_turn_exact(self, cli: AIPerfCLI):
        """Test request count that aligns with complete sessions.

        10 users, 3 turns each, request_count=30 = exactly 10 complete sessions.
        """
        num_users = 10
        turns_per_session = 3
        request_count = 30  # Exactly 10 sessions × 3 turns
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

    def test_request_count_multi_turn_partial(self, cli: AIPerfCLI):
        """Test request count that stops mid-session.

        10 users, 3 turns each, but request_count=25 stops before all
        sessions complete their turns.
        """
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

        assert result.request_count == request_count, (
            f"Expected exactly {request_count} requests (partial sessions), "
            f"got {result.request_count}"
        )

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_request_count_1_5x_users(self, cli: AIPerfCLI):
        """Test 1.5x users worth of requests (single turn).

        10 users, request_count=15 means 15 requests total.
        """
        num_users = 10
        request_count = 15  # 1.5x users
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
        )
        result = cli.run_sync(cmd, timeout=60.0)

        assert result.request_count == request_count

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_request_count_multiple_cycles(self, cli: AIPerfCLI):
        """Test request count spanning multiple user replacement cycles.

        5 users, request_count=50 = 10 full cycles.
        """
        num_users = 5
        request_count = 50  # 10x users
        qps = 100.0

        cmd = build_user_centric_command(
            num_users=num_users,
            request_count=request_count,
            qps=qps,
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

    @pytest.mark.parametrize(
        "num_users,num_sessions",
        [
            (10, 5),  # 2x fewer sessions
            (20, 1),  # Single session for many users
            (50, 25),  # Half the sessions
            (100, 50),  # Large scale
        ],
    )
    def test_various_invalid_num_sessions(
        self, cli: AIPerfCLI, num_users: int, num_sessions: int
    ):
        """Verify various invalid num_sessions configurations fail."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            num_sessions=num_sessions,
        )
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        assert result.exit_code == 1, (
            f"Expected failure for num_users={num_users}, num_sessions={num_sessions}"
        )

    @pytest.mark.parametrize(
        "num_users,request_count",
        [
            (10, 5),  # 2x fewer requests
            (20, 1),  # Single request for many users
            (50, 25),  # Half the requests
            (100, 50),  # Large scale
        ],
    )
    def test_various_invalid_request_count(
        self, cli: AIPerfCLI, num_users: int, request_count: int
    ):
        """Verify various invalid request_count configurations fail."""
        cmd = build_user_centric_command(
            num_users=num_users,
            qps=100.0,
            request_count=request_count,
        )
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        assert result.exit_code == 1, (
            f"Expected failure for num_users={num_users}, request_count={request_count}"
        )

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
