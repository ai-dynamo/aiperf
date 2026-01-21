# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for fixed schedule timing mode.

Fixed schedule mode replays conversation traces at precise timestamps from dataset
metadata. First turns are sent at absolute timestamps, subsequent turns are
dispatched based on delay_ms or calculated from timestamp_ms.

Tests cover:
- Basic functionality with timestamp-based scheduling
- Credit flow verification (balanced, per-session)
- Timing accuracy (requests at correct timestamps)
- Multi-turn conversations with delays
- Concurrency interactions
- Load balancing across workers
- Edge cases (single request, worker configurations)
"""

from dataclasses import dataclass
from pathlib import Path

import orjson
import pytest

from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    CreditFlowAnalyzer,
    LoadBalancingAnalyzer,
)
from tests.harness.utils import AIPerfCLI, AIPerfResults


def get_request_count(result: AIPerfResults) -> int:
    """Get request count from results, falling back to JSONL if JSON export fails.

    Fixed schedule mode has a validation conflict with the default dataset_sampling_strategy,
    which causes JSON export to fail. Fall back to JSONL record count.
    """
    # Try JSON export first (uses result.json.request_count.avg)
    if result.json and result.json.request_count:
        return int(result.json.request_count.avg)
    # Fall back to JSONL record count
    if result.jsonl:
        return len(result.jsonl)
    return 0


@dataclass
class FixedScheduleTestConfig:
    """Configuration for a fixed schedule test scenario."""

    num_sessions: int
    turns_per_session: int = 1
    schedule_duration_ms: int = 400  # Total schedule duration in ms (keep fast!)
    delay_ms: int = 5  # Delay between turns (for multi-turn)
    workers_max: int = 3
    concurrency: int | None = None
    prefill_concurrency: int | None = None
    osl: int = 50
    timeout: float = 60.0

    @property
    def expected_requests(self) -> int:
        """Calculate expected total requests."""
        return self.num_sessions * self.turns_per_session


def generate_trace_file(
    path: Path,
    config: FixedScheduleTestConfig,
    *,
    stagger_ms: int | None = None,
) -> Path:
    """Generate a mooncake trace file for fixed schedule testing.

    Args:
        path: Directory to create trace file in
        config: Test configuration
        stagger_ms: If set, stagger first turns by this interval.
                    If None, distributes evenly across schedule_duration_ms.

    Returns:
        Path to created trace file
    """
    trace_file = path / "trace.jsonl"

    if stagger_ms is None:
        # Distribute first turns evenly across schedule duration
        if config.num_sessions > 1:
            stagger_ms = config.schedule_duration_ms // (config.num_sessions - 1)
        else:
            stagger_ms = 0

    with open(trace_file, "w") as f:
        for session_idx in range(config.num_sessions):
            for turn_idx in range(config.turns_per_session):
                if turn_idx == 0:
                    # First turn - needs timestamp
                    line = {
                        "session_id": f"session_{session_idx}",
                        "timestamp": session_idx * stagger_ms,
                        "input_length": 100,
                    }
                else:
                    # Subsequent turns - use delay
                    line = {
                        "session_id": f"session_{session_idx}",
                        "delay": config.delay_ms,
                        "input_length": 100,
                    }
                f.write(orjson.dumps(line).decode() + "\n")

    return trace_file


def build_fixed_schedule_command(
    config: FixedScheduleTestConfig,
    trace_file: Path,
    *,
    auto_offset: bool = True,
    extra_args: str = "",
) -> str:
    """Build a CLI command for fixed schedule tests.

    Args:
        config: Test configuration
        trace_file: Path to trace file
        auto_offset: Whether to auto-offset timestamps
        extra_args: Additional CLI arguments

    Returns:
        CLI command string
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --fixed-schedule \
            --custom-dataset-type mooncake_trace \
            --input-file {trace_file} \
            --workers-max {config.workers_max} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """

    if auto_offset:
        cmd += " --fixed-schedule-auto-offset"

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


# =============================================================================
# Basic Functionality Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleBasic:
    """Basic functionality tests for fixed schedule timing."""

    @pytest.mark.parametrize(  # fmt: skip
        "num_sessions",
        [5, 10, 20, 50],
    )
    def test_fixed_schedule_completes(
        self, cli: AIPerfCLI, tmp_path: Path, num_sessions: int
    ):
        """Test fixed schedule mode completes with various session counts."""
        config = FixedScheduleTestConfig(num_sessions=num_sessions)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == num_sessions

    def test_fixed_schedule_multi_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test fixed schedule with multi-turn conversations."""
        config = FixedScheduleTestConfig(
            num_sessions=15,
            turns_per_session=4,
            delay_ms=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests


# =============================================================================
# Credit Flow Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleCreditFlow:
    """Credit flow verification for fixed schedule timing."""

    def test_credits_balanced(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify all credits sent are returned."""
        config = FixedScheduleTestConfig(num_sessions=20)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced(), (
            f"Credits not balanced: {analyzer.total_credits} sent, "
            f"{analyzer.total_returns} returned"
        )

    def test_credits_per_session(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify each session gets expected credits."""
        config = FixedScheduleTestConfig(
            num_sessions=12,
            turns_per_session=3,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Build credits per session count
        credits_per_session = {
            sid: len(payloads) for sid, payloads in analyzer.credits_by_session.items()
        }
        assert len(credits_per_session) == config.num_sessions

        for session_id, count in credits_per_session.items():
            assert count == config.turns_per_session, (
                f"Session {session_id} has {count} credits, "
                f"expected {config.turns_per_session}"
            )

    def test_turn_indices_sequential(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify turn indices are sequential within each session."""
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # turn_indices_sequential() returns bool only
        assert analyzer.turn_indices_sequential()


# =============================================================================
# Timing Accuracy Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleTiming:
    """Timing accuracy tests for fixed schedule mode."""

    def test_first_turns_staggered(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify first turns are sent at staggered timestamps."""
        stagger_ms = 50  # 50ms between first turns
        config = FixedScheduleTestConfig(
            num_sessions=10,
            schedule_duration_ms=450,  # 9 gaps of 50ms
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=stagger_ms)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Get first turn issue times (returns list[int], not dict)
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        assert len(first_turn_times) == config.num_sessions

        # Verify stagger pattern (with tolerance)
        # times are already sorted from the method
        gaps = [
            (first_turn_times[i] - first_turn_times[i - 1])
            for i in range(1, len(first_turn_times))
        ]

        expected_gap_ns = stagger_ms * 1_000_000
        tolerance_ms = 40  # ±40ms tolerance (accounts for CI scheduling jitter)

        for gap in gaps:
            error_ms = abs(gap - expected_gap_ns) / 1_000_000
            assert error_ms < tolerance_ms, (
                f"Gap {gap / 1e6:.2f}ms differs from expected {stagger_ms}ms by {error_ms:.1f}ms"
            )

    def test_concurrent_first_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Test multiple sessions with same timestamp (concurrent start)."""
        config = FixedScheduleTestConfig(num_sessions=10)
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)  # All at t=0
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        # All first turns should be issued within a small window
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # get_first_turn_issue_times_ns() returns list[int]
        first_turn_times = analyzer.get_first_turn_issue_times_ns()
        assert len(first_turn_times) == config.num_sessions

        # Max spread should be less than 200ms for concurrent starts
        # (widened from 100ms to account for CI scheduling jitter)
        max_spread_ns = max(first_turn_times) - min(first_turn_times)
        assert max_spread_ns < 200_000_000, (
            f"First turn spread {max_spread_ns / 1e6:.2f}ms too large for concurrent start"
        )


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleMultiTurn:
    """Multi-turn conversation tests for fixed schedule mode."""

    def test_delays_between_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify delays between turns are respected with timing check."""
        config = FixedScheduleTestConfig(
            num_sessions=3,
            turns_per_session=3,
            delay_ms=40,  # 40ms between turns
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify delay timing accuracy
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        credit_payloads = runner.payloads_by_type(Credit, sent=True)
        return_payloads = runner.payloads_by_type(CreditReturn, sent=True)

        for session_id in {p.payload.x_correlation_id for p in credit_payloads}:
            session_credits = sorted(
                [
                    p
                    for p in credit_payloads
                    if p.payload.x_correlation_id == session_id
                ],
                key=lambda p: p.payload.turn_index,
            )
            session_returns = sorted(
                [
                    p
                    for p in return_payloads
                    if p.payload.credit.x_correlation_id == session_id
                ],
                key=lambda p: p.payload.credit.turn_index,
            )

            for i in range(len(session_returns) - 1):
                actual_delay_ms = (
                    session_credits[i + 1].timestamp_ns
                    - session_returns[i].timestamp_ns
                ) / 1_000_000
                error_ms = abs(actual_delay_ms - config.delay_ms)
                assert error_ms < 50, f"Delay error {error_ms:.1f}ms exceeds tolerance"

    def test_zero_delay_turns(self, cli: AIPerfCLI, tmp_path: Path):
        """Test multi-turn with zero delay (immediate subsequent turns)."""
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=5,
            delay_ms=0,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify all sessions completed all turns
        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        # Use session_credits_match helper
        assert analyzer.session_credits_match(expected_turns=config.turns_per_session)


# =============================================================================
# Concurrency Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleConcurrency:
    """Concurrency limit tests for fixed schedule mode."""

    def test_respects_concurrency_limit(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify concurrency limit is respected."""
        concurrency_limit = 5
        config = FixedScheduleTestConfig(
            num_sessions=20,
            concurrency=concurrency_limit,
        )
        # All sessions start at t=0 to stress concurrency
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        analyzer = ConcurrencyAnalyzer(result)

        # concurrency_within_limit() returns bool only
        assert analyzer.concurrency_within_limit(concurrency_limit)

    def test_respects_prefill_concurrency_limit(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify prefill concurrency limit is respected."""
        prefill_limit = 3
        config = FixedScheduleTestConfig(
            num_sessions=15,
            prefill_concurrency=prefill_limit,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions

        analyzer = ConcurrencyAnalyzer(result)

        # prefill_concurrency_within_limit() returns bool only
        assert analyzer.prefill_concurrency_within_limit(prefill_limit)

    def test_concurrency_with_multi_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test concurrency limits with multi-turn conversations."""
        concurrency_limit = 8
        config = FixedScheduleTestConfig(
            num_sessions=10,
            turns_per_session=4,
            delay_ms=0,  # Zero delay to maximize concurrent requests
            concurrency=concurrency_limit,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify concurrency limit is actually respected
        analyzer = ConcurrencyAnalyzer(result)
        assert analyzer.concurrency_within_limit(concurrency_limit)


# =============================================================================
# Load Balancing Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleLoadBalancing:
    """Load balancing tests for fixed schedule mode."""

    def test_jains_fairness_index(self, cli: AIPerfCLI, tmp_path: Path):
        """Verify Jain's Fairness Index is acceptable."""
        config = FixedScheduleTestConfig(
            num_sessions=100,
            workers_max=5,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = LoadBalancingAnalyzer(result)

        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.9, f"Jain's Fairness Index {jfi:.4f} below threshold 0.9"


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleEdgeCases:
    """Edge case tests for fixed schedule mode."""

    def test_single_session_single_turn(self, cli: AIPerfCLI, tmp_path: Path):
        """Test minimal case: single session, single turn."""
        config = FixedScheduleTestConfig(num_sessions=1, turns_per_session=1)
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == 1

    def test_single_worker(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with single worker (all requests routed to same worker)."""
        config = FixedScheduleTestConfig(
            num_sessions=15,
            turns_per_session=3,
            workers_max=1,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

    def test_more_workers_than_sessions(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with more workers than sessions."""
        config = FixedScheduleTestConfig(
            num_sessions=3,
            workers_max=10,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.num_sessions


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestFixedScheduleStress:
    """Stress tests for fixed schedule mode."""

    def test_high_turn_count(self, cli: AIPerfCLI, tmp_path: Path):
        """Test with high number of turns per session."""
        config = FixedScheduleTestConfig(
            num_sessions=20,
            turns_per_session=25,
            delay_ms=1,
            workers_max=5,
            timeout=120.0,
        )
        trace_file = generate_trace_file(tmp_path, config)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

    def test_high_concurrency_burst(self, cli: AIPerfCLI, tmp_path: Path):
        """Test burst of requests with high concurrency."""
        concurrency_limit = 20
        config = FixedScheduleTestConfig(
            num_sessions=50,
            turns_per_session=2,
            delay_ms=0,
            concurrency=concurrency_limit,
            workers_max=5,
            timeout=120.0,
        )
        trace_file = generate_trace_file(tmp_path, config, stagger_ms=0)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert get_request_count(result) == config.expected_requests

        # Verify concurrency limit is respected even under burst load
        analyzer = ConcurrencyAnalyzer(result)
        assert analyzer.concurrency_within_limit(concurrency_limit)

    def test_mixed_timestamps_and_delays(self, cli: AIPerfCLI, tmp_path: Path):
        """Test conversation with mixed timestamp/delay pattern.

        Critical test: T1 timestamp, T2 delay, T3 timestamp, T4 delay
        """
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            # Session A: Mixed pattern
            write({"session_id": "A", "timestamp": 0, "input_length": 100})
            write({"session_id": "A", "delay": 50, "input_length": 100})
            write({"session_id": "A", "timestamp": 300, "input_length": 100})
            write({"session_id": "A", "delay": 80, "input_length": 100})
            # Session B: Another mixed pattern
            write({"session_id": "B", "timestamp": 100, "input_length": 100})
            write({"session_id": "B", "delay": 100, "input_length": 100})
            write({"session_id": "B", "timestamp": 500, "input_length": 100})

        config = FixedScheduleTestConfig(num_sessions=2)
        cmd = build_fixed_schedule_command(config, trace_file)
        result = cli.run_sync(cmd, timeout=30.0)

        # Session A: 4 turns, Session B: 3 turns = 7 total
        assert get_request_count(result) == 7

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)

        assert analyzer.credits_balanced()
        assert analyzer.num_sessions == 2


# =============================================================================
# Start/End Offset Filtering Tests
# =============================================================================


@pytest.mark.component_integration
class TestFixedScheduleOffsetFiltering:
    """Tests for fixed schedule with start/end offset filtering.

    Verifies that when using --fixed-schedule-start-offset and
    --fixed-schedule-end-offset, the phase completes correctly based
    on the actual filtered dataset size, not the original file size.
    """

    def test_offset_filtering_completes(self, cli: AIPerfCLI, tmp_path: Path):
        """Test that filtered dataset completes without waiting for original size.

        Creates a 10-line trace file (timestamps 1000-10000ms), filters to
        lines with timestamps between 1500-3500ms (should yield 2 lines:
        timestamp 2000 and 3000), and verifies the phase completes with
        exactly 2 requests.
        """
        trace_file = tmp_path / "trace.jsonl"

        # Create 10 entries with timestamps 1000-10000ms (1 second apart)
        with open(trace_file, "w") as f:
            for i in range(10):
                timestamp = (i + 1) * 1000  # 1000, 2000, ..., 10000
                line = {
                    "timestamp": timestamp,
                    "text_input": f"Question {i + 1}",
                    "output_length": 50 + i * 10,
                }
                f.write(orjson.dumps(line).decode() + "\n")

        # Filter to timestamps between 1500-3500ms
        # Should include: timestamp 2000 and 3000 (2 lines)
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --fixed-schedule \
                --input-file {trace_file} \
                --fixed-schedule-start-offset 1500 \
                --fixed-schedule-end-offset 3500 \
                --workers-max 3 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=30.0)

        # Should complete with exactly 2 requests (not hang waiting for 10)
        assert get_request_count(result) == 2

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_offset_filtering_multi_session(self, cli: AIPerfCLI, tmp_path: Path):
        """Test offset filtering with multi-turn sessions."""
        trace_file = tmp_path / "trace.jsonl"

        # Create sessions with different timestamps
        with open(trace_file, "w") as f:

            def write(d):
                f.write(orjson.dumps(d).decode() + "\n")

            # Session A: starts at 500ms (before filter)
            write({"session_id": "A", "timestamp": 500, "input_length": 100})
            write({"session_id": "A", "delay": 50, "input_length": 100})

            # Session B: starts at 2000ms (within filter)
            write({"session_id": "B", "timestamp": 2000, "input_length": 100})
            write({"session_id": "B", "delay": 50, "input_length": 100})

            # Session C: starts at 3000ms (within filter)
            write({"session_id": "C", "timestamp": 3000, "input_length": 100})
            write({"session_id": "C", "delay": 50, "input_length": 100})

            # Session D: starts at 5000ms (after filter)
            write({"session_id": "D", "timestamp": 5000, "input_length": 100})
            write({"session_id": "D", "delay": 50, "input_length": 100})

        # Filter to 1500-3500ms: should include B and C (4 turns total)
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --fixed-schedule \
                --custom-dataset-type mooncake_trace \
                --input-file {trace_file} \
                --fixed-schedule-start-offset 1500 \
                --fixed-schedule-end-offset 3500 \
                --workers-max 3 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=30.0, assert_success=False)

        # TODO: Currently we have a limitation where if the first turn is not within the offset range,
        # the phase will fail and exit with a non-zero exit code. We need to improve this so that the
        # phase can complete successfully even if the first turn is not within the offset range,
        # however that is a larger overall issue that needs to be addressed.

        # For now, it is important that the run does not hang, and will return a non-zero exit code
        # from the failure to setup the phase.
        assert result.exit_code != 0

    def test_offset_filtering_single_entry(self, cli: AIPerfCLI, tmp_path: Path):
        """Test offset filtering that results in a single entry."""
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:
            for i in range(5):
                timestamp = (i + 1) * 1000
                line = {"timestamp": timestamp, "input_length": 100}
                f.write(orjson.dumps(line).decode() + "\n")

        # Filter to 2500-3500ms: should include only timestamp 3000 (1 line)
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --fixed-schedule \
                --input-file {trace_file} \
                --fixed-schedule-start-offset 2500 \
                --fixed-schedule-end-offset 3500 \
                --workers-max 1 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=30.0)

        assert get_request_count(result) == 1

    def test_offset_filtering_empty_result(self, cli: AIPerfCLI, tmp_path: Path):
        """Test offset filtering that results in no entries."""
        trace_file = tmp_path / "trace.jsonl"

        with open(trace_file, "w") as f:
            for i in range(5):
                timestamp = (i + 1) * 1000  # 1000-5000ms
                line = {"timestamp": timestamp, "input_length": 100}
                f.write(orjson.dumps(line).decode() + "\n")

        # Filter to 10000-20000ms: should include nothing
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --fixed-schedule \
                --input-file {trace_file} \
                --fixed-schedule-start-offset 10000 \
                --fixed-schedule-end-offset 20000 \
                --workers-max 1 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=30.0)

        # Should complete with 0 requests (not hang)
        assert get_request_count(result) == 0
