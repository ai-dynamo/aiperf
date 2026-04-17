# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for concurrency schedule ingestion.

Validates the full pipeline end-to-end:
- `schedule.jsonl` file consumed by `--concurrency-schedule-file`
- `ScheduleFollowerStrategy` drives `set_session_limit` at each tick
- `DynamicConcurrencyLimit` ramps up immediately and drains gracefully on
  ramp-down (in-flight requests are never killed)

Tests run aiperf in burst mode against the in-process FakeTransport mock
endpoint with TTFT=5ms / ITL=1ms (see `realistic_latency` fixture). With
OSL=50 every request takes ~55ms, giving enough concurrency pressure to
observe the schedule's effect on dispatch while keeping tests fast.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import ConcurrencyAnalyzer
from tests.harness.utils import AIPerfCLI, AIPerfResults


def _write_schedule(path: Path, ticks: list[tuple[float, int]]) -> Path:
    """Write a schedule.jsonl file with the given (time_sec, concurrency) ticks."""
    path.write_bytes(
        b"\n".join(
            orjson.dumps({"time_sec": float(t), "concurrency": int(c)})
            for t, c in ticks
        )
        + b"\n"
    )
    return path


def _build_schedule_command(
    schedule_path: Path,
    *,
    num_sessions: int,
    concurrency_upper_bound: int,
    osl: int = 50,
) -> str:
    """Build an aiperf burst command bound to a concurrency schedule file."""
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {num_sessions} \
            --concurrency {concurrency_upper_bound} \
            --concurrency-schedule-file {schedule_path} \
            --osl {osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """


def _concurrent_at_time(intervals: list[tuple[int, int]], sample_time_ns: int) -> int:
    """Count how many request intervals are in flight at a given monotonic timestamp."""
    return sum(1 for start, end in intervals if start <= sample_time_ns <= end)


def _phase_start_ns(result: AIPerfResults) -> int:
    """Return the earliest credit issue timestamp as the logical phase start."""
    intervals = ConcurrencyAnalyzer(result).get_request_intervals()
    return min(start for start, _ in intervals)


@pytest.mark.component_integration
class TestConcurrencySchedule:
    """End-to-end coverage: schedule.jsonl drives live concurrency changes."""

    def test_ramp_up_then_down_1_to_2_to_1(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        """A 1 → 2 → 1 schedule must:
        - cap peak concurrency at 2 (never exceed the highest tick)
        - actually reach 2 during the C=2 window (the ramp-up is exercised)
        - stay at 1 during the opening C=1 window
        - drain to 1 after the ramp-down (no requests killed mid-flight).
        """
        schedule_path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [(0.0, 1), (2.0, 2), (4.0, 1), (6.0, 1)],
        )

        cmd = _build_schedule_command(
            schedule_path,
            num_sessions=500,
            concurrency_upper_bound=2,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        analyzer = ConcurrencyAnalyzer(result)
        intervals = analyzer.get_request_intervals()
        assert intervals, "expected at least one request to fire"

        peak = analyzer.get_max_concurrent()
        assert peak <= 2, f"peak concurrency {peak} exceeded schedule cap of 2"
        assert peak == 2, (
            f"schedule's C=2 window should have been exercised; peak was {peak}"
        )

        phase_start = _phase_start_ns(result)
        one_sec = 1_000_000_000

        # Sample the opening C=1 window (the schedule initial limit is 1).
        opening_max = max(
            _concurrent_at_time(intervals, phase_start + int(t_ms * 1_000_000))
            for t_ms in (100, 500, 1000, 1500, 1900)
        )
        assert opening_max <= 1, f"opening C=1 window saw concurrency {opening_max} > 1"

        # Sample inside the C=2 window (tick fires at 2s, leave slack for dispatch
        # latency + request duration so the slot is actually filled).
        mid_max = max(
            _concurrent_at_time(
                intervals, phase_start + 2 * one_sec + int(t_ms * 1_000_000)
            )
            for t_ms in (300, 800, 1200, 1700)
        )
        assert mid_max == 2, f"C=2 window should drive concurrency to 2, saw {mid_max}"

        # Ramp-down: after a grace window the debt has drained and C is 1 again.
        # Sample late in the C=1 closing window (t=4→6); leave slack for the
        # last C=2 request (max ~55ms) to complete.
        closing_max = max(
            _concurrent_at_time(intervals, phase_start + int(t_ms * 1_000_000))
            for t_ms in (4500, 5000, 5500, 5800)
        )
        assert closing_max <= 1, (
            f"closing C=1 window saw concurrency {closing_max} > 1 — "
            "ramp-down should have drained debt by now"
        )

    def test_ramp_down_does_not_kill_in_flight(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        """A steep ramp from C=5 → C=1 must drain gracefully: every credit
        issued while the cap was 5 eventually returns (no in-flight kills)."""
        schedule_path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [(0.0, 5), (2.0, 5), (2.01, 1), (4.0, 1)],
        )

        cmd = _build_schedule_command(
            schedule_path,
            num_sessions=500,
            concurrency_upper_bound=5,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        analyzer = ConcurrencyAnalyzer(result)
        intervals = analyzer.get_request_intervals()
        assert intervals, "expected at least one request to fire"

        # Every issued credit (Credit message) must have a matching CreditReturn,
        # i.e. every start has a valid end. ConcurrencyAnalyzer only builds
        # intervals where both sides match, so the analyzer's interval count
        # tells us how many credits were dispatched AND completed cleanly.
        runner = result.runner_result
        from aiperf.credit.messages import CreditReturn
        from aiperf.credit.structs import Credit

        dispatched = [
            p.payload.id for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]
        returned = {
            p.payload.credit.id
            for p in runner.sent_payloads
            if isinstance(p.payload, CreditReturn)
        }
        leaked = [cid for cid in dispatched if cid not in returned]
        assert not leaked, (
            f"{len(leaked)} credits dispatched during C=5 were never returned "
            "— ramp-down should not kill in-flight requests"
        )

        # The peak must have been ≤ 5 (schedule cap) but ≥ 2 so we actually
        # exercised the steep drop.
        peak = analyzer.get_max_concurrent()
        assert peak <= 5
        assert peak >= 2, (
            f"didn't reach enough concurrency to test a ramp-down (peak={peak})"
        )

    def test_schedule_ends_drive_phase_shutdown(
        self, cli: AIPerfCLI, tmp_path: Path
    ) -> None:
        """When the user doesn't pass --benchmark-duration, the phase should
        stop at the last tick's timestamp. Confirm no credit fires meaningfully
        after the schedule terminates."""
        schedule_end_sec = 3.0
        schedule_path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [(0.0, 2), (schedule_end_sec, 2)],
        )

        cmd = _build_schedule_command(
            schedule_path,
            num_sessions=1000,
            concurrency_upper_bound=2,
        )
        result = cli.run_sync(cmd, timeout=30.0)

        analyzer = ConcurrencyAnalyzer(result)
        intervals = analyzer.get_request_intervals()
        assert intervals

        phase_start = _phase_start_ns(result)
        last_issue_ns = max(start for start, _ in intervals)
        last_issue_relative_sec = (last_issue_ns - phase_start) / 1_000_000_000

        # Grace window: credits issued after schedule end would indicate the
        # phase kept running. 1.0s grace accounts for ramper shutdown latency.
        assert last_issue_relative_sec <= schedule_end_sec + 1.0, (
            f"credit issued {last_issue_relative_sec:.2f}s after start, "
            f"schedule ends at {schedule_end_sec}s"
        )
