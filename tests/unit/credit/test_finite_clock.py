# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for finite replay clock synchronization across controller and workers.

Verifies clock synchronization invariants for finite agentic replay:
- PhaseLifecycle shares the controller router's MonotonicClock instance so
  preflight pongs, phase lifecycle start, and credit issuance share an identical
  time domain.
- System wall-clock jumps (e.g. NTP step adjustments) between router bootstrap
  and phase start are ignored by deriving wall timestamps from perf_counter deltas.
- Worker ClockOffsetTracker preflight calibration remains valid upon phase start
  without outlier rejection, preserving zero-early wire scheduling.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.monotonic_clock import MonotonicClock
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker


def _make_phase_config(
    phase: CreditPhase = CreditPhase.PROFILING,
    finite_replay: bool = True,
) -> CreditPhaseConfig:
    """Helper to construct a CreditPhaseConfig for finite replay testing."""
    return CreditPhaseConfig(
        phase=phase,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        expected_duration_sec=60.0,
        finite_replay=finite_replay,
    )


class TestFiniteClockSynchronization:
    """Verifies clock coherence between StickyCreditRouter, PhaseLifecycle, and Workers."""

    def test_preflight_and_phase_share_single_monotonic_clock_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both TimePong.router_sent_wall_ns and Credit.issued_at_ns must share
        the exact same monotonic clock frame.

        Verifies that when a single MonotonicClock is passed to PhaseLifecycle,
        the phase's start wall timestamp and now_ns() reads advance from the
        same anchor established at router initialization.
        """
        router_clock = MonotonicClock(
            perf_anchor_ns=1_000_000,
            wall_anchor_ns=1_700_000_000_000_000_000,
        )
        lifecycle = PhaseLifecycle(config=_make_phase_config(), clock=router_clock)

        # Freeze perf_counter_ns to eliminate live CPU clock jitter between reads
        current_perf = 2_000_000
        monkeypatch.setattr(time, "perf_counter_ns", lambda: current_perf)

        # Before phase start, now_ns() directly reflects router_clock.now_ns()
        assert lifecycle.now_ns() == router_clock.now_ns()

        # Start the phase
        lifecycle.start()
        assert lifecycle.started_at_perf_ns == current_perf
        assert lifecycle.started_at_ns == router_clock.wall_time_for_perf_ns(
            current_perf
        )

        # Subsequent now_ns() calls remain locked to the router clock
        current_perf += 500
        assert lifecycle.now_ns() == router_clock.now_ns()

    @pytest.mark.asyncio
    async def test_forward_wall_clock_step_between_router_init_and_phase_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that a forward NTP step (+100 ms) on the controller host
        between router construction and phase start does not trigger outlier
        rejection or skew converted event timestamps.

        Sequence:
        1. Router starts at T0, captures (perf_0, wall_0).
        2. Worker performs preflight ping/pong and calibrates its ClockOffsetTracker.
        3. Host wall clock steps forward by +100 ms (simulated via time.time_ns).
        4. Phase starts via lifecycle.start() and issues the first credit.
        5. Worker receives credit and evaluates the offset sample.

        Invariants verified:
        - The worker's ClockOffsetTracker does NOT reject the credit sample as an outlier.
        - Calibration remains valid.
        - CreditCallbackHandler._controller_perf_ns converts worker timestamps
          back to exact controller perf_counter time with 0 ns early bias.
        """
        initial_perf = 10_000_000
        initial_wall = 1_700_000_000_000_000_000
        router_clock = MonotonicClock(
            perf_anchor_ns=initial_perf,
            wall_anchor_ns=initial_wall,
        )

        lifecycle = PhaseLifecycle(config=_make_phase_config(), clock=router_clock)

        # Worker setup: worker clock is offset by +500 ms (500_000_000 ns) ahead of controller
        worker_tracker = ClockOffsetTracker(
            logger_name="test_worker",
            min_samples=5,
            finite_preflight=True,
        )
        worker_clock_offset_ns = 500_000_000

        # Step 1: Preflight RTT probe calibration
        current_perf = initial_perf + 5_000_000  # +5 ms elapsed
        for _seq in range(10):
            current_perf += 1_000_000  # +1 ms per probe
            monkeypatch.setattr(time, "perf_counter_ns", lambda p=current_perf: p)
            router_sent_wall = router_clock.now_ns()

            # Worker receives probe with its own local clock reading
            worker_now = router_sent_wall + worker_clock_offset_ns
            worker_tracker.observe(
                issued_at_ns=router_sent_wall, received_at_ns=worker_now
            )

        assert worker_tracker.is_calibrated is True
        assert worker_tracker.correction_ns == worker_clock_offset_ns
        assert worker_tracker.rejected_sample_count == 0

        # Step 2: Controller host wall clock jumps forward by +100 ms (NTP step)
        ntp_jump_ns = 100_000_000  # +100 ms
        monkeypatch.setattr(
            time, "time_ns", lambda: initial_wall + 100_000_000 + ntp_jump_ns
        )

        # Step 3: Phase starts
        current_perf += 20_000_000  # +20 ms later
        monkeypatch.setattr(time, "perf_counter_ns", lambda: current_perf)
        lifecycle.start()

        # started_at_ns must NOT absorb the +100 ms NTP jump
        expected_phase_start_wall = router_clock.wall_time_for_perf_ns(current_perf)
        assert lifecycle.started_at_ns == expected_phase_start_wall

        # Step 4: Controller issues credit in the phase frame
        current_perf += 2_000_000  # +2 ms later
        monkeypatch.setattr(time, "perf_counter_ns", lambda: current_perf)
        credit_issued_at_ns = lifecycle.now_ns()

        # Step 5: Worker receives credit
        worker_receive_wall = credit_issued_at_ns + worker_clock_offset_ns
        updated_offset = worker_tracker.observe(
            issued_at_ns=credit_issued_at_ns, received_at_ns=worker_receive_wall
        )

        # The credit sample must NOT be rejected as an outlier
        assert worker_tracker.rejected_sample_count == 0
        assert updated_offset == worker_clock_offset_ns

        # Step 6: Worker dispatches request and sends TransportDispatched
        event_perf_on_controller = current_perf + 15_000_000  # wire event at +15 ms
        worker_event_wall = (
            router_clock.wall_time_for_perf_ns(event_perf_on_controller)
            + worker_clock_offset_ns
        )

        # Controller converts worker wall timestamp back to controller perf time
        handler_ctx = MagicMock()
        handler_ctx.lifecycle = lifecycle
        converted_perf_ns = CreditCallbackHandler._controller_perf_ns(
            handler=handler_ctx,
            worker_wall_ns=worker_event_wall,
            correction_ns=worker_tracker.correction_ns,
        )

        # Converted perf time must EXACTLY equal the actual controller perf timestamp
        assert converted_perf_ns == event_perf_on_controller
        # Specifically, verify it is NOT 100 ms early!
        assert converted_perf_ns != event_perf_on_controller - ntp_jump_ns

    @pytest.mark.asyncio
    async def test_phase_runner_wires_router_clock_to_lifecycle(self) -> None:
        """PhaseRunner must extract the router's MonotonicClock and inject it
        into PhaseLifecycle so that preflight pongs, credits, and event
        conversion share the same time frame.

        This is an integration-level wiring test: the actual ``PhaseRunner``
        constructor reads ``credit_router.clock`` and threads the instance
        through to ``self._lifecycle._clock``.  A missing or broken handoff
        silently falls back to ``time.time_ns()`` in PhaseLifecycle, which is
        the exact dual-anchor vulnerability AIP-1401 prohibits.
        """
        from aiperf.timing.phase.runner import PhaseRunner

        router_clock = MonotonicClock(
            perf_anchor_ns=1_000_000,
            wall_anchor_ns=1_700_000_000_000_000_000,
        )

        mock_router = MagicMock()
        mock_router.clock = router_clock
        mock_router.send_credit = AsyncMock()
        mock_router.cancel_all_credits = AsyncMock()
        mock_router.wait_for_workers = AsyncMock()
        mock_router.mark_credits_complete = MagicMock()

        mock_conv_src = MagicMock()
        mock_conv_src.dataset_metadata = None

        runner = PhaseRunner(
            config=_make_phase_config(),
            conversation_source=mock_conv_src,
            phase_publisher=MagicMock(),
            credit_router=mock_router,
            concurrency_manager=MagicMock(),
            cancellation_policy=MagicMock(),
            callback_handler=MagicMock(),
        )

        # The lifecycle must hold the exact same MonotonicClock object
        assert runner._lifecycle._clock is router_clock

    @pytest.mark.asyncio
    async def test_backward_wall_clock_step_between_router_init_and_phase_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that a backward NTP step (-100 ms) between router init and
        phase start does not trigger outlier rejection or skew timestamps.

        Mirrors the forward-step test but in the opposite direction: the host
        wall clock jumps backward by 100 ms (e.g. NTP step correction after
        a fast-running local oscillator).  The MonotonicClock-based derivation
        must be immune in both directions.

        Invariants verified:
        - Worker ClockOffsetTracker does NOT reject the credit sample as an outlier.
        - _controller_perf_ns round-trips to exact controller perf time.
        - The converted timestamp is NOT 100 ms late (the backward-step analog
          of the forward-step's 100 ms-early violation).
        """
        initial_perf = 10_000_000
        initial_wall = 1_700_000_000_000_000_000
        router_clock = MonotonicClock(
            perf_anchor_ns=initial_perf,
            wall_anchor_ns=initial_wall,
        )

        lifecycle = PhaseLifecycle(config=_make_phase_config(), clock=router_clock)

        worker_tracker = ClockOffsetTracker(
            logger_name="test_worker_backward",
            min_samples=5,
            finite_preflight=True,
        )
        worker_clock_offset_ns = 500_000_000

        # Preflight calibration
        current_perf = initial_perf + 5_000_000
        for _seq in range(10):
            current_perf += 1_000_000
            monkeypatch.setattr(time, "perf_counter_ns", lambda p=current_perf: p)
            router_sent_wall = router_clock.now_ns()
            worker_now = router_sent_wall + worker_clock_offset_ns
            worker_tracker.observe(
                issued_at_ns=router_sent_wall, received_at_ns=worker_now
            )

        assert worker_tracker.is_calibrated is True
        assert worker_tracker.rejected_sample_count == 0

        # Host wall clock jumps BACKWARD by 100 ms
        ntp_jump_ns = -100_000_000
        monkeypatch.setattr(
            time, "time_ns", lambda: initial_wall + 100_000_000 + ntp_jump_ns
        )

        # Phase starts — lifecycle.start() derives from router_clock, not time.time_ns
        current_perf += 20_000_000
        monkeypatch.setattr(time, "perf_counter_ns", lambda: current_perf)
        lifecycle.start()

        expected_phase_start_wall = router_clock.wall_time_for_perf_ns(current_perf)
        assert lifecycle.started_at_ns == expected_phase_start_wall

        # Issue credit
        current_perf += 2_000_000
        monkeypatch.setattr(time, "perf_counter_ns", lambda: current_perf)
        credit_issued_at_ns = lifecycle.now_ns()

        # Worker receives credit
        worker_receive_wall = credit_issued_at_ns + worker_clock_offset_ns
        updated_offset = worker_tracker.observe(
            issued_at_ns=credit_issued_at_ns, received_at_ns=worker_receive_wall
        )

        assert worker_tracker.rejected_sample_count == 0
        assert updated_offset == worker_clock_offset_ns

        # Worker dispatches and controller converts back
        event_perf_on_controller = current_perf + 15_000_000
        worker_event_wall = (
            router_clock.wall_time_for_perf_ns(event_perf_on_controller)
            + worker_clock_offset_ns
        )

        handler_ctx = MagicMock()
        handler_ctx.lifecycle = lifecycle
        converted_perf_ns = CreditCallbackHandler._controller_perf_ns(
            handler=handler_ctx,
            worker_wall_ns=worker_event_wall,
            correction_ns=worker_tracker.correction_ns,
        )

        assert converted_perf_ns == event_perf_on_controller
        # Must NOT be 100 ms late (backward-step analog of forward-step's early bias)
        assert converted_perf_ns != event_perf_on_controller - ntp_jump_ns

    def test_fallback_path_without_clock_is_vulnerable_to_ntp_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative test: prove that PhaseLifecycle without a shared
        MonotonicClock IS vulnerable to a wall-clock step between construction
        and start().

        When no ``clock`` is injected, ``start()`` captures ``time.time_ns()``
        directly.  If the host wall clock jumps between the router's anchor
        capture and ``start()``, the phase anchor absorbs the jump.  Worker
        timestamps calibrated against the router's pong frame then disagree
        with the phase's ``started_at_ns`` by exactly the NTP step, causing
        ``_controller_perf_ns`` to produce a result that is offset by the jump.

        This test confirms the vulnerability exists in the fallback path,
        validating that the clock-injection fix is structurally necessary.
        """
        initial_perf = 10_000_000
        initial_wall = 1_700_000_000_000_000_000

        router_clock = MonotonicClock(
            perf_anchor_ns=initial_perf,
            wall_anchor_ns=initial_wall,
        )

        # NO clock injected — exercises the fallback path
        lifecycle_no_clock = PhaseLifecycle(config=_make_phase_config(), clock=None)

        ntp_jump_ns = 100_000_000  # +100 ms

        # Advance perf_counter to simulate real elapsed time
        phase_start_perf = initial_perf + 25_000_000
        monkeypatch.setattr(time, "perf_counter_ns", lambda: phase_start_perf)
        # Wall clock has jumped by +100 ms beyond what perf_counter would predict
        monkeypatch.setattr(
            time,
            "time_ns",
            lambda: initial_wall + 25_000_000 + ntp_jump_ns,
        )

        lifecycle_no_clock.start()

        # With clock injected, started_at_ns would equal router_clock's prediction
        expected_if_fixed = router_clock.wall_time_for_perf_ns(phase_start_perf)
        # Without clock, started_at_ns absorbed the NTP jump
        assert lifecycle_no_clock.started_at_ns != expected_if_fixed
        assert lifecycle_no_clock.started_at_ns == expected_if_fixed + ntp_jump_ns

        # _controller_perf_ns using this vulnerable lifecycle produces wrong result
        credit_perf = phase_start_perf + 2_000_000
        monkeypatch.setattr(time, "perf_counter_ns", lambda: credit_perf)

        # Simulate a worker event at a known controller perf time
        event_perf_on_controller = credit_perf + 15_000_000
        worker_clock_offset_ns = 500_000_000
        # Worker wall timestamp derived from the router clock (the correct frame)
        worker_event_wall = (
            router_clock.wall_time_for_perf_ns(event_perf_on_controller)
            + worker_clock_offset_ns
        )

        handler_ctx = MagicMock()
        handler_ctx.lifecycle = lifecycle_no_clock
        converted_perf_ns = CreditCallbackHandler._controller_perf_ns(
            handler=handler_ctx,
            worker_wall_ns=worker_event_wall,
            correction_ns=worker_clock_offset_ns,
        )

        # Without the fix, converted perf time is 100 ms EARLY (the AIP-1401 violation)
        assert converted_perf_ns != event_perf_on_controller
        assert converted_perf_ns == event_perf_on_controller - ntp_jump_ns
