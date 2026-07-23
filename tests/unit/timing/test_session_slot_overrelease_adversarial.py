# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: agentic mid-trace resume acquires/releases the session slot balanced."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker

_LIMIT = 2


def _profiling_config() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=1000,
        expected_num_sessions=None,
        expected_duration_sec=None,
        concurrency=_LIMIT,
        prefill_concurrency=None,
    )


def _build_issuer_with_real_concurrency() -> tuple[CreditIssuer, ConcurrencyManager]:
    config = _profiling_config()
    cm = ConcurrencyManager()
    cm.configure_for_phase(
        CreditPhase.PROFILING, config.concurrency, config.prefill_concurrency
    )

    lifecycle = PhaseLifecycle(config)
    lifecycle.start()
    progress = PhaseProgressTracker(config)
    stop_checker = StopConditionChecker(
        config=config, lifecycle=lifecycle, counter=progress.counter
    )

    cancellation = MagicMock()
    cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)
    router = MagicMock()
    router.send_credit = AsyncMock()

    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=cm,
        credit_router=router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
    )
    return issuer, cm


def _session_effective_slots(cm: ConcurrencyManager) -> int:
    return cm._session_limiter._phase_limits[CreditPhase.PROFILING].effective_slots


def _turn(
    turn_index: int, *, corr: str, num: int = 4, session_start: bool = False
) -> TurnToSend:
    return TurnToSend(
        conversation_id="trace",
        x_correlation_id=corr,
        turn_index=turn_index,
        num_turns=num,
        is_session_start=session_start,
    )


def test_recycled_session_started_at_turn_zero_is_slot_balanced() -> None:
    """Baseline: a recycled session (turn 0) acquires then releases one slot."""

    async def body() -> int:
        issuer, cm = _build_issuer_with_real_concurrency()
        assert _session_effective_slots(cm) == _LIMIT
        await issuer.issue_credit(_turn(0, corr="recycled"))
        assert _session_effective_slots(cm) == _LIMIT - 1
        cm.release_session_slot(CreditPhase.PROFILING)
        return _session_effective_slots(cm)

    assert asyncio.run(body()) == _LIMIT


def test_mid_trace_root_acquires_and_releases_session_slot_balanced() -> None:
    """A mid-trace resume (turn_index > 0, is_session_start) acquires a session"""

    async def body() -> tuple[int, int]:
        issuer, cm = _build_issuer_with_real_concurrency()
        before = _session_effective_slots(cm)

        await issuer.issue_credit(_turn(3, corr="resumed-root", session_start=True))
        held = before - _session_effective_slots(cm)

        cm.release_session_slot(CreditPhase.PROFILING)
        return held, _session_effective_slots(cm)

    held, after_release = asyncio.run(body())
    assert held == 1, "a mid-trace resume must acquire a session slot"
    assert after_release == _LIMIT, "release is balanced; no over-subscription"


def test_lane_credit_acquires_and_releases_one_session_slot_balanced() -> None:
    """A rootless/gated lane holds its session slot via the lane-credit path --"""

    async def body() -> tuple[int, int, int]:
        issuer, cm = _build_issuer_with_real_concurrency()
        before = _session_effective_slots(cm)
        acquired = await issuer.acquire_lane_credit("lane-root", root_pending=False)
        held = before - _session_effective_slots(cm)
        issuer.release_lane_credit()
        return int(bool(acquired)), held, _session_effective_slots(cm)

    acquired, held, after_release = asyncio.run(body())
    assert acquired == 1, "lane credit acquisition must succeed when a slot is free"
    assert held == 1, "a lane credit must hold exactly one session slot"
    assert after_release == _LIMIT, "release is balanced; no over-subscription"


def test_gated_parent_lane_accounts_session_via_tracker() -> None:
    """A gated parent lane (root_pending=True, session_turns>0) routes session"""

    async def body() -> tuple[int, int, int]:
        issuer, _cm = _build_issuer_with_real_concurrency()
        counter = issuer._progress.counter
        acquired = await issuer.acquire_lane_credit(
            "gated-root", root_pending=True, session_turns=2
        )
        return (
            int(bool(acquired)),
            counter.sent_sessions,
            counter.total_session_turns,
        )

    acquired, sent_sessions, total_turns = asyncio.run(body())
    assert acquired == 1, "gated-parent lane acquisition must succeed"
    assert sent_sessions == 1, "the gated parent's session is counted as sent"
    assert total_turns == 2, "the gated parent's remaining turns are accounted"


def test_gated_parent_lane_keeps_in_flight_sessions_non_negative() -> None:
    """End-to-end through the issuer: a gated parent lane that accounts its"""

    async def body() -> int:
        issuer, _cm = _build_issuer_with_real_concurrency()
        counter = issuer._progress.counter
        await issuer.acquire_lane_credit(
            "gated-root", root_pending=True, session_turns=2
        )
        counter.increment_returned(is_final_turn=True, cancelled=False)
        return counter.in_flight_sessions

    assert asyncio.run(body()) == 0


def test_lane_credit_counts_against_the_session_concurrency_limit() -> None:
    """Lane credits draw from the same budget as root credits: with LIMIT=2,"""

    async def body() -> tuple[bool, bool, int]:
        issuer, cm = _build_issuer_with_real_concurrency()
        first = await issuer.acquire_lane_credit("lane-root-a", root_pending=False)
        second = await issuer.acquire_lane_credit("lane-root-b", root_pending=False)
        return bool(first), bool(second), _session_effective_slots(cm)

    first, second, slots = asyncio.run(body())
    assert first and second, "both lane credits acquire within the limit"
    assert slots == 0, "two lane credits exhaust a LIMIT=2 session budget"
