# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: agentic mid-trace resume acquires/releases the session slot balanced.

Session-slot symmetry contract (issuer.py): a session slot is acquired on a
session's first credit in the phase and released on its final turn.

Agentic replay resumes a sampled trajectory at ``turn_index = k_i + 1`` (>= 1)
in PROFILING, so the resumed root's first PROFILING credit has ``turn_index > 0``.
The FIX flags that credit ``is_session_start=True`` (via
``SampledSession.build_turn_at_index``), so ``CreditIssuer.issue_credit``
acquires a session slot for it (``needs_session_slot = is_session_start and not
is_child``). On the root's final turn
``CreditCallbackHandler._release_slots_for_return`` releases it, keeping the
session semaphore balanced.

Before the fix the resume credit acquired NO slot (the gate was
``turn_index == 0``) yet the final-turn release still fired, over-releasing the
(unbounded ``asyncio.Semaphore``) session limiter and admitting sessions above
``--concurrency``. Agentic replay enables the session limiter
(config.py sets ``concurrency=loadgen.concurrency`` for both phases), so the
over-subscription was real.
"""

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
        total_expected_requests=1000,  # large: never the gating factor here
        expected_num_sessions=None,
        expected_duration_sec=None,
        concurrency=_LIMIT,
        prefill_concurrency=None,  # prefill limiting disabled (agentic default)
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
        # Recycled session starts at turn 0 -> acquires a session slot.
        await issuer.issue_credit(_turn(0, corr="recycled"))
        assert _session_effective_slots(cm) == _LIMIT - 1
        # Final turn return releases it (callback_handler path, agent_depth==0).
        cm.release_session_slot(CreditPhase.PROFILING)
        return _session_effective_slots(cm)

    assert asyncio.run(body()) == _LIMIT


def test_mid_trace_root_acquires_and_releases_session_slot_balanced() -> None:
    """A mid-trace resume (turn_index > 0, is_session_start) acquires a session
    slot on its first credit and releases it on its final turn -- balanced, so
    it can never over-release and admit a session above --concurrency.

    The resume credit carries is_session_start=True (set by
    SampledSession.build_turn_at_index); the strategy only emits these at a
    phase's initial dispatch, so the acquisition always finds a free slot.
    """

    async def body() -> tuple[int, int]:
        issuer, cm = _build_issuer_with_real_concurrency()
        before = _session_effective_slots(cm)  # == _LIMIT

        # Resumed trajectory at phase start: first credit is turn 3 (k_i+1) but
        # is a session start -> must acquire a session slot.
        await issuer.issue_credit(_turn(3, corr="resumed-root", session_start=True))
        held = before - _session_effective_slots(cm)

        # Final-turn return releases the slot (callback handler, agent_depth==0).
        cm.release_session_slot(CreditPhase.PROFILING)
        return held, _session_effective_slots(cm)

    held, after_release = asyncio.run(body())
    assert held == 1, "a mid-trace resume must acquire a session slot"
    assert after_release == _LIMIT, "release is balanced; no over-subscription"
