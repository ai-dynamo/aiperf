# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuation send failures must terminate count-only phases with their cause."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import TurnMetadata
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.runner import PhaseRunner
from tests.harness.time_traveler import TimeTraveler
from tests.unit.timing.conftest import MockCreditRouter


@pytest.mark.asyncio
@pytest.mark.looptime
@pytest.mark.parametrize("policy", ["rate_slot", "direct"])
@pytest.mark.parametrize("delay_ms", [None, 0, 25])
@pytest.mark.parametrize("failed_turn", [1, 2])
async def test_continuation_send_failure_reaches_phase_runner(
    time_traveler_no_patch_sleep: TimeTraveler,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    delay_ms: float | None,
    failed_turn: int,
) -> None:
    """A three-turn session fails on turn two instead of waiting forever for three."""
    monkeypatch.setattr(Environment.TIMING, "HIGH_RES_TIMER", False)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        continuation_dispatch=policy,
        total_expected_requests=3,
        expected_num_sessions=1,
        request_rate=100,
        arrival_pattern=ArrivalPattern.CONSTANT,
    )
    source = MagicMock()
    source.dataset_metadata = None
    source.next.return_value.build_first_turn.return_value = TurnToSend(
        conversation_id="root", x_correlation_id="root", turn_index=0, num_turns=3
    )
    source.get_next_turn_metadata.return_value = TurnMetadata(delay_ms=delay_ms)
    concurrency = ConcurrencyManager()
    callbacks = CreditCallbackHandler(concurrency)
    router = MockCreditRouter(auto_return=True)
    router.set_return_callback(callbacks.on_credit_return)
    publisher = MagicMock()
    publisher.publish_phase_start = AsyncMock()
    publisher.publish_phase_sending_complete = AsyncMock()
    publisher.publish_phase_complete = AsyncMock()
    publisher.publish_progress = AsyncMock()
    publisher.publish_phase_baseline_request = AsyncMock()
    publisher.server_metrics_runtime_active = False
    error = RuntimeError("injected continuation router send failure")
    send_credit = router.send_credit
    attempted_turns: list[int] = []

    async def fail_continuation(credit: Credit) -> None:
        attempted_turns.append(credit.turn_index)
        if credit.turn_index == failed_turn:
            raise error
        await send_credit(credit)

    monkeypatch.setattr(router, "send_credit", fail_continuation)
    runner = PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=publisher,
        credit_router=router,
        concurrency_manager=concurrency,
        cancellation_policy=None,
        callback_handler=callbacks,
    )
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await asyncio.wait_for(runner.run(is_final_phase=True), timeout=1.0)
        assert exc_info.value is error
        assert attempted_turns == list(range(failed_turn + 1))
        assert runner._lifecycle.was_cancelled
        assert runner._lifecycle.is_complete
        publisher.publish_phase_complete.assert_awaited_once()
    finally:
        runner.cancel()
        if router._pending:
            await asyncio.gather(*router._pending)
