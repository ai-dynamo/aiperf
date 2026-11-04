# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root session-arrival dispatch with real credit, stop, and concurrency gates."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import TurnMetadata
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig, RequestCancellationConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.request_cancellation import RequestCancellationSimulator
from aiperf.timing.strategies.request_rate import RequestRateStrategy
from tests.harness.time_traveler import TimeTraveler


class SessionArrivalRig:
    """Wire the production issuer and return handler around an in-memory router."""

    def __init__(
        self,
        *,
        delay_ms: float | None = None,
        requests: int | None = 10,
        sessions: int | None = None,
        pattern: ArrivalPattern = ArrivalPattern.CONSTANT,
        policy: Literal["rate_slot", "direct"] = "direct",
    ) -> None:
        self.config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=1.0,
            arrival_pattern=pattern,
            continuation_dispatch=policy,
            total_expected_requests=requests,
            expected_num_sessions=sessions,
            concurrency=2,
            prefill_concurrency=2,
        )
        self.scheduler = LoopScheduler()
        self.lifecycle = PhaseLifecycle(self.config)
        self.lifecycle.start()
        self.progress = PhaseProgressTracker(self.config)
        self.stop_checker = StopConditionChecker(
            config=self.config, lifecycle=self.lifecycle, counter=self.progress.counter
        )
        self.concurrency = ConcurrencyManager()
        self.concurrency.configure_for_phase(CreditPhase.PROFILING, 2, 2)
        self.sent: list[Credit] = []
        self.outbox: asyncio.Queue[Credit] = asyncio.Queue()
        self.auto_return = False
        self.source = MagicMock()
        self.source.get_next_turn_metadata.return_value = TurnMetadata(
            delay_ms=delay_ms
        )
        self.source.next.return_value.build_first_turn.side_effect = lambda: self.turn(
            f"session-{len(self.sent)}", num_turns=3
        )
        router = MagicMock()
        router.send_credit = self.send_credit
        self.issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=self.stop_checker,
            progress=self.progress,
            concurrency_manager=self.concurrency,
            credit_router=router,
            cancellation_policy=RequestCancellationSimulator(
                RequestCancellationConfig()
            ),
            lifecycle=self.lifecycle,
        )
        self.strategy = RequestRateStrategy(
            config=self.config,
            conversation_source=self.source,
            scheduler=self.scheduler,
            stop_checker=self.stop_checker,
            credit_issuer=self.issuer,
            lifecycle=self.lifecycle,
        )
        self.callbacks = CreditCallbackHandler(self.concurrency)
        self.callbacks.register_phase(
            phase=CreditPhase.PROFILING,
            progress=self.progress,
            lifecycle=self.lifecycle,
            stop_checker=self.stop_checker,
            strategy=self.strategy,
        )

    @staticmethod
    def turn(name: str, *, num_turns: int = 2) -> TurnToSend:
        """Build a root whose remaining turns inherit the same session slot."""
        return TurnToSend(
            conversation_id=name,
            x_correlation_id=name,
            turn_index=0,
            num_turns=num_turns,
        )

    async def send_credit(self, credit: Credit) -> None:
        """Capture wire credits and optionally simulate an immediate worker return."""
        self.sent.append(credit)
        self.outbox.put_nowait(credit)
        if self.auto_return:
            self.scheduler.schedule_later(0, self.return_credit(credit))

    async def return_credit(
        self, credit: Credit, *, first_token_sent: bool = False
    ) -> None:
        """Deliver a real return through counting, slot release, and scheduling."""
        await self.callbacks.on_credit_return(
            "worker", CreditReturn(credit=credit, first_token_sent=first_token_sent)
        )

    async def next_credit(self) -> Credit:
        """Wait for a send with a bounded virtual-time failure deadline."""
        return await asyncio.wait_for(self.outbox.get(), timeout=5)


@pytest.fixture
async def session_arrival_rig(
    time_traveler_no_patch_sleep: TimeTraveler,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Callable[..., SessionArrivalRig], None]:
    """Use existing virtual clocks and cancel only work owned by each rig."""
    monkeypatch.setattr(Environment.TIMING, "HIGH_RES_TIMER", False)
    rigs: list[SessionArrivalRig] = []

    def create(**kwargs: Any) -> SessionArrivalRig:
        rig = SessionArrivalRig(**kwargs)
        rigs.append(rig)
        return rig

    yield create
    for rig in rigs:
        tasks = rig.scheduler.cancel_all()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delay_ms",
    [param(None, id="no-delay"), param(0, id="zero-delay"), param(500, id="think-time")],
)  # fmt: skip
async def test_direct_continuation_honors_think_time_and_retains_session_slot(
    session_arrival_rig: Callable[..., SessionArrivalRig], delay_ms: float | None
) -> None:
    rig = session_arrival_rig(delay_ms=delay_ms, requests=2)
    assert await rig.issuer.issue_credit(rig.turn("root"))
    root = await rig.next_credit()
    returned_at_ns = time.time_ns()
    await rig.return_credit(root)
    continuation = await rig.next_credit()

    assert continuation.turn_index == 1
    assert continuation.x_correlation_id == root.x_correlation_id
    assert continuation.issued_at_ns - returned_at_ns >= (delay_ms or 0) * 1_000_000
    assert rig.progress.all_credits_sent_event.is_set()
    stats = rig.concurrency.get_session_stats(CreditPhase.PROFILING)
    assert stats.acquire_count == 1
    assert stats.release_count == 0
    await rig.return_credit(continuation)
    assert stats.release_count == 1
    assert rig.progress.check_all_returned_or_cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_ms", [param(None, id="none"), param(0, id="zero")])  # fmt: skip
async def test_direct_continuation_prefill_wait_does_not_block_return_handler(
    session_arrival_rig: Callable[..., SessionArrivalRig], delay_ms: float | None
) -> None:
    rig = session_arrival_rig(delay_ms=delay_ms)
    rig.concurrency.configure_for_phase(CreditPhase.PROFILING, 2, 1)
    assert await rig.issuer.issue_credit(rig.turn("first"))
    first = await rig.next_credit()
    await rig.callbacks.on_first_token(
        FirstToken(credit_id=first.id, phase=first.phase, ttft_ns=1)
    )
    assert await rig.issuer.issue_credit(rig.turn("second"))
    second = await rig.next_credit()

    await asyncio.wait_for(
        rig.return_credit(first, first_token_sent=True), timeout=0.01
    )
    await asyncio.sleep(0)
    assert len(rig.sent) == 2
    await rig.callbacks.on_first_token(
        FirstToken(credit_id=second.id, phase=second.phase, ttft_ns=1)
    )
    continuation = await rig.next_credit()
    assert continuation.x_correlation_id == "first"
    assert rig.concurrency.get_session_stats().acquire_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop",
    [param("cancel", id="cancelled"), param("sending", id="sending-complete")],
)  # fmt: skip
async def test_direct_delayed_continuation_rechecks_phase_stop_before_send(
    session_arrival_rig: Callable[..., SessionArrivalRig], stop: str
) -> None:
    rig = session_arrival_rig(delay_ms=500)
    assert await rig.issuer.issue_credit(rig.turn("root"))
    await rig.return_credit(await rig.next_credit())
    if stop == "cancel":
        rig.lifecycle.cancel()
    else:
        rig.lifecycle.mark_sending_complete()
    await asyncio.sleep(0.6)
    assert len(rig.sent) == 1
    stats = rig.concurrency.get_session_stats()
    assert stats.acquire_count == 1
    assert stats.release_count == 0
    prefill_stats = rig.concurrency._prefill_limiter.global_stats
    assert prefill_stats.acquire_count == 2
    assert prefill_stats.release_count == 2


@pytest.mark.asyncio
async def test_direct_simultaneous_continuations_respect_exact_request_budget(
    session_arrival_rig: Callable[..., SessionArrivalRig],
) -> None:
    rig = session_arrival_rig(requests=3)
    for name in ("first", "second"):
        assert await rig.issuer.issue_credit(rig.turn(name))
    roots = [await rig.next_credit(), await rig.next_credit()]
    for root in roots:
        await rig.return_credit(root)
    final = await rig.next_credit()
    await asyncio.sleep(0)

    assert len(rig.sent) == 3
    assert final.turn_index == 1
    assert rig.progress.all_credits_sent_event.is_set()
    stats = rig.concurrency.get_session_stats()
    assert stats.acquire_count == 2
    assert stats.release_count == 0
    await rig.return_credit(final)
    assert stats.release_count == 2
    assert rig.progress.check_all_returned_or_cancelled()


@pytest.mark.asyncio
async def test_direct_pending_think_time_is_cancelled_at_sending_cutoff(
    session_arrival_rig: Callable[..., SessionArrivalRig],
) -> None:
    rig = session_arrival_rig(delay_ms=500)
    assert await rig.issuer.issue_credit(rig.turn("root"))
    await rig.return_credit(await rig.next_credit())
    rig.lifecycle.mark_sending_complete()
    rig.scheduler.cancel_all_pending()
    await asyncio.sleep(0.6)
    assert len(rig.sent) == 1
    assert rig.concurrency._prefill_limiter.global_stats.acquire_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [param(ArrivalPattern.CONSTANT, id="constant"), param(ArrivalPattern.POISSON, id="poisson")],
)  # fmt: skip
async def test_direct_rate_slots_start_sessions_and_session_limit_drains_all_turns(
    session_arrival_rig: Callable[..., SessionArrivalRig],
    pattern: ArrivalPattern,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = session_arrival_rig(requests=None, sessions=3, pattern=pattern)
    rig.auto_return = True
    sent_before_tick: list[int] = []

    async def wait_for_virtual_tick(_target: float, _now: float) -> bool:
        # The causal slot assertion does not depend on fractional wall-clock waits.
        sent_before_tick.append(len(rig.sent))
        await asyncio.sleep(1)
        return False

    monkeypatch.setattr(rig.strategy, "_wait_until_next_target", wait_for_virtual_tick)
    await asyncio.wait_for(rig.strategy.execute_phase(), timeout=15)
    await asyncio.sleep(0)

    assert [credit.turn_index for credit in rig.sent] == [0, 1, 2] * 3
    assert (
        len(sent_before_tick) == 4
    )  # Three admission ticks and the terminal stop check.
    assert len({credit.x_correlation_id for credit in rig.sent}) == 3
    assert rig.progress.all_credits_sent_event.is_set()
    assert rig.progress.check_all_returned_or_cancelled()
    assert rig.strategy._continuation_turns.empty()
    stats = rig.concurrency.get_session_stats()
    assert stats.acquire_count == stats.release_count == 3


@pytest.mark.asyncio
async def test_direct_policy_preserves_child_dispatch_and_refusal_cleanup(
    session_arrival_rig: Callable[..., SessionArrivalRig],
) -> None:
    rig = session_arrival_rig()
    orchestrator = MagicMock()
    orchestrator.on_child_stopped = AsyncMock()
    rig.strategy._branch_orchestrator = orchestrator
    rig.lifecycle.cancel()
    child = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="child",
        x_correlation_id="child",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        agent_depth=1,
        parent_correlation_id="parent",
    )
    await rig.strategy.handle_credit_return(child)
    orchestrator.on_child_stopped.assert_awaited_once_with("child")
    assert rig.sent == []
    assert rig.concurrency.get_session_stats().acquire_count == 0
