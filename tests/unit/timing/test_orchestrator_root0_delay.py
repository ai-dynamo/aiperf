# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator root's turn-0 think-time (root0 pre-delay) must be applied
before round 0's branches fire.

Between-round waits ride the gated join, but round 0 has no join, so its authored
delay would otherwise be dropped (observed: 250 ms authored, ~2.5 ms applied).
``_maybe_apply_root0_think_ms`` closes that gap on the turn-0 return.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from aiperf.common.models import ConversationMetadata, TurnMetadata
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _orch(delay_ms: float) -> BranchOrchestrator:
    orch = BranchOrchestrator.__new__(BranchOrchestrator)
    orch._think_time_by_conv = {}  # fixed think-time (no sampled distribution)
    cs = MagicMock()
    cs.get_metadata.return_value = ConversationMetadata(
        conversation_id="start",
        turns=[TurnMetadata(timestamp_ms=0.0, no_request=True, delay_ms=delay_ms)],
        agent_depth=0,
        is_orchestrator=True,
    )
    cs.sample_ordinal.return_value = 0
    orch._cs = cs
    orch._issuer = MagicMock()
    return orch


def _credit(**kw) -> MagicMock:
    base = dict(
        turn_index=0, no_request=True, conversation_id="start", x_correlation_id="c0"
    )
    base.update(kw)
    return MagicMock(**base)


def _capture_think(orch: BranchOrchestrator) -> list[float]:
    """Capture the seconds passed to the (interruptible) think-time sleep."""
    slept: list[float] = []

    async def _fake(seconds: float, _still_sendable: object) -> None:
        slept.append(seconds)

    orch._sleep_think_ms = _fake
    return slept


@pytest.mark.asyncio
async def test_root0_delay_applied_on_turn0_orchestrator_credit():
    orch = _orch(250.0)
    slept = _capture_think(orch)
    await orch._maybe_apply_root0_think_ms(_credit())
    assert slept == [0.25]  # 250 ms authored -> 0.25 s


def test_sample_think_ms_rejects_nonfinite_draw(monkeypatch):
    """A large median (Cristian uses 31 s) times ``exp(700)`` overflows to inf;
    the draw must never reach asyncio.sleep -- it falls back to the finite
    median."""
    import math

    from aiperf.common.models.dataset_models import ThinkTimeSpec
    from aiperf.timing import branch_orchestrator as bo

    orch = BranchOrchestrator.__new__(BranchOrchestrator)
    orch._think_time_by_conv = {"start": ThinkTimeSpec(sigma=1.0)}
    cs = MagicMock()
    cs.sample_ordinal.return_value = 0
    orch._cs = cs
    stream = MagicMock()
    stream.normal.return_value = 700.0  # force exp(700) -> overflow with big median
    monkeypatch.setattr(bo._rng, "derive", lambda key: stream)

    result = orch._sample_think_ms("start", "c0", 0, 1e6)  # 1e6 * exp(700) == inf
    assert math.isfinite(result)
    assert result == 1e6  # fell back to the finite median


@pytest.mark.asyncio
async def test_root0_delay_skipped_for_later_turns_and_real_roots():
    orch = _orch(250.0)
    slept = _capture_think(orch)
    # A gated (join) turn: its wait is handled by _release_blocked_join, not here.
    await orch._maybe_apply_root0_think_ms(_credit(turn_index=1))
    # A normal (request-producing) root: paced by the strategy, not the spine.
    await orch._maybe_apply_root0_think_ms(_credit(no_request=False))
    assert slept == []


def _bare_sleeper() -> BranchOrchestrator:
    orch = BranchOrchestrator.__new__(BranchOrchestrator)
    orch._cleaning_up = False
    orch._think_time_wake = asyncio.Event()
    return orch


@pytest.mark.asyncio
async def test_sleep_think_ms_interrupted_by_cleanup():
    """A pending think-time sleep must return early once cleanup fires, so
    shutdown doesn't wait out a full (possibly large) interval."""
    orch = _bare_sleeper()
    task = asyncio.create_task(orch._sleep_think_ms(1000.0, lambda: True))
    await asyncio.sleep(0)
    orch._cleaning_up = True
    orch._wake_think_time_sleepers()
    # A 1000s think-time must return promptly; wrap in wait_for so a hang fails.
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_sleep_think_ms_elapses_when_not_cleaned_up():
    """Without cleanup, the sleep runs its full (here tiny) interval normally."""
    orch = _bare_sleeper()
    await orch._sleep_think_ms(0.001, lambda: True)  # timeout elapses -> returns


@pytest.mark.asyncio
async def test_sleep_think_ms_skipped_when_turn_would_be_refused():
    orch = _bare_sleeper()
    await asyncio.wait_for(orch._sleep_think_ms(1000.0, lambda: False), timeout=1.0)


@pytest.mark.asyncio
async def test_sleep_think_ms_woken_but_still_sendable_keeps_think_time():
    """A wake at the sending cutoff must not shorten the think-time of a turn
    that will still be sent (nested spine after --num-conversations)."""
    orch = _bare_sleeper()
    loop = asyncio.get_running_loop()
    start = loop.time()
    task = asyncio.create_task(orch._sleep_think_ms(0.2, lambda: True))
    await asyncio.sleep(0)
    orch._wake_think_time_sleepers()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done()
    await asyncio.wait_for(task, timeout=1.0)
    assert loop.time() - start >= 0.19


@pytest.mark.asyncio
async def test_sleep_think_ms_woken_and_refused_returns_at_once():
    orch = _bare_sleeper()
    sendable = True
    task = asyncio.create_task(orch._sleep_think_ms(1000.0, lambda: sendable))
    await asyncio.sleep(0)
    sendable = False
    orch._wake_think_time_sleepers()
    await asyncio.wait_for(task, timeout=1.0)
