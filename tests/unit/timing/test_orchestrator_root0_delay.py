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
    return orch


def _credit(**kw) -> MagicMock:
    base = dict(
        turn_index=0, no_request=True, conversation_id="start", x_correlation_id="c0"
    )
    base.update(kw)
    return MagicMock(**base)


def _capture_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    return slept


@pytest.mark.asyncio
async def test_root0_delay_applied_on_turn0_orchestrator_credit(monkeypatch):
    orch = _orch(250.0)
    slept = _capture_sleep(monkeypatch)
    await orch._maybe_apply_root0_think_ms(_credit())
    assert slept == [0.25]  # 250 ms authored -> 0.25 s


@pytest.mark.asyncio
async def test_root0_delay_skipped_for_later_turns_and_real_roots(monkeypatch):
    orch = _orch(250.0)
    slept = _capture_sleep(monkeypatch)
    # A gated (join) turn: its wait is handled by _release_blocked_join, not here.
    await orch._maybe_apply_root0_think_ms(_credit(turn_index=1))
    # A normal (request-producing) root: paced by the strategy, not the spine.
    await orch._maybe_apply_root0_think_ms(_credit(no_request=False))
    assert slept == []
