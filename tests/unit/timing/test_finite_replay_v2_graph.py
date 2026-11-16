# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite replay of an explicitly timed, multi-root play."""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    CreditPhase,
    ReplayDependencyEvent,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    ReplayTurnReference,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator


class ManualScheduler:
    def __init__(self) -> None:
        self.scheduled = []

    def schedule_at_perf_ns(self, deadline_ns, coroutine):
        self.scheduled.append((deadline_ns, coroutine))
        return len(self.scheduled)

    def close(self) -> None:
        for _, coroutine in self.scheduled:
            coroutine.close()


@pytest.mark.asyncio
async def test_explicit_play_uses_independent_floors_and_transport_events(monkeypatch):
    root = ConversationMetadata(
        conversation_id="play",
        is_orchestrator=True,
        turns=[TurnMetadata(timestamp_ms=0, no_request=True, branch_ids=["play:0"])],
        branches=[
            ConversationBranchInfo(
                branch_id="play:0",
                child_conversation_ids=["a", "b", "c"],
                mode=ConversationBranchMode.SPAWN,
                is_background=True,
            )
        ],
    )
    children = [
        ConversationMetadata(
            conversation_id=name,
            is_root=False,
            agent_depth=1,
            parent_conversation_id="play",
            turns=[TurnMetadata(timestamp_ms=floor, replay_predecessors=refs)],
        )
        for name, floor, refs in [
            ("a", 0, []),
            ("b", 10.0000001, []),
            (
                "c",
                0,
                [
                    ReplayTurnReference(
                        conversation_id="a",
                        turn_index=0,
                        event=ReplayDependencyEvent.DISPATCH,
                        delay_ns=5_000_000,
                    ),
                    ReplayTurnReference(
                        conversation_id="b",
                        turn_index=0,
                        event=ReplayDependencyEvent.COMPLETION,
                        delay_ns=2_000_000,
                    ),
                ],
            ),
        ]
    ]
    scheduler = ManualScheduler()
    coordinator = ReplayBarrierCoordinator(
        DatasetMetadata(
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            conversations=[root, *children],
        ),
        strict_finite=True,
        scheduler=scheduler,
    )
    coordinator.activate()
    monkeypatch.setattr(
        "aiperf.timing.replay_dependencies.time.perf_counter_ns", lambda: 1_000_000_000
    )

    def credit(name, number, *, virtual=False):
        return Credit(
            id=number,
            phase=CreditPhase.PROFILING,
            conversation_id=name,
            x_correlation_id=name,
            root_correlation_id="play-instance",
            turn_index=0,
            num_turns=1,
            issued_at_ns=0,
            agent_depth=0 if virtual else 1,
            finite_replay=True,
            no_request=virtual,
        )

    coordinator.register_credit(credit("play", 1, virtual=True))
    try:
        for name in ("a", "b", "c"):
            await coordinator.submit(
                TurnToSend(
                    conversation_id=name,
                    x_correlation_id=name,
                    root_correlation_id="play-instance",
                    turn_index=0,
                    num_turns=1,
                    agent_depth=1,
                ),
                AsyncMock(return_value=True),
            )
        assert [deadline for deadline, _ in scheduler.scheduled] == [
            1_000_000_000,
            1_010_000_001,
        ]
        coordinator.record_dispatch(credit("a", 2), 1_002_000_000, 0)
        coordinator.record_dispatch(credit("b", 3), 1_011_000_000, 0)
        coordinator.record_completion(
            credit("b", 3), 1_018_000_000, clock_spread_ns=0, failed=False
        )
        assert scheduler.scheduled[-1][0] == 1_020_000_000
    finally:
        scheduler.close()
