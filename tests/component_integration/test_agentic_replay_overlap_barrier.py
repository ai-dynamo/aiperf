# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import DatasetMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator
from tests.unit.dataset.loader.test_weka_overlap_groups import _load

pytestmark = pytest.mark.component_integration


@pytest.mark.asyncio
async def test_loaded_abc_group_joins_before_d() -> None:
    conversations = _load()
    metadata = DatasetMetadata(
        conversations=[conversation.to_metadata() for conversation in conversations],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    coordinator = ReplayBarrierCoordinator(metadata)
    coordinator.activate()
    root = next(
        conversation for conversation in metadata.conversations if conversation.is_root
    )
    children = [
        conversation
        for conversation in metadata.conversations
        if not conversation.is_root
    ]
    issued: list[tuple[str, int]] = []

    async def submit(conversation_id: str, turn_index: int) -> None:
        turn = TurnToSend(
            conversation_id=conversation_id,
            x_correlation_id=f"corr:{conversation_id}",
            turn_index=turn_index,
            num_turns=len(
                next(
                    c
                    for c in metadata.conversations
                    if c.conversation_id == conversation_id
                ).turns
            ),
            root_correlation_id="root",
        )

        async def issue() -> bool:
            issued.append((conversation_id, turn_index))
            return True

        await coordinator.submit(turn, issue)

    await submit(root.conversation_id, 0)
    for child in children:
        await submit(child.conversation_id, 0)
    await submit(root.conversation_id, 1)
    assert issued == [
        (root.conversation_id, 0),
        *[(child.conversation_id, 0) for child in children],
    ]

    coordinator.complete(_credit(root.conversation_id, 0, len(root.turns), "corr:root"))
    await asyncio.sleep(0)
    assert (root.conversation_id, 1) not in issued
    coordinator.complete(_credit(children[0].conversation_id, 0, 1, "corr:child-0"))
    await asyncio.sleep(0)
    assert (root.conversation_id, 1) not in issued
    coordinator.complete(_credit(children[1].conversation_id, 0, 1, "corr:child-1"))
    await asyncio.sleep(0)
    assert issued[-1] == (root.conversation_id, 1)


def _credit(
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    correlation_id: str,
) -> Credit:
    return Credit(
        id=turn_index,
        phase=CreditPhase.PROFILING,
        conversation_id=conversation_id,
        x_correlation_id=correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        root_correlation_id="root",
    )
