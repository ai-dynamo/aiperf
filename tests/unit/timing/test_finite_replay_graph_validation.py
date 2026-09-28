# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pre-flight structural validation of finite replay dependency graphs."""

import pytest

from aiperf.common.enums import ConversationBranchMode, ReplayDependencyEvent
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    ReplayTurnReference,
    TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator


def _coordinator(conversations: list[ConversationMetadata]) -> ReplayBarrierCoordinator:
    md = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=conversations,
    )
    return ReplayBarrierCoordinator(md, strict_finite=True, scheduler=LoopScheduler())


@pytest.mark.asyncio
async def test_duplicate_conversation_ids_raise_error() -> None:
    c1 = ConversationMetadata(
        conversation_id="dup", turns=[TurnMetadata(timestamp_ms=0.0)]
    )
    c2 = ConversationMetadata(
        conversation_id="dup", turns=[TurnMetadata(timestamp_ms=10.0)]
    )
    with pytest.raises(RuntimeError, match="conversation IDs must be unique"):
        _coordinator([c1, c2])


@pytest.mark.asyncio
async def test_missing_parent_conversation_raises_error() -> None:
    c = ConversationMetadata(
        conversation_id="child",
        parent_conversation_id="nonexistent_parent",
        turns=[TurnMetadata(timestamp_ms=10.0)],
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="is missing"):
        _coordinator([c])


@pytest.mark.asyncio
async def test_conversation_parent_cycle_raises_error() -> None:
    c1 = ConversationMetadata(
        conversation_id="c1",
        parent_conversation_id="c2",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        is_root=False,
    )
    c2 = ConversationMetadata(
        conversation_id="c2",
        parent_conversation_id="c1",
        turns=[TurnMetadata(timestamp_ms=10.0)],
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="conversation parent cycle"):
        _coordinator([c1, c2])


@pytest.mark.asyncio
async def test_root_with_no_turns_raises_error() -> None:
    c = ConversationMetadata(conversation_id="root", turns=[], is_root=True)
    with pytest.raises(RuntimeError, match="has no turns"):
        _coordinator([c])


@pytest.mark.asyncio
async def test_root_turn0_missing_timestamp_raises_error() -> None:
    c = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=None)],
        is_root=True,
    )
    with pytest.raises(RuntimeError, match="has no timestamp_ms"):
        _coordinator([c])


@pytest.mark.asyncio
async def test_root_turn0_non_finite_timestamp_raises_error() -> None:
    c = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=float("inf"))],
        is_root=True,
    )
    with pytest.raises(RuntimeError, match="timestamp_ms is not finite"):
        _coordinator([c])


@pytest.mark.asyncio
async def test_pre_session_branch_raises_error() -> None:
    branch = ConversationBranchInfo(
        branch_id="b0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    parent = ConversationMetadata(
        conversation_id="parent",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        branches=[branch],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=10.0)],
        parent_conversation_id="parent",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="does not support pre-session branches"):
        _coordinator([parent, child])


@pytest.mark.asyncio
async def test_spawn_parent_without_request_raises_error() -> None:
    branch = ConversationBranchInfo(
        branch_id="b0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    parent = ConversationMetadata(
        conversation_id="parent",
        turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["b0"], no_request=True)],
        branches=[branch],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=10.0)],
        parent_conversation_id="parent",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="does not issue an HTTP request"):
        _coordinator([parent, child])


@pytest.mark.asyncio
async def test_spawn_branch_crossing_root_traces_raises_error() -> None:
    branch = ConversationBranchInfo(
        branch_id="b0",
        child_conversation_ids=["child_of_other"],
        mode=ConversationBranchMode.SPAWN,
    )
    root1 = ConversationMetadata(
        conversation_id="root1",
        turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["b0"])],
        branches=[branch],
        is_root=True,
    )
    root2 = ConversationMetadata(
        conversation_id="root2",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        is_root=True,
    )
    child_of_other = ConversationMetadata(
        conversation_id="child_of_other",
        turns=[TurnMetadata(timestamp_ms=10.0)],
        parent_conversation_id="root2",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="crosses root traces"):
        _coordinator([root1, root2, child_of_other])


@pytest.mark.asyncio
async def test_spawn_child_starting_before_declaring_turn_raises_error() -> None:
    branch = ConversationBranchInfo(
        branch_id="b0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    parent = ConversationMetadata(
        conversation_id="parent",
        turns=[TurnMetadata(timestamp_ms=100.0, branch_ids=["b0"])],
        branches=[branch],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=50.0)],  # starts before parent turn (100.0)
        parent_conversation_id="parent",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="starts before its declaring turn"):
        _coordinator([parent, child])


@pytest.mark.asyncio
async def test_root_turn0_having_predecessors_raises_error() -> None:
    ref = ReplayTurnReference(
        conversation_id="child", turn_index=0, event=ReplayDependencyEvent.COMPLETION
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0, replay_predecessors=[ref])],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        parent_conversation_id="root",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="turn 0 cannot have predecessors"):
        _coordinator([root, child])


@pytest.mark.asyncio
async def test_dependency_predecessor_starting_after_dependent_raises_error() -> None:
    ref = ReplayTurnReference(
        conversation_id="child", turn_index=0, event=ReplayDependencyEvent.COMPLETION
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=50.0, replay_predecessors=[ref]),
        ],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=60.0)],  # starts at 60.0 > 50.0
        parent_conversation_id="root",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="starts after"):
        _coordinator([root, child])


@pytest.mark.asyncio
async def test_dependency_on_itself_raises_error() -> None:
    ref = ReplayTurnReference(
        conversation_id="root", turn_index=0, event=ReplayDependencyEvent.COMPLETION
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=50.0, replay_predecessors=[ref]),
        ],
        is_root=True,
    )
    ref_self = ReplayTurnReference(
        conversation_id="root", turn_index=1, event=ReplayDependencyEvent.COMPLETION
    )
    root.turns[1].replay_predecessors.append(ref_self)

    with pytest.raises(RuntimeError, match="depends on itself"):
        _coordinator([root])


@pytest.mark.asyncio
async def test_dependency_cycle_raises_error() -> None:
    ref_b = ReplayTurnReference(
        conversation_id="b", turn_index=0, event=ReplayDependencyEvent.COMPLETION
    )
    ref_a = ReplayTurnReference(
        conversation_id="a", turn_index=1, event=ReplayDependencyEvent.COMPLETION
    )
    root = ConversationMetadata(
        conversation_id="a",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=50.0, replay_predecessors=[ref_b]),
        ],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="b",
        turns=[TurnMetadata(timestamp_ms=50.0, replay_predecessors=[ref_a])],
        parent_conversation_id="a",
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="dependency cycle"):
        _coordinator([root, child])


@pytest.mark.asyncio
async def test_root_turn0_no_request_raises_error() -> None:
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0, no_request=True)],
        is_root=True,
    )
    with pytest.raises(RuntimeError, match="turn 0 must issue an HTTP request"):
        _coordinator([root])


@pytest.mark.asyncio
async def test_dependency_referencing_nonexistent_turn_index_raises_error() -> None:
    ref_missing = ReplayTurnReference(
        conversation_id="root", turn_index=5, event=ReplayDependencyEvent.COMPLETION
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=50.0, replay_predecessors=[ref_missing]),
        ],
        is_root=True,
    )
    with pytest.raises(RuntimeError, match="is missing"):
        _coordinator([root])


@pytest.mark.asyncio
async def test_dependency_referencing_no_request_turn_raises_error() -> None:
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        parent_conversation_id="root",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=10.0, no_request=True),
            TurnMetadata(
                timestamp_ms=20.0,
                replay_predecessors=[
                    ReplayTurnReference(
                        conversation_id="child",
                        turn_index=1,
                        event=ReplayDependencyEvent.COMPLETION,
                    )
                ],
            ),
        ],
        is_root=False,
    )
    with pytest.raises(
        RuntimeError, match="references a turn without a transport event"
    ):
        _coordinator([root, child])


@pytest.mark.asyncio
async def test_turn_timestamp_preceding_root_start_raises_error() -> None:
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=50.0)],
        is_root=True,
    )
    child = ConversationMetadata(
        conversation_id="child",
        parent_conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=20.0)],
        is_root=False,
    )
    with pytest.raises(RuntimeError, match="has an invalid root floor"):
        _coordinator([root, child])
