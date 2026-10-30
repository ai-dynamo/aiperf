# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-turn root and subagent trace progression in finite replay."""

import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource


def _make_multiturn_metadata() -> DatasetMetadata:
    branch = ConversationBranchInfo(
        branch_id="root-0:b0",
        child_conversation_ids=["child-0"],
        mode=ConversationBranchMode.SPAWN,
        start_timestamp_ms=10.0,
    )
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="root-0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0, branch_ids=[branch.branch_id]),
                    TurnMetadata(timestamp_ms=100.0),
                    TurnMetadata(timestamp_ms=200.0),
                ],
                branches=[branch],
                is_root=True,
            ),
            ConversationMetadata(
                conversation_id="child-0",
                turns=[
                    TurnMetadata(timestamp_ms=10.0),
                    TurnMetadata(timestamp_ms=60.0),
                ],
                parent_conversation_id="root-0",
                agent_depth=1,
                is_root=False,
            ),
            ConversationMetadata(
                conversation_id="root-1",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=50.0),
                ],
                is_root=True,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_finite_multiturn_root_and_subagent_dispatch_progression() -> None:
    """Multi-turn roots and child continuations advance through their turns and drain."""
    metadata = _make_multiturn_metadata()
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(["root-0", "root-1"]),
        concurrency=1,
        random_seed=42,
        start_min_ratio=0,
        start_max_ratio=0,
        finite_replay=True,
    )
    config = MagicMock(
        phase=CreditPhase.PROFILING,
        concurrency=1,
        finite_replay=True,
        agentic_cache_warmup_duration_sec=None,
    )
    scheduler = MagicMock(pending_count=0, running_count=0)

    scheduled: list[asyncio.Task[Any]] = []

    def execute(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        scheduled.append(task)
        return task

    scheduler.execute_async.side_effect = execute

    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    issuer.dispatch_child_turn = AsyncMock(return_value=True)
    issuer.replay_gate.has_pending_finite_work.return_value = False

    branch_orchestrator = MagicMock()
    branch_orchestrator.has_pending_branch_work.return_value = False

    concurrency_manager = MagicMock()
    registry = SessionTreeRegistry(concurrency_manager)
    progress = MagicMock(in_flight=0)
    progress.all_credits_sent_event = asyncio.Event()

    strategy = AgenticReplayStrategy(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=branch_orchestrator,
        session_tree_registry=registry,
        progress=progress,
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Initial admission: only root-0 turn 0 is dispatched
    assert issuer.issue_credit.await_count == 1
    call = issuer.issue_credit.await_args_list[0]
    turn: TurnToSend = call.args[0]
    assert turn.conversation_id == "root-0" and turn.turn_index == 0

    root0_corr = turn.effective_root_correlation_id
    registry.open_tree(root0_corr, CreditPhase.PROFILING, root_pending=True)

    # Dispatch next turn for root (turn 1)
    credit_root0_t0 = Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id="root-0",
        x_correlation_id="corr-root0",
        root_correlation_id=root0_corr,
        issued_at_ns=1,
        turn_index=0,
        num_turns=3,
        agent_depth=0,
        finite_replay=True,
    )
    await strategy._dispatch_next_turn(credit_root0_t0)
    assert issuer.issue_credit.await_count == 2
    assert issuer.issue_credit.await_args_list[1].args[0].turn_index == 1

    # Dispatch next turn for subagent (turn 1) -> must call dispatch_child_turn
    credit_child_t0 = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="child-0",
        x_correlation_id="corr-child0",
        root_correlation_id=root0_corr,
        issued_at_ns=1,
        turn_index=0,
        num_turns=2,
        agent_depth=1,
        finite_replay=True,
    )
    await strategy._dispatch_next_turn(credit_child_t0)
    assert issuer.dispatch_child_turn.await_count == 1
    assert issuer.dispatch_child_turn.await_args_list[0].args[0].turn_index == 1

    # Close tree and verify root-1 is admitted
    registry.on_root_terminal(root0_corr)
    await asyncio.gather(*scheduled)

    admitted_roots = [
        call.args[0].conversation_id
        for call in issuer.issue_credit.await_args_list
        if call.args[0].turn_index == 0
    ]
    assert admitted_roots == ["root-0", "root-1"]


@pytest.mark.asyncio
async def test_concurrency_exceeding_trace_pool_drains_cleanly() -> None:
    """When concurrency > trace count, lanes cap at pool size and finish without hanging."""
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="only-root",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                is_root=True,
            )
        ],
    )
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(["only-root"]),
        concurrency=10,  # 10 lanes for 1 trace
        random_seed=42,
        finite_replay=True,
    )
    config = MagicMock(
        phase=CreditPhase.PROFILING,
        concurrency=10,
        finite_replay=True,
        agentic_cache_warmup_duration_sec=None,
    )
    scheduler = LoopScheduler()
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    issuer.replay_gate.has_pending_finite_work.return_value = False

    progress = MagicMock(in_flight=0)
    progress.all_credits_sent_event = asyncio.Event()

    registry = SessionTreeRegistry(MagicMock())
    strategy = AgenticReplayStrategy(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        session_tree_registry=registry,
        progress=progress,
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Only 1 root dispatched, not 10
    assert issuer.issue_credit.await_count == 1
    assert not strategy._finite_roots

    root = issuer.issue_credit.await_args.args[0].effective_root_correlation_id
    registry.open_tree(root, CreditPhase.PROFILING, root_pending=True)
    registry.register_descendants(root)
    assert not registry.on_root_terminal(root)
    assert not progress.all_credits_sent_event.is_set()

    assert registry.on_descendant_done(root)
    await asyncio.wait_for(progress.all_credits_sent_event.wait(), timeout=1)
    assert issuer.issue_credit.await_count == 1
    assert not strategy._finite_active_lanes
