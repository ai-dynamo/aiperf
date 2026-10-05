# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite root admission ends only after every admitted tree drains."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource


@pytest.mark.asyncio
async def test_roots_are_admitted_once_and_background_child_holds_completion() -> None:
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id=f"root-{index}",
                turns=[TurnMetadata(timestamp_ms=0)],
            )
            for index in range(3)
        ],
    )
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(
            [conversation.conversation_id for conversation in metadata.conversations]
        ),
        concurrency=2,
        random_seed=42,
        start_min_ratio=0,
        start_max_ratio=0,
        finite_replay=True,
    )
    config = MagicMock(
        phase=CreditPhase.PROFILING,
        concurrency=2,
        finite_replay=True,
        agentic_cache_warmup_duration_sec=None,
    )
    scheduler = MagicMock(pending_count=0, running_count=0)
    scheduled: list[asyncio.Task] = []

    def execute(coroutine):
        task = asyncio.create_task(coroutine)
        scheduled.append(task)
        return task

    scheduler.execute_async.side_effect = execute
    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)
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
    initial = [call.args[0] for call in issuer.issue_credit.await_args_list]
    assert {turn.conversation_id for turn in initial} == {"root-0", "root-1"}
    assert not progress.all_credits_sent_event.is_set()

    first = initial[0].effective_root_correlation_id
    second = initial[1].effective_root_correlation_id
    registry.open_tree(first, CreditPhase.PROFILING, root_pending=True)
    registry.open_tree(second, CreditPhase.PROFILING, root_pending=True)
    registry.register_descendants(first)
    assert not registry.on_root_terminal(first)
    assert [
        call.args[0].conversation_id for call in issuer.issue_credit.await_args_list
    ] == [
        "root-0",
        "root-1",
    ]
    assert registry.on_descendant_done(first)
    await asyncio.gather(*scheduled)
    admitted = [call.args[0] for call in issuer.issue_credit.await_args_list]
    assert [turn.conversation_id for turn in admitted].count("root-2") == 1
    assert not progress.all_credits_sent_event.is_set()

    third = admitted[-1].effective_root_correlation_id
    registry.open_tree(third, CreditPhase.PROFILING, root_pending=True)
    branch_orchestrator.has_pending_branch_work.return_value = True
    assert registry.on_root_terminal(second)
    assert registry.on_root_terminal(third)
    await asyncio.gather(*scheduled)
    await asyncio.sleep(0)
    assert not progress.all_credits_sent_event.is_set()

    branch_orchestrator.has_pending_branch_work.return_value = False
    issuer.replay_gate.has_pending_finite_work.return_value = True
    strategy._maybe_finish_finite_graph()
    assert not progress.all_credits_sent_event.is_set()

    issuer.replay_gate.has_pending_finite_work.return_value = False
    scheduler.pending_count = 1
    strategy._maybe_finish_finite_graph()
    assert not progress.all_credits_sent_event.is_set()

    scheduler.pending_count = 0
    strategy._maybe_finish_finite_graph()
    assert progress.all_credits_sent_event.is_set()
    assert sorted(turn.conversation_id for turn in admitted) == [
        "root-0",
        "root-1",
        "root-2",
    ]
    assert concurrency_manager.release_session_slot.call_count == 3
    assert registry.peak_open == 2
