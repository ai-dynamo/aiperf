# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An explicit play drains every request through the finite replay pipeline."""

import asyncio
import time

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
    ReplayDependencyEvent,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    ReplayTurnReference,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.credit.messages import CreditReturn, TransportDispatched
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.component_integration.test_finite_replay_e2e import _build_finite_test_rig

pytestmark = pytest.mark.component_integration


@pytest.mark.asyncio
async def test_multi_root_play_drains_after_all_dependencies() -> None:
    branch = ConversationBranchInfo(
        branch_id="play:0",
        child_conversation_ids=["join", "a", "b"],
        mode=ConversationBranchMode.SPAWN,
        is_background=False,
    )
    root = ConversationMetadata(
        conversation_id="play",
        is_orchestrator=True,
        turns=[
            TurnMetadata(
                timestamp_ms=0, no_request=True, branch_ids=[branch.branch_id]
            ),
            TurnMetadata(
                timestamp_ms=0,
                no_request=True,
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch.branch_id
                    )
                ],
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(
            conversation_id=name,
            parent_conversation_id="play",
            is_root=False,
            agent_depth=1,
            turns=[TurnMetadata(timestamp_ms=0, replay_predecessors=dependencies)],
        )
        for name, dependencies in [
            ("a", []),
            ("b", []),
            (
                "join",
                [
                    ReplayTurnReference(
                        conversation_id="a",
                        turn_index=0,
                        event=ReplayDependencyEvent.DISPATCH,
                    ),
                    ReplayTurnReference(
                        conversation_id="b",
                        turn_index=0,
                        event=ReplayDependencyEvent.COMPLETION,
                    ),
                ],
            ),
        ]
    ]
    rig = _build_finite_test_rig(
        DatasetMetadata(
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            conversations=[root, *children],
        )
    )
    # A free prefill slot ensures the dependency, not concurrency, holds join.
    rig.concurrency_manager.configure_for_phase(0, 1, prefill_concurrency=4)
    dispatched: list[str] = []
    return_tasks: list[asyncio.Task[None]] = []
    a_completed = asyncio.Event()
    b_dispatched = asyncio.Event()
    release_b = asyncio.Event()
    join_dispatched = asyncio.Event()

    async def fake_send_credit(credit: Credit) -> None:
        if credit.no_request:
            return_tasks.append(
                asyncio.create_task(
                    rig.callback_handler.on_credit_return(
                        "worker",
                        CreditReturn(credit=credit, cancelled=False, error=None),
                    )
                )
            )
            return
        dispatched.append(credit.conversation_id)
        if credit.conversation_id == "join":
            join_dispatched.set()
        now = time.perf_counter_ns()
        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker",
                transport_start_wall_ns=rig.lifecycle.now_ns(),
                transport_start_perf_ns=now,
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )
        if credit.conversation_id == "b":
            b_dispatched.set()

        async def finish() -> None:
            if credit.conversation_id == "b":
                await release_b.wait()
            await rig.callback_handler.on_credit_return(
                "worker",
                CreditReturn(
                    credit=credit,
                    transport_eof_perf_ns=time.perf_counter_ns(),
                    clock_offset_spread_ns=0,
                    cancelled=False,
                    error=None,
                ),
            )
            if credit.conversation_id == "a":
                a_completed.set()

        return_tasks.append(asyncio.create_task(finish()))

    rig.router.send_credit.side_effect = fake_send_credit
    rig.lifecycle.start()
    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()
    await asyncio.wait_for(
        asyncio.gather(a_completed.wait(), b_dispatched.wait()), timeout=5
    )
    assert rig.branch_orchestrator.stats.graphs_admitted == 1
    assert rig.branch_orchestrator.stats.graphs_completed_to_end == 0
    try:
        await asyncio.wait_for(join_dispatched.wait(), timeout=0.1)
        joined_before_b_completed = True
    except TimeoutError:
        joined_before_b_completed = False
    release_b.set()
    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=5)
    await asyncio.gather(*return_tasks)

    assert not joined_before_b_completed
    assert sorted(dispatched) == ["a", "b", "join"]
    assert dispatched.index("join") > dispatched.index("a")
    assert dispatched.index("join") > dispatched.index("b")
    assert rig.progress.fatal_error is None
    assert rig.branch_orchestrator.stats.graphs_completed_to_end == 1
    assert rig.barrier.finite_diagnostics()[0] == len(dispatched)
    assert rig.concurrency_manager.get_session_stats(0).release_count == 1
    assert not rig.barrier.has_pending_finite_work()


def _two_plays() -> DatasetMetadata:
    conversations = []
    for name in ("first", "second"):
        child = f"{name}-request"
        branch = ConversationBranchInfo(
            branch_id=f"{name}:0",
            child_conversation_ids=[child],
            mode=ConversationBranchMode.SPAWN,
        )
        conversations.extend(
            [
                ConversationMetadata(
                    conversation_id=name,
                    is_orchestrator=True,
                    turns=[
                        TurnMetadata(
                            timestamp_ms=0,
                            no_request=True,
                            branch_ids=[branch.branch_id],
                        ),
                        TurnMetadata(
                            timestamp_ms=0,
                            no_request=True,
                            prerequisites=[
                                TurnPrerequisite(
                                    kind=PrerequisiteKind.SPAWN_JOIN,
                                    branch_id=branch.branch_id,
                                )
                            ],
                        ),
                    ],
                    branches=[branch],
                ),
                ConversationMetadata(
                    conversation_id=child,
                    parent_conversation_id=name,
                    is_root=False,
                    agent_depth=1,
                    turns=[TurnMetadata(timestamp_ms=0)],
                ),
            ]
        )
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=conversations,
    )


@pytest.mark.asyncio
async def test_play_slot_is_reused_only_after_all_requests_finish() -> None:
    rig = _build_finite_test_rig(_two_plays(), concurrency=1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    dispatched = []
    return_tasks = []

    async def fake_send_credit(credit):
        if credit.no_request:
            return_tasks.append(
                asyncio.create_task(
                    rig.callback_handler.on_credit_return(
                        "worker",
                        CreditReturn(credit=credit, cancelled=False, error=None),
                    )
                )
            )
            return
        dispatched.append(credit.conversation_id)
        start = time.perf_counter_ns()
        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker",
                transport_start_wall_ns=rig.lifecycle.now_ns(),
                transport_start_perf_ns=start,
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )

        async def finish():
            if credit.conversation_id == "first-request":
                first_started.set()
                await release_first.wait()
            await rig.callback_handler.on_credit_return(
                "worker",
                CreditReturn(
                    credit=credit,
                    transport_eof_perf_ns=time.perf_counter_ns(),
                    clock_offset_spread_ns=0,
                    cancelled=False,
                    error=None,
                ),
            )

        return_tasks.append(asyncio.create_task(finish()))

    rig.router.send_credit.side_effect = fake_send_credit
    rig.lifecycle.start()
    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()
    await asyncio.wait_for(first_started.wait(), timeout=5)
    assert dispatched == ["first-request"]
    assert rig.concurrency_manager.get_session_stats(0).release_count == 0
    assert rig.branch_orchestrator.stats.graphs_admitted == 1
    assert rig.branch_orchestrator.stats.graphs_completed_to_end == 0
    release_first.set()
    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=5)
    await asyncio.gather(*return_tasks)
    assert dispatched == ["first-request", "second-request"]
    assert rig.concurrency_manager.get_session_stats(0).release_count == 2
    assert rig.branch_orchestrator.stats.graphs_admitted == 2
    assert rig.branch_orchestrator.stats.graphs_completed_to_end == 2
    assert rig.progress.fatal_error is None


@pytest.mark.asyncio
async def test_failed_play_does_not_admit_next_play() -> None:
    rig = _build_finite_test_rig(_two_plays(), concurrency=1)
    dispatched = []
    return_tasks = []

    async def fake_send_credit(credit):
        if credit.no_request:
            return_tasks.append(
                asyncio.create_task(
                    rig.callback_handler.on_credit_return(
                        "worker",
                        CreditReturn(credit=credit, cancelled=False, error=None),
                    )
                )
            )
            return
        dispatched.append(credit.conversation_id)
        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker",
                transport_start_wall_ns=rig.lifecycle.now_ns(),
                transport_start_perf_ns=time.perf_counter_ns(),
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )
        return_tasks.append(
            asyncio.create_task(
                rig.callback_handler.on_credit_return(
                    "worker",
                    CreditReturn(credit=credit, cancelled=False, error="server failed"),
                )
            )
        )

    rig.router.send_credit.side_effect = fake_send_credit
    rig.lifecycle.start()
    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()
    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=5)
    await asyncio.gather(*return_tasks)
    assert dispatched == ["first-request"]
    assert rig.progress.fatal_error is not None
    assert rig.branch_orchestrator.stats.graphs_admitted == 1
    assert rig.branch_orchestrator.stats.graphs_completed_to_end == 0
    rig.branch_orchestrator.cleanup()
    await rig.issuer.replay_gate.cancel(notify_refused=False)
    await asyncio.gather(*rig.scheduler.cancel_all(), return_exceptions=True)
