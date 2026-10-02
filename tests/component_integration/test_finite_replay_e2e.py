# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration test for finite replay end-to-end flow.

Exercises the full pipeline:
DatasetMetadata -> TrajectorySource -> AgenticReplayStrategy ->
SessionTreeRegistry -> BranchOrchestrator -> ReplayBarrierCoordinator ->
CreditCallbackHandler.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from msgspec.structs import replace
from pytest import param

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn, TransportDispatched
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


def _make_finite_dataset() -> DatasetMetadata:
    """Create a 2-root dataset: root-0 has a spawned subagent, root-1 is a 2-turn root."""
    branch = ConversationBranchInfo(
        branch_id="root-0:branch-1",
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
                    TurnMetadata(
                        timestamp_ms=0.0,
                        api_time_ms=50.0,
                        branch_ids=[branch.branch_id],
                    ),
                    TurnMetadata(timestamp_ms=100.0, api_time_ms=30.0),
                ],
                branches=[branch],
                is_root=True,
            ),
            ConversationMetadata(
                conversation_id="child-0",
                turns=[TurnMetadata(timestamp_ms=10.0, api_time_ms=40.0)],
                parent_conversation_id="root-0",
                agent_depth=1,
                is_root=False,
            ),
            ConversationMetadata(
                conversation_id="root-1",
                turns=[
                    TurnMetadata(timestamp_ms=0.0, api_time_ms=20.0),
                    TurnMetadata(timestamp_ms=50.0, api_time_ms=20.0),
                ],
                is_root=True,
            ),
        ],
    )


def _build_finite_test_rig(
    dataset: DatasetMetadata,
    *,
    concurrency: int = 1,
    sampler: SequentialSampler | None = None,
    random_seed: int = 42,
) -> SimpleNamespace:
    """Build and wire a fresh finite replay execution pipeline."""
    if sampler is None:
        sampler = SequentialSampler(
            [c.conversation_id for c in dataset.conversations if c.is_root]
        )

    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=random_seed,
        finite_replay=True,
    )
    phase_cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        concurrency=concurrency,
        finite_replay=True,
    )

    scheduler = LoopScheduler()
    lifecycle = PhaseLifecycle(phase_cfg)
    progress = PhaseProgressTracker(phase_cfg)
    stop_checker = StopConditionChecker(
        config=phase_cfg,
        lifecycle=lifecycle,
        counter=progress.counter,
    )

    concurrency_manager = ConcurrencyManager()
    concurrency_manager.configure_for_phase(0, concurrency, prefill_concurrency=1)

    tree_registry = SessionTreeRegistry(concurrency_manager)
    router = MagicMock()

    barrier = ReplayBarrierCoordinator(
        dataset,
        strict_finite=True,
        scheduler=scheduler,
    )

    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        profiling_index=0,
        phase_name="profiling",
        phase_kind="profiling",
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=concurrency_manager,
        credit_router=router,
        cancellation_policy=MagicMock(
            next_cancellation_delay_ns=MagicMock(return_value=None)
        ),
        lifecycle=lifecycle,
        session_tree_registry=tree_registry,
        session_tree_registry_enabled=True,
        replay_barrier=barrier,
        finite_replay=True,
    )

    branch_orchestrator = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        session_tree_registry=tree_registry,
    )

    strategy = AgenticReplayStrategy(
        config=phase_cfg,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
        branch_orchestrator=branch_orchestrator,
        session_tree_registry=tree_registry,
        progress=progress,
    )

    callback_handler = CreditCallbackHandler(
        concurrency_manager=concurrency_manager, session_tree_registry=tree_registry
    )
    callback_handler.register_phase(
        phase=CreditPhase.PROFILING,
        phase_index=0,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop_checker,
        strategy=strategy,
    )
    callback_handler.set_branch_orchestrator(
        branch_orchestrator, phase=CreditPhase.PROFILING, phase_index=0
    )
    issuer.replay_gate.set_child_refused(branch_orchestrator.on_child_stopped)
    issuer.replay_gate.set_credit_issued(branch_orchestrator.on_credit_issued)

    return SimpleNamespace(
        dataset=dataset,
        source=source,
        phase_cfg=phase_cfg,
        scheduler=scheduler,
        lifecycle=lifecycle,
        progress=progress,
        stop_checker=stop_checker,
        concurrency_manager=concurrency_manager,
        tree_registry=tree_registry,
        router=router,
        barrier=barrier,
        issuer=issuer,
        branch_orchestrator=branch_orchestrator,
        strategy=strategy,
        callback_handler=callback_handler,
    )


@pytest.mark.asyncio
async def test_finite_replay_e2e_lifecycle_execution_to_completion() -> None:
    """A multi-root, multi-subagent finite dataset executes faithfully to completion."""
    dataset = _make_finite_dataset()
    rig = _build_finite_test_rig(dataset, concurrency=1)

    # Simulated worker that answers credits automatically
    dispatched_credits: list[Credit] = []

    async def fake_send_credit(credit: Credit) -> None:
        dispatched_credits.append(credit)
        now_wall = rig.lifecycle.now_ns()

        # Worker sends TransportDispatched
        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker-0",
                transport_start_wall_ns=now_wall,
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )

        # Worker sends CreditReturn with EOF
        ret = CreditReturn(
            credit=credit,
            transport_eof_wall_ns=now_wall + 50_000_000,
            clock_offset_ns=0,
            clock_offset_spread_ns=0,
            cancelled=False,
            error=None,
        )
        await rig.callback_handler.on_credit_return("worker-0", ret)

    rig.router.send_credit.side_effect = fake_send_credit

    # Run the strategy setup and execution
    rig.lifecycle.start()

    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()

    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=2)

    assert rig.progress.all_credits_sent_event.is_set()
    assert {c.conversation_id for c in dispatched_credits} == {
        "root-0",
        "child-0",
        "root-1",
    }
    assert rig.concurrency_manager.get_session_stats(0).release_count == 2
    assert rig.progress.fatal_error is None
    assert sorted(
        (c.conversation_id, c.turn_index) for c in dispatched_credits
    ) == sorted(
        (conversation.conversation_id, index)
        for conversation in dataset.conversations
        for index in range(len(conversation.turns))
    )
    assert not rig.barrier.has_pending_finite_work()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,cancelled,missing_field,expected_error",
    [
        param("Inference server 500 internal error", False, None, "terminal request failed", id="server-error"),
        param(None, True, None, "terminal request failed", id="cancelled"),
        param(None, False, "transport_eof_wall_ns", "response EOF missing", id="missing-eof"),
        param(None, False, "clock_offset_ns", "EOF has no clock correction", id="missing-offset"),
        param(None, False, "clock_offset_spread_ns", "EOF has no clock spread", id="missing-spread"),
    ],
)  # fmt: skip
async def test_finite_replay_e2e_invalid_return_aborts_before_next_turn(
    error: str | None,
    cancelled: bool,
    missing_field: str | None,
    expected_error: str,
) -> None:
    """An invalid non-final return must not advance its root or admit later roots."""
    dataset = _make_finite_dataset()
    rig = _build_finite_test_rig(dataset, concurrency=1)

    dispatched_credits: list[Credit] = []

    async def fake_failing_credit(credit: Credit) -> None:
        dispatched_credits.append(credit)
        now_wall = rig.lifecycle.now_ns()

        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker-0",
                transport_start_wall_ns=now_wall,
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )

        ret = CreditReturn(
            credit=credit,
            transport_eof_wall_ns=now_wall + 50_000_000,
            clock_offset_ns=0,
            clock_offset_spread_ns=0,
            cancelled=cancelled,
            error=error,
        )
        if missing_field is not None:
            ret = replace(ret, **{missing_field: None})
        await rig.callback_handler.on_credit_return("worker-0", ret)

    rig.router.send_credit.side_effect = fake_failing_credit

    rig.lifecycle.start()

    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()

    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=2)

    assert rig.progress.fatal_error is not None
    assert expected_error in str(rig.progress.fatal_error)
    assert rig.progress.all_credits_sent_event.is_set()
    assert [(c.conversation_id, c.turn_index) for c in dispatched_credits] == [
        ("root-0", 0)
    ]
    rig.branch_orchestrator.cleanup()
    await rig.issuer.replay_gate.cancel(notify_refused=False)
    await asyncio.gather(*rig.scheduler.cancel_all(), return_exceptions=True)


@pytest.mark.asyncio
async def test_finite_replay_e2e_multi_subagent_concurrent_branches_drain() -> None:
    """A root with multiple concurrent subagents holds session slot until all finish."""
    branch_spawn = ConversationBranchInfo(
        branch_id="root-0:spawn-1",
        child_conversation_ids=["child-spawn"],
        mode=ConversationBranchMode.SPAWN,
        start_timestamp_ms=10.0,
    )
    branch_fork = ConversationBranchInfo(
        branch_id="root-0:fork-1",
        child_conversation_ids=["child-fork"],
        mode=ConversationBranchMode.FORK,
        start_timestamp_ms=20.0,
    )
    dataset = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="root-0",
                turns=[
                    TurnMetadata(
                        timestamp_ms=0.0,
                        api_time_ms=50.0,
                        branch_ids=[branch_spawn.branch_id, branch_fork.branch_id],
                    ),
                    TurnMetadata(timestamp_ms=100.0, api_time_ms=30.0),
                ],
                branches=[branch_spawn, branch_fork],
                is_root=True,
            ),
            ConversationMetadata(
                conversation_id="child-spawn",
                turns=[TurnMetadata(timestamp_ms=10.0, api_time_ms=40.0)],
                parent_conversation_id="root-0",
                agent_depth=1,
                is_root=False,
            ),
            ConversationMetadata(
                conversation_id="child-fork",
                turns=[TurnMetadata(timestamp_ms=20.0, api_time_ms=30.0)],
                parent_conversation_id="root-0",
                agent_depth=1,
                is_root=False,
            ),
        ],
    )
    rig = _build_finite_test_rig(dataset, concurrency=1)

    dispatched_credits: list[Credit] = []

    async def fake_send_credit(credit: Credit) -> None:
        dispatched_credits.append(credit)
        now_wall = rig.lifecycle.now_ns()

        await rig.callback_handler.on_transport_dispatched(
            TransportDispatched(
                credit_id=credit.id,
                phase=credit.phase,
                phase_index=credit.phase_index,
                worker_id="worker-0",
                transport_start_wall_ns=now_wall,
                clock_offset_ns=0,
                clock_offset_spread_ns=0,
            )
        )

        ret = CreditReturn(
            credit=credit,
            transport_eof_wall_ns=now_wall + 50_000_000,
            clock_offset_ns=0,
            clock_offset_spread_ns=0,
            cancelled=False,
            error=None,
        )
        await rig.callback_handler.on_credit_return("worker-0", ret)

    rig.router.send_credit.side_effect = fake_send_credit

    rig.lifecycle.start()

    await rig.strategy.setup_phase()
    await rig.strategy.execute_phase()

    await asyncio.wait_for(rig.progress.all_credits_sent_event.wait(), timeout=2)

    assert rig.progress.all_credits_sent_event.is_set()
    assert {c.conversation_id for c in dispatched_credits} == {
        "root-0",
        "child-spawn",
        "child-fork",
    }
    assert rig.concurrency_manager.get_session_stats(0).release_count == 1
    assert rig.progress.fatal_error is None
    assert sorted(
        (c.conversation_id, c.turn_index) for c in dispatched_credits
    ) == sorted(
        (conversation.conversation_id, index)
        for conversation in dataset.conversations
        for index in range(len(conversation.turns))
    )
    assert not rig.barrier.has_pending_finite_work()
