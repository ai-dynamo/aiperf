# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end wiring test at the DAG orchestrator seam.

Drives the REAL CreditIssuer + REAL BranchOrchestrator + REAL
SessionTreeRegistry + REAL ConversationSource through the full tree lifecycle
(open -> register -> root terminal -> descendant done) and asserts the
transitions produce correct lineage finality on the ISSUED credits. No
MagicMock for the registry or any credit -- only the phase scalars
(progress/lifecycle/stop_checker) and the concurrency slots are faked.
"""

import time
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.session_tree import SessionTreeRegistry


class _FakeConcurrency:
    """Slots always granted; releases are no-ops."""

    async def acquire_session_slot(self, phase, can_proceed) -> bool:
        return True

    async def acquire_prefill_slot(self, phase, can_proceed) -> bool:
        return True

    def release_session_slot(self, phase) -> None:
        pass


class _CapturingRouter:
    def __init__(self) -> None:
        self.sent: list[Credit] = []

    async def send_credit(self, *, credit: Credit) -> None:
        self.sent.append(credit)


def _mk_source() -> ConversationSource:
    """Root (2 turns, SPAWN branch on turn 0) -> one child (2 turns)."""
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[
                    TurnMetadata(timestamp_ms=0.0, branch_ids=["root-conv:0"]),
                    TurnMetadata(timestamp_ms=0.0),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="root-conv:0",
                        child_conversation_ids=["child-conv"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="post",
                    ),
                ],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="child-conv",
                turns=[TurnMetadata(timestamp_ms=0.0), TurnMetadata(timestamp_ms=0.0)],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations]
    )
    return ConversationSource(ds, sampler)


def _make_issuer(
    registry: SessionTreeRegistry, router: _CapturingRouter
) -> CreditIssuer:
    progress = MagicMock()
    progress.increment_sent = MagicMock(return_value=(1, False))
    progress.freeze_sent_counts = MagicMock()
    progress.all_credits_sent_event = MagicMock()

    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    stop_checker.can_start_new_session = MagicMock(return_value=True)
    stop_checker.can_send_dag_child_turn = MagicMock(return_value=True)

    cancellation = MagicMock()
    cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)

    lifecycle = MagicMock()
    lifecycle.started_at_ns = time.time_ns()
    lifecycle.started_at_perf_ns = time.perf_counter_ns()

    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=_FakeConcurrency(),
        credit_router=router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
        session_tree_registry=registry,
    )


@pytest.mark.asyncio
async def test_orchestrator_seam_drives_tree_finality_transitions():
    registry = SessionTreeRegistry()
    router = _CapturingRouter()
    source = _mk_source()
    issuer = _make_issuer(registry, router)
    orch = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        session_tree_registry=registry,
    )

    # 1. OPEN: issuing the root's turn-0 credit opens the tree (issuer seam).
    root_turn0 = TurnToSend(
        conversation_id="root-conv",
        x_correlation_id="root-x",
        turn_index=0,
        num_turns=2,
    )
    await issuer.issue_credit(root_turn0)
    assert registry.has_tree("root-x")
    root_credit0 = router.sent[0]
    assert root_credit0.agent_depth == 0
    # Turn 0 of 2 is not final -> conservative False.
    assert root_credit0.is_tree_final is False

    # 2. REGISTER: intercepting the root turn-0 return spawns the child and
    #    registers it as a descendant (orchestrator seam).
    await orch.intercept(root_credit0)
    child_credits = [c for c in router.sent if c.agent_depth == 1]
    assert len(child_credits) == 1
    child_credit0 = child_credits[0]
    assert child_credit0.root_correlation_id == "root-x"
    # Root still pending + descendant outstanding -> child not tree-final, and
    # parent (== root) not yet final.
    assert child_credit0.is_parent_final is False
    assert child_credit0.is_tree_final is False
    assert registry.open_count() == 1
    child_x = child_credit0.x_correlation_id

    # 3. ROOT TERMINAL: issue the root's final turn, then intercept it so the
    #    orchestrator clears root_pending (orchestrator seam via intercept).
    root_turn1 = TurnToSend(
        conversation_id="root-conv",
        x_correlation_id="root-x",
        turn_index=1,
        num_turns=2,
    )
    await issuer.issue_credit(root_turn1)
    root_credit1 = router.sent[-1]
    # Child still outstanding when the root's final turn is issued.
    assert root_credit1.is_tree_final is False
    await orch.intercept(root_credit1)
    assert registry.root_terminal("root-x") is True

    # 4. FINALITY ON ISSUE: the child's final continuation, issued after the
    #    root terminal with the child as the sole outstanding descendant, is
    #    provably the tree's last request AND its parent (the root) is final.
    child_turn1 = TurnToSend(
        conversation_id="child-conv",
        x_correlation_id=child_x,
        turn_index=1,
        num_turns=2,
        agent_depth=1,
        parent_correlation_id="root-x",
        root_correlation_id="root-x",
    )
    await issuer.dispatch_child_turn(child_turn1)
    child_credit1 = router.sent[-1]
    assert child_credit1.x_correlation_id == child_x
    assert child_credit1.is_parent_final is True
    assert child_credit1.is_tree_final is True

    # 5. DESCENDANT DONE: the child's terminal return drains + retires the tree
    #    (orchestrator seam).
    await orch.on_child_leaf_reached(child_x)
    assert not registry.has_tree("root-x")
    assert registry.open_count() == 0


# =============================================================================
# Conservative-contract regressions: SPAWN branches must gate finality
# =============================================================================


def _mk_single_turn_spawn_source() -> ConversationSource:
    """Single-turn root whose ONLY turn declares a terminal SPAWN branch."""
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["root-conv:0"])],
                branches=[
                    ConversationBranchInfo(
                        branch_id="root-conv:0",
                        child_conversation_ids=["child-conv"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="post",
                    ),
                ],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="child-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations]
    )
    return ConversationSource(ds, sampler)


@pytest.mark.asyncio
async def test_single_turn_root_with_terminal_spawn_branch_not_tree_final():
    """(a) A single-turn root declaring a terminal SPAWN branch must stamp
    is_tree_final=False on the root credit: its children spawn at
    return-intercept, AFTER issue-time stamping, so the registry shows nothing
    outstanding yet. Previously stamped a wrong True (has_forks is FORK-only)."""
    from aiperf.timing.conversation_source import SampledSession

    registry = SessionTreeRegistry()
    router = _CapturingRouter()
    source = _mk_single_turn_spawn_source()
    issuer = _make_issuer(registry, router)
    orch = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        session_tree_registry=registry,
    )

    session = SampledSession(
        conversation_id="root-conv",
        metadata=source.get_metadata("root-conv"),
        x_correlation_id="root-a",
    )
    root_turn0 = session.build_first_turn()
    # End-to-end stamp: SPAWN-only branch -> has_branches True, has_forks False.
    assert root_turn0.has_branches is True
    assert root_turn0.has_forks is False

    await issuer.issue_credit(root_turn0)
    root_credit = router.sent[0]
    assert root_credit.is_final_turn is True
    assert root_credit.is_tree_final is False

    # The spawn then registers + the tree drains cleanly through the child.
    await orch.intercept(root_credit)
    child_credits = [c for c in router.sent if c.agent_depth == 1]
    assert len(child_credits) == 1
    await orch.on_child_leaf_reached(child_credits[0].x_correlation_id)
    assert not registry.has_tree("root-a")
    assert registry.late_events == 0


def _mk_grandchild_spawn_source() -> ConversationSource:
    """Single-turn root -> child (2 turns; final turn declares a SPAWN
    branch to a grandchild)."""
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["root-conv:0"])],
                branches=[
                    ConversationBranchInfo(
                        branch_id="root-conv:0",
                        child_conversation_ids=["child-conv"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="post",
                    ),
                ],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="child-conv",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=0.0, branch_ids=["child-conv:1"]),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="child-conv:1",
                        child_conversation_ids=["grandchild-conv"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="post",
                    ),
                ],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
            ConversationMetadata(
                conversation_id="grandchild-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=2,
                parent_conversation_id="child-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations]
    )
    return ConversationSource(ds, sampler)


@pytest.mark.asyncio
async def test_child_final_turn_spawning_grandchild_not_tree_final():
    """(b) A child's final turn that declares a SPAWN branch (grandchild
    pending) must stamp is_tree_final=False even when the child is the sole
    outstanding descendant and the root is already terminal -- previously the
    exact wrong-True scenario."""
    registry = SessionTreeRegistry()
    router = _CapturingRouter()
    source = _mk_grandchild_spawn_source()
    issuer = _make_issuer(registry, router)
    orch = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        session_tree_registry=registry,
    )

    # Root turn 0 (single turn, spawning): opens tree; intercept spawns the
    # child, registers it, then marks the root terminal.
    root_turn0 = TurnToSend(
        conversation_id="root-conv",
        x_correlation_id="root-b",
        turn_index=0,
        num_turns=1,
        has_branches=True,
    )
    await issuer.issue_credit(root_turn0)
    root_credit = router.sent[0]
    assert root_credit.is_tree_final is False
    await orch.intercept(root_credit)
    assert registry.root_terminal("root-b") is True
    child_credit0 = next(c for c in router.sent if c.agent_depth == 1)
    child_x = child_credit0.x_correlation_id

    # Child's final turn, built from real metadata (branch_ids on turn 1),
    # issued while the child is the sole outstanding descendant.
    next_meta = source.get_next_turn_metadata(child_credit0)
    child_turn1 = TurnToSend.from_previous_credit(child_credit0, next_meta)
    assert child_turn1.has_branches is True
    assert child_turn1.has_forks is False
    await issuer.dispatch_child_turn(child_turn1)
    child_credit1 = router.sent[-1]
    assert child_credit1.is_final_turn is True
    assert child_credit1.is_parent_final is True
    assert child_credit1.is_tree_final is False  # grandchild pending

    # Production return order: leaf-reached fires BEFORE intercept spawns the
    # grandchild (callback handler step 4b before step 5). The child is the last
    # outstanding descendant, so its leaf-reached decrement drains and retires
    # the tree; intercept's register_descendants then RESURRECTS it (root already
    # terminal). The grandchild -- a single-turn, sole-remaining, genuinely-last
    # request -- is therefore correctly stamped tree-final (previously the
    # resurrect was missing and it was wrongly under-fired to False).
    await orch.on_child_leaf_reached(child_x)
    await orch.intercept(child_credit1)
    grandchild_credits = [c for c in router.sent if c.agent_depth == 2]
    assert len(grandchild_credits) == 1
    assert grandchild_credits[0].root_correlation_id == "root-b"
    assert grandchild_credits[0].is_tree_final is True
    await orch.on_child_leaf_reached(grandchild_credits[0].x_correlation_id)
    assert not registry.has_tree("root-b")
    assert registry.late_events == 0


def _mk_pre_session_source() -> ConversationSource:
    """Two-turn root with a pre-session SPAWN branch attached to turn 0."""
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[
                    TurnMetadata(timestamp_ms=0.0, branch_ids=["root-conv:pre"]),
                    TurnMetadata(timestamp_ms=0.0),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="root-conv:pre",
                        child_conversation_ids=["pre-child-conv"],
                        mode=ConversationBranchMode.SPAWN,
                        dispatch_timing="pre",
                    ),
                ],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="pre-child-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations]
    )
    return ConversationSource(ds, sampler)


@pytest.mark.asyncio
async def test_pre_session_child_live_blocks_root_final_turn_tree_final():
    """(c) A live pre-session SPAWN child must keep the root's final turn from
    stamping is_tree_final=True. Pre-dispatch runs before sampling (no root id
    exists yet), so the orchestrator folds the live pre children into the root
    instance's tree at its turn-0 return -- before any final-turn issue."""
    registry = SessionTreeRegistry()
    router = _CapturingRouter()
    source = _mk_pre_session_source()
    issuer = _make_issuer(registry, router)
    orch = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        session_tree_registry=registry,
    )

    # Pre-dispatch fires the background child before any root credit exists.
    await orch.dispatch_pre_session_branches()
    pre_credits = [c for c in router.sent if c.agent_depth == 1]
    assert len(pre_credits) == 1
    pre_child_x = pre_credits[0].x_correlation_id
    assert pre_credits[0].is_tree_final is False

    # Root turn 0: opens the tree; its return-intercept folds the live
    # pre-session child into this root's tree.
    root_turn0 = TurnToSend(
        conversation_id="root-conv",
        x_correlation_id="root-c",
        turn_index=0,
        num_turns=2,
        has_branches=True,  # turn 0 declares the pre branch
    )
    await issuer.issue_credit(root_turn0)
    root_credit0 = [c for c in router.sent if c.agent_depth == 0][0]
    await orch.intercept(root_credit0)

    # Root's FINAL turn (declares no branches) with the pre child still live:
    # must stamp False. Previously wrong-True (pre children were never
    # registered anywhere).
    next_meta = source.get_next_turn_metadata(root_credit0)
    root_turn1 = TurnToSend.from_previous_credit(root_credit0, next_meta)
    assert root_turn1.has_branches is False
    await issuer.issue_credit(root_turn1)
    root_credit1 = router.sent[-1]
    assert root_credit1.is_final_turn is True
    assert root_credit1.is_tree_final is False  # pre-session child still live

    # Root terminal, then the pre child's terminal drains + retires the tree.
    await orch.intercept(root_credit1)
    assert registry.root_terminal("root-c") is True
    assert registry.has_tree("root-c")
    await orch.on_child_leaf_reached(pre_child_x)
    assert not registry.has_tree("root-c")
    assert registry.late_events == 0
