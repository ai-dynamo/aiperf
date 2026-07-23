# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 unit tests for delayed joins in :class:`BranchOrchestrator`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from tests.unit.timing._shared_helpers import _mk_conv, _mk_source


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _k5_metadata() -> list[ConversationMetadata]:
    """Parent conv with 6 turns: spawn on turn 0, gate on turn 5 (K=5)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0", "c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    c1 = _mk_conv("c1", [TurnMetadata()], [])
    return [root, c0, c1]


@pytest.mark.asyncio
async def test_delayed_join_k5_parent_progresses():
    """Spawn at T=0, gate at T=5. Parent returns from turns 0..3 without"""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False
    assert "corr-root" in orch._future_joins
    assert 5 in orch._future_joins["corr-root"]
    assert orch.stats.parents_suspended == 0

    for t in range(1, 4):
        assert await orch.intercept(_mk_credit("root", "corr-root", t)) is False
    assert orch.stats.parents_suspended == 0

    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    assert "corr-root" in orch._active_joins
    assert orch.stats.parents_suspended == 1

    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_delayed_join_children_finish_before_parent_arrives():
    """Children complete before the parent returns from turn 4. When the"""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is False
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    assert orch.stats.parents_suspended == 0
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_delayed_join_k1_regression_via_new_architecture():
    """K=1 auto-desugared case: spawn on turn 0, gate on turn 1. Parent's"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    cs = _mk_source([root, c0])

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert orch.stats.parents_suspended == 1

    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_overlap_branch_dispatches_from_parent_start_not_return() -> None:
    branch = ConversationBranchInfo(
        branch_id="root:overlap",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
        start_timestamp_ms=0.0,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0,
                api_time_ms=5000.0,
                branch_ids=[branch.branch_id],
            ),
            TurnMetadata(
                timestamp_ms=6000.0,
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch.branch_id,
                    )
                ],
            ),
        ],
        [branch],
    )
    root.replay_scope_id = "root"
    child = _mk_conv("child", [TurnMetadata(timestamp_ms=0.0)], [])
    cs = _mk_source([root, child])
    child_session = MagicMock(
        x_correlation_id="corr-child",
        metadata=child,
        effective_root_correlation_id="corr-root",
    )
    cs.start_branch_child.return_value = child_session
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = _mk_credit("root", "corr-root", 0)
    credit.phase = CreditPhase.PROFILING
    credit.effective_root_correlation_id = "corr-root"

    await orch.on_credit_issued(credit)

    issuer.dispatch_first_turn.assert_awaited_once_with(child_session)
    assert await orch.intercept(credit) is True
    issuer.dispatch_first_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_overlap_dispatch_skips_fork_branches() -> None:
    """FORK branches must not dispatch at credit-issue even when their"""
    branch = ConversationBranchInfo(
        branch_id="root:fork-overlap",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.FORK,
        start_timestamp_ms=0.0,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0,
                api_time_ms=5000.0,
                branch_ids=[branch.branch_id],
            ),
            TurnMetadata(),
        ],
        [branch],
    )
    root.replay_scope_id = "root"
    child = _mk_conv("child", [TurnMetadata(timestamp_ms=0.0)], [])
    cs = _mk_source([root, child])
    cs.start_branch_child = MagicMock()
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = _mk_credit("root", "corr-root", 0)
    credit.phase = CreditPhase.PROFILING
    credit.effective_root_correlation_id = "corr-root"

    await orch.on_credit_issued(credit)

    cs.start_branch_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_awaited()
    assert orch._overlap_dispatched_branches == set()


@pytest.mark.asyncio
async def test_delayed_join_stop_condition_fires_during_gap_suppresses_join():
    """If the issuer reports ``dispatch_join_turn`` returned False (stop"""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 4))

    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0


@pytest.mark.asyncio
async def test_delayed_join_fail_fast_aborts_siblings_mid_gap(monkeypatch):
    """With ``AIPERF_DAG_FAIL_FAST=true`` and a child erroring during the"""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock()
    issuer.abort_session = AsyncMock()

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    await orch.on_child_errored("corr-c0")
    assert orch.stats.parents_failed_due_to_child_error == 1
    issuer.abort_session.assert_any_await("corr-root")
    issuer.abort_session.assert_any_await("corr-c1")
    assert "corr-root" not in orch._future_joins
    assert "corr-root" not in orch._active_joins


@pytest.mark.asyncio
async def test_delayed_join_multiple_branches_different_k_values_accepted_phase2():
    """Phase 2: declaring two gated branches on the same spawning turn with"""
    from aiperf.common.validators.orchestrator_v1 import (
        validate_for_orchestrator_v1,
    )

    branch_a = ConversationBranchInfo(
        branch_id="r:0a",
        child_conversation_ids=["ca"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="r:0b",
        child_conversation_ids=["cb"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "r",
        [
            TurnMetadata(branch_ids=["r:0a", "r:0b"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0a")
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0b")
                ]
            ),
        ],
        [branch_a, branch_b],
    )
    ca = _mk_conv("ca", [TurnMetadata()], [])
    cb = _mk_conv("cb", [TurnMetadata()], [])
    md = DatasetMetadata(
        conversations=[conv, ca, cb],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)
