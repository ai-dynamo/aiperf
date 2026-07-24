# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: parent spawns two SPAWN children, parent suspends at spawn,
both children drain, parent's gated turn dispatches via dispatch_join_turn.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator

pytestmark = pytest.mark.component_integration


def _mk_credit(
    conv_id: str, x_corr: str, turn_index: int = 0, agent_depth: int = 0
) -> Credit:
    c = MagicMock(spec=Credit)
    c.conversation_id = conv_id
    c.x_correlation_id = x_corr
    c.turn_index = turn_index
    c.agent_depth = agent_depth
    c.parent_correlation_id = None
    return c


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    lookup = {c.conversation_id: c for c in conversations}
    cs.get_metadata.side_effect = lambda cid: lookup[cid]
    return cs


@pytest.mark.asyncio
async def test_parent_resumes_after_all_children_complete():
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    c1 = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    c2 = ConversationMetadata(conversation_id="c2", turns=[TurnMetadata()])

    cs = _mk_source([root, c1, c2])

    # start_branch_child returns a fake SampledSession with a unique x_correlation_id.
    child_corrs = iter(["corr-c1", "corr-c2"])

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **_kw
    ):
        s = MagicMock()
        s.x_correlation_id = next(child_corrs)
        return s

    cs.start_branch_child.side_effect = _start

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Parent completes turn 0.
    parent_credit = _mk_credit("root", "corr-root", turn_index=0)
    suppressed = await orch.intercept(parent_credit)
    assert suppressed is True
    # Parent is blocked at its next turn (gated on turn 1).
    assert "corr-root" in orch._active_joins
    assert orch._active_joins["corr-root"].gated_turn_index == 1

    # Children complete one at a time.
    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c2")

    # Join dispatched exactly once with the correct PendingBranchJoin.
    issuer.dispatch_join_turn.assert_awaited_once()
    sent = issuer.dispatch_join_turn.call_args.args[0]
    assert sent.parent_x_correlation_id == "corr-root"
    assert sent.parent_conversation_id == "root"
    assert sent.gated_turn_index == 1
    assert orch.stats.parents_resumed == 1
    assert orch.stats.joins_suppressed == 0


@pytest.mark.asyncio
async def test_join_suppressed_when_issuer_returns_false():
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    c1 = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, c1])
    cs.start_branch_child.return_value = MagicMock(x_correlation_id="corr-c1")

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "corr-root"))
    await orch.on_child_leaf_reached("corr-c1")

    assert orch.stats.parents_resumed == 0
    assert orch.stats.joins_suppressed == 1


@pytest.mark.asyncio
async def test_release_blocked_join_applies_think_time_before_dispatch(monkeypatch):
    """A gated turn carrying a per-round think-time sleeps for that duration
    before its join turn is dispatched (the coordinator's inter-round wait)."""
    from aiperf.timing import branch_orchestrator as bo_mod
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)

    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(bo_mod.asyncio, "sleep", _fake_sleep)

    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-spine",
        parent_conversation_id="spine",
        parent_num_turns=3,
        gated_turn_index=1,
        parent_delay_ms_on_gated_turn=50.0,
        parent_no_request_on_gated_turn=True,  # spine gate -> think-time applies
    )
    await orch._release_blocked_join(pending)

    assert slept == [0.05]  # 50 ms -> 0.05 s, applied before dispatch
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_blocked_join_no_sleep_when_think_time_zero(monkeypatch):
    from aiperf.timing import branch_orchestrator as bo_mod
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)

    slept: list[float] = []
    monkeypatch.setattr(bo_mod.asyncio, "sleep", lambda s: slept.append(s) or _noop())

    async def _noop() -> None:
        return None

    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-spine",
        parent_conversation_id="spine",
        parent_num_turns=3,
        gated_turn_index=1,
        parent_delay_ms_on_gated_turn=0.0,
    )
    await orch._release_blocked_join(pending)

    assert slept == []
    issuer.dispatch_join_turn.assert_awaited_once()


def _spine_pending(round_idx: int, x: str = "i1", median: float = 200.0):
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    return PendingBranchJoin(
        parent_x_correlation_id=x,
        parent_conversation_id="spine",
        parent_num_turns=3,
        gated_turn_index=round_idx,
        parent_delay_ms_on_gated_turn=median,
        parent_no_request_on_gated_turn=True,  # spine gates are request-free
    )


def test_resolve_think_ms_zero_for_normal_dag_join_turn():
    """Regression: a NORMAL (non-request-free) DAG join turn carrying an authored
    delay_ms must NOT incur a pre-join think-sleep -- agentx fires such joins
    immediately. Think-time is exclusive to request-free orchestrator spines."""
    from aiperf.common.models import ConversationMetadata
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    source = _mk_source([ConversationMetadata(conversation_id="spine")])
    orch = BranchOrchestrator(conversation_source=source, credit_issuer=MagicMock())
    normal_gate = PendingBranchJoin(
        parent_x_correlation_id="i1",
        parent_conversation_id="spine",
        parent_num_turns=3,
        gated_turn_index=1,
        parent_delay_ms_on_gated_turn=200.0,  # authored trace delay
        parent_no_request_on_gated_turn=False,  # NOT a spine gate
    )
    assert orch._resolve_think_ms(normal_gate) == 0.0


def test_resolve_think_ms_samples_lognormal_per_instance_round():
    """A sampled spine draws an independent, reproducible lognormal think-time
    per (instance, round) around the stamped median."""
    from aiperf.common.models import ConversationMetadata
    from aiperf.common.models.dataset_models import ThinkTimeSpec

    source = _mk_source(
        [
            ConversationMetadata(
                conversation_id="spine", think_time=ThinkTimeSpec(sigma=0.5)
            )
        ]
    )
    orch = BranchOrchestrator(conversation_source=source, credit_issuer=MagicMock())

    v1 = orch._resolve_think_ms(_spine_pending(1))
    assert v1 > 0.0 and v1 != 200.0  # sampled, not the raw median
    assert (
        orch._resolve_think_ms(_spine_pending(1)) == v1
    )  # reproducible per (instance, round)
    assert orch._resolve_think_ms(_spine_pending(2)) != v1  # independent per round
    assert (
        orch._resolve_think_ms(_spine_pending(1, x="i2")) != v1
    )  # independent per instance


def test_resolve_think_ms_fixed_when_no_distribution():
    from aiperf.common.models import ConversationMetadata

    source = _mk_source([ConversationMetadata(conversation_id="spine")])
    orch = BranchOrchestrator(conversation_source=source, credit_issuer=MagicMock())
    assert orch._resolve_think_ms(_spine_pending(1)) == 200.0


def test_resolve_think_ms_clamps_to_max():
    from aiperf.common.models import ConversationMetadata
    from aiperf.common.models.dataset_models import ThinkTimeSpec

    source = _mk_source(
        [
            ConversationMetadata(
                conversation_id="spine", think_time=ThinkTimeSpec(sigma=0.5, max_ms=1.0)
            )
        ]
    )
    orch = BranchOrchestrator(conversation_source=source, credit_issuer=MagicMock())
    assert (
        orch._resolve_think_ms(_spine_pending(1)) == 1.0
    )  # median 200 clamped to max 1
