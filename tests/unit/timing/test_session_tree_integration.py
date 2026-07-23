# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration of SessionTreeRegistry with a real ConcurrencyManager and the"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.session_tree import SessionTreeRegistry

PROFILING = CreditPhase.PROFILING


def _held(cm: ConcurrencyManager) -> int:
    """Session slots currently acquired (not yet released) for PROFILING."""
    return cm._session_limiter.get_held_slots(PROFILING)


def _build_orchestrator_with_background_child(registry):
    """Root conversation whose single turn spawns one BACKGROUND subagent."""
    root_meta = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(timestamp_ms=0.0, branch_ids=["bg"])],
        branches=[
            ConversationBranchInfo(
                branch_id="bg",
                child_conversation_ids=["child"],
                mode=ConversationBranchMode.SPAWN,
                is_background=True,
            )
        ],
    )
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="root",
    )

    class _Source:
        dataset_metadata = DatasetMetadata(
            conversations=[root_meta, child_meta],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        def get_metadata(self, conversation_id):
            return {"root": root_meta, "child": child_meta}[conversation_id]

        def start_branch_child(self, **kwargs):
            # Deterministic child id so the test can drive its completion.
            return SampledSession(
                conversation_id=kwargs["child_conversation_id"],
                metadata=child_meta,
                x_correlation_id="child-corr",
                agent_depth=kwargs["agent_depth"],
                parent_correlation_id=kwargs["parent_correlation_id"],
                root_correlation_id=kwargs["root_correlation_id"],
                branch_mode=kwargs["branch_mode"],
            )

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(
        conversation_source=_Source(),
        credit_issuer=issuer,
        session_tree_registry=registry,
    )
    return orch


def _root_final_turn_credit() -> MagicMock:
    credit = MagicMock()
    credit.x_correlation_id = "root-corr"
    credit.conversation_id = "root"
    credit.turn_index = 0
    credit.agent_depth = 0
    credit.phase = PROFILING
    credit.root_correlation_id = None
    credit.effective_root_correlation_id = "root-corr"
    return credit


@pytest.mark.asyncio
async def test_background_subagent_outliving_root_holds_slot_until_drain():
    cm = ConcurrencyManager()
    cm.configure_for_phase(PROFILING, concurrency=2, prefill_concurrency=None)
    registry = SessionTreeRegistry(cm)
    drained: list[tuple[str, CreditPhase]] = []
    registry.set_drain_callback(lambda root, phase: drained.append((root, phase)))
    orch = _build_orchestrator_with_background_child(registry)

    # 1. Issuer admits the root: acquire the physical slot + open the tree.
    assert await cm.acquire_session_slot(PROFILING, lambda: True) is True
    registry.open_tree("root-corr", PROFILING, root_pending=True)
    assert _held(cm) == 1
    assert registry.open_count(PROFILING) == 1

    # 2. Root's (final) turn returns -> orchestrator spawns the background child,
    #    which registers a descendant against the tree.
    credit = _root_final_turn_credit()
    assert await orch.intercept(credit) is False
    assert registry._trees["root-corr"].outstanding == 1

    # 3. Callback handler marks the root terminal AFTER intercept. The child is
    #    still in flight, so the slot must be HELD (not released) -- a new root
    #    must NOT be admittable on its behalf yet.
    assert registry.on_root_terminal("root-corr") is False
    assert _held(cm) == 1, "slot held while a background subagent is still running"
    assert registry.open_count(PROFILING) == 1
    assert drained == []

    # 4. The background subagent finishes -> the tree drains -> the slot is
    #    released exactly once and the lane recycle is signalled.
    await orch.on_child_leaf_reached("child-corr")
    assert _held(cm) == 0, "slot released only after the whole tree drained"
    assert registry.open_count(PROFILING) == 0
    assert drained == [("root-corr", PROFILING)]
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_root_with_no_descendants_releases_slot_immediately_on_terminal():
    """A plain root (no subagents) drains the instant its final turn returns."""
    cm = ConcurrencyManager()
    cm.configure_for_phase(PROFILING, concurrency=1, prefill_concurrency=None)
    registry = SessionTreeRegistry(cm)
    drained: list[str] = []
    registry.set_drain_callback(lambda root, phase: drained.append(root))

    assert await cm.acquire_session_slot(PROFILING, lambda: True) is True
    registry.open_tree("root-corr", PROFILING, root_pending=True)
    assert _held(cm) == 1

    assert registry.on_root_terminal("root-corr") is True
    assert _held(cm) == 0
    assert drained == ["root-corr"]


@pytest.mark.asyncio
async def test_seed_snapshot_registers_grandchildren_under_tree_root():
    """Depth≥2 snapshot children must register against the depth-0 root,
    matching live spawn / ``_tree_descendant_done`` decrement keying.
    """
    from aiperf.timing.trajectory_source import ConversationState

    cm = ConcurrencyManager()
    cm.configure_for_phase(PROFILING, concurrency=1, prefill_concurrency=None)
    registry = SessionTreeRegistry(cm)
    drained: list[str] = []
    registry.set_drain_callback(lambda root, phase: drained.append(root))

    root_meta = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata()],
        branches=[],
    )
    mid_meta = ConversationMetadata(
        conversation_id="mid",
        turns=[TurnMetadata()],
        branches=[],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="root",
    )
    leaf_meta = ConversationMetadata(
        conversation_id="leaf",
        turns=[TurnMetadata()],
        branches=[],
        is_root=False,
        agent_depth=2,
        parent_conversation_id="mid",
    )
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=[root_meta, mid_meta, leaf_meta],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: {
        "root": root_meta,
        "mid": mid_meta,
        "leaf": leaf_meta,
    }[cid]
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        session_tree_registry=registry,
    )

    assert await cm.acquire_session_slot(PROFILING, lambda: True) is True
    registry.open_tree("root-corr", PROFILING, root_pending=True)

    orch.seed_snapshot(
        (
            ConversationState(
                conversation_id="root",
                x_correlation_id="root-corr",
                next_turn_index=0,
                root_correlation_id="root-corr",
            ),
            ConversationState(
                conversation_id="mid",
                x_correlation_id="mid-corr",
                next_turn_index=0,
                agent_depth=1,
                parent_correlation_id="root-corr",
                root_correlation_id="root-corr",
                branch_mode=ConversationBranchMode.SPAWN,
            ),
            ConversationState(
                conversation_id="leaf",
                x_correlation_id="leaf-corr",
                next_turn_index=0,
                agent_depth=2,
                parent_correlation_id="mid-corr",
                root_correlation_id="root-corr",
                branch_mode=ConversationBranchMode.SPAWN,
            ),
        )
    )

    # Both descendants keyed under the depth-0 root, not mid-corr.
    assert registry._trees["root-corr"].outstanding == 2
    assert "mid-corr" not in registry._trees
    assert orch._child_root["leaf-corr"] == "root-corr"
    assert orch._descendant_counts["mid-corr"] == 1  # per-parent drain key
    assert orch._descendant_counts["root-corr"] == 1

    await orch.on_child_leaf_reached("leaf-corr")
    assert registry._trees["root-corr"].outstanding == 1
    assert _held(cm) == 1

    await orch.on_child_leaf_reached("mid-corr")
    registry.on_root_terminal("root-corr")
    assert _held(cm) == 0
    assert drained == ["root-corr"]
