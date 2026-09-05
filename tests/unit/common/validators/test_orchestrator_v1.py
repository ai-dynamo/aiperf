# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for orchestrator synthesized-turn branch acceptance in
``validate_for_orchestrator_v1``.

A request-less orchestrator conversation has ONE synthesized ``no_request``
turn carrying a post-timing SPAWN branch that fans out to its children. The
validator must accept this shape through the normal branch path while still
enforcing that spawned children resolve to real conversations.

Builds real ``ConversationMetadata`` (never ``MagicMock``): a MagicMock
auto-creates whatever attribute path is read and would silently hide a
validation regression this suite guards.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy


def _dataset(*convs: ConversationMetadata) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=list(convs),
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _post_spawn(branch_id: str, children: list[str]) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=children,
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="post",
    )


def _build_orchestrator_metadata(
    session_id: str, children: list[str]
) -> ConversationMetadata:
    """A request-less orchestrator: ONE synthesized ``no_request`` turn
    declaring a post-timing SPAWN branch that fans out to its children."""
    branch_id = f"{session_id}:0"
    return ConversationMetadata(
        conversation_id=session_id,
        turns=[TurnMetadata(branch_ids=[branch_id], prerequisites=[], no_request=True)],
        branches=[_post_spawn(branch_id, children)],
        is_root=True,
        agent_depth=0,
        is_orchestrator=True,
    )


def _build_child_metadata(session_id: str) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=session_id,
        turns=[TurnMetadata(branch_ids=[], prerequisites=[])],
        is_root=False,
        agent_depth=1,
    )


def test_orchestrator_synthesized_branch_passes_validation():
    """An orchestrator whose synthesized turn declares a post-timing SPAWN
    branch must validate cleanly through the normal branch path."""
    meta = _build_orchestrator_metadata(session_id="start", children=["fan-out-a"])
    child = _build_child_metadata(session_id="fan-out-a")
    validate_for_orchestrator_v1(_dataset(meta, child))


def test_orchestrator_spawn_child_must_exist():
    """Child existence is still enforced for orchestrators: a spawn naming a
    conversation absent from the dataset must reject."""
    meta = _build_orchestrator_metadata(session_id="start", children=["missing"])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(meta))
    msg = str(exc.value)
    assert "start" in msg
    assert "missing" in msg
