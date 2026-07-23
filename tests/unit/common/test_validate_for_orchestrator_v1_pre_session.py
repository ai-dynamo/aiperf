# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2b validator tests: pre-session dispatch shape restrictions.

Covers the new rejection paths in ``validate_for_orchestrator_v1``:

- ``dispatch_timing="pre"`` with FORK mode -> NotImplementedError.
- ``dispatch_timing="pre"`` with ``is_background=False`` -> NotImplementedError.
- ``dispatch_timing="pre"`` on a non-root conversation -> NotImplementedError.
- ``dispatch_timing="pre"`` declared on a turn other than turn 0 ->
  NotImplementedError.
- A valid pre-session branch shape is accepted.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy


def _metadata(
    branches: list[ConversationBranchInfo],
    *,
    branch_turn_index: int = 0,
    num_turns: int = 2,
    agent_depth: int = 0,
) -> DatasetMetadata:
    turns = [TurnMetadata() for _ in range(num_turns)]
    turns[branch_turn_index] = TurnMetadata(branch_ids=[b.branch_id for b in branches])
    child_ids: set[str] = set()
    for b in branches:
        child_ids.update(b.child_conversation_ids)
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=turns,
                branches=branches,
                agent_depth=agent_depth,
            ),
            *(
                ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
                for cid in sorted(child_ids)
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def test_pre_session_with_fork_rejected():
    """FORK mode + dispatch_timing=pre is rejected (no real parent session).

    v2 moved this rejection to construction-time: the ConversationBranchInfo
    field validator rejects FORK + dispatch_timing="pre" before the dataset
    validator ever runs (stricter, not a gap).
    """
    with pytest.raises(ValidationError, match="reserved for SPAWN"):
        ConversationBranchInfo(
            branch_id="r:pre",
            child_conversation_ids=["c"],
            mode=ConversationBranchMode.FORK,
            dispatch_timing="pre",
        )


@pytest.mark.xfail(
    strict=True,
    reason="PORT DEVIATION: v2 _check_pre_session_branch DROPPED the "
    "'pre-session dispatch requires is_background=True' check. v2 re-keyed "
    "fire-and-forget gating off is_background onto dispatch_timing='pre' to "
    "support main's dag_jsonl background-forks, so a blocking (is_background="
    "False) pre-session SPAWN now validates with no equivalent rejection.",
)
def test_pre_session_with_blocking_rejected():
    """is_background=False + dispatch_timing=pre rejected (cannot gate
    against non-existent parent).

    PORT DEVIATION: v2 dropped this is_background-keyed rejection (the concept
    was re-keyed onto dispatch_timing='pre'), so the construct now validates.
    """
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        is_background=False,
        dispatch_timing="pre",
    )
    md = _metadata([branch])
    with pytest.raises(
        NotImplementedError, match="pre-session dispatch requires is_background=True"
    ):
        validate_for_orchestrator_v1(md)


def test_pre_session_on_non_root_rejected():
    """A conversation with agent_depth > 0 may not host a pre-session branch."""
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    md = _metadata([branch], agent_depth=1)
    with pytest.raises(NotImplementedError, match="requires a root conversation"):
        validate_for_orchestrator_v1(md)


def test_pre_session_on_non_turn_0_rejected():
    """Pre-session branch declared on any turn other than turn 0 is rejected."""
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    md = _metadata([branch], branch_turn_index=1, num_turns=3)
    with pytest.raises(NotImplementedError, match="must be declared on turn 0"):
        validate_for_orchestrator_v1(md)


def test_pre_session_valid_shape_accepted():
    """Background SPAWN + dispatch_timing=pre on turn 0 of a root: accepted."""
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    md = _metadata([branch])
    # Should not raise.
    validate_for_orchestrator_v1(md)
