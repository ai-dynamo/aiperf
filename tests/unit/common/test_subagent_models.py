# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import Conversation, Turn


def test_conversation_branch_info_defaults():
    s = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["a", "b"],
        mode=ConversationBranchMode.FORK,
    )
    assert s.is_background is False


def test_conversation_carries_subagent_spawns():
    c = Conversation(
        session_id="s",
        turns=[Turn(raw_payload={"messages": []})],
        branches=[
            ConversationBranchInfo(
                branch_id="s:0",
                child_conversation_ids=["x"],
                mode=ConversationBranchMode.FORK,
            )
        ],
    )
    assert c.branches[0].branch_id == "s:0"


def test_turn_carries_spawn_ids():
    t = Turn(raw_payload={"messages": []}, branch_ids=["s:0"])
    assert t.branch_ids == ["s:0"]


def test_metadata_projection_copies_dag_fields():
    c = Conversation(
        session_id="root",
        turns=[Turn(raw_payload={"messages": []}, branch_ids=["root:0"])],
        branches=[
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["a"],
                mode=ConversationBranchMode.FORK,
            )
        ],
    )
    meta = c.metadata()
    assert meta.branches[0].branch_id == "root:0"
    assert meta.turns[0].branch_ids == ["root:0"]


def test_conversation_dag_field_defaults():
    c = Conversation(session_id="s", turns=[Turn(raw_payload={"messages": []})])
    assert c.agent_depth == 0
    assert c.subagent_type is None
    assert c.parent_conversation_id is None


def test_metadata_projection_propagates_new_dag_fields():
    from aiperf.common.enums.enums import SubagentType

    c = Conversation(
        session_id="child",
        turns=[Turn(raw_payload={"messages": []})],
        agent_depth=2,
        subagent_type=SubagentType.EXPLORE,
        parent_conversation_id="parent",
    )
    meta = c.metadata()
    assert meta.agent_depth == 2
    assert meta.subagent_type == SubagentType.EXPLORE
    assert meta.parent_conversation_id == "parent"


def test_subagent_type_enum_is_case_insensitive():
    from aiperf.common.enums.enums import SubagentType

    assert SubagentType("explore") == SubagentType.EXPLORE
    assert SubagentType("GENERAL") == SubagentType.GENERAL
    assert SubagentType("Plan") == SubagentType.PLAN


def test_conversation_branch_mode_is_case_insensitive():
    assert ConversationBranchMode("fork") == ConversationBranchMode.FORK
    assert ConversationBranchMode("SPAWN") == ConversationBranchMode.SPAWN
    assert ConversationBranchMode("Fork") == ConversationBranchMode.FORK


def test_branch_info_allows_background_on_fork():
    s = ConversationBranchInfo(
        branch_id="x:0",
        child_conversation_ids=["y"],
        mode=ConversationBranchMode.FORK,
        is_background=True,
    )
    assert s.is_background is True


def test_branch_info_rejects_subagent_type_on_fork():
    import pytest

    from aiperf.common.enums.enums import SubagentType

    with pytest.raises(ValueError, match="subagent_type"):
        ConversationBranchInfo(
            branch_id="x:0",
            child_conversation_ids=["y"],
            mode=ConversationBranchMode.FORK,
            subagent_type=SubagentType.EXPLORE,
        )


def test_branch_info_allows_background_on_spawn():
    s = ConversationBranchInfo(
        branch_id="x:0",
        child_conversation_ids=["y"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
    )
    assert s.is_background is True


def test_branch_info_allows_subagent_type_on_spawn():
    from aiperf.common.enums.enums import SubagentType

    s = ConversationBranchInfo(
        branch_id="x:0",
        child_conversation_ids=["y"],
        mode=ConversationBranchMode.SPAWN,
        subagent_type=SubagentType.EXPLORE,
    )
    assert s.subagent_type == SubagentType.EXPLORE


def test_conversation_branch_info_has_no_join_turn_index():
    from aiperf.common.enums import ConversationBranchMode
    from aiperf.common.models import ConversationBranchInfo

    s = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    assert not hasattr(s, "join_turn_index")
