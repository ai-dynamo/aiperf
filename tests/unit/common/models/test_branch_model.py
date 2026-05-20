# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models.branch import ConversationBranchInfo


class TestConversationBranchInfoDefaults:
    def test_fork_default_dispatch_post(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.FORK,
        )
        assert b.dispatch_timing == "post"
        assert b.mode is ConversationBranchMode.FORK

    def test_spawn_default_dispatch_post(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
        )
        assert b.dispatch_timing == "post"

    def test_spawn_can_set_pre(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        )
        assert b.dispatch_timing == "pre"


class TestConversationBranchInfoValidator:
    def test_fork_rejects_pre(self):
        with pytest.raises(ValidationError) as exc_info:
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["c1"],
                mode=ConversationBranchMode.FORK,
                dispatch_timing="pre",
            )
        assert "SPAWN" in str(exc_info.value) or "spawn" in str(exc_info.value)

    def test_invalid_dispatch_value(self):
        with pytest.raises(ValidationError):
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["c1"],
                mode=ConversationBranchMode.SPAWN,
                dispatch_timing="bogus",
            )


class TestConversationBranchInfoSerialization:
    def test_round_trip(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1", "c2"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        )
        dumped = b.model_dump()
        restored = ConversationBranchInfo.model_validate(dumped)
        assert restored == b


class TestConversationBranchInfoSubagentType:
    def test_spawn_can_set_subagent_type(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            subagent_type="Explore",
        )
        assert b.subagent_type == "Explore"

    def test_spawn_accepts_arbitrary_recorded_string(self):
        """Free-form so corpus values like 'Codex Subagent' pass through verbatim."""
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            subagent_type="Codex Subagent",
        )
        assert b.subagent_type == "Codex Subagent"

    def test_spawn_default_subagent_type_is_none(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
        )
        assert b.subagent_type is None

    def test_fork_rejects_subagent_type(self):
        with pytest.raises(ValidationError) as exc_info:
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["c1"],
                mode=ConversationBranchMode.FORK,
                subagent_type="general-purpose",
            )
        assert "SPAWN" in str(exc_info.value) or "spawn" in str(exc_info.value)

    def test_fork_with_explicit_none_subagent_type_is_valid(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.FORK,
            subagent_type=None,
        )
        assert b.subagent_type is None

    def test_subagent_type_round_trip(self):
        b = ConversationBranchInfo(
            branch_id="root:0",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.SPAWN,
            subagent_type="Plan",
        )
        restored = ConversationBranchInfo.model_validate(b.model_dump())
        assert restored == b
        assert restored.subagent_type == "Plan"
