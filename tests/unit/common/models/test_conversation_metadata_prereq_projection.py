# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import (
    Conversation,
    Text,
    Turn,
    TurnPrerequisite,
)


class TestConversationMetadataProjectsPrereqs:
    def test_metadata_carries_prereqs(self):
        prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
        conv = Conversation(
            session_id="conv-a",
            turns=[
                Turn(texts=[Text(contents=["root"])]),
                Turn(texts=[Text(contents=["join"])], prerequisites=[prereq]),
            ],
        )
        meta = conv.metadata()
        assert meta.conversation_id == "conv-a"
        assert len(meta.turns) == 2
        assert meta.turns[0].prerequisites == []
        assert meta.turns[1].prerequisites == [prereq]

    def test_metadata_default_empty_prereqs(self):
        conv = Conversation(
            session_id="conv-b",
            turns=[Turn(texts=[Text(contents=["only"])])],
        )
        meta = conv.metadata()
        assert meta.turns[0].prerequisites == []
