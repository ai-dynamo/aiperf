# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import (
    Conversation,
    Text,
    Turn,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook


def _roundtrip(obj, cls):
    return msgspec.convert(
        msgspec.to_builtins(obj, enc_hook=_msgspec_enc_hook),
        cls,
        dec_hook=_msgspec_dec_hook,
    )


class TestTurnPrerequisitesField:
    def test_default_is_empty_list(self):
        t = Turn(texts=[Text(contents=["hi"])])
        assert t.prerequisites == []

    def test_round_trip_with_prereqs(self):
        prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
        t = Turn(texts=[Text(contents=["hi"])], prerequisites=[prereq])
        restored = _roundtrip(t, Turn)
        assert restored.prerequisites == [prereq]


class TestTurnMetadataHasForks:
    def test_default_false(self):
        m = TurnMetadata()
        assert m.has_forks is False

    def test_set_true(self):
        m = TurnMetadata(has_forks=True)
        assert m.has_forks is True

    def test_round_trip(self):
        m = TurnMetadata(has_forks=True, timestamp_ms=1000.0)
        restored = _roundtrip(m, TurnMetadata)
        assert restored.has_forks is True
        assert restored.timestamp_ms == 1000.0


class TestConversationAgentDepth:
    def test_default_zero(self):
        c = Conversation(session_id="s1", turns=[Turn(texts=[Text(contents=["hi"])])])
        assert c.agent_depth == 0

    def test_set_depth(self):
        c = Conversation(
            session_id="s1",
            turns=[Turn(texts=[Text(contents=["hi"])])],
            agent_depth=2,
        )
        assert c.agent_depth == 2

    def test_round_trip(self):
        c = Conversation(
            session_id="s1",
            turns=[Turn(texts=[Text(contents=["hi"])])],
            agent_depth=3,
        )
        restored = _roundtrip(c, Conversation)
        assert restored.agent_depth == 3
