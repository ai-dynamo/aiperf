# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import (
    Conversation,
    Text,
    Turn,
    TurnMetadata,
    TurnPrerequisite,
)


class TestTurnPrerequisitesField:
    def test_default_is_empty_list(self):
        t = Turn(texts=[Text(contents=["hi"])])
        assert t.prerequisites == []

    def test_round_trip_with_prereqs(self):
        prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
        t = Turn(texts=[Text(contents=["hi"])], prerequisites=[prereq])
        dumped = t.model_dump()
        restored = Turn.model_validate(dumped)
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
        dumped = m.model_dump()
        restored = TurnMetadata.model_validate(dumped)
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
        dumped = c.model_dump()
        restored = Conversation.model_validate(dumped)
        assert restored.agent_depth == 3


class TestTurnMetadataProjectsMaxTokensAndAudioDuration:
    """Turn.metadata() projects ``max_tokens`` and ``audio_duration_seconds``.

    These fields land on ``TurnMetadata`` so the records pipeline can read
    per-turn caps and ASR audio duration without holding the full Turn list.
    """

    def test_max_tokens_projected_into_metadata(self):
        t = Turn(role="user", max_tokens=5)
        assert t.metadata().max_tokens == 5

    def test_audio_duration_seconds_projected_into_metadata(self):
        t = Turn(role="user", max_tokens=5, audio_duration_seconds=1.5)
        meta = t.metadata()
        assert meta.audio_duration_seconds == 1.5
        assert meta.max_tokens == 5

    def test_both_fields_none_projects_none(self):
        t = Turn(role="user")
        meta = t.metadata()
        assert meta.max_tokens is None
        assert meta.audio_duration_seconds is None

    def test_conversation_metadata_projection_carries_max_tokens_and_audio(self):
        """``Conversation.metadata()`` mirrors the same projection for both fields."""
        c = Conversation(
            session_id="s1",
            turns=[
                Turn(role="user", max_tokens=10, audio_duration_seconds=2.25),
                Turn(role="user"),
            ],
        )
        conv_meta = c.metadata()
        assert conv_meta.turns[0].max_tokens == 10
        assert conv_meta.turns[0].audio_duration_seconds == 2.25
        assert conv_meta.turns[1].max_tokens is None
        assert conv_meta.turns[1].audio_duration_seconds is None

    def test_turn_metadata_round_trip_preserves_new_fields(self):
        m = TurnMetadata(max_tokens=7, audio_duration_seconds=0.5)
        dumped = m.model_dump()
        restored = TurnMetadata.model_validate(dumped)
        assert restored.max_tokens == 7
        assert restored.audio_duration_seconds == 0.5
