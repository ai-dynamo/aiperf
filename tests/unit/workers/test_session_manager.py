# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for UserSessionManager to ensure Credit.num_turns is respected.

These tests ensure that the worker properly uses Credit.num_turns instead of
len(conversation.turns), which is critical for ramp-up users who start mid-session.
"""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.dataset_models import DatasetMetadata
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.workers.session_manager import (
    ContentSession,
    RawPayloadSession,
    UserSession,
    UserSessionManager,
)


@pytest.fixture
def session_manager():
    """Create a UserSessionManager instance."""
    return UserSessionManager()


@pytest.fixture
def sample_conversation():
    """Create a sample conversation with 5 turns."""
    return Conversation(
        conversation_id="test-conv",
        turns=[
            Turn(messages=[{"role": "user", "content": f"Question {i + 1}"}])
            for i in range(5)
        ],
    )


class TestUserSessionManager:
    """Tests for UserSessionManager Credit.num_turns handling."""

    def test_create_session_uses_credit_num_turns_not_conversation_length(
        self, session_manager, sample_conversation
    ):
        """Ensure UserSession.num_turns comes from Credit, not conversation.

        This is critical for ramp-up users who may only execute 1 turn even though
        the conversation template has 5 turns available.
        """
        # Conversation has 5 turns, but Credit says only do 1
        session = session_manager.create_content_session(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=1,  # Artificial cap from Credit
        )

        # UserSession should use Credit.num_turns (1), not len(conversation.turns) (5)
        assert session.num_turns == 1
        assert len(session.conversation.turns) == 5  # Conversation still has all turns

    def test_advance_turn_validates_against_credit_num_turns(
        self, session_manager, sample_conversation
    ):
        """Ensure turn validation uses Credit.num_turns."""
        session = session_manager.create_content_session(
            x_correlation_id="test-corr-id",
            conversation=sample_conversation,
            num_turns=2,  # Only 2 turns allowed
        )

        # Should be able to advance to turn 0 and 1
        session.advance_turn(0)
        assert session.turn_index == 0

        session.advance_turn(1)
        assert session.turn_index == 1

        # Should reject turn 2 (out of range for num_turns=2)
        with pytest.raises(
            ValueError,
            match="Turn index 2 is out of range for conversation with 2 turns",
        ):
            session.advance_turn(2)

    def test_ramp_up_user_single_turn_scenario(
        self, session_manager, sample_conversation
    ):
        """Test ramp-up user who only executes 1 turn (e.g., User 1 starting at Turn 5).

        This simulates multi-round-qa's ramp-up behavior where some users are
        initialized mid-session and only complete their final turn.
        """
        # User 1 in ramp-up: starts at question_id=5, only does 1 turn
        session = session_manager.create_content_session(
            x_correlation_id="ramp-up-user-1",
            conversation=sample_conversation,
            num_turns=1,  # Only 1 turn to execute
        )

        # Advance to turn 0 (their only turn)
        turn = session.advance_turn(0)

        # Should access first turn of conversation (conversation has all 5 turns available)
        assert turn.messages[0]["content"] == "Question 1"

        # After turn 0, is_final_turn should be True (0 == 1-1)
        # This would be determined by Credit.is_final_turn, which we validate here
        assert session.turn_index == 0
        assert session.num_turns == 1
        # Credit.is_final_turn would be: turn_index (0) == num_turns (1) - 1 → True

    def test_full_session_uses_all_conversation_turns(
        self, session_manager, sample_conversation
    ):
        """Test normal user who executes all turns (e.g., steady-state users)."""
        session = session_manager.create_content_session(
            x_correlation_id="full-session-user",
            conversation=sample_conversation,
            num_turns=5,  # All turns
        )

        assert session.num_turns == 5

        # Should be able to advance through all 5 turns
        for turn_idx in range(5):
            turn = session.advance_turn(turn_idx)
            assert turn.messages[0]["content"] == f"Question {turn_idx + 1}"

    def test_partial_session_mid_conversation(
        self, session_manager, sample_conversation
    ):
        """Test user who starts mid-session and does partial turns (e.g., User 4 doing 3 turns)."""
        session = session_manager.create_content_session(
            x_correlation_id="partial-user",
            conversation=sample_conversation,
            num_turns=3,  # Only 3 turns (simulating User 4 at question_id=3)
        )

        assert session.num_turns == 3

        # Can advance turns 0, 1, 2
        for turn_idx in range(3):
            turn = session.advance_turn(turn_idx)
            assert turn is not None

        # Turn 3 should fail (out of range)
        with pytest.raises(ValueError, match="out of range"):
            session.advance_turn(3)

    def test_url_index_stored_for_multi_url_load_balancing(
        self, session_manager, sample_conversation
    ):
        """Test that url_index is stored in session for multi-URL load balancing.

        When using multiple --url endpoints with multi-turn conversations, the first
        turn gets a url_index from the round-robin sampler. All subsequent turns must
        use the same url_index to ensure the entire conversation hits the same backend.
        """
        # First turn: Credit provides url_index=2 from round-robin
        session = session_manager.create_content_session(
            x_correlation_id="multi-url-session",
            conversation=sample_conversation,
            num_turns=3,
            url_index=2,  # From Credit on first turn
        )

        # Session stores the url_index for subsequent turns
        assert session.url_index == 2

        # All turns should use this stored url_index (worker reads from session)
        for turn_idx in range(3):
            session.advance_turn(turn_idx)
            # Worker would use session.url_index (2) for every turn
            assert session.url_index == 2

    def test_url_index_none_for_single_url_mode(
        self, session_manager, sample_conversation
    ):
        """Test that url_index can be None when only one URL is configured."""
        session = session_manager.create_content_session(
            x_correlation_id="single-url-session",
            conversation=sample_conversation,
            num_turns=2,
            url_index=None,  # No multi-URL load balancing
        )

        assert session.url_index is None


# ============================================================
# Fixtures for context mode tests
# ============================================================


def _make_session(
    context_mode: ConversationContextMode | None = None,
    num_turns: int = 3,
    default_context_mode: ConversationContextMode | None = None,
) -> UserSession:
    """Create a UserSession with the given context_mode on its conversation."""
    conversation = Conversation(
        conversation_id="ctx-conv",
        context_mode=context_mode,
        turns=[
            Turn(messages=[{"role": "user", "content": f"Q{i}"}])
            for i in range(num_turns)
        ],
    )
    mgr = UserSessionManager()
    mgr.set_default_context_mode(default_context_mode)
    return mgr.create_content_session(
        x_correlation_id="ctx-test",
        conversation=conversation,
        num_turns=num_turns,
    )


# ============================================================
# Context Mode Resolution
# ============================================================


class TestUserSessionContextModeResolution:
    """Verify context_mode resolves: conversation > dataset default > DELTAS_WITHOUT_RESPONSES."""

    @pytest.mark.parametrize(
        "conversation_mode,expected",
        [
            (None, ConversationContextMode.DELTAS_WITHOUT_RESPONSES),
            (ConversationContextMode.DELTAS_WITHOUT_RESPONSES, ConversationContextMode.DELTAS_WITHOUT_RESPONSES),
            (ConversationContextMode.DELTAS_WITH_RESPONSES, ConversationContextMode.DELTAS_WITH_RESPONSES),
            (ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES, ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES),
        ],
    )  # fmt: skip
    def test_context_mode_resolves_correctly(
        self,
        conversation_mode: ConversationContextMode | None,
        expected: ConversationContextMode,
    ) -> None:
        session = _make_session(context_mode=conversation_mode)
        assert session.context_mode == expected

    def test_dataset_default_used_when_conversation_has_none(self) -> None:
        session = _make_session(
            context_mode=None,
            default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        )
        assert (
            session.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )

    def test_conversation_overrides_dataset_default(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        )
        assert session.context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES

    def test_global_default_when_both_none(self) -> None:
        session = _make_session(context_mode=None, default_context_mode=None)
        assert session.context_mode == ConversationContextMode.DELTAS_WITHOUT_RESPONSES


# ============================================================
# should_store_response
# ============================================================


class TestUserSessionShouldStoreResponse:
    """Verify should_store_response gates on context mode."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (ConversationContextMode.DELTAS_WITHOUT_RESPONSES, True),
            (ConversationContextMode.DELTAS_WITH_RESPONSES, False),
            (ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES, False),
            param(None, True, id="default-deltas-without-responses"),
        ],
    )  # fmt: skip
    def test_should_store_response_per_mode(
        self, mode: ConversationContextMode | None, expected: bool
    ) -> None:
        session = _make_session(context_mode=mode)
        assert session.should_store_response() is expected


# ============================================================
# turn_list with context mode
# ============================================================


class TestUserSessionTurnList:
    """Verify turn_list contains correct turns based on context mode."""

    def test_deltas_without_responses_returns_full_history(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        )
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3  # Q0, A0, Q1
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "A0"
        assert turns[2].messages[0]["content"] == "Q1"

    def test_deltas_with_responses_returns_dataset_turns_only(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES
        )
        session.advance_turn(0)
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2  # Q0, Q1 (no assistant responses stored)
        assert turns[0].messages[0]["content"] == "Q0"
        assert turns[1].messages[0]["content"] == "Q1"

    def test_message_array_returns_only_last(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
        )
        session.advance_turn(0)
        session.advance_turn(1)
        session.advance_turn(2)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q2"

    def test_message_array_single_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            num_turns=1,
        )
        session.advance_turn(0)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q0"

    def test_default_mode_returns_full_history(self) -> None:
        session = _make_session(context_mode=None)
        session.advance_turn(0)
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 3


# ============================================================
# Integration: context mode + should_store_response together
# ============================================================


class TestUserSessionContextModeWorkflow:
    """Verify the full workflow of context mode with store_response gating."""

    def test_deltas_without_responses_stores_responses_and_sends_full_history(
        self,
    ) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITHOUT_RESPONSES, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is True
        session.store_response(Turn(messages=[{"role": "assistant", "content": "A0"}]))
        session.advance_turn(1)

        assert len(session.turn_list) == 3

    def test_deltas_with_responses_skips_live_responses_sends_dataset_turns(
        self,
    ) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES, num_turns=2
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        # Worker would NOT call store_response based on should_store_response()
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 2
        assert all(t.messages[0]["role"] == "user" for t in turns)

    def test_message_array_skips_responses_sends_only_current_turn(self) -> None:
        session = _make_session(
            context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
            num_turns=2,
        )
        session.advance_turn(0)
        assert session.should_store_response() is False
        session.advance_turn(1)

        turns = session.turn_list
        assert len(turns) == 1
        assert turns[0].messages[0]["content"] == "Q1"


# ============================================================
# message_array_without_responses rejected
# ============================================================


class TestMessageArrayWithoutResponsesRejected:
    """MESSAGE_ARRAY_WITHOUT_RESPONSES is reserved and must be rejected early."""

    def test_conversation_rejects_unsupported_mode(self) -> None:
        with pytest.raises(ValidationError, match="not yet supported"):
            Conversation(
                context_mode=ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES,
            )

    def test_dataset_metadata_rejects_unsupported_default_mode(self) -> None:
        with pytest.raises(ValidationError, match="not yet supported"):
            DatasetMetadata(
                sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
                default_context_mode=ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES,
            )


# ============================================================
# RawPayloadSession
# ============================================================


class TestRawPayloadSession:
    """Tests for the PAYLOAD_BYTES-format session type.

    ``RawPayloadSession`` carries only routing fields (conversation_id +
    num_turns + turn_index). The wire payload bytes live in mmap and the
    records-process resolves them through its own client — no
    ``Conversation`` body, no ``turn_list`` to accumulate, no response
    storage. FORK + PAYLOAD_BYTES is refused upstream, so fork-related
    behaviors are intentionally absent.
    """

    def test_create_raw_payload_session_returns_raw_payload_type(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_raw_payload_session(
            x_correlation_id="raw-x",
            conversation_id="conv-raw",
            num_turns=3,
        )
        assert isinstance(session, RawPayloadSession)
        assert not isinstance(session, ContentSession)
        assert session.conversation_id == "conv-raw"
        assert session.num_turns == 3
        assert session.turn_index == 0

    def test_advance_turn_bumps_index_without_turn_list(
        self, session_manager: UserSessionManager
    ) -> None:
        """No Turn objects to instantiate, no list to mutate — just the index."""
        session = session_manager.create_raw_payload_session(
            x_correlation_id="raw-adv",
            conversation_id="conv-raw",
            num_turns=4,
        )
        session.advance_turn(2)
        assert session.turn_index == 2
        # RawPayloadSession deliberately exposes no ``turn_list`` attribute.
        assert not hasattr(session, "turn_list")

    def test_advance_turn_rejects_out_of_range(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_raw_payload_session(
            x_correlation_id="raw-bad",
            conversation_id="conv-raw",
            num_turns=2,
        )
        with pytest.raises(ValueError, match="out of range"):
            session.advance_turn(2)

    def test_advance_turn_rejects_negative(
        self, session_manager: UserSessionManager
    ) -> None:
        session = session_manager.create_raw_payload_session(
            x_correlation_id="raw-neg",
            conversation_id="conv-raw",
            num_turns=2,
        )
        with pytest.raises(ValueError, match="negative"):
            session.advance_turn(-1)

    def test_url_index_propagates(self, session_manager: UserSessionManager) -> None:
        session = session_manager.create_raw_payload_session(
            x_correlation_id="raw-url",
            conversation_id="conv-raw",
            num_turns=1,
            url_index=3,
        )
        assert session.url_index == 3

    def test_get_returns_stored_raw_payload_session(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.create_raw_payload_session(
            x_correlation_id="raw-get",
            conversation_id="conv-raw",
            num_turns=1,
        )
        looked_up = session_manager.get("raw-get")
        assert isinstance(looked_up, RawPayloadSession)
        assert looked_up.conversation_id == "conv-raw"

    def test_evict_removes_raw_payload_session(
        self, session_manager: UserSessionManager
    ) -> None:
        session_manager.create_raw_payload_session(
            x_correlation_id="raw-evict",
            conversation_id="conv-raw",
            num_turns=1,
        )
        session_manager.evict("raw-evict")
        assert session_manager.get("raw-evict") is None

    def test_evict_if_unpinned_removes_raw_payload_session(
        self, session_manager: UserSessionManager
    ) -> None:
        """RawPayloadSession can never be FORK-pinned, so unpinned eviction
        always succeeds immediately."""
        session_manager.create_raw_payload_session(
            x_correlation_id="raw-unpin",
            conversation_id="conv-raw",
            num_turns=1,
        )
        session_manager.evict_if_unpinned("raw-unpin")
        assert session_manager.get("raw-unpin") is None

    def test_pin_for_fork_child_rejects_raw_payload_session(
        self, session_manager: UserSessionManager
    ) -> None:
        """FORK + PAYLOAD_BYTES is refused at dataset-format selection;
        defensively reject at the pin call site too."""
        session_manager.create_raw_payload_session(
            x_correlation_id="raw-pin",
            conversation_id="conv-raw",
            num_turns=1,
        )
        with pytest.raises(TypeError, match="not a ContentSession"):
            session_manager.pin_for_fork_child("raw-pin")

    def test_seed_from_parent_noop_when_parent_is_raw_payload(
        self, session_manager: UserSessionManager
    ) -> None:
        """Seeding a child from a RawPayloadSession parent is structurally
        impossible (no turn_list to copy). It must not raise — just no-op
        — so the request still goes out without seed context."""
        # Create a raw payload "parent" and a content-shaped "child"; this
        # combination should never happen in practice (FORK + PAYLOAD_BYTES
        # is refused upstream), but seed_from_parent is best-effort.
        session_manager.create_raw_payload_session(
            x_correlation_id="raw-parent",
            conversation_id="conv-raw",
            num_turns=1,
        )
        conversation = Conversation(
            conversation_id="child-conv",
            turns=[Turn(messages=[{"role": "user", "content": "Q"}])],
        )
        child = session_manager.create_content_session(
            x_correlation_id="content-child",
            conversation=conversation,
            num_turns=1,
        )
        session_manager.seed_from_parent("content-child", "raw-parent")
        # Child's turn_list should still be empty — seed silently no-op'd.
        assert child.turn_list == []


# ============================================================
# UserSessionManager factory behaviour
# ============================================================


class TestUserSessionManagerFactories:
    """Both factories store the session in the same cache under the same key."""

    def test_create_content_session_stores_by_correlation_id(
        self, session_manager: UserSessionManager
    ) -> None:
        conversation = Conversation(
            conversation_id="c-1",
            turns=[Turn(messages=[{"role": "user", "content": "Q"}])],
        )
        created = session_manager.create_content_session(
            x_correlation_id="corr-content",
            conversation=conversation,
            num_turns=1,
        )
        assert isinstance(created, ContentSession)
        assert session_manager.get("corr-content") is created

    def test_create_raw_payload_session_stores_by_correlation_id(
        self, session_manager: UserSessionManager
    ) -> None:
        created = session_manager.create_raw_payload_session(
            x_correlation_id="corr-raw",
            conversation_id="conv-raw",
            num_turns=2,
        )
        assert isinstance(created, RawPayloadSession)
        assert session_manager.get("corr-raw") is created
