# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker discard_assistant_response behavior."""

import pytest

from aiperf.common.models import Conversation, Text, Turn
from aiperf.workers.session_manager import UserSession


class TestDiscardAssistantResponse:
    """Test that discard_assistant_response on Conversation controls whether responses are stored."""

    @pytest.fixture
    def normal_session(self) -> UserSession:
        """Session with normal conversation (responses should be stored)."""
        conv = Conversation(
            session_id="normal",
            turns=[Turn(texts=[Text(contents=["hello"])])],
        )
        return UserSession(
            x_correlation_id="corr-1",
            num_turns=1,
            conversation=conv,
        )

    @pytest.fixture
    def discard_session(self) -> UserSession:
        """Session with discard_assistant_response=True (responses should not be stored)."""
        conv = Conversation(
            session_id="agentic-coding",
            turns=[Turn(raw_messages=[{"role": "user", "content": "hi"}])],
            discard_assistant_response=True,
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        return UserSession(
            x_correlation_id="corr-2",
            num_turns=1,
            conversation=conv,
        )

    def test_normal_conversation_stores_response(self, normal_session):
        """Normal conversation stores assistant response in turn_list."""
        resp_turn = Turn(role="assistant", texts=[Text(contents=["Hi there!"])])

        # Simulate the worker logic: store if not discard
        if not normal_session.conversation.discard_assistant_response:
            normal_session.store_response(resp_turn)

        assert len(normal_session.turn_list) == 1
        assert normal_session.turn_list[0].role == "assistant"

    def test_discard_assistant_response_skips_store(self, discard_session):
        """With discard_assistant_response=True, response is not stored."""
        resp_turn = Turn(
            role="assistant", texts=[Text(contents=["I'll look at the code."])]
        )

        # Simulate the worker logic: store if not discard
        if not discard_session.conversation.discard_assistant_response:
            discard_session.store_response(resp_turn)

        assert len(discard_session.turn_list) == 0

    def test_discard_session_has_tools(self, discard_session):
        """Verify tools are accessible on the conversation."""
        assert discard_session.conversation.tools is not None
        assert len(discard_session.conversation.tools) == 1
        assert discard_session.conversation.tools[0]["type"] == "function"
