# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChatEndpoint raw_messages bypass and tools support."""

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.fixture
def model_endpoint():
    return create_model_endpoint(EndpointType.CHAT)


@pytest.fixture
def endpoint(model_endpoint):
    return ChatEndpoint(model_endpoint)


class TestFormatPayloadRawMessages:
    """Tests for raw_messages bypass in format_payload."""

    def test_uses_raw_messages_when_present(self, endpoint, model_endpoint):
        raw_msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "I'll look at the code.",
                "tool_calls": [{"id": "1", "function": {"name": "read_file"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "file contents here"},
        ]
        turn = Turn(raw_messages=raw_msgs)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_msgs
        assert payload["model"] == "test-model"

    def test_ignores_system_message_with_raw_messages(self, endpoint, model_endpoint):
        """System message in request_info should not be prepended when raw_messages is used."""
        raw_msgs = [
            {"role": "user", "content": "hello"},
        ]
        turn = Turn(raw_messages=raw_msgs)
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="This should be ignored",
            user_context_message="This too",
        )
        payload = endpoint.format_payload(request_info)

        # Should use raw_messages directly, not prepend system/user context
        assert payload["messages"] == raw_msgs
        assert len(payload["messages"]) == 1

    def test_multi_turn_concatenates_delta_messages(self, endpoint, model_endpoint):
        """Multi-turn raw_messages are concatenated from all turns (delta format)."""
        turn0 = Turn(
            raw_messages=[
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "Fix the bug."},
            ]
        )
        turn1 = Turn(
            raw_messages=[
                {"role": "assistant", "content": "I'll look at the code."},
                {"role": "tool", "tool_call_id": "1", "content": "file contents"},
            ]
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn0, turn1]
        )
        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I'll look at the code."},
            {"role": "tool", "tool_call_id": "1", "content": "file contents"},
        ]

    def test_normal_turns_unaffected(self, endpoint, model_endpoint):
        """Regular turns without raw_messages should work as before."""
        turn = Turn(texts=[Text(contents=["What is AI?"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == [
            {"role": "user", "name": "", "content": "What is AI?"}
        ]


class TestFormatPayloadTools:
    """Tests for tools in format_payload."""

    def test_includes_tools(self, endpoint, model_endpoint):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}},
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        ]
        turn = Turn(texts=[Text(contents=["Find the bug"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        request_info.tools = tools
        payload = endpoint.format_payload(request_info)

        assert payload["tools"] == tools

    def test_no_tools_when_none(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)

        assert "tools" not in payload

    def test_raw_messages_with_tools(self, endpoint, model_endpoint):
        """raw_messages and tools work together."""
        raw_msgs = [{"role": "user", "content": "Use search"}]
        tools = [{"type": "function", "function": {"name": "search"}}]
        turn = Turn(raw_messages=raw_msgs)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        request_info.tools = tools
        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_msgs
        assert payload["tools"] == tools
