# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MooncakeTrace messages field validation."""

import pytest
from pydantic import ValidationError

from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.plugin.enums import CustomDatasetType


class TestMooncakeMessagesValidation:
    """Test MooncakeTrace model validation for the messages field."""

    def test_valid_messages_simple(self):
        """Test that a valid messages list is accepted."""
        messages = [{"role": "user", "content": "Hello"}]
        trace = MooncakeTrace(messages=messages)
        assert trace.type == CustomDatasetType.MOONCAKE_TRACE
        assert trace.messages == messages
        assert trace.text_input is None
        assert trace.input_length is None

    def test_valid_messages_with_output_length(self):
        """Test messages with output_length."""
        messages = [{"role": "user", "content": "Hello"}]
        trace = MooncakeTrace(messages=messages, output_length=50)
        assert trace.output_length == 50

    def test_valid_messages_with_timestamp(self):
        """Test messages with timestamp."""
        messages = [{"role": "user", "content": "Hello"}]
        trace = MooncakeTrace(messages=messages, timestamp=1000)
        assert trace.timestamp == 1000

    def test_valid_messages_with_delay(self):
        """Test messages with delay."""
        messages = [{"role": "user", "content": "Hello"}]
        trace = MooncakeTrace(messages=messages, delay=500)
        assert trace.delay == 500

    def test_valid_messages_multi_turn_conversation(self):
        """Test messages with a full multi-turn conversation including tool calls."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "72F sunny"},
            {"role": "assistant", "content": "It's 72F and sunny!"},
        ]  # fmt: skip
        trace = MooncakeTrace(messages=messages, output_length=50)
        assert trace.messages is not None
        assert len(trace.messages) == 5

    def test_invalid_messages_with_input_length(self):
        """Test that messages + input_length is rejected."""
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValidationError, match="mutually exclusive"):
            MooncakeTrace(messages=messages, input_length=100)

    def test_invalid_messages_with_text_input(self):
        """Test that messages + text_input is rejected."""
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValidationError, match="mutually exclusive"):
            MooncakeTrace(messages=messages, text_input="Hello")

    def test_invalid_messages_with_hash_ids(self):
        """Test that messages + hash_ids is rejected."""
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            ValidationError, match=r"hash_ids.*(not allowed|only allowed)"
        ):
            MooncakeTrace(messages=messages, hash_ids=[1, 2, 3])

    def test_invalid_messages_empty_list(self):
        """Test that an empty messages list is rejected."""
        with pytest.raises(ValidationError, match="non-empty"):
            MooncakeTrace(messages=[])

    def test_invalid_messages_missing_role(self):
        """Test that a message without 'role' is rejected."""
        with pytest.raises(ValidationError, match="role"):
            MooncakeTrace(messages=[{"content": "Hello"}])

    def test_valid_messages_without_content(self):
        """Test that a message with role but no content is valid (tool-call assistant messages)."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}]},
        ]  # fmt: skip
        trace = MooncakeTrace(messages=messages)
        assert trace.messages is not None

    def test_valid_tools_with_messages(self):
        """Test that tools are accepted when messages is provided."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ]
        trace = MooncakeTrace(messages=messages, tools=tools, output_length=50)
        assert trace.tools == tools
        assert trace.messages == messages

    def test_invalid_tools_without_messages(self):
        """Test that tools are rejected when messages is not provided."""
        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ]
        with pytest.raises(ValidationError, match="tools.*only allowed when.*messages"):
            MooncakeTrace(input_length=100, tools=tools)

    def test_invalid_tools_empty_list(self):
        """Test that an empty tools list is rejected."""
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValidationError, match="tools.*non-empty"):
            MooncakeTrace(messages=messages, tools=[])
