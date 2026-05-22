# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import (
    ToolCallResponseData,
)
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    TextResponseData,
)
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def streaming_endpoint():
    ep_info = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)


@pytest.fixture
def endpoint():
    ep_info = create_model_endpoint(EndpointType.RESPONSES)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)


class TestFunctionCallArgumentsDelta:
    def test_delta_event_emits_tool_call_data(self, streaming_endpoint):
        event = {
            "type": "response.function_call_arguments.delta",
            "delta": '{"city":"SF"',
        }
        parsed = streaming_endpoint._parse_streaming_event(event, perf_ns=1)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == '{"city":"SF"'

    def test_delta_event_with_no_delta_returns_none(self, streaming_endpoint):
        event = {"type": "response.function_call_arguments.delta"}
        parsed = streaming_endpoint._parse_streaming_event(event, perf_ns=1)
        assert parsed is None


class TestExtractResponseContentFunctionCallWalk:
    def test_function_call_alone_emits_tool_call_data(self, endpoint):
        json_obj = {
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                }
            ],
        }
        data = endpoint._extract_response_content(json_obj)
        assert isinstance(data, ToolCallResponseData)
        assert "get_weather" in data.tool_call_text
        assert '"city":"SF"' in data.tool_call_text
        assert data.content is None

    def test_message_and_function_call_message_wins(self, endpoint):
        json_obj = {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Sure!"}],
                },
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": "{}",
                },
            ],
        }
        data = endpoint._extract_response_content(json_obj)
        assert isinstance(data, TextResponseData)
        assert data.text == "Sure!"

    def test_reasoning_wins_over_message_and_function_call(self, endpoint):
        json_obj = {
            "object": "response",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                {
                    "type": "function_call",
                    "name": "f",
                    "arguments": "{}",
                },
            ],
        }
        data = endpoint._extract_response_content(json_obj)
        assert isinstance(data, ReasoningResponseData)
        assert data.reasoning == "thinking..."
        assert data.content == "answer"


class TestInstructionsHandling:
    def test_instructions_no_longer_inserted_as_system_message(self, endpoint):
        from aiperf.common.models import Text, Turn

        turn = Turn(texts=[Text(contents=["Hi"])], model="test-model")
        ep_info = create_model_endpoint(
            EndpointType.RESPONSES,
            extra=[("instructions", "You are helpful.")],
        )
        ep = create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)
        request_info = create_request_info(model_endpoint=ep_info, turns=[turn])
        payload = ep.format_payload(request_info)
        for item in payload.get("input", []):
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    assert content != "You are helpful."
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            assert part.get("text") != "You are helpful."
