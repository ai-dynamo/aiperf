# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.models import (
    RequestRecord,
    TextResponse,
)
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    TextResponseData,
    ToolCallResponseData,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


@pytest.fixture
def chat_endpoint():
    ep_info = create_model_endpoint(EndpointType.CHAT, streaming=True)
    return create_endpoint_with_mock_transport(ChatEndpoint, ep_info)


class TestParseChunkContentAndToolCalls:
    def test_chunk_with_content_only(self, chat_endpoint):
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "Hello"}}],
        }
        data = chat_endpoint.extract_chat_response_data(json_obj)
        assert isinstance(data, TextResponseData)
        assert data.text == "Hello"

    def test_chunk_with_tool_calls_only(self, chat_endpoint):
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"SF"}',
                                },
                            }
                        ]
                    }
                }
            ],
        }
        data = chat_endpoint.extract_chat_response_data(json_obj)
        assert isinstance(data, ToolCallResponseData)
        assert "get_weather" in data.tool_call_text
        assert '"city":"SF"' in data.tool_call_text
        assert data.content is None

    def test_chunk_with_both_content_and_tool_calls(self, chat_endpoint):
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {
                        "content": "Let me check. ",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"SF"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        data = chat_endpoint.extract_chat_response_data(json_obj)
        assert isinstance(data, ToolCallResponseData)
        assert data.content == "Let me check. "
        assert "get_weather" in data.tool_call_text
        assert data.get_text() == "Let me check. " + "get_weather" + '{"city":"SF"}'

    def test_chunk_with_reasoning_wins(self, chat_endpoint):
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {
                        "content": "answer",
                        "reasoning": "thinking...",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
        }
        data = chat_endpoint.extract_chat_response_data(json_obj)
        assert isinstance(data, ReasoningResponseData)
        assert data.reasoning == "thinking..."
        assert data.content == "answer"


class TestBuildAssistantTurnReassemblesToolCallDeltas:
    def test_streaming_tool_calls_index_keyed_concat(self, chat_endpoint):
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"city":'},
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"SF"}'},
                                }
                            ]
                        }
                    }
                ],
            },
        ]
        responses = [
            TextResponse(perf_ns=i, text=orjson.dumps(c).decode())
            for i, c in enumerate(chunks)
        ]
        record = RequestRecord(responses=responses)
        turn = chat_endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        assert turn.raw_messages is not None
        assert len(turn.raw_messages) == 1
        msg = turn.raw_messages[0]
        assert msg["role"] == "assistant"
        tool_calls = msg.get("tool_calls")
        assert tool_calls and len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city":"SF"}'


def _reasoning_reply_chunks() -> list[dict]:
    """Streaming SSE chunks for one reasoning-model reply (vLLM/SGLang shape)."""
    deltas = [
        {"role": "assistant", "content": ""},
        {"reasoning_content": "I should walk "},
        {"reasoning_content": "the list."},
        {"content": "def rev(head): "},
        {"content": "..."},
    ]
    return [
        {"object": "chat.completion.chunk", "choices": [{"delta": d}]} for d in deltas
    ]


def _record_from(json_objs: list[dict]) -> RequestRecord:
    return RequestRecord(
        responses=[
            TextResponse(perf_ns=i, text=orjson.dumps(o).decode())
            for i, o in enumerate(json_objs)
        ]
    )


class TestBuildAssistantTurnReasoningReplay:
    def test_streaming_reasoning_reply_replays_answer_only(self, chat_endpoint):
        turn = chat_endpoint.build_assistant_turn(
            _record_from(_reasoning_reply_chunks())
        )
        assert turn is not None
        assert turn.texts[0].contents == ["def rev(head): ..."]

    def test_streaming_reasoning_without_answer_replays_reasoning(self, chat_endpoint):
        chunks = [
            c
            for c in _reasoning_reply_chunks()
            if "content" not in c["choices"][0]["delta"]
        ]
        turn = chat_endpoint.build_assistant_turn(_record_from(chunks))
        assert turn is not None
        assert turn.texts[0].contents == ["I should walk the list."]

    def test_streaming_and_non_streaming_replay_identical_turn(self, chat_endpoint):
        non_streaming = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "I should walk the list.",
                        "content": "def rev(head): ...",
                    }
                }
            ],
        }
        streamed = chat_endpoint.build_assistant_turn(
            _record_from(_reasoning_reply_chunks())
        )
        whole = chat_endpoint.build_assistant_turn(_record_from([non_streaming]))
        assert streamed is not None and whole is not None
        assert streamed.model_dump() == whole.model_dump()
