# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestRecord,
    TextResponse,
    TextResponseData,
    Turn,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class _StubEndpoint(BaseEndpoint):
    """Minimum viable concrete BaseEndpoint subclass for assistant-turn tests.

    Bypasses ``BaseEndpoint.__init__`` (no ModelEndpointInfo needed) and
    treats each ``TextResponse.text`` as a token of pre-canned parsed-data
    selector via the ``_canned`` map keyed by ``perf_ns``.
    """

    def __init__(self, canned: dict[int, ParsedResponse | None] | None = None):
        self._canned = canned or {}

    def format_payload(self, request_info):
        raise NotImplementedError

    def parse_response(self, response: InferenceServerResponse):
        return self._canned.get(response.perf_ns)


def _text_response(perf_ns: int) -> TextResponse:
    """Build a real TextResponse so RequestRecord's pydantic validator accepts it."""
    return TextResponse(perf_ns=perf_ns, text="{}", content_type="application/json")


class TestBuildAssistantTurnDefault:
    def test_text_only_record(self):
        canned = {
            10: ParsedResponse(perf_ns=10, data=TextResponseData(text="Hello")),
            20: ParsedResponse(perf_ns=20, data=TextResponseData(text=", world")),
        }
        ep = _StubEndpoint(canned)
        record = RequestRecord(
            responses=[_text_response(10), _text_response(20)],
            start_perf_ns=0,
            end_perf_ns=20,
        )
        turn = ep.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        assert len(turn.texts) == 1
        assert turn.texts[0].contents == ["Hello, world"]

    def test_reasoning_drops_reasoning_keeps_content(self):
        canned = {
            10: ParsedResponse(
                perf_ns=10,
                data=ReasoningResponseData(
                    content="visible answer",
                    reasoning="hidden chain of thought",
                ),
            ),
        }
        ep = _StubEndpoint(canned)
        record = RequestRecord(
            responses=[_text_response(10)],
            start_perf_ns=0,
            end_perf_ns=10,
        )
        turn = ep.build_assistant_turn(record)
        assert turn is not None
        assert turn.texts[0].contents == ["visible answer"]

    def test_reasoning_with_only_reasoning_falls_back_to_reasoning_text(self):
        # Some servers (and the mock) return all output as ``reasoning``
        # with empty ``content``; without this fallback FORK-mode DAG
        # children would inherit a parent context with no captured
        # assistant turn.
        canned = {
            10: ParsedResponse(
                perf_ns=10,
                data=ReasoningResponseData(
                    content=None,
                    reasoning="Let me think about this...",
                ),
            ),
        }
        ep = _StubEndpoint(canned)
        record = RequestRecord(
            responses=[_text_response(10)],
            start_perf_ns=0,
            end_perf_ns=10,
        )
        turn = ep.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        assert turn.texts[0].contents == ["Let me think about this..."]

    def test_mixed_reasoning_and_text_combines_content(self):
        canned = {
            10: ParsedResponse(perf_ns=10, data=TextResponseData(text="Hello")),
            20: ParsedResponse(
                perf_ns=20,
                data=ReasoningResponseData(reasoning="Thinking...", content="World"),
            ),
        }
        ep = _StubEndpoint(canned)
        record = RequestRecord(
            responses=[_text_response(10), _text_response(20)],
            start_perf_ns=0,
            end_perf_ns=20,
        )
        turn = ep.build_assistant_turn(record)
        assert turn is not None
        assert turn.texts[0].contents == ["HelloWorld"]

    def test_empty_record_returns_none(self):
        ep = _StubEndpoint()
        record = RequestRecord(
            responses=[],
            start_perf_ns=0,
            end_perf_ns=0,
        )
        assert ep.build_assistant_turn(record) is None

    def test_responses_with_no_data_return_none(self):
        canned = {10: ParsedResponse(perf_ns=10, data=None)}
        ep = _StubEndpoint(canned)
        record = RequestRecord(
            responses=[_text_response(10)],
            start_perf_ns=0,
            end_perf_ns=10,
        )
        assert ep.build_assistant_turn(record) is None

    def test_unparseable_responses_return_none(self):
        # parse_response returns None for every response - record is unusable.
        ep = _StubEndpoint(canned={})
        record = RequestRecord(
            responses=[_text_response(10), _text_response(20)],
            start_perf_ns=0,
            end_perf_ns=20,
        )
        assert ep.build_assistant_turn(record) is None


class TestBuildMessagesSkeleton:
    @pytest.fixture
    def endpoint(self):
        return _StubEndpoint()

    def test_empty_turns_returns_empty_list(self, endpoint):
        assert endpoint.build_messages([]) == []

    def test_raw_messages_spliced_verbatim(self, endpoint):
        raw = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "ping"},
        ]
        turn = Turn(role="user", raw_messages=raw)
        out = endpoint.build_messages([turn])
        assert out == raw

    def test_single_text_turn_renders_string_content(self, endpoint):
        from aiperf.common.models import Text

        turn = Turn(role="user", texts=[Text(contents=["hello"])])
        out = endpoint.build_messages([turn])
        assert out == [{"role": "user", "content": "hello"}]

    def test_default_role_used_when_turn_role_none(self, endpoint):
        from aiperf.common.models import Text

        turn = Turn(role=None, texts=[Text(contents=["hi"])])
        out = endpoint.build_messages([turn])
        assert out[0]["role"] == "user"

    def test_multi_text_turn_renders_parts(self, endpoint):
        from aiperf.common.models import Text

        turn = Turn(role="user", texts=[Text(contents=["a", "b"])])
        out = endpoint.build_messages([turn])
        assert out[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]


class TestExtractPayloadInputs:
    @pytest.fixture
    def endpoint(self):
        return _StubEndpoint()

    def test_chat_messages_string_content(self, endpoint):
        payload = {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hello"},
            ]
        }
        out = endpoint.extract_payload_inputs(payload)
        assert out.texts == ["be helpful", "hello"]
        assert out.image_count == 0
        assert out.audio_count == 0
        assert out.video_count == 0

    def test_chat_messages_content_parts_count_modalities(self, endpoint):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "x", "format": "wav"},
                        },
                        {"type": "video_url", "video_url": {"url": "https://v"}},
                    ],
                },
            ]
        }
        out = endpoint.extract_payload_inputs(payload)
        assert out.texts == ["describe this"]
        assert out.image_count == 1
        assert out.audio_count == 1
        assert out.video_count == 1

    def test_completions_prompt_string(self, endpoint):
        out = endpoint.extract_payload_inputs({"prompt": "hello world"})
        assert out.texts == ["hello world"]

    def test_completions_prompt_list(self, endpoint):
        out = endpoint.extract_payload_inputs({"prompt": ["a", "b"]})
        assert out.texts == ["a", "b"]

    def test_embeddings_input_string(self, endpoint):
        out = endpoint.extract_payload_inputs({"input": "embed me"})
        assert out.texts == ["embed me"]

    def test_embeddings_input_list(self, endpoint):
        out = endpoint.extract_payload_inputs({"input": ["a", "b", "c"]})
        assert out.texts == ["a", "b", "c"]

    def test_rankings_query_passages(self, endpoint):
        payload = {
            "query": "what",
            "passages": ["doc one", {"text": "doc two"}],
        }
        out = endpoint.extract_payload_inputs(payload)
        assert out.texts == ["what", "doc one", "doc two"]

    def test_huggingface_inputs(self, endpoint):
        out = endpoint.extract_payload_inputs({"inputs": "hf prompt"})
        assert out.texts == ["hf prompt"]

    def test_chat_assistant_tool_calls_collected(self, endpoint):
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    ],
                }
            ]
        }
        out = endpoint.extract_payload_inputs(payload)
        assert "lookup" in out.texts
        assert '{"q":"x"}' in out.texts

    def test_tools_schema_collected(self, endpoint):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "weather lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        out = endpoint.extract_payload_inputs(payload)
        assert "hi" in out.texts
        assert "get_weather" in out.texts
        assert "weather lookup" in out.texts
        # parameters serialised
        assert any('"type"' in t and '"object"' in t for t in out.texts)
