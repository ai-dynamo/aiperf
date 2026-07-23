# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``BaseEndpoint.extract_payload_inputs`` and its overrides.

Covers the single-pass walk that feeds ISL tokenisation
(``ExtractedPayload.texts``) and per-record ``MediaCounts``
(``image_count``/``audio_count``/``video_count``) from the wire-ready
JSON payload. Endpoints may extend this walk by setting ``PART_TYPES``
(chat-shape content-part type names) or overriding
``extract_payload_inputs`` directly.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint


def _chat() -> ChatEndpoint:
    return ChatEndpoint(model_endpoint=create_model_endpoint(EndpointType.CHAT))


def _responses() -> ResponsesEndpoint:
    return ResponsesEndpoint(
        model_endpoint=create_model_endpoint(EndpointType.RESPONSES)
    )


def _image_retrieval() -> ImageRetrievalEndpoint:
    return ImageRetrievalEndpoint(
        model_endpoint=create_model_endpoint(EndpointType.IMAGE_RETRIEVAL)
    )


class TestChatShapeDispatch:
    """``PART_TYPES`` default dispatch for the chat-completions payload shape."""

    def test_empty_payload_yields_empty_result(self):
        result = _chat().extract_payload_inputs({})
        assert result.texts == []
        assert result.image_count == 0
        assert result.audio_count == 0
        assert result.video_count == 0

    def test_plain_string_content(self):
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.texts == ["Hello", "Hi there"]
        assert result.image_count == 0

    def test_part_list_text_and_image(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:abc"}},
                    ],
                },
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.texts == ["Describe this"]
        assert result.image_count == 1
        assert result.audio_count == 0
        assert result.video_count == 0

    def test_multiple_media_types_counted(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A"},
                        {"type": "image_url", "image_url": {"url": "a"}},
                        {"type": "image_url", "image_url": {"url": "b"}},
                        {"type": "input_audio", "input_audio": {"data": "x"}},
                        {"type": "video_url", "video_url": {"url": "v"}},
                        {"type": "text", "text": "B"},
                    ],
                },
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.texts == ["A", "B"]
        assert result.image_count == 2
        assert result.audio_count == 1
        assert result.video_count == 1

    def test_unknown_part_types_ignored(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "keep"},
                        {"type": "something_new", "data": "ignored"},
                    ],
                }
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.texts == ["keep"]
        assert result.image_count == 0


class TestItemsArrayDisambiguation:
    """The base walker disambiguates Responses/chat ``input``/``messages``
    (dicts with ``role``) from embeddings ``input: [str, ...]``."""

    def test_flat_input_strings_falls_through_to_flat_shape(self):
        payload = {"input": ["a", "b", "c"]}
        result = _chat().extract_payload_inputs(payload)
        # Embeddings shape — flat-field walker handles it.
        assert result.texts == ["a", "b", "c"]
        assert result.image_count == 0

    def test_input_with_role_treated_as_items_array(self):
        payload = {"input": [{"role": "user", "content": "hello from input array"}]}
        result = _chat().extract_payload_inputs(payload)
        assert result.texts == ["hello from input array"]


class TestFlatFieldFallbacks:
    """Completions / embeddings / rankings / HuggingFace flat shapes.

    Each shape early-returns so a plugin that accidentally emits two
    shapes doesn't silently double-count.
    """

    def test_completions_prompt_string(self):
        result = _chat().extract_payload_inputs({"prompt": "one shot"})
        assert result.texts == ["one shot"]

    def test_completions_prompt_list(self):
        result = _chat().extract_payload_inputs({"prompt": ["a", "b"]})
        assert result.texts == ["a", "b"]

    def test_embeddings_input_string(self):
        result = _chat().extract_payload_inputs({"input": "to embed"})
        assert result.texts == ["to embed"]

    def test_rankings_query_and_passages(self):
        result = _chat().extract_payload_inputs(
            {
                "query": "my question",
                "passages": ["p1", {"text": "p2"}, "p3"],
            }
        )
        assert result.texts == ["my question", "p1", "p2", "p3"]

    def test_huggingface_inputs_string(self):
        result = _chat().extract_payload_inputs({"inputs": "hf text"})
        assert result.texts == ["hf text"]

    def test_prompt_wins_over_later_shapes(self):
        """Regression: if a plugin erroneously emits both ``prompt`` and
        ``input`` (flat), the walker must not double-count."""
        result = _chat().extract_payload_inputs(
            {"prompt": "P", "input": "I", "inputs": "HF"}
        )
        assert result.texts == ["P"]

    def test_input_wins_over_query_when_prompt_absent(self):
        result = _chat().extract_payload_inputs(
            {"input": "I", "query": "Q", "passages": ["p"]}
        )
        assert result.texts == ["I"]


class TestResponsesEndpointOverride:
    """The Responses override prepends the top-level ``instructions`` field.

    ``instructions`` is the Responses-API system-prompt equivalent; the
    base walker does not know about it, so the override's job is to
    prepend it once.
    """

    def test_instructions_prepended(self):
        payload = {
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": "hi"}],
        }
        result = _responses().extract_payload_inputs(payload)
        assert result.texts[0] == "You are a helpful assistant."
        assert "hi" in result.texts

    def test_instructions_missing_is_noop(self):
        payload = {"input": [{"role": "user", "content": "hi"}]}
        result = _responses().extract_payload_inputs(payload)
        assert result.texts == ["hi"]

    def test_responses_part_types_dispatch(self):
        """Responses overrides ``PART_TYPES`` with ``input_text`` /
        ``input_image`` / ``input_audio``; the inherited walker dispatches
        those instead of chat's ``text`` / ``image_url`` / ``input_audio``."""
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "data:abc"},
                        {"type": "input_audio", "input_audio": {"data": "x"}},
                    ],
                }
            ]
        }
        result = _responses().extract_payload_inputs(payload)
        assert result.texts == ["describe"]
        assert result.image_count == 1
        assert result.audio_count == 1

    def test_chat_style_part_types_not_counted_by_responses(self):
        """Responses' ``PART_TYPES`` doesn't include chat's ``image_url``
        type name — chat-shape parts in a Responses payload should NOT
        be counted as images."""
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "a"}}],
                }
            ]
        }
        result = _responses().extract_payload_inputs(payload)
        assert result.image_count == 0


class TestImageRetrievalOverride:
    """NIM image retrieval overrides ``extract_payload_inputs`` to handle
    its flat ``input: [...]`` list of parts with no role wrapper."""

    def test_image_retrieval_counts_images(self):
        payload = {
            "input": [
                {"type": "image_url", "image_url": {"url": "a"}},
                {"type": "image_url", "image_url": {"url": "b"}},
                {"type": "image_url", "image_url": {"url": "c"}},
            ]
        }
        result = _image_retrieval().extract_payload_inputs(payload)
        assert result.image_count == 3
        assert result.texts == []

    def test_image_retrieval_empty_input(self):
        result = _image_retrieval().extract_payload_inputs({"input": []})
        assert result.image_count == 0


class MinimalEndpoint(BaseEndpoint):
    """Concrete subclass for testing base behaviour without other overrides."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        return {}

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return None


class TestBaseExtractionDefaults:
    def test_result_is_extractedpayload_instance(self):
        endpoint = MinimalEndpoint(
            model_endpoint=create_model_endpoint(EndpointType.CHAT)
        )
        result = endpoint.extract_payload_inputs(
            {"messages": [{"role": "user", "content": "x"}]}
        )
        assert isinstance(result, ExtractedPayload)
        assert result.texts == ["x"]


class TestChatMessagesField:
    """``ExtractedPayload.messages`` carries the chat-shape role/content
    view used by the record processor's ``apply_chat_template`` path.
    Populated only for chat/Responses message arrays; ``None`` for flat
    completions/embeddings/rankings/HF shapes (templating doesn't apply)."""

    def test_chat_string_content_populates_messages(self):
        payload = {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.messages == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_chat_mixed_content_concatenates_text_parts(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe "},
                        {"type": "image_url", "image_url": {"url": "data:abc"}},
                        {"type": "text", "text": "this image"},
                    ],
                }
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.messages == [{"role": "user", "content": "describe this image"}]
        assert result.image_count == 1

    def test_tool_call_only_assistant_turn_preserves_tool_calls(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ]
        payload = {
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": tool_calls},
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.messages == [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]

    def test_extract_payload_inputs_message_tool_calls_excluded_from_tool_texts(self):
        """tool_calls riding in ``messages`` are excluded from ``tool_texts``.

        The chat-template ISL path renders ``messages`` (tool_calls included)
        and tokenises ``tool_texts`` on top; keeping the same call text in both
        would count it twice. It must still appear in ``texts`` for the
        bare-text path.
        """
        payload = {
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                },
            ]
        }
        result = _chat().extract_payload_inputs(payload)
        assert result.tool_texts == []
        assert "get_weather" in result.texts
        assert '{"city": "SF"}' in result.texts

    def test_extract_payload_inputs_responses_function_call_in_tool_texts(self):
        """Responses ``function_call`` items have no role, so they are absent
        from ``messages`` and must remain in ``tool_texts`` for the
        chat-template path to account for them."""
        payload = {
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                },
            ]
        }
        result = _responses().extract_payload_inputs(payload)
        assert "get_weather" in result.tool_texts
        assert '{"city": "SF"}' in result.tool_texts

    def test_flat_shapes_leave_messages_none(self):
        for payload in (
            {"prompt": "hi"},
            {"input": "embed me"},
            {"input": ["a", "b"]},
            {"query": "q", "passages": ["p"]},
            {"inputs": "hf"},
        ):
            result = _chat().extract_payload_inputs(payload)
            assert result.messages is None, payload

    def test_empty_payload_leaves_messages_none(self):
        result = _chat().extract_payload_inputs({})
        assert result.messages is None

    def test_responses_instructions_prepended_to_messages(self):
        payload = {
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": "hi"}],
        }
        result = _responses().extract_payload_inputs(payload)
        assert result.messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]

    def test_responses_no_instructions_no_system_prepend(self):
        payload = {"input": [{"role": "user", "content": "hi"}]}
        result = _responses().extract_payload_inputs(payload)
        assert result.messages == [{"role": "user", "content": "hi"}]
