# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that chat/completions endpoints capture the raw spec-decode payload.

``parse_response`` must lift ``choices[0].speculative_decoding_stats`` onto the
``ParsedResponse`` uninterpreted, including on a streaming finish-reason chunk
whose delta is empty (which otherwise carries no data and would be dropped), and
must not choke on a malformed non-list ``choices``.
"""

from typing import Any, TypeVar
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType

STATS = {
    "mean_acceptance_length": 1.5,
    "draft_acceptance_rate": 0.25,
    "acceptance_histogram": {"0": 2, "2": 2},
    "num_spec_steps": 4,
    "num_accepted_draft_tokens": 4,
    "num_draft_tokens": 16,
    "num_spec_tokens": 3,
}


_EndpointT = TypeVar("_EndpointT", bound=BaseEndpoint)


def _make_endpoint(
    endpoint_type: EndpointType, endpoint_cls: type[_EndpointT]
) -> _EndpointT:
    model_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="m")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(type=endpoint_type, base_url="http://localhost:8000"),
    )
    with patch("aiperf.plugin.plugins.get_class") as mock_get_class:
        mock_get_class.return_value = MagicMock()
        return endpoint_cls(model_endpoint=model_endpoint)


def _mock_response(json_obj: dict[str, Any]) -> Mock:
    mock_response = Mock(spec=InferenceServerResponse)
    mock_response.perf_ns = 123
    mock_response.get_json.return_value = json_obj
    return mock_response


@pytest.fixture
def chat_endpoint() -> ChatEndpoint:
    return _make_endpoint(EndpointType.CHAT, ChatEndpoint)


@pytest.fixture
def completions_endpoint() -> CompletionsEndpoint:
    return _make_endpoint(EndpointType.COMPLETIONS, CompletionsEndpoint)


class TestChatSpecDecodeCapture:
    def test_parse_response_non_streaming_captures_stats(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        json_obj = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                    "speculative_decoding_stats": STATS,
                }
            ],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_streaming_empty_delta_retains_stats(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """The finish chunk has no content but must still surface the stats."""
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "speculative_decoding_stats": STATS,
                }
            ],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.data is None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_absent_stats_returns_none(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None

    def test_parse_response_malformed_choices_returns_none(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """A non-list ``choices`` (e.g. an error envelope) must not raise."""
        json_obj = {"object": "error", "choices": {"0": {"message": {}}}}
        assert chat_endpoint.parse_response(_mock_response(json_obj)) is None

    def test_parse_response_multi_choice_suppresses_stats(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """n > 1 (multiple stats-bearing choices): stats suppressed, not mixed."""
        json_obj = {
            "object": "chat.completion",
            "choices": [
                {"message": {"content": "a"}, "speculative_decoding_stats": STATS},
                {"message": {"content": "b"}, "speculative_decoding_stats": STATS},
            ],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None


class TestCompletionsSpecDecodeCapture:
    def test_parse_response_non_streaming_captures_stats(
        self, completions_endpoint: CompletionsEndpoint
    ) -> None:
        json_obj = {
            "object": "text_completion",
            "choices": [{"text": "hi", "speculative_decoding_stats": STATS}],
        }
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_streaming_empty_text_retains_stats(
        self, completions_endpoint: CompletionsEndpoint
    ) -> None:
        json_obj = {
            "object": "text_completion",
            "choices": [
                {
                    "text": "",
                    "finish_reason": "stop",
                    "speculative_decoding_stats": STATS,
                }
            ],
        }
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_absent_stats_returns_none(
        self, completions_endpoint: CompletionsEndpoint
    ) -> None:
        json_obj = {"object": "text_completion", "choices": [{"text": "hi"}]}
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None
