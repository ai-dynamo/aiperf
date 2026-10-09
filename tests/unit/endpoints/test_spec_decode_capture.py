# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that chat/completions endpoints capture the raw spec-decode payload.

``parse_response`` must lift ``metrics.speculative_decoding`` from the response
root onto the ``ParsedResponse`` uninterpreted -- including on a streaming
trailing usage chunk whose ``choices`` is empty (it carries no content and would
otherwise be dropped) -- and must not choke on a malformed non-list ``choices``.
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
    "acceptance_histogram": [2, 0, 2, 0],
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
                }
            ],
            "metrics": {"speculative_decoding": STATS},
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_streaming_usage_chunk_retains_stats(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """The trailing usage chunk has empty choices but carries stats at root."""
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"completion_tokens": 50},
            "metrics": {"speculative_decoding": STATS},
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

    def test_parse_response_metrics_without_spec_decode_yields_none(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """``metrics`` present (timing only, or ``n > 1`` where vLLM nulls the
        ``speculative_decoding`` sub-object): no spec-decode payload captured."""
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"content": "hi"}}],
            "metrics": {"time_to_first_token_ms": 12.0, "speculative_decoding": None},
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
            "choices": [{"text": "hi"}],
            "metrics": {"speculative_decoding": STATS},
        }
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_streaming_usage_chunk_retains_stats(
        self, completions_endpoint: CompletionsEndpoint
    ) -> None:
        json_obj = {
            "object": "text_completion",
            "choices": [],
            "usage": {"completion_tokens": 50},
            "metrics": {"speculative_decoding": STATS},
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


LLAMACPP_TIMINGS = {
    "predicted_n": 20,
    "draft_n": 15,
    "draft_n_accepted": 14,
    "predicted_ms": 76.2,
    "prompt_n": 11,
}


class TestLlamaCppTimingsCapture:
    """llama.cpp timings path: top-level ``timings`` carries spec-decode counters."""

    def test_parse_response_llamacpp_timings_captured(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """Top-level timings with draft_n/draft_n_accepted are captured as spec_decode_stats."""
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "timings": LLAMACPP_TIMINGS,
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == LLAMACPP_TIMINGS

    def test_parse_response_llamacpp_timings_streaming_finish_chunk(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """Timings on the finish-reason streaming chunk are captured."""
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "finish_reason": "length"}],
            "timings": LLAMACPP_TIMINGS,
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == LLAMACPP_TIMINGS

    def test_parse_response_vllm_stats_take_priority_over_timings(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """When both metrics.speculative_decoding and timings are present, vLLM wins."""
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "metrics": {"speculative_decoding": STATS},
            "timings": LLAMACPP_TIMINGS,
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_parse_response_timings_without_draft_fields_not_captured(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """Timings that carry no draft_n/draft_n_accepted (non-spec run) are not captured."""
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "timings": {"predicted_n": 20, "predicted_ms": 76.2},
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None
