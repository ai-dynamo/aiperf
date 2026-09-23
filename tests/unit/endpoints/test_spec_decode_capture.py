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


# TensorRT-LLM attaches its payload per choice rather than at the response root,
# using its own field names. Shape mirrors the TRT-LLM adapter's tests.
TRTLLM_STATS = {
    "acceptance_rate": 0.5,
    "total_accepted_draft_tokens": 30,
    "total_draft_tokens": 60,
    "num_spec_steps": 20,
    "acceptance_histogram": [8, 0, 6, 6],
    "num_spec_tokens": 3,
}


class TestPerChoiceCapture:
    """The per-choice placement TensorRT-LLM uses, on both endpoints.

    One case per endpoint rather than per endpoint-and-stream-shape: the capture
    helper is a pure function of the response body and never inspects the
    endpoint type, ``object``, or whether the payload arrived on a stream chunk.
    Both endpoints are kept because they have separate ``_parse_json_response``
    implementations, each calling the helper independently, so a single case
    would leave the other call site untested.
    """

    def test_chat_non_streaming(self, chat_endpoint: ChatEndpoint) -> None:
        json_obj = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                    "speculative_decoding": TRTLLM_STATS,
                }
            ],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == TRTLLM_STATS

    def test_completions_non_streaming(
        self, completions_endpoint: CompletionsEndpoint
    ) -> None:
        json_obj = {
            "object": "text_completion",
            "choices": [
                {
                    "text": "hi",
                    "finish_reason": "stop",
                    "speculative_decoding": TRTLLM_STATS,
                }
            ],
        }
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == TRTLLM_STATS

    def test_multi_choice_is_suppressed(self, chat_endpoint: ChatEndpoint) -> None:
        """``n > 1`` non-streaming: a single per-request record cannot attribute
        request-level usage to one sequence, so no payload is captured."""
        json_obj = {
            "object": "chat.completion",
            "choices": [
                {"message": {"content": "a"}, "speculative_decoding": TRTLLM_STATS},
                {"message": {"content": "b"}, "speculative_decoding": TRTLLM_STATS},
            ],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None

    def test_root_placement_wins_when_both_present(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        """Not expected on the wire, but the precedence must be deterministic."""
        json_obj = {
            "object": "chat.completion",
            "choices": [
                {"message": {"content": "hi"}, "speculative_decoding": TRTLLM_STATS}
            ],
            "metrics": {"speculative_decoding": STATS},
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_non_dict_per_choice_payload_ignored(
        self, chat_endpoint: ChatEndpoint
    ) -> None:
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"content": "hi"}, "speculative_decoding": "x"}],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None
