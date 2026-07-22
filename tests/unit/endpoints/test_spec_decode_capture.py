# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that chat/completions endpoints capture the raw spec-decode payload.

``parse_response`` must lift ``choices[0].speculative_decoding_stats`` onto the
``ParsedResponse`` uninterpreted, including on a streaming finish-reason chunk
whose delta is empty (which otherwise carries no data and would be dropped).
"""

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


def _make_endpoint(endpoint_type, endpoint_cls):
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


def _mock_response(json_obj):
    mock_response = Mock(spec=InferenceServerResponse)
    mock_response.perf_ns = 123
    mock_response.get_json.return_value = json_obj
    return mock_response


@pytest.fixture
def chat_endpoint():
    return _make_endpoint(EndpointType.CHAT, ChatEndpoint)


@pytest.fixture
def completions_endpoint():
    return _make_endpoint(EndpointType.COMPLETIONS, CompletionsEndpoint)


class TestChatSpecDecodeCapture:
    def test_non_streaming_captures_stats(self, chat_endpoint):
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

    def test_streaming_terminal_chunk_with_empty_delta_is_retained(self, chat_endpoint):
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

    def test_absent_stats_leaves_field_none(self, chat_endpoint):
        json_obj = {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        parsed = chat_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None


class TestCompletionsSpecDecodeCapture:
    def test_non_streaming_captures_stats(self, completions_endpoint):
        json_obj = {
            "object": "text_completion",
            "choices": [{"text": "hi", "speculative_decoding_stats": STATS}],
        }
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == STATS

    def test_streaming_terminal_chunk_with_empty_text_is_retained(
        self, completions_endpoint
    ):
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

    def test_absent_stats_leaves_field_none(self, completions_endpoint):
        json_obj = {"object": "text_completion", "choices": [{"text": "hi"}]}
        parsed = completions_endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None
