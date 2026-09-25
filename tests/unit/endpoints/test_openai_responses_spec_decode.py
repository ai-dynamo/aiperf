# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the Responses API (`/v1/responses`) captures the spec-decode payload.

vLLM nests `metrics.speculative_decoding` on the response object: at the root
non-streaming, and under the `response.completed` event's `response` key
streaming. `ResponsesEndpoint` must lift it onto the `ParsedResponse` at both
placements, and only the terminal event may carry it -- the full-response-bearing
`response.created` / `response.in_progress` events must NOT, so exactly one
`ParsedResponse` per record carries stats (the invariant the parser relies on to
build a single engine-neutral acceptance record).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import orjson
import pytest

from aiperf.common.models import (
    RequestRecord,
    SpecDecodeAcceptanceRecord,
)
from aiperf.common.models.record_models import InferenceServerResponse, TextResponse
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)

# A self-consistent vLLM ``summary`` payload (mirrors the shape from vLLM
# PR #48915). Dense acceptance_histogram [39, 1, 0, 3]: 39 steps accepted 0
# drafts, 1 accepted 1, 3 accepted 3 -> 43 steps, 10 accepted, mean 1.2325...
SPEC_PAYLOAD: dict[str, Any] = {
    "mean_acceptance_length": 1.2325581395348837,
    "draft_acceptance_rate": 0.07751937984496124,
    "acceptance_histogram": [39, 1, 0, 3],
    "num_spec_steps": 43,
    "num_accepted_draft_tokens": 10,
    "num_draft_tokens": 129,
    "num_spec_tokens": 3,
}


@pytest.fixture
def endpoint() -> ResponsesEndpoint:
    me = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, me)


def _mock_response(json_obj: dict[str, Any]) -> Mock:
    mock_response = Mock(spec=InferenceServerResponse)
    mock_response.perf_ns = 123
    mock_response.get_json.return_value = json_obj
    return mock_response


def _record(events: list[dict]) -> RequestRecord:
    """Wrap Responses-API event dicts as an SSE-ordered RequestRecord."""
    responses = [
        TextResponse(perf_ns=i, text=orjson.dumps(event).decode())
        for i, event in enumerate(events)
    ]
    return RequestRecord(responses=responses)


class TestResponsesSpecDecodeCapture:
    def test_non_streaming_captures_stats(self, endpoint: ResponsesEndpoint) -> None:
        json_obj = {
            "object": "response",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 8},
            "metrics": {"speculative_decoding": SPEC_PAYLOAD},
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats == SPEC_PAYLOAD

    def test_streaming_completed_event_captures_stats(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        """Stats ride the terminal event, nested under its ``response`` key."""
        json_obj = {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 5, "output_tokens": 8},
                "metrics": {"speculative_decoding": SPEC_PAYLOAD},
            },
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.data is None
        assert parsed.spec_decode_stats == SPEC_PAYLOAD

    def test_streaming_completed_event_stats_without_usage_still_captured(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        """A terminal event carrying spec-decode but no ``usage`` still yields a record."""
        json_obj = {
            "type": "response.completed",
            "response": {"metrics": {"speculative_decoding": SPEC_PAYLOAD}},
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.usage is None
        assert parsed.spec_decode_stats == SPEC_PAYLOAD

    def test_non_streaming_stats_without_data_or_usage_still_captured(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        """A full response with only spec-decode (no output, no usage) returns a
        ParsedResponse, not None."""
        json_obj = {
            "object": "response",
            "metrics": {"speculative_decoding": SPEC_PAYLOAD},
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is None
        assert parsed.spec_decode_stats == SPEC_PAYLOAD

    @pytest.mark.parametrize("event_type", ["response.created", "response.in_progress"])
    def test_intermediate_events_do_not_capture_stats(
        self, endpoint: ResponsesEndpoint, event_type: str
    ) -> None:
        """`created`/`in_progress` embed a full response but must NOT carry stats
        -- otherwise two carriers would trip the parser's >1 suppression."""
        json_obj = {
            "type": event_type,
            "response": {"metrics": {"speculative_decoding": SPEC_PAYLOAD}},
        }
        assert endpoint.parse_response(_mock_response(json_obj)) is None

    def test_non_streaming_absent_stats(self, endpoint: ResponsesEndpoint) -> None:
        json_obj = {
            "object": "response",
            "output": [],
            "usage": {"output_tokens": 8},
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None

    def test_metrics_without_spec_decode_yields_none(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        json_obj = {
            "object": "response",
            "output": [],
            "usage": {"output_tokens": 8},
            "metrics": {"time_to_first_token_ms": 12.0, "speculative_decoding": None},
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None

    def test_malformed_metrics_does_not_raise(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        json_obj = {
            "object": "response",
            "output": [],
            "usage": {"output_tokens": 8},
            "metrics": "oops",
        }
        parsed = endpoint.parse_response(_mock_response(json_obj))
        assert parsed is not None
        assert parsed.spec_decode_stats is None


class TestResponsesSpecDecodeRecordLevel:
    def test_streaming_record_yields_single_acceptance_record(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        """A realistic event stream must produce exactly one stats carrier and a
        non-null engine-neutral ``SpecDecodeAcceptanceRecord`` via the adapter path."""
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.in_progress", "response": {}},
            {"type": "response.output_text.delta", "delta": "The "},
            {"type": "response.output_text.delta", "delta": "quick"},
            {"type": "response.output_text.done", "text": "The quick"},
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 8},
                    "metrics": {"speculative_decoding": SPEC_PAYLOAD},
                },
            },
        ]

        parsed = endpoint.extract_response_data(_record(events))

        # Exactly one carrier -- the terminal completed event. This is what the
        # parser's "exactly one" guard depends on; more than one would suppress.
        carriers = [p for p in parsed if p.spec_decode_stats]
        assert len(carriers) == 1

        record = InferenceResultParser._extract_spec_decode_acceptance(parsed)
        assert isinstance(record, SpecDecodeAcceptanceRecord)
        assert record.engine == "vllm"
        assert record.num_spec_steps == 43
        assert record.num_accepted_draft_tokens == 10
        assert record.mean_acceptance_length == pytest.approx(1.2325581395348837)
        assert record.completion_tokens == 8  # from usage.output_tokens

    def test_stream_without_stats_yields_no_record(
        self, endpoint: ResponsesEndpoint
    ) -> None:
        """No spec-decode payload anywhere -> no carrier, no record."""
        events = [
            {"type": "response.created", "response": {}},
            {"type": "response.output_text.delta", "delta": "hi"},
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "output_tokens": 2}},
            },
        ]

        parsed = endpoint.extract_response_data(_record(events))
        assert [p for p in parsed if p.spec_decode_stats] == []
        assert InferenceResultParser._extract_spec_decode_acceptance(parsed) is None
