# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.models import RequestRecord, TextResponse
from aiperf.common.models.record_models import ReasoningResponseData
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from aiperf.records.inference_result_parser import InferenceResultParser
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.mark.asyncio
@pytest.mark.parametrize("include_answer", [False, True])
async def test_streaming_reasoning_summary_counts_tokens_and_accepts_record(
    *,
    setup_inference_parser: InferenceResultParser,
    include_answer: bool,
) -> None:
    model_endpoint = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    endpoint = ResponsesEndpoint(model_endpoint=model_endpoint)
    info = create_request_info(model_endpoint=model_endpoint)
    info.payload_bytes = orjson.dumps(endpoint.format_payload(info))
    events = [
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "Plan ",
            "summary_index": 0,
            "output_index": 0,
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "the response",
            "summary_index": 0,
            "output_index": 0,
        },
        {
            "type": "response.reasoning_summary_text.done",
            "text": "Plan the response",
            "summary_index": 0,
            "output_index": 0,
        },
    ]
    if include_answer:
        events.append({"type": "response.output_text.delta", "delta": "Final answer"})
    record = RequestRecord(
        request_info=info,
        model_name="test-model",
        start_perf_ns=1,
        responses=[
            TextResponse(perf_ns=i + 2, text=orjson.dumps(event).decode())
            for i, event in enumerate(events)
        ],
    )
    first = endpoint.parse_response(record.responses[0])
    assert first is not None
    assert isinstance(first.data, ReasoningResponseData)
    assert first.data.reasoning == "Plan "
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
    setup_inference_parser.endpoint = endpoint
    setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

    parsed = await setup_inference_parser.parse_request_record(record)

    assert record.error is None
    assert parsed.token_counts.reasoning == 3
    assert parsed.token_counts.output == (2 if include_answer else None)
    assert parsed.responses[0].perf_ns == 2
    assert len(parsed.responses) == (3 if include_answer else 2)


@pytest.mark.parametrize("delta", [None, ""])
def test_empty_reasoning_summary_delta_has_no_content(delta: str | None) -> None:
    endpoint = ResponsesEndpoint(
        model_endpoint=create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    )
    response = TextResponse(
        perf_ns=1,
        text=orjson.dumps(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": delta,
            }
        ).decode(),
    )

    assert endpoint.parse_response(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_first", [False, True])
@pytest.mark.parametrize("include_item_id", [False, True])
async def test_reasoning_summary_falls_back_per_item_without_combining_sources(
    *,
    setup_inference_parser: InferenceResultParser,
    raw_first: bool,
    include_item_id: bool,
) -> None:
    model_endpoint = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    endpoint = ResponsesEndpoint(model_endpoint=model_endpoint)
    info = create_request_info(model_endpoint=model_endpoint)
    info.payload_bytes = orjson.dumps(endpoint.format_payload(info))
    identity = {"output_index": 0}
    if include_item_id:
        identity["item_id"] = "rs-1"
    raw = {
        "type": "response.reasoning_text.delta",
        "delta": "Full reasoning text ",
        **identity,
    }
    summary = {
        "type": "response.reasoning_summary_text.delta",
        "delta": "Short summary ",
        **identity,
    }
    events = [raw, summary] if raw_first else [summary, raw]
    events.append(
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "Separate item summary",
            "item_id": "rs-2",
            "output_index": 1,
        }
    )
    record = RequestRecord(
        request_info=info,
        model_name="test-model",
        start_perf_ns=1,
        responses=[
            TextResponse(perf_ns=i + 2, text=orjson.dumps(event).decode())
            for i, event in enumerate(events)
        ],
    )
    assert endpoint.parse_response(record.responses[0]) is not None
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
    setup_inference_parser.endpoint = endpoint
    setup_inference_parser.get_tokenizer = AsyncMock(return_value=tokenizer)

    parsed = await setup_inference_parser.parse_request_record(record)

    assert record.error is None
    assert parsed.token_counts.reasoning == 6
    assert parsed.responses[0].perf_ns == 2


@pytest.mark.parametrize("raw_delta", [None, ""])
def test_reasoning_summary_remains_fallback_when_raw_delta_has_no_content(
    raw_delta: str | None,
) -> None:
    endpoint = ResponsesEndpoint(
        model_endpoint=create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    )
    events = [
        {
            "type": "response.reasoning_text.delta",
            "delta": raw_delta,
            "item_id": "rs-1",
            "output_index": 0,
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "delta": "Summary",
            "item_id": "rs-1",
            "output_index": 0,
        },
    ]
    record = RequestRecord(
        responses=[
            TextResponse(perf_ns=i + 1, text=orjson.dumps(event).decode())
            for i, event in enumerate(events)
        ]
    )

    parsed = endpoint.extract_response_data(record)

    assert [response.data.reasoning for response in parsed] == ["Summary"]
