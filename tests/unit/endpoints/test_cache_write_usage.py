# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-write accounting must survive endpoint parsing and final usage selection."""

import pytest
from pytest import param

from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponse, ParsedResponseRecord, RequestRecord
from aiperf.common.models.record_models import TextResponseData, TokenCounts
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.anthropic_messages import MessagesEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.usage_cache_metrics import UsagePromptCacheWriteTokensMetric
from aiperf.metrics.types.usage_total_metrics import (
    TotalUsagePromptCacheWriteTokensMetric,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_mock_response, create_model_endpoint


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("write", [None, 0, 8192])
@pytest.mark.parametrize(
    "endpoint_type,endpoint_class,write_key",
    [
        param(EndpointType.CHAT, ChatEndpoint, "cache_write_tokens", id="chat"),
        param(EndpointType.CHAT, ChatEndpoint, "created_cache_tokens", id="vllm"),
        param(EndpointType.RESPONSES, ResponsesEndpoint, "cache_write_tokens", id="responses"),
        param(EndpointType.MESSAGES, MessagesEndpoint, "cache_creation_input_tokens", id="messages"),
    ],
)  # fmt: skip
def test_cache_write_usage_reaches_record_and_total_metrics(
    endpoint_type: EndpointType,
    endpoint_class: type,
    write_key: str,
    write: int | None,
    streaming: bool,
) -> None:
    endpoint = endpoint_class(
        model_endpoint=create_model_endpoint(endpoint_type, streaming=streaming)
    )
    if endpoint_type == EndpointType.CHAT:
        usage = {
            "prompt_tokens": 8448,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        if write is not None:
            usage["prompt_tokens_details"][write_key] = write
        frames = [
            {
                "object": "chat.completion.chunk" if streaming else "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta" if streaming else "message": {"content": "OK"},
                    }
                ],
                "usage": usage,
            }
        ]
    elif endpoint_type == EndpointType.RESPONSES:
        usage = {
            "input_tokens": 8448,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
        }
        if write is not None:
            usage["input_tokens_details"][write_key] = write
        response = {
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": usage,
        }
        frames = (
            [
                {
                    "type": "response.output_text.delta",
                    "delta": "OK",
                    "output_index": 0,
                    "content_index": 0,
                },
                {"type": "response.completed", "response": response},
            ]
            if streaming
            else [response]
        )
    else:
        usage = {
            "input_tokens": 8448 - (write or 0),
            "cache_read_input_tokens": 0,
            "output_tokens": 8,
        }
        if write is not None:
            usage["cache_creation_input_tokens"] = write
        frames = (
            [
                {
                    "type": "message_start",
                    "message": {"usage": {**usage, "output_tokens": 0}},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "OK"},
                },
                {"type": "message_delta", "usage": {"output_tokens": 8}},
            ]
            if streaming
            else [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "OK"}],
                    "usage": usage,
                }
            ]
        )

    request = RequestRecord(start_perf_ns=1, end_perf_ns=10)
    request.responses = [create_mock_response(json_data=f) for f in frames]
    record = ParsedResponseRecord(
        request=request,
        responses=endpoint.extract_response_data(request),
        token_counts=TokenCounts(input=8448, output=8),
    )
    assert record.final_usage.prompt_tokens == 8448
    assert record.final_usage.completion_tokens == 8
    assert record.valid
    metric = UsagePromptCacheWriteTokensMetric()
    if write is None:
        with pytest.raises(NoMetricValue):
            metric.parse_record(record, MetricRecordDict())
    else:
        value = metric.parse_record(record, MetricRecordDict())
        assert value == write
        assert (
            TotalUsagePromptCacheWriteTokensMetric._derive_value(
                MetricResultsDict({metric.tag: value})
            )
            == write
        )


@pytest.mark.parametrize(
    "details_key", ["prompt_tokens_details", "input_tokens_details"]
)
@pytest.mark.parametrize("write_key", ["cache_write_tokens", "created_cache_tokens"])
@pytest.mark.parametrize("write", [None, 0, 8192])
def test_nested_write_preserves_null_zero_and_precedence(
    details_key: str,
    write_key: str,
    write: int | None,
) -> None:
    usage = Usage({details_key: {write_key: write}, "cache_creation_input_tokens": 999})
    assert usage.prompt_cache_write_tokens == write


def test_canonical_write_precedes_vllm_alias() -> None:
    usage = Usage(
        {
            "prompt_tokens_details": {
                "cache_write_tokens": 0,
                "created_cache_tokens": 123,
            }
        }
    )
    assert usage.prompt_cache_write_tokens == 0


def test_unreported_placeholder_does_not_become_measured_zero() -> None:
    usage = Usage(
        {
            "input_tokens_details": {
                "cache_write_tokens": 0,
                "cache_write_tokens_reported": False,
            },
            "cache_creation_input_tokens": 999,
        }
    )
    assert usage.prompt_cache_write_tokens is None


def test_final_unknown_write_does_not_restore_an_earlier_count() -> None:
    record = ParsedResponseRecord(
        request=RequestRecord(start_perf_ns=1, end_perf_ns=10),
        responses=[
            ParsedResponse(
                perf_ns=2,
                data=TextResponseData(text="OK"),
                usage={"prompt_tokens_details": {"cache_write_tokens": 8192}},
            ),
            ParsedResponse(
                perf_ns=9,
                usage={
                    "prompt_tokens_details": {
                        "cache_write_tokens": 0,
                        "cache_write_tokens_reported": False,
                    }
                },
            ),
        ],
        token_counts=TokenCounts(input=8448, output=8),
    )
    assert record.valid
    with pytest.raises(NoMetricValue):
        UsagePromptCacheWriteTokensMetric().parse_record(record, MetricRecordDict())


@pytest.mark.parametrize("details", [None, {}, "invalid"])
def test_absent_nested_write_retains_legacy_vendor_fallback(details: object) -> None:
    usage = Usage({"prompt_tokens_details": details, "cacheWriteInputTokens": 12})
    assert usage.prompt_cache_write_tokens == 12
