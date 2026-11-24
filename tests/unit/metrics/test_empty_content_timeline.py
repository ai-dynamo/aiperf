# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import orjson
import pytest

from aiperf.common.models import (
    ParsedResponseRecord,
    RequestRecord,
    SSEMessage,
    TokenCounts,
)
from aiperf.common.models.record_models import first_content_chunk_completion_tokens
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.metrics.types.decode_duration_metric import (
    DecodeDurationMetric,
    FullDecodeDurationMetric,
)
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.inter_token_latency_metric import (
    FullResponseInterTokenLatencyMetric,
    InterTokenLatencyMetric,
)
from aiperf.metrics.types.output_token_throughput_metrics import (
    FullResponseOutputTokenThroughputPerUserMetric,
    OutputTokenThroughputPerUserMetric,
)
from aiperf.metrics.types.prefill_throughput_per_user import (
    PrefillThroughputPerUserMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.time_to_first_output_token_metric import (
    TimeToFirstOutputTokenMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.metrics.types.ttst_metric import TTSTMetric
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint
from tests.unit.metrics.conftest import run_simple_metrics_pipeline

START_NS = 1_000_000_000


def _event(perf_ns: int, payload: dict[str, object]) -> SSEMessage:
    return SSEMessage.parse(
        b"data: " + orjson.dumps(payload) + b"\n\n",
        perf_ns=perf_ns,
    )


def _chat_chunk(
    delta: dict[str, object], finish_reason: str | None = None
) -> dict[str, object]:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
    }


def _record(*, enabled: bool, output_tokens: int = 4) -> ParsedResponseRecord:
    model_endpoint = create_model_endpoint(EndpointType.CHAT, streaming=True)
    model_endpoint.endpoint.allow_empty_content = enabled
    endpoint = ChatEndpoint(model_endpoint)
    request = RequestRecord(
        model_name="test-model",
        start_perf_ns=START_NS,
        end_perf_ns=1_600_000_000,
        responses=[
            _event(1_050_000_000, _chat_chunk({"role": "assistant"})),
            _event(1_100_000_000, _chat_chunk({"content": ""})),
            _event(1_200_000_000, _chat_chunk({"content": "hello"})),
            _event(1_300_000_000, _chat_chunk({"reasoning_content": ""})),
            _event(
                1_400_000_000,
                _chat_chunk({"content": ""}, finish_reason="stop"),
            ),
            _event(1_500_000_000, {"usage": {"completion_tokens": output_tokens}}),
        ],
    )
    return ParsedResponseRecord(
        request=request,
        responses=endpoint.extract_response_data(request),
        token_counts=TokenCounts(input=100, output=output_tokens),
    )


def _empty_only_record() -> ParsedResponseRecord:
    model_endpoint = create_model_endpoint(EndpointType.CHAT, streaming=True)
    model_endpoint.endpoint.allow_empty_content = True
    endpoint = ChatEndpoint(model_endpoint)
    request = RequestRecord(
        model_name="test-model",
        start_perf_ns=START_NS,
        end_perf_ns=1_200_000_000,
        responses=[
            _event(1_100_000_000, _chat_chunk({"content": ""})),
        ],
    )
    return ParsedResponseRecord(
        request=request,
        responses=endpoint.extract_response_data(request),
        token_counts=TokenCounts(input=100, output=0),
    )


def test_enabled_empty_responses_drive_one_metric_timeline() -> None:
    record = _record(enabled=True)
    results = run_simple_metrics_pipeline(
        [record],
        TTFTMetric.tag,
        TTSTMetric.tag,
        TimeToFirstOutputTokenMetric.tag,
        RequestLatencyMetric.tag,
        DecodeDurationMetric.tag,
        FullDecodeDurationMetric.tag,
        InterChunkLatencyMetric.tag,
        InterTokenLatencyMetric.tag,
        FullResponseInterTokenLatencyMetric.tag,
        OutputTokenThroughputPerUserMetric.tag,
        FullResponseOutputTokenThroughputPerUserMetric.tag,
        PrefillThroughputPerUserMetric.tag,
    )

    assert [response.perf_ns for response in record.content_responses] == [
        1_100_000_000,
        1_200_000_000,
        1_300_000_000,
        1_400_000_000,
    ]
    assert results[TTFTMetric.tag] == [100_000_000]
    assert results[TTSTMetric.tag] == [100_000_000]
    assert results[TimeToFirstOutputTokenMetric.tag] == [200_000_000]
    assert results[RequestLatencyMetric.tag] == [400_000_000]
    assert results[DecodeDurationMetric.tag] == [300_000_000]
    assert results[FullDecodeDurationMetric.tag] == [500_000_000]
    assert results[InterChunkLatencyMetric.tag] == [
        [100_000_000, 100_000_000, 100_000_000]
    ]
    # The empty TTFT chunk carries no token, so all 4 output tokens fall in the
    # decode window.
    assert results[InterTokenLatencyMetric.tag] == pytest.approx([300_000_000 / 4])
    assert results[FullResponseInterTokenLatencyMetric.tag] == pytest.approx(
        [500_000_000 / 4]
    )
    assert results[OutputTokenThroughputPerUserMetric.tag] == pytest.approx([4 / 0.3])
    assert results[FullResponseOutputTokenThroughputPerUserMetric.tag] == pytest.approx(
        [4 / 0.5]
    )
    assert results[PrefillThroughputPerUserMetric.tag] == pytest.approx([1_000.0])


def test_disabled_empty_responses_preserve_non_empty_timeline() -> None:
    record = _record(enabled=False)
    results = run_simple_metrics_pipeline(
        [record],
        TTFTMetric.tag,
        TTSTMetric.tag,
        TimeToFirstOutputTokenMetric.tag,
        RequestLatencyMetric.tag,
        DecodeDurationMetric.tag,
        FullDecodeDurationMetric.tag,
        InterChunkLatencyMetric.tag,
    )

    assert [response.perf_ns for response in record.content_responses] == [
        1_200_000_000
    ]
    assert results[TTFTMetric.tag] == [200_000_000]
    assert results[TimeToFirstOutputTokenMetric.tag] == [200_000_000]
    assert results[RequestLatencyMetric.tag] == [200_000_000]
    assert results[DecodeDurationMetric.tag] == [0]
    assert results[FullDecodeDurationMetric.tag] == [400_000_000]
    assert TTSTMetric.tag not in results
    assert InterChunkLatencyMetric.tag not in results


def test_enabled_empty_only_response_is_valid_but_has_no_token_metrics() -> None:
    record = _empty_only_record()

    assert record.valid
    assert record.token_counts is not None
    assert record.token_counts.output == 0
    results = run_simple_metrics_pipeline(
        [record], TTFTMetric.tag, InterTokenLatencyMetric.tag
    )
    assert results[TTFTMetric.tag] == [100_000_000]
    assert InterTokenLatencyMetric.tag not in results


def _per_chunk_usage_record(*, empty_chunk_usage: bool) -> ParsedResponseRecord:
    """Empty TTFT chunk followed by cumulative per-chunk usage of 4 then 8 tokens."""
    model_endpoint = create_model_endpoint(EndpointType.CHAT, streaming=True)
    model_endpoint.endpoint.allow_empty_content = True
    endpoint = ChatEndpoint(model_endpoint)
    empty_chunk = _chat_chunk({"content": ""})
    if empty_chunk_usage:
        empty_chunk["usage"] = {"completion_tokens": 0}
    request = RequestRecord(
        model_name="test-model",
        start_perf_ns=START_NS,
        end_perf_ns=1_500_000_000,
        responses=[
            _event(1_100_000_000, empty_chunk),
            _event(
                1_200_000_000,
                {**_chat_chunk({"content": "abcd"}), "usage": {"completion_tokens": 4}},
            ),
            _event(
                1_400_000_000,
                {
                    **_chat_chunk({"content": "efgh"}, finish_reason="stop"),
                    "usage": {"completion_tokens": 8},
                },
            ),
        ],
    )
    responses = endpoint.extract_response_data(request)
    return ParsedResponseRecord(
        request=request,
        responses=responses,
        token_counts=TokenCounts(
            input=100,
            output=8,
            first_content_chunk_tokens=first_content_chunk_completion_tokens(responses),
        ),
    )


@pytest.mark.parametrize("empty_chunk_usage", [True, False])
def test_per_chunk_usage_counts_all_tokens_after_empty_ttft_chunk(
    empty_chunk_usage: bool,
) -> None:
    record = _per_chunk_usage_record(empty_chunk_usage=empty_chunk_usage)
    results = run_simple_metrics_pipeline(
        [record],
        TTFTMetric.tag,
        RequestLatencyMetric.tag,
        FullDecodeDurationMetric.tag,
        InterTokenLatencyMetric.tag,
        FullResponseInterTokenLatencyMetric.tag,
        OutputTokenThroughputPerUserMetric.tag,
        FullResponseOutputTokenThroughputPerUserMetric.tag,
    )

    assert record.token_counts is not None
    assert record.token_counts.first_content_chunk_tokens == 0
    assert results[TTFTMetric.tag] == [100_000_000]
    assert results[InterTokenLatencyMetric.tag] == pytest.approx([300_000_000 / 8])
    assert results[FullResponseInterTokenLatencyMetric.tag] == pytest.approx(
        [400_000_000 / 8]
    )
    assert results[OutputTokenThroughputPerUserMetric.tag] == pytest.approx([8 / 0.3])
    assert results[FullResponseOutputTokenThroughputPerUserMetric.tag] == pytest.approx(
        [8 / 0.4]
    )
