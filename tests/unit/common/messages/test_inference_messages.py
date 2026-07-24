# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.messages import InferenceResultsMessage
from aiperf.common.messages.inference_messages import encode_parsed_responses
from aiperf.common.models import (
    BaseResponseData,
    EmbeddingResponseData,
    ImageDataItem,
    ImageResponseData,
    ImageRetrievalResponseData,
    ParsedResponse,
    RAGSources,
    RankingsResponseData,
    ReasoningResponseData,
    RequestRecord,
    TextResponseData,
    ToolCallResponseData,
    VideoResponseData,
)


def test_inference_results_parsed_responses_round_trip_builtin_types():
    responses = [
        ParsedResponse(perf_ns=1, data=BaseResponseData()),
        ParsedResponse(perf_ns=2, data=EmbeddingResponseData([[1.0, 2.0]])),
        ParsedResponse(
            perf_ns=3,
            data=ImageResponseData(
                images=[ImageDataItem(url="https://example.test/image.png")],
                size="1024x1024",
            ),
        ),
        ParsedResponse(perf_ns=4, data=ImageRetrievalResponseData([{"id": "image-1"}])),
        ParsedResponse(perf_ns=5, data=RankingsResponseData([{"index": 0}])),
        ParsedResponse(
            perf_ns=6,
            data=ReasoningResponseData(content="answer", reasoning="thought"),
        ),
        ParsedResponse(
            perf_ns=7,
            data=TextResponseData("hello"),
            usage={"prompt_tokens": 11, "completion_tokens": 7},
            sources=RAGSources({"document": "source"}),
            metadata={"finish_reason": "stop"},
        ),
        ParsedResponse(perf_ns=8, data=ToolCallResponseData("tool()")),
        ParsedResponse(perf_ns=9, data=VideoResponseData(video_id="video-1")),
        ParsedResponse(perf_ns=10, usage={"completion_tokens": 7}),
    ]
    payloads = encode_parsed_responses(responses)
    assert payloads is not None

    message = InferenceResultsMessage(
        service_id="worker-1",
        record=RequestRecord(start_perf_ns=1, end_perf_ns=11),
        parsed_responses=payloads,
        last_response_perf_ns=10,
        raw_response_count=11,
        responses_compacted=True,
    )

    restored = InferenceResultsMessage.from_json(message.to_json_bytes())
    restored_responses = [
        payload.to_parsed_response() for payload in restored.parsed_responses or []
    ]

    assert restored_responses == responses
    assert restored.last_response_perf_ns == 10
    assert restored.raw_response_count == 11
    assert restored.responses_compacted is True


def test_inference_results_legacy_message_defaults_to_raw_path():
    message = InferenceResultsMessage(
        service_id="worker-1",
        record=RequestRecord(start_perf_ns=1, end_perf_ns=2),
    )

    legacy_data = message.model_dump(
        mode="json",
        exclude={
            "parsed_responses",
            "last_response_perf_ns",
            "raw_response_count",
            "responses_compacted",
        },
    )
    restored = InferenceResultsMessage.from_json(legacy_data)

    assert restored.parsed_responses is None
    assert restored.last_response_perf_ns is None
    assert restored.raw_response_count is None
    assert restored.responses_compacted is False
