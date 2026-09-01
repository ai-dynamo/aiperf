# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.models import (
    RequestRecord,
    SSEField,
    SSEMessage,
    TokenIdsResponseData,
    Turn,
)
from aiperf.endpoints.sglang_generate import SGLangGenerateEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


def _sse(json_data: dict, perf_ns: int = 123) -> SSEMessage:
    return SSEMessage(
        perf_ns=perf_ns,
        packets=[SSEField(name="data", value=orjson.dumps(json_data).decode())],
    )


@pytest.fixture
def model_endpoint():
    return create_model_endpoint(
        EndpointType.SGLANG_GENERATE,
        streaming=True,
    )


@pytest.fixture
def endpoint(model_endpoint):
    return SGLangGenerateEndpoint(model_endpoint)


def test_metadata_requires_token_ids_and_streaming():
    metadata = plugins.get_endpoint_metadata(EndpointType.SGLANG_GENERATE)
    assert metadata.endpoint_path == "/generate"
    assert metadata.supports_streaming
    assert metadata.produces_tokens
    assert metadata.tokenizes_input
    assert metadata.requires_token_ids


def test_format_payload_accumulates_context_and_maps_priority(endpoint, model_endpoint):
    turns = [
        Turn(token_ids=[1, 2], max_tokens=3),
        Turn(role="assistant", token_ids=[4, 5]),
        Turn(
            token_ids=[6],
            max_tokens=7,
            extra_body={
                "nvext": {"agent_hints": {"strict_priority": 192}},
                "sampling_params": {"temperature": 0.2},
                "return_logprob": True,
            },
        ),
    ]
    request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

    payload = endpoint.format_payload(request_info)

    assert payload == {
        "rid": "test-request-id",
        "input_ids": [1, 2, 4, 5, 6],
        "sampling_params": {
            "ignore_eos": True,
            "max_new_tokens": 7,
            "temperature": 0.2,
        },
        "stream": True,
        "priority": 192,
        "return_logprob": True,
    }


def test_format_payload_reset_context(endpoint, model_endpoint):
    turns = [
        Turn(token_ids=[1, 2]),
        Turn(token_ids=[9], reset_context=True, max_tokens=2),
    ]
    payload = endpoint.format_payload(
        create_request_info(model_endpoint=model_endpoint, turns=turns)
    )
    assert payload["input_ids"] == [9]


def test_format_payload_requires_streaming_and_token_ids(endpoint, model_endpoint):
    model_endpoint.endpoint.streaming = False
    with pytest.raises(ValueError, match="requires streaming"):
        endpoint.format_payload(
            create_request_info(
                model_endpoint=model_endpoint, turns=[Turn(token_ids=[1])]
            )
        )

    model_endpoint.endpoint.streaming = True
    with pytest.raises(ValueError, match="requires token_ids"):
        endpoint.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[Turn()])
        )


def test_parse_response_preserves_ids_usage_and_metadata(endpoint):
    parsed = endpoint.parse_response(
        _sse(
            {
                "output_ids": [42],
                "meta_info": {
                    "id": "req-1",
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "finish_reason": {"type": "length"},
                },
            }
        )
    )

    assert parsed is not None
    assert isinstance(parsed.data, TokenIdsResponseData)
    assert parsed.data.token_ids == [42]
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens == 10
    assert parsed.usage.completion_tokens == 3
    assert parsed.usage.total_tokens == 13
    assert parsed.metadata["id"] == "req-1"
    assert parsed.metadata["finish_reason"] == {"type": "length"}


def test_done_and_invalid_output_ids_are_ignored(endpoint):
    done = SSEMessage.parse("data: [DONE]", perf_ns=123)
    assert endpoint.parse_response(done) is None
    assert endpoint.parse_response(_sse({"output_ids": [True]})) is None


def test_assistant_turn_replays_streamed_output_ids(endpoint):
    record = RequestRecord(
        responses=[
            _sse(
                {
                    "output_ids": [10],
                    "meta_info": {"prompt_tokens": 4, "completion_tokens": 1},
                },
                perf_ns=100,
            ),
            _sse(
                {
                    "output_ids": [11, 12],
                    "meta_info": {"prompt_tokens": 4, "completion_tokens": 3},
                },
                perf_ns=200,
            ),
        ]
    )

    turn = endpoint.build_assistant_turn(record)

    assert turn is not None
    assert turn.role == "assistant"
    assert turn.token_ids == [10, 11, 12]


def test_extract_payload_inputs_counts_pretokenized_ids(endpoint):
    extracted = endpoint.extract_payload_inputs({"input_ids": [1, 2, 3]})
    assert extracted.pretokenised_token_count == 3
    assert extracted.texts == []
