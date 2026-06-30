# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.models import RequestRecord, TextResponse, Turn
from aiperf.endpoints.vllm_generate import VllmGenerateEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.fixture
def endpoint():
    return VllmGenerateEndpoint(
        create_model_endpoint(EndpointType.VLLM_GENERATE, model_name="test-model")
    )


def test_metadata():
    metadata = plugins.get_endpoint_metadata(EndpointType.VLLM_GENERATE)
    assert metadata.endpoint_path == "/inference/v1/generate"
    assert metadata.supports_streaming is False
    assert metadata.produces_tokens is True
    assert metadata.tokenizes_input is False


def test_format_payload(endpoint):
    request = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        max_tokens=17,
        extra_body={"token_ids": [1, 2, 3], "sampling_params": {"temperature": 0}},
    )

    payload = endpoint.format_payload(request)

    assert payload == {
        "model": "test-model",
        "token_ids": [1, 2, 3],
        "sampling_params": {"temperature": 0, "max_tokens": 17},
        "stream": False,
        "request_id": "test-request-id",
    }


def test_format_payload_rejects_missing_tokens(endpoint):
    request = create_request_info(model_endpoint=endpoint.model_endpoint)
    with pytest.raises(ValueError, match="token_ids"):
        endpoint.format_payload(request)


def test_extract_payload_inputs_counts_exact_ids(endpoint):
    extracted = endpoint.extract_payload_inputs({"token_ids": [10, 11, 12, 13]})
    assert extracted.pretokenised_token_count == 4


def test_extract_response_data_reconstructs_usage(endpoint):
    response = TextResponse(
        perf_ns=123,
        text=orjson.dumps(
            {
                "request_id": "req-1",
                "choices": [
                    {"index": 0, "token_ids": [20, 21], "finish_reason": "stop"}
                ],
            }
        ).decode(),
        content_type="application/json",
    )
    record = RequestRecord(
        model_name="test-model",
        responses=[response],
        turns=[Turn(role="user", raw_payload={"token_ids": [1, 2, 3]})],
    )

    parsed = endpoint.extract_response_data(record)

    assert len(parsed) == 1
    assert parsed[0].usage.prompt_tokens == 3
    assert parsed[0].usage.completion_tokens == 2
    assert parsed[0].usage.total_tokens == 5
    assert parsed[0].metadata["completion_token_ids"] == [20, 21]
