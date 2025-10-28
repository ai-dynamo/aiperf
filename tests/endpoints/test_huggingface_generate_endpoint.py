# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint


class TestHuggingFaceGenerateEndpoint:
    """Tests for the Hugging Face TGI /generate endpoint."""

    @pytest.fixture
    def model_endpoint(self):
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="HuggingFaceH4/zephyr-7b-beta")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.HUGGINGFACE_GENERATE,
                base_url="http://localhost:8080",
                custom_endpoint="/generate",
            ),
        )

    def test_format_payload_basic(self, model_endpoint, sample_conversations):
        """Basic text-only request formatting."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        expected_payload = {
            "inputs": turn.texts[0].contents[0],
            "parameters": {},
        }
        assert payload == expected_payload

    def test_format_payload_with_max_tokens(self, model_endpoint, sample_conversations):
        """Verify max_new_tokens parameter is included."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turn.max_tokens = 50
        turns = [turn]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)
        assert payload["parameters"]["max_new_tokens"] == 50
        assert payload["inputs"] == turn.texts[0].contents[0]

    def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        """Extra parameters (temperature, top_p, etc.) get merged."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        model_endpoint.endpoint.extra = {"temperature": 0.8, "top_p": 0.9}
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)
        assert payload["parameters"]["temperature"] == 0.8
        assert payload["parameters"]["top_p"] == 0.9

    def test_parse_response_dict(self, model_endpoint):
        """Response comes back as a single dict."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        fake_response = type(
            "Resp",
            (),
            {
                "perf_ns": 123,
                "get_json": lambda _: {"generated_text": "Hello from Zephyr!"},
            },
        )()

        parsed = endpoint.parse_response(fake_response)
        assert parsed.data.text == "Hello from Zephyr!"

    def test_parse_response_list(self, model_endpoint):
        """Response comes back as a list of dicts."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        fake_response = type(
            "Resp",
            (),
            {"perf_ns": 123, "get_json": lambda _: [{"generated_text": "Hi there!"}]},
        )()

        parsed = endpoint.parse_response(fake_response)
        assert parsed.data.text == "Hi there!"

    def test_parse_response_empty(self, model_endpoint):
        """Empty or invalid JSON returns None."""
        endpoint = HuggingFaceGenerateEndpoint(model_endpoint)
        fake_response = type("Resp", (), {"perf_ns": 123, "get_json": lambda _: {}})()
        assert endpoint.parse_response(fake_response) is None
