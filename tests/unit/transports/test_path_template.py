# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for path template substitution in AioHttpTransport.get_url()."""

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin.enums import EndpointType
from aiperf.transports.aiohttp_transport import AioHttpTransport


def _create_model_endpoint(
    endpoint_type: EndpointType = EndpointType.CHAT,
    model_name: str = "my-model",
    base_url: str = "http://localhost:8000",
    custom_endpoint: str | None = None,
    streaming: bool = False,
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_urls=[base_url],
            custom_endpoint=custom_endpoint,
            streaming=streaming,
        ),
    )


def _create_request_info(model_endpoint: ModelEndpointInfo) -> RequestInfo:
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation-id",
    )


class TestPathTemplateSubstitution:
    """Tests for {model_name} template substitution in get_url()."""

    @pytest.mark.parametrize(
        "custom_endpoint,model_name,expected_path",
        [
            (
                "/v2/models/{model_name}/infer",
                "triton-llm",
                "http://localhost:8000/v2/models/triton-llm/infer",
            ),
            (
                "/v1/models/{model_name}:predict",
                "sklearn-iris",
                "http://localhost:8000/v1/models/sklearn-iris:predict",
            ),
            (
                "/v1/chat/completions",
                "my-model",
                "http://localhost:8000/v1/chat/completions",
            ),
        ],
        ids=["v2-infer-template", "v1-predict-template", "no-template"],
    )
    def test_custom_endpoint_path_template(
        self, custom_endpoint, model_name, expected_path
    ):
        """Test path template substitution in custom endpoint paths."""
        model_endpoint = _create_model_endpoint(
            model_name=model_name,
            custom_endpoint=custom_endpoint,
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == expected_path

    def test_metadata_path_template_v2_infer(self):
        """Test path template substitution via endpoint metadata (kserve_v2_infer)."""
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_V2_INFER,
            model_name="triton-trtllm",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/v2/models/triton-trtllm/infer"

    def test_metadata_path_template_v1_predict(self):
        """Test path template substitution via endpoint metadata (kserve_v1_predict)."""
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_V1_PREDICT,
            model_name="sklearn-iris",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/v1/models/sklearn-iris:predict"

    def test_kserve_chat_url(self):
        """Test kserve_chat produces /openai/v1/chat/completions path."""
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_CHAT,
            model_name="vllm-qwen",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/openai/v1/chat/completions"

    def test_kserve_completions_url(self):
        """Test kserve_completions produces /openai/v1/completions path."""
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_COMPLETIONS,
            model_name="vllm-model",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/openai/v1/completions"

    def test_kserve_embeddings_url(self):
        """Test kserve_embeddings produces /openai/v1/embeddings path."""
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_EMBEDDINGS,
            model_name="embed-model",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/openai/v1/embeddings"

    def test_v1_predict_with_v1_base_url_dedup(self):
        """Test that /v1 base URL dedup works correctly for kserve_v1_predict.

        When base_url ends in /v1 and path starts with v1/,
        the v1/ prefix is removed to avoid /v1/v1/... duplication.
        """
        model_endpoint = _create_model_endpoint(
            endpoint_type=EndpointType.KSERVE_V1_PREDICT,
            model_name="sklearn-iris",
            base_url="http://localhost:8000/v1",
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)
        request_info = _create_request_info(model_endpoint)

        url = transport.get_url(request_info)

        assert url == "http://localhost:8000/v1/models/sklearn-iris:predict"
