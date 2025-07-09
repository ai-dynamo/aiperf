#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.config.config_defaults import EndPointDefaults
from aiperf.common.enums import EndpointType, Modality, ModelSelectionStrategy


class TestModelInfo:
    """Test cases for ModelInfo class."""

    def test_model_info_creation(self):
        """Test ModelInfo can be created with valid parameters."""
        model_info = ModelInfo(
            name="gpt-4", version="2024-01-01", modality=Modality.TEXT
        )
        assert model_info.name == "gpt-4"
        assert model_info.version == "2024-01-01"
        assert model_info.modality == Modality.TEXT

    def test_model_info_defaults(self):
        """Test ModelInfo defaults are set correctly."""
        model_info = ModelInfo(name="gpt-4")
        assert model_info.name == "gpt-4"
        assert model_info.version is None
        assert model_info.modality == Modality.TEXT

    def test_model_info_with_different_modalities(self):
        """Test ModelInfo with different modalities."""
        modalities = [
            Modality.TEXT,
            Modality.IMAGE,
            Modality.AUDIO,
            Modality.MULTIMODAL,
        ]
        for modality in modalities:
            model_info = ModelInfo(name="test-model", modality=modality)
            assert model_info.modality == modality

    def test_model_info_validation_empty_name(self):
        """Test ModelInfo validates empty name."""
        with pytest.raises(ValueError):
            ModelInfo(name="")

    def test_model_info_serialization(self):
        """Test ModelInfo can be serialized/deserialized."""
        model_info = ModelInfo(
            name="gpt-4", version="2024-01-01", modality=Modality.TEXT
        )
        data = model_info.model_dump()
        reconstructed = ModelInfo.model_validate(data)
        assert reconstructed.name == model_info.name
        assert reconstructed.version == model_info.version
        assert reconstructed.modality == model_info.modality


class TestModelListInfo:
    """Test cases for ModelListInfo class."""

    def test_model_list_info_creation(self, sample_model_info):
        """Test ModelListInfo can be created with valid parameters."""
        model_list_info = ModelListInfo(
            models=[sample_model_info],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        )
        assert len(model_list_info.models) == 1
        assert model_list_info.models[0] == sample_model_info
        assert (
            model_list_info.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        )

    def test_model_list_info_multiple_models(self):
        """Test ModelListInfo with multiple models."""
        models = [
            ModelInfo(name="gpt-4", modality=Modality.TEXT),
            ModelInfo(name="gpt-3.5-turbo", modality=Modality.TEXT),
            ModelInfo(name="dall-e-3", modality=Modality.IMAGE),
        ]
        model_list_info = ModelListInfo(
            models=models, model_selection_strategy=ModelSelectionStrategy.RANDOM
        )
        assert len(model_list_info.models) == 3
        assert model_list_info.model_selection_strategy == ModelSelectionStrategy.RANDOM

    def test_model_list_info_from_user_config(self):
        """Test ModelListInfo.from_user_config method."""
        user_config = Mock()
        user_config.model_names = ["gpt-4", "gpt-3.5-turbo"]
        user_config.endpoint.model_selection_strategy = (
            ModelSelectionStrategy.ROUND_ROBIN
        )

        model_list_info = ModelListInfo.from_user_config(user_config)

        assert len(model_list_info.models) == 2
        assert model_list_info.models[0].name == "gpt-4"
        assert model_list_info.models[1].name == "gpt-3.5-turbo"
        assert (
            model_list_info.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        )

    def test_model_list_info_validation_empty_models(self):
        """Test ModelListInfo validates empty models list."""
        with pytest.raises(ValueError):
            ModelListInfo(
                models=[], model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN
            )

    def test_model_list_info_different_strategies(self):
        """Test ModelListInfo with different selection strategies."""
        model_info = ModelInfo(name="gpt-4")
        strategies = [
            ModelSelectionStrategy.ROUND_ROBIN,
            ModelSelectionStrategy.RANDOM,
            ModelSelectionStrategy.MODALITY_AWARE,
        ]
        for strategy in strategies:
            model_list_info = ModelListInfo(
                models=[model_info], model_selection_strategy=strategy
            )
            assert model_list_info.model_selection_strategy == strategy


class TestEndpointInfo:
    """Test cases for EndpointInfo class."""

    def test_endpoint_info_creation(self):
        """Test EndpointInfo can be created with valid parameters."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            base_url="https://api.openai.com",
            streaming=False,
            api_key="test-api-key",
            timeout=30.0,
        )
        assert endpoint_info.type == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert endpoint_info.base_url == "https://api.openai.com"
        assert endpoint_info.streaming is False
        assert endpoint_info.api_key == "test-api-key"
        assert endpoint_info.timeout == 30.0

    def test_endpoint_info_defaults(self):
        """Test EndpointInfo defaults are set correctly."""
        endpoint_info = EndpointInfo()
        assert endpoint_info.type == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert endpoint_info.base_url is None
        assert endpoint_info.custom_endpoint is None
        assert endpoint_info.streaming is False
        assert endpoint_info.headers is None
        assert endpoint_info.api_key is None
        assert endpoint_info.timeout == EndPointDefaults.TIMEOUT

    def test_endpoint_info_with_custom_endpoint(self):
        """Test EndpointInfo with custom endpoint."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            base_url="https://custom-api.com",
            custom_endpoint="/custom/chat",
            streaming=True,
        )
        assert endpoint_info.type == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert endpoint_info.base_url == "https://custom-api.com"
        assert endpoint_info.custom_endpoint == "/custom/chat"
        assert endpoint_info.streaming is True

    def test_endpoint_info_with_headers_and_params(self):
        """Test EndpointInfo with headers and URL parameters."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_EMBEDDINGS,
            base_url="https://api.openai.com",
            headers={"Authorization": "Bearer token", "X-Custom": "value"},
            url_params={"dimensions": 1536, "encoding_format": "float"},
            extra={"temperature": 0.7},
        )
        assert endpoint_info.headers == {
            "Authorization": "Bearer token",
            "X-Custom": "value",
        }
        assert endpoint_info.url_params == {
            "dimensions": 1536,
            "encoding_format": "float",
        }
        assert endpoint_info.extra == {"temperature": 0.7}

    def test_endpoint_info_from_user_config(self):
        """Test EndpointInfo.from_user_config method."""
        user_config = Mock()
        user_config.endpoint.type = EndpointType.OPENAI_CHAT_COMPLETIONS
        user_config.endpoint.url = "https://api.openai.com"
        user_config.endpoint.custom = "/custom/endpoint"
        user_config.endpoint.streaming = True
        user_config.endpoint.timeout = 60.0
        user_config.endpoint.api_key = "test-key"
        user_config.input.headers = {"Custom-Header": "value"}
        user_config.input.extra = {"temperature": 0.8}

        endpoint_info = EndpointInfo.from_user_config(user_config)

        assert endpoint_info.type == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert endpoint_info.base_url == "https://api.openai.com"
        assert endpoint_info.custom_endpoint == "/custom/endpoint"
        assert endpoint_info.streaming is True
        assert endpoint_info.timeout == 60.0
        assert endpoint_info.api_key == "test-key"
        assert endpoint_info.headers == {"Custom-Header": "value"}
        assert endpoint_info.extra == {"temperature": 0.8}

    def test_endpoint_info_different_types(self):
        """Test EndpointInfo with different endpoint types."""
        endpoint_types = [
            EndpointType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_COMPLETIONS,
            EndpointType.OPENAI_EMBEDDINGS,
            EndpointType.OPENAI_RESPONSES,
        ]
        for endpoint_type in endpoint_types:
            endpoint_info = EndpointInfo(type=endpoint_type)
            assert endpoint_info.type == endpoint_type

    def test_endpoint_info_ssl_options(self):
        """Test EndpointInfo with SSL options."""
        ssl_options = {"verify": False, "cert": "/path/to/cert.pem"}
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS, ssl_options=ssl_options
        )
        assert endpoint_info.ssl_options == ssl_options


class TestModelEndpointInfo:
    """Test cases for ModelEndpointInfo class."""

    def test_model_endpoint_info_creation(
        self, sample_model_list_info, sample_endpoint_info
    ):
        """Test ModelEndpointInfo can be created with valid parameters."""
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=sample_endpoint_info
        )
        assert model_endpoint_info.models == sample_model_list_info
        assert model_endpoint_info.endpoint == sample_endpoint_info

    def test_model_endpoint_info_from_user_config(self, sample_user_config):
        """Test ModelEndpointInfo.from_user_config method."""
        model_endpoint_info = ModelEndpointInfo.from_user_config(sample_user_config)

        assert len(model_endpoint_info.models.models) == 1
        assert model_endpoint_info.models.models[0].name == "gpt-4"
        assert model_endpoint_info.endpoint.type == EndpointType.OPENAI_CHAT_COMPLETIONS
        assert model_endpoint_info.endpoint.base_url == "https://api.openai.com"
        assert model_endpoint_info.endpoint.api_key == "test-api-key"

    def test_model_endpoint_info_url_property(self, sample_model_endpoint_info):
        """Test ModelEndpointInfo.url property."""
        url = sample_model_endpoint_info.url
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_model_endpoint_info_url_with_custom_endpoint(self, sample_model_list_info):
        """Test ModelEndpointInfo.url property with custom endpoint."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            base_url="https://custom-api.com",
            custom_endpoint="/custom/chat",
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )
        url = model_endpoint_info.url
        assert url == "https://custom-api.com/custom/chat"

    def test_model_endpoint_info_url_without_base_url(self, sample_model_list_info):
        """Test ModelEndpointInfo.url property without base URL."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS, base_url=None
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )
        url = model_endpoint_info.url
        assert url == "/v1/chat/completions"

    def test_model_endpoint_info_url_with_trailing_slash(self, sample_model_list_info):
        """Test ModelEndpointInfo.url property with trailing slash in base URL."""
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_CHAT_COMPLETIONS,
            base_url="https://api.openai.com/",
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )
        url = model_endpoint_info.url
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_model_endpoint_info_primary_model_property(
        self, sample_model_endpoint_info
    ):
        """Test ModelEndpointInfo.primary_model property."""
        primary_model = sample_model_endpoint_info.primary_model
        assert primary_model.name == "gpt-4"
        assert primary_model.version == "2024-01-01"
        assert primary_model.modality == Modality.TEXT

    def test_model_endpoint_info_primary_model_name_property(
        self, sample_model_endpoint_info
    ):
        """Test ModelEndpointInfo.primary_model_name property."""
        primary_model_name = sample_model_endpoint_info.primary_model_name
        assert primary_model_name == "gpt-4"

    def test_model_endpoint_info_primary_model_multiple_models(
        self, sample_model_list_info
    ):
        """Test ModelEndpointInfo.primary_model with multiple models."""
        # Add additional models
        additional_model = ModelInfo(name="gpt-3.5-turbo", modality=Modality.TEXT)
        sample_model_list_info.models.append(additional_model)

        endpoint_info = EndpointInfo(type=EndpointType.OPENAI_CHAT_COMPLETIONS)
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        # Primary model should be the first one
        primary_model = model_endpoint_info.primary_model
        assert primary_model.name == "gpt-4"

    def test_model_endpoint_info_different_endpoint_types(self, sample_model_list_info):
        """Test ModelEndpointInfo with different endpoint types."""
        endpoint_types_and_paths = [
            (EndpointType.OPENAI_CHAT_COMPLETIONS, "/v1/chat/completions"),
            (EndpointType.OPENAI_COMPLETIONS, "/v1/completions"),
            (EndpointType.OPENAI_EMBEDDINGS, "/v1/embeddings"),
            (EndpointType.OPENAI_RESPONSES, "/v1/responses"),
        ]

        for endpoint_type, expected_path in endpoint_types_and_paths:
            endpoint_info = EndpointInfo(
                type=endpoint_type, base_url="https://api.openai.com"
            )
            model_endpoint_info = ModelEndpointInfo(
                models=sample_model_list_info, endpoint=endpoint_info
            )
            url = model_endpoint_info.url
            assert url == f"https://api.openai.com{expected_path}"

    def test_model_endpoint_info_serialization(self, sample_model_endpoint_info):
        """Test ModelEndpointInfo can be serialized/deserialized."""
        data = sample_model_endpoint_info.model_dump()
        reconstructed = ModelEndpointInfo.model_validate(data)
        assert (
            reconstructed.models.models[0].name
            == sample_model_endpoint_info.models.models[0].name
        )
        assert reconstructed.endpoint.type == sample_model_endpoint_info.endpoint.type
        assert (
            reconstructed.endpoint.base_url
            == sample_model_endpoint_info.endpoint.base_url
        )

    def test_model_endpoint_info_deep_copy(self, sample_model_endpoint_info):
        """Test ModelEndpointInfo can be deep copied."""
        copied = sample_model_endpoint_info.model_copy(deep=True)
        assert (
            copied.models.models[0].name
            == sample_model_endpoint_info.models.models[0].name
        )
        assert copied.endpoint.type == sample_model_endpoint_info.endpoint.type

        # Verify it's a deep copy
        copied.models.models[0].name = "different-model"
        assert sample_model_endpoint_info.models.models[0].name == "gpt-4"
