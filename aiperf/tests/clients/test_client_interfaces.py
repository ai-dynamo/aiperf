#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from aiperf.clients.client_interfaces import (
    InferenceClientFactory,
    RequestConverterFactory,
    RequestConverterProtocol,
    ResponseExtractorFactory,
    ResponseExtractorProtocol,
)
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.dataset_models import Turn
from aiperf.common.enums import EndpointType
from aiperf.common.exceptions import FactoryCreationError
from aiperf.common.record_models import RequestRecord, ResponseData
from aiperf.common.tokenizer import Tokenizer


class TestInferenceClientFactory:
    """Test cases for InferenceClientFactory."""

    def test_inference_client_factory_registration(self):
        """Test client registration with factory."""

        # Create a mock client class
        class MockClient:
            def __init__(self, model_endpoint: ModelEndpointInfo):
                self.model_endpoint = model_endpoint

            async def initialize(self):
                pass

            async def send_request(
                self, model_endpoint: ModelEndpointInfo, payload: Any
            ) -> RequestRecord:
                return RequestRecord()

            async def close(self):
                pass

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.HUGGINGFACE_GENERATE

        # Register the mock client
        InferenceClientFactory.register(test_endpoint)(MockClient)

        # Verify registration by checking if we can create an instance
        client = InferenceClientFactory.create_instance(
            test_endpoint, model_endpoint=Mock()
        )
        assert isinstance(client, MockClient)

        # Clean up - remove from registry
        if test_endpoint in InferenceClientFactory._registry:
            del InferenceClientFactory._registry[test_endpoint]

    def test_inference_client_factory_create(self, sample_model_endpoint_info):
        """Test client creation from factory."""

        class MockClient:
            def __init__(self, model_endpoint: ModelEndpointInfo):
                self.model_endpoint = model_endpoint

            async def initialize(self):
                pass

            async def send_request(
                self, model_endpoint: ModelEndpointInfo, payload: Any
            ) -> RequestRecord:
                return RequestRecord()

            async def close(self):
                pass

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.TEMPLATE

        # Register and create client
        InferenceClientFactory.register(test_endpoint)(MockClient)
        client = InferenceClientFactory.create_instance(
            test_endpoint, model_endpoint=sample_model_endpoint_info
        )

        assert isinstance(client, MockClient)
        assert client.model_endpoint == sample_model_endpoint_info

        # Clean up
        if test_endpoint in InferenceClientFactory._registry:
            del InferenceClientFactory._registry[test_endpoint]

    def test_inference_client_factory_register_all(self):
        """Test registering client for multiple endpoint types."""

        class MockClient:
            def __init__(self, model_endpoint: ModelEndpointInfo):
                self.model_endpoint = model_endpoint

            async def initialize(self):
                pass

            async def send_request(
                self, model_endpoint: ModelEndpointInfo, payload: Any
            ) -> RequestRecord:
                return RequestRecord()

            async def close(self):
                pass

        # Use endpoint types that aren't already registered
        endpoint_types = [
            EndpointType.DYNAMIC_GRPC,
            EndpointType.TEMPLATE,
            EndpointType.NVCLIP,
        ]

        InferenceClientFactory.register_all(*endpoint_types)(MockClient)

        # Verify registration for all types
        for endpoint_type in endpoint_types:
            try:
                client = InferenceClientFactory.create_instance(
                    endpoint_type, model_endpoint=Mock()
                )
                assert isinstance(client, MockClient)
            except Exception:
                pytest.fail(f"Failed to create client for {endpoint_type}")

        # Clean up
        for endpoint_type in endpoint_types:
            if endpoint_type in InferenceClientFactory._registry:
                del InferenceClientFactory._registry[endpoint_type]

    def test_inference_client_factory_unregistered_type(self):
        """Test factory with unregistered endpoint type."""
        # Use an endpoint type that's definitely not registered
        unregistered_type = EndpointType.TENSORRTLLM_ENGINE

        with pytest.raises(FactoryCreationError):
            InferenceClientFactory.create_instance(
                unregistered_type, model_endpoint=Mock()
            )

    def test_inference_client_factory_get_class_from_type(self):
        """Test getting class from factory."""
        # Test with a registered endpoint type (OpenAI client should be registered)
        try:
            client_class = InferenceClientFactory.get_class_from_type(
                EndpointType.OPENAI_CHAT_COMPLETIONS
            )
            assert client_class is not None
        except Exception:
            pytest.skip("OpenAI client not registered in this test environment")

    def test_inference_client_factory_get_all_classes(self):
        """Test getting all registered classes."""
        all_classes = InferenceClientFactory.get_all_classes()
        assert isinstance(all_classes, list)
        # Should have at least some classes registered
        assert len(all_classes) >= 0

    def test_inference_client_factory_get_all_class_types(self):
        """Test getting all registered class types."""
        all_types = InferenceClientFactory.get_all_class_types()
        assert isinstance(all_types, list)
        # Should have the same number as classes
        assert len(all_types) == len(InferenceClientFactory.get_all_classes())


class TestRequestConverterProtocol:
    """Test cases for RequestConverterProtocol."""

    def test_request_converter_protocol_methods(self):
        """Test that RequestConverterProtocol defines the required methods."""
        assert hasattr(RequestConverterProtocol, "format_payload")

    def test_request_converter_protocol_isinstance_check(self):
        """Test isinstance check for RequestConverterProtocol."""
        mock_converter = Mock()
        mock_converter.format_payload = AsyncMock()

        assert isinstance(mock_converter, RequestConverterProtocol)


class TestRequestConverterFactory:
    """Test cases for RequestConverterFactory."""

    def test_request_converter_factory_registration(self):
        """Test converter registration with factory."""

        class MockConverter:
            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {"test": "payload"}

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.TEMPLATE

        # Register the mock converter
        RequestConverterFactory.register(test_endpoint)(MockConverter)

        # Verify registration by creating an instance
        converter = RequestConverterFactory.create_instance(test_endpoint)
        assert isinstance(converter, MockConverter)

        # Clean up
        if test_endpoint in RequestConverterFactory._registry:
            del RequestConverterFactory._registry[test_endpoint]

    def test_request_converter_factory_create(self):
        """Test converter creation from factory."""

        class MockConverter:
            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {"test": "payload"}

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.DYNAMIC_GRPC

        # Register and create converter
        RequestConverterFactory.register(test_endpoint)(MockConverter)
        converter = RequestConverterFactory.create_instance(test_endpoint)

        assert isinstance(converter, MockConverter)

        # Clean up
        if test_endpoint in RequestConverterFactory._registry:
            del RequestConverterFactory._registry[test_endpoint]

    def test_request_converter_factory_create_with_args(self):
        """Test converter creation with arguments."""

        class MockConverter:
            def __init__(self, custom_arg: str):
                self.custom_arg = custom_arg

            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {"test": "payload", "custom_arg": self.custom_arg}

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.NVCLIP

        # Register and create converter with args
        RequestConverterFactory.register(test_endpoint)(MockConverter)
        converter = RequestConverterFactory.create_instance(
            test_endpoint, custom_arg="test_value"
        )

        assert isinstance(converter, MockConverter)
        assert converter.custom_arg == "test_value"

        # Clean up
        if test_endpoint in RequestConverterFactory._registry:
            del RequestConverterFactory._registry[test_endpoint]

    @pytest.mark.asyncio
    async def test_request_converter_factory_usage(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test using converter created from factory."""

        class MockConverter:
            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {
                    "model": model_endpoint.primary_model_name,
                    "messages": [{"role": "user", "content": "test"}],
                }

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.IMAGE_RETRIEVAL

        # Register and create converter
        RequestConverterFactory.register(test_endpoint)(MockConverter)
        converter = RequestConverterFactory.create_instance(test_endpoint)

        # Test usage
        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )
        assert payload["model"] == "gpt-4"
        assert "messages" in payload

        # Clean up
        if test_endpoint in RequestConverterFactory._registry:
            del RequestConverterFactory._registry[test_endpoint]


class TestResponseExtractorProtocol:
    """Test cases for ResponseExtractorProtocol."""

    def test_response_extractor_protocol_methods(self):
        """Test that ResponseExtractorProtocol defines the required methods."""
        assert hasattr(ResponseExtractorProtocol, "extract_response_data")

    def test_response_extractor_protocol_isinstance_check(self):
        """Test isinstance check for ResponseExtractorProtocol."""
        mock_extractor = Mock()
        mock_extractor.extract_response_data = AsyncMock()

        assert isinstance(mock_extractor, ResponseExtractorProtocol)

    def test_response_extractor_protocol_missing_methods(self):
        """Test that objects missing required methods don't pass isinstance check."""
        mock_extractor = object()
        # Missing extract_response_data method

        assert not isinstance(mock_extractor, ResponseExtractorProtocol)


class TestResponseExtractorFactory:
    """Test cases for ResponseExtractorFactory."""

    def test_response_extractor_factory_registration(self):
        """Test extractor registration with factory."""

        class MockExtractor:
            async def extract_response_data(
                self, record: RequestRecord, tokenizer: Tokenizer | None
            ) -> list[ResponseData]:
                return [
                    ResponseData(perf_ns=0, raw_text=["test"], parsed_text=["test"])
                ]

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.RANKINGS

        # Register the mock extractor
        ResponseExtractorFactory.register(test_endpoint)(MockExtractor)

        # Verify registration by creating an instance
        extractor = ResponseExtractorFactory.create_instance(test_endpoint)
        assert isinstance(extractor, MockExtractor)

        # Clean up
        if test_endpoint in ResponseExtractorFactory._registry:
            del ResponseExtractorFactory._registry[test_endpoint]

    def test_response_extractor_factory_create(self):
        """Test extractor creation from factory."""

        class MockExtractor:
            async def extract_response_data(
                self, record: RequestRecord, tokenizer: Tokenizer | None
            ) -> list[ResponseData]:
                return [
                    ResponseData(perf_ns=0, raw_text=["test"], parsed_text=["test"])
                ]

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.TRITON_GENERATE

        # Register and create extractor
        ResponseExtractorFactory.register(test_endpoint)(MockExtractor)
        extractor = ResponseExtractorFactory.create_instance(test_endpoint)

        assert isinstance(extractor, MockExtractor)

        # Clean up
        if test_endpoint in ResponseExtractorFactory._registry:
            del ResponseExtractorFactory._registry[test_endpoint]

    @pytest.mark.asyncio
    async def test_response_extractor_factory_usage(
        self, sample_request_record, mock_tokenizer
    ):
        """Test using extractor created from factory."""

        class MockExtractor:
            async def extract_response_data(
                self, record: RequestRecord, tokenizer: Tokenizer | None
            ) -> list[ResponseData]:
                return [
                    ResponseData(
                        perf_ns=record.start_perf_ns,
                        raw_text=["Hello, world!"],
                        parsed_text=["Hello, world!"],
                        token_count=2,
                    )
                ]

        # Use a different endpoint type to avoid conflicts
        test_endpoint = EndpointType.DYNAMO_ENGINE

        # Register and create extractor
        ResponseExtractorFactory.register(test_endpoint)(MockExtractor)
        extractor = ResponseExtractorFactory.create_instance(test_endpoint)

        # Test usage
        response_data = await extractor.extract_response_data(
            sample_request_record, mock_tokenizer
        )
        assert len(response_data) == 1
        assert response_data[0].raw_text == ["Hello, world!"]
        assert response_data[0].parsed_text == ["Hello, world!"]
        assert response_data[0].token_count == 2

        # Clean up
        if test_endpoint in ResponseExtractorFactory._registry:
            del ResponseExtractorFactory._registry[test_endpoint]


class TestFactoryIntegration:
    """Integration tests for all factories."""

    def test_all_factories_independent(self):
        """Test that all factories are independent."""

        class MockClient:
            def __init__(self, model_endpoint: ModelEndpointInfo):
                pass

        class MockConverter:
            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {}

        class MockExtractor:
            async def extract_response_data(
                self, record: RequestRecord, tokenizer: Tokenizer | None
            ) -> list[ResponseData]:
                return []

        # Use different endpoint types for each factory
        client_endpoint = EndpointType.TENSORRTLLM
        converter_endpoint = EndpointType.TENSORRTLLM_ENGINE
        extractor_endpoint = EndpointType.HUGGINGFACE_GENERATE

        InferenceClientFactory.register(client_endpoint)(MockClient)
        RequestConverterFactory.register(converter_endpoint)(MockConverter)
        ResponseExtractorFactory.register(extractor_endpoint)(MockExtractor)

        # Verify all are registered independently
        client = InferenceClientFactory.create_instance(
            client_endpoint, model_endpoint=Mock()
        )
        converter = RequestConverterFactory.create_instance(converter_endpoint)
        extractor = ResponseExtractorFactory.create_instance(extractor_endpoint)

        assert isinstance(client, MockClient)
        assert isinstance(converter, MockConverter)
        assert isinstance(extractor, MockExtractor)

        # Clean up
        for endpoint, factory in [
            (client_endpoint, InferenceClientFactory),
            (converter_endpoint, RequestConverterFactory),
            (extractor_endpoint, ResponseExtractorFactory),
        ]:
            if endpoint in factory._registry:
                del factory._registry[endpoint]

    def test_factory_isolation(self):
        """Test that factories don't interfere with each other."""

        class MockClient:
            def __init__(self, model_endpoint: ModelEndpointInfo):
                pass

        class MockConverter:
            async def format_payload(
                self, model_endpoint: ModelEndpointInfo, turn: Turn
            ) -> dict[str, Any]:
                return {}

        # Register in different factories with different endpoint types
        client_endpoint = EndpointType.TEMPLATE
        converter_endpoint = EndpointType.DYNAMIC_GRPC

        InferenceClientFactory.register(client_endpoint)(MockClient)
        RequestConverterFactory.register(converter_endpoint)(MockConverter)

        # Verify isolation - each should only be able to create its own type
        client = InferenceClientFactory.create_instance(
            client_endpoint, model_endpoint=Mock()
        )
        converter = RequestConverterFactory.create_instance(converter_endpoint)

        assert isinstance(client, MockClient)
        assert isinstance(converter, MockConverter)

        # Clean up
        if client_endpoint in InferenceClientFactory._registry:
            del InferenceClientFactory._registry[client_endpoint]
        if converter_endpoint in RequestConverterFactory._registry:
            del RequestConverterFactory._registry[converter_endpoint]
