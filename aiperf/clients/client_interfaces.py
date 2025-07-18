# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, Protocol, runtime_checkable

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import EndpointType
from aiperf.common.factories import FactoryMixin
from aiperf.common.models import RequestRecord, ResponseData, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.types import RequestInputT, RequestOutputT

################################################################################
# Inference Clients
################################################################################


@runtime_checkable
class InferenceClientProtocol(Protocol, Generic[RequestInputT]):
    """Protocol for an inference server client.

    This protocol defines the methods that must be implemented by any inference server client
    implementation that is compatible with the AIPerf framework.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo) -> None:
        """Create a new inference server client based on the provided configuration."""
        ...

    async def initialize(self) -> None:
        """Initialize the inference server client in an asynchronous context."""
        ...

    async def send_request(
        self,
        model_endpoint: ModelEndpointInfo,
        payload: RequestInputT,
    ) -> RequestRecord:
        """Send a request to the inference server.

        This method is used to send a request to the inference server.

        Args:
            model_endpoint: The endpoint to send the request to.
            payload: The payload to send to the inference server.
        Returns:
            The raw response from the inference server.
        """
        ...

    async def close(self) -> None:
        """Close the client."""
        ...


class InferenceClientFactory(FactoryMixin[EndpointType, InferenceClientProtocol]):
    """Factory for registering and creating InferenceClientProtocol instances based on the specified endpoint type.
    see: :class:`FactoryMixin` for more details.
    """


################################################################################
# Request Converters (Input Converters)
################################################################################


@runtime_checkable
class RequestConverterProtocol(Protocol, Generic[RequestOutputT]):
    """Protocol for a request converter that converts a raw request to a formatted request for the inference server."""

    async def format_payload(
        self, model_endpoint: ModelEndpointInfo, turn: Turn
    ) -> RequestOutputT:
        """Format the turn for the inference server."""
        ...


class RequestConverterFactory(FactoryMixin[EndpointType, RequestConverterProtocol]):
    """Factory for registering and creating RequestConverterProtocol instances based on the specified request payload type.
    see: :class:`FactoryMixin` for more details.
    """


################################################################################
# Response Extractors (Output Converters)
################################################################################


@runtime_checkable
class ResponseExtractorProtocol(Protocol):
    """Protocol for a response extractor that extracts the response data from a raw inference server
    response and converts it to a list of ResponseData objects."""

    async def extract_response_data(
        self, record: RequestRecord, tokenizer: Tokenizer | None
    ) -> list[ResponseData]:
        """Extract the response data from a raw inference server response and convert it to a list of ResponseData objects."""
        ...


class ResponseExtractorFactory(FactoryMixin[EndpointType, ResponseExtractorProtocol]):
    """Factory for registering and creating ResponseExtractorProtocol instances based on the specified response extractor type.
    see: :class:`FactoryMixin` for more details.
    """
