# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request converters for OpenAI requests.

The converters are registered with the RequestConverterFactory and are used to convert
the input data into the format expected by the OpenAI API.
"""

import logging
from typing import Any

from aiperf.clients.client_interfaces import (
    RequestConverterFactory,
    RequestConverterProtocol,
)
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.dataset_models import Turn
from aiperf.common.enums import EndpointType


@RequestConverterFactory.register(EndpointType.OPENAI_CHAT_COMPLETIONS)
class OpenAIChatCompletionRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI chat completion requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for a chat completion request."""

        messages = [
            {
                "role": text.role or "user",
                "name": text.name,
                "content": content,
            }
            for text in turn.text
            for content in text.content
            if content
        ]

        payload = {
            "messages": messages,
            "model": model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload


@RequestConverterFactory.register(EndpointType.OPENAI_COMPLETIONS)
class OpenAICompletionRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI completion requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for a completion request."""

        payload = {
            "prompt": turn.text,
            "model": model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload


@RequestConverterFactory.register(EndpointType.OPENAI_EMBEDDINGS)
class OpenAIEmbeddingsRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI Embeddings requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for an embeddings request."""

        payload = {
            "input": turn.text,
            "model": model_endpoint.primary_model_name,
            "dimensions": model_endpoint.endpoint.url_params.get("dimensions", 1536)
            if model_endpoint.endpoint.url_params
            else 1536,
            "encoding_format": model_endpoint.endpoint.url_params.get(
                "encoding_format", "float"
            )
            if model_endpoint.endpoint.url_params
            else "float",
            "user": model_endpoint.endpoint.url_params.get("user", "")
            if model_endpoint.endpoint.url_params
            else "",
            "stream": model_endpoint.endpoint.streaming,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload


@RequestConverterFactory.register(EndpointType.OPENAI_RESPONSES)
class OpenAIResponsesRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI Responses requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for a responses request."""

        payload = {
            "input": turn.text,
            "model": model_endpoint.primary_model_name,
            "max_output_tokens": model_endpoint.endpoint.url_params.get(
                "max_output_tokens", 1000
            )
            if model_endpoint.endpoint.url_params
            else 1000,
            "stream": model_endpoint.endpoint.streaming,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload
