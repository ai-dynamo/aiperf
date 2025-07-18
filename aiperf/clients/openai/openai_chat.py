# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from aiperf.clients.client_interfaces import (
    RequestConverterFactory,
    RequestConverterProtocol,
)
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import EndpointType
from aiperf.common.models import Turn

DEFAULT_ROLE = "user"


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

        # TODO: Do we need to support image and audio inputs?
        messages = [
            {
                "role": turn.role or DEFAULT_ROLE,
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
